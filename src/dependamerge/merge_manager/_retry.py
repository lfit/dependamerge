# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The merge dispatch retry loop.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from enum import Enum, auto

from ..github_async import PermissionError as GitHubPermissionError
from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._models import (
    _merge_already_in_progress,
)


class _RetryDecision(Enum):
    """
    What the retry loop should do once an error has been classified.

    A classifier cannot ``continue`` or ``break`` on the loop's behalf,
    so it returns one of these members instead and the loop keeps its
    own control flow visible.  A classifier returning ``None`` has
    declined to settle the matter, and the loop falls through to the
    shared tail that decides between backing off and giving up.
    """

    RETRY = auto()
    STOP = auto()


class _RetryMixin(_MergeManagerBase):
    """The merge dispatch retry loop."""

    async def _merge_pr_with_retry(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """
        Merge a PR with retry logic for transient failures.

        Args:
            pr_info: Pull request information
            owner: Repository owner
            repo: Repository name

        Returns:
            True if merged successfully, False otherwise
        """
        if not self._github_client:
            raise RuntimeError("GitHub client not initialized")

        for attempt in range(self.max_retries + 1):
            try:
                # Check if PR has already been closed/merged before attempting
                if attempt > 0:
                    recheck = await self._recheck_pr_before_retry(
                        owner, repo, pr_info, attempt
                    )
                    if recheck is not None:
                        return recheck

                # Use pre-determined merge method for this repository
                cache_key = f"{owner}/{repo}"
                merge_method = self._pr_merge_methods.get(
                    cache_key, self.default_merge_method
                )

                # Attempt the merge
                self.log.debug(
                    f"Attempting merge for {owner}/{repo}#{pr_info.number} with method={merge_method}"
                )
                merged = await self._github_client.merge_pull_request(
                    owner, repo, pr_info.number, merge_method
                )
                self.log.debug(
                    f"Merge API returned {merged} for {owner}/{repo}#{pr_info.number}"
                )

                if merged:
                    return True

                # Merge failed, check if we can fix it
                self.log.warning(
                    f"⚠️ Merge API returned false for PR {owner}/{repo}#{pr_info.number} (attempt {attempt + 1})"
                )
                if attempt < self.max_retries:
                    if await self._retry_after_merge_failure(
                        pr_info, owner, repo, attempt
                    ):
                        continue
                    break

            except GitHubPermissionError:
                # Token cannot merge on this repo — propagate to the
                # caller so the PermissionError handler in
                # ``_merge_single_pr`` reports it cleanly and records
                # the repo for fast-fail of remaining PRs in the
                # batch.  Retrying or breaking silently here would
                # downgrade the failure into a generic
                # "merge failed: clean" reason that misleads users.
                raise
            except Exception as e:
                error_msg = str(e)
                pr_key = self._record_merge_exception(pr_info, owner, repo, e)

                # "Merge already in progress" is unambiguous on its own,
                # so it is matched *before* the 405 branch rather than
                # inside it.  Gating it on the literal "Method Not
                # Allowed" text as well would mean an upstream rewording
                # silently reinstated the fail-after-six-seconds
                # behaviour this handler exists to remove --- the very
                # fragility ``_merge_already_in_progress`` guards against.
                if _merge_already_in_progress(error_msg):
                    # GitHub has already accepted a merge for this PR
                    # (typically auto-merge armed earlier in this run)
                    # and is completing it.  Dispatching again is
                    # pointless and the previous short backoff --- 3s
                    # then 6s --- expired well before GitHub finished,
                    # so these were reported as failures despite
                    # merging moments later: 4 of the 5 such PRs in
                    # the run analysed in
                    # ``docs/BULK_RUN_PERFORMANCE_AUDIT.md`` had
                    # merged.  Watch for completion instead.
                    if await self._await_in_progress_merge(
                        owner, repo, pr_info, pr_key
                    ):
                        return True
                    break

                decision = await self._classify_merge_error(
                    error_msg, pr_info, owner, repo, attempt
                )
                if decision is _RetryDecision.RETRY:
                    continue
                if decision is _RetryDecision.STOP:
                    break

                if self._error_ends_retries(error_msg, pr_info, owner, repo, attempt):
                    break

                # Wait a bit before retrying
                await asyncio.sleep(1.0)

        self.log.debug(
            f"_merge_pr_with_retry returning False for {owner}/{repo}#{pr_info.number} after all retries"
        )
        return False

    async def _retry_after_merge_failure(
        self, pr_info: PullRequestInfo, owner: str, repo: str, attempt: int
    ) -> bool:
        """
        Decide whether a merge that returned false is worth retrying.

        Split out so the "is anything fixable?" question and the two
        log lines that announce its answer sit together, while the
        caller keeps the ``continue``/``break`` and the loop's control
        flow stays visible.

        Args:
            pr_info: Pull request information
            owner: Repository owner
            repo: Repository name
            attempt: Zero-based index of the attempt that just failed

        Returns:
            True to retry the merge, False to stop trying
        """
        should_retry = await self._handle_merge_failure(pr_info, owner, repo)
        if should_retry:
            self.log.info(
                f"Retrying merge for PR {owner}/{repo}#{pr_info.number} (attempt {attempt + 2})"
            )
            return True
        self.log.info(
            f"Not retrying PR {owner}/{repo}#{pr_info.number} - no fixable issues found"
        )
        return False

    def _record_merge_exception(
        self, pr_info: PullRequestInfo, owner: str, repo: str, error: Exception
    ) -> str:
        """
        Record a failed merge attempt's exception for later reporting.

        Every attempt records the exception and the head SHA it applied
        to before the error is classified, so the failure summary can
        explain the outcome even when a later attempt takes a different
        path out of the loop.  Kept separate because it is bookkeeping
        rather than part of the retry decision.

        Args:
            pr_info: Pull request information
            owner: Repository owner
            repo: Repository name
            error: Exception raised by the merge attempt

        Returns:
            The ``owner/repo#number`` key the exception was stored under
        """
        # Store exception for better error reporting
        pr_key = f"{owner}/{repo}#{pr_info.number}"
        self._last_merge_exception[pr_key] = error
        self._last_merge_exception_head[pr_key] = pr_info.head_sha
        self.log.debug(
            f"Stored exception for {pr_key}: {type(error).__name__}: {str(error)[:200]}"
        )
        return pr_key

    async def _classify_merge_error(
        self,
        error_msg: str,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        attempt: int,
    ) -> _RetryDecision | None:
        """
        Map a failed merge attempt's error onto a retry decision.

        Enhanced error handling with specific status code checks: 405
        carries several distinct meanings and gets its own classifier,
        while 403 and 422 are terminal.  Anything else is left
        undecided so the caller's retry budget and backoff apply.

        Args:
            error_msg: Text of the exception raised by the merge attempt
            pr_info: Pull request information
            owner: Repository owner
            repo: Repository name
            attempt: Zero-based index of the attempt that just failed

        Returns:
            RETRY or STOP when the status code settles the matter, None
            when it does not
        """
        # Enhanced error handling with specific status code checks
        if "405" in error_msg and "Method Not Allowed" in error_msg:
            # Don't log here - will be handled in failure summary
            return await self._classify_method_not_allowed(
                error_msg, pr_info, owner, repo, attempt
            )
        if "403" in error_msg and "Forbidden" in error_msg:
            return _RetryDecision.STOP
        if "422" in error_msg:
            return _RetryDecision.STOP

        # Only log for debugging purposes
        self.log.debug(
            f"Merge attempt {attempt + 1} failed for PR {owner}/{repo}#{pr_info.number}: {error_msg}"
        )
        return None

    async def _classify_method_not_allowed(
        self,
        error_msg: str,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        attempt: int,
    ) -> _RetryDecision | None:
        """
        Decide what a 405 "Method Not Allowed" rejection means here.

        GitHub reuses 405 for a concurrency race on the base branch, for
        a PR that is behind its base, for a transient refusal of a PR
        that ought to merge, and for a genuinely blocked one.  Each
        wants a different delay and a different state refresh, so the
        whole sub-classification lives here rather than nested three
        levels deep inside the retry loop.

        Args:
            error_msg: Text of the exception raised by the merge attempt
            pr_info: Pull request information
            owner: Repository owner
            repo: Repository name
            attempt: Zero-based index of the attempt that just failed

        Returns:
            RETRY or STOP when the rejection settles the matter, None
            for a behind PR that ``fix_out_of_date`` may still rescue
        """
        pr_key = f"{owner}/{repo}#{pr_info.number}"

        if "base branch was modified" in error_msg.lower():
            # Pure concurrency race: in a same-repo batch a
            # sibling PR merged and advanced the base branch
            # between GitHub computing this PR's merge commit
            # and applying it, so GitHub returns 405 "Base
            # branch was modified. Review and try the merge
            # again."  It is always transient and unrelated to
            # the PR's own mergeability (no rebase or approval
            # is needed), so a short delay lets GitHub recompute
            # against the new base head, then we retry.
            if attempt < self.max_retries:
                retry_delay = 2.0 * (attempt + 1)
                self.log.info(
                    f"Base branch moved under {pr_key} (concurrent "
                    f"merge); waiting {retry_delay}s before retry "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})…"
                )
                await asyncio.sleep(retry_delay)
                return _RetryDecision.RETRY
            return _RetryDecision.STOP

        if "behind" in error_msg.lower() and self.fix_out_of_date:
            # Allow retry for behind PRs
            return None

        if pr_info.mergeable_state in ("clean", "unstable"):
            # The PR should be mergeable but GitHub returned 405 —
            # this is a transient API error (often follows a 502
            # during GitHub degradation).  Re-fetch state and retry.
            if attempt < self.max_retries:
                retry_delay = 3.0 * (attempt + 1)
                self.log.info(
                    f"Transient 405 on mergeable PR {pr_key} "
                    f"(state={pr_info.mergeable_state}), "
                    f"waiting {retry_delay}s before retry "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})…"
                )
                await asyncio.sleep(retry_delay)
                # Refresh PR state in case something changed
                await self._refresh_pr_mergeable(owner, repo, pr_info, pr_key)
                return _RetryDecision.RETRY
            return _RetryDecision.STOP

        if pr_info.mergeable_state == "blocked":
            # If we just approved this PR, the branch protection
            # evaluator may not have caught up yet.  Re-fetch the
            # PR state and, if it has become "clean", allow a retry
            # instead of giving up immediately.
            if pr_key in self._recently_approved and attempt < self.max_retries:
                if await self._blocked_pr_became_clean(owner, repo, pr_info, pr_key):
                    # Approval has propagated — retry the merge
                    return _RetryDecision.RETRY
                self._recently_approved.discard(pr_key)
            # Still blocked after re-check (or not recently approved)
            return _RetryDecision.STOP

        # Don't retry 405 errors unless they're "behind" issues
        return _RetryDecision.STOP

    def _error_ends_retries(
        self,
        error_msg: str,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        attempt: int,
    ) -> bool:
        """
        Decide whether an unclassified error ends the retry loop.

        Reached once the status-code classification has declined to
        settle the matter.  It applies the retry budget and the
        permanent error conditions that would make another dispatch
        pointless, so the caller is left with a single backoff.

        Args:
            error_msg: Text of the exception raised by the merge attempt
            pr_info: Pull request information
            owner: Repository owner
            repo: Repository name
            attempt: Zero-based index of the attempt that just failed

        Returns:
            True to stop retrying, False to back off and try again
        """
        if attempt >= self.max_retries:
            return True

        # Don't retry certain error types that are unlikely to be transient
        # Exception: Allow retry for 405 errors on "behind" PRs if fix_out_of_date is enabled
        if ("405" in error_msg and "behind" not in error_msg.lower()) or (
            "422" in error_msg and "not mergeable" in error_msg.lower()
        ):
            self.log.info(
                f"Not retrying PR {owner}/{repo}#{pr_info.number} due to permanent error condition"
            )
            return True
        if (
            "405" in error_msg
            and "behind" in error_msg.lower()
            and not self.fix_out_of_date
        ):
            self.log.info(
                f"Not retrying PR {owner}/{repo}#{pr_info.number} - behind base branch but --no-fix is set"
            )
            return True

        return False
