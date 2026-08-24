# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The retry loop around the merge call itself.
"""

from __future__ import annotations

import asyncio
from enum import Enum

from ..github_async import PermissionError as GitHubPermissionError
from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._types import _merge_already_in_progress


class _RetryDecision(Enum):
    """What the retry loop should do after an attempt failed.

    ``BACKOFF`` is the "no decision yet" answer: the error was not
    conclusive, so the shared permanent-error checks and the one-second
    delay at the foot of the loop body decide instead.
    """

    MERGED = "merged"
    RETRY = "retry"
    STOP = "stop"
    BACKOFF = "backoff"


class _MergeRetryMixin(_MergeManagerBase):
    """Retrying a merge through its transient failure modes."""

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

                if await self._dispatch_merge(pr_info, owner, repo):
                    return True

                # Merge failed, check if we can fix it
                self.log.warning(
                    f"⚠️ Merge API returned false for PR {owner}/{repo}#{pr_info.number} (attempt {attempt + 1})"
                )
                if attempt < self.max_retries:
                    decision = await self._retry_after_false_merge(
                        pr_info, owner, repo, attempt
                    )
                    if decision is _RetryDecision.STOP:
                        break
                    continue

            except GitHubPermissionError:
                # Token cannot merge on this repo — propagate to the
                # caller so the PermissionError handler in
                # ``_merge_single_pr`` reports it cleanly and records
                # the repo for fast-fail of remaining PRs in the
                # batch.  Retrying or breaking silently here would
                # downgrade the failure into a generic
                # "merge failed: clean" reason that misleads users.
                raise
            except Exception as exc:
                decision = await self._handle_merge_exception(
                    pr_info, owner, repo, exc, attempt
                )
                if decision is _RetryDecision.MERGED:
                    return True
                if decision is _RetryDecision.STOP:
                    break

        self.log.debug(
            f"_merge_pr_with_retry returning False for {owner}/{repo}#{pr_info.number} after all retries"
        )
        return False

    async def _dispatch_merge(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Call the merge API with the method chosen for this repository."""
        if not self._github_client:
            raise RuntimeError("GitHub client not initialized")

        # Use pre-determined merge method for this repository
        cache_key = f"{owner}/{repo}"
        merge_method = self._pr_merge_methods.get(cache_key, self.default_merge_method)

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
        return merged

    async def _retry_after_false_merge(
        self, pr_info: PullRequestInfo, owner: str, repo: str, attempt: int
    ) -> _RetryDecision:
        """Decide whether a merge that returned false is worth another go."""
        should_retry = await self._handle_merge_failure(pr_info, owner, repo)
        if should_retry:
            self.log.info(
                f"Retrying merge for PR {owner}/{repo}#{pr_info.number} (attempt {attempt + 2})"
            )
            return _RetryDecision.RETRY
        self.log.info(
            f"Not retrying PR {owner}/{repo}#{pr_info.number} - no fixable issues found"
        )
        return _RetryDecision.STOP

    async def _handle_merge_exception(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        exc: Exception,
        attempt: int,
    ) -> _RetryDecision:
        """Classify a merge exception into the loop's next move."""
        error_msg = str(exc)

        # Store exception for better error reporting
        pr_key = f"{owner}/{repo}#{pr_info.number}"
        self._last_merge_exception[pr_key] = exc
        self._last_merge_exception_head[pr_key] = pr_info.head_sha
        self.log.debug(
            f"Stored exception for {pr_key}: {type(exc).__name__}: {str(exc)[:200]}"
        )

        # "Merge already in progress" is unambiguous on its own,
        # so it is matched *before* the 405 branch rather than
        # inside it.  Gating it on the literal "Method Not
        # Allowed" text as well would mean an upstream rewording
        # silently reinstated the fail-after-six-seconds
        # behaviour this handler exists to remove --- the very
        # fragility ``_merge_already_in_progress`` guards against.
        if _merge_already_in_progress(error_msg):
            return await self._watch_in_progress_merge(pr_info, owner, repo, pr_key)

        # Enhanced error handling with specific status code checks
        if "405" in error_msg and "Method Not Allowed" in error_msg:
            decision = await self._decide_after_405(
                pr_info, owner, repo, error_msg, attempt
            )
            if decision is not _RetryDecision.BACKOFF:
                return decision
        elif "403" in error_msg and "Forbidden" in error_msg:
            return _RetryDecision.STOP
        elif "422" in error_msg:
            return _RetryDecision.STOP
        else:
            # Only log for debugging purposes
            self.log.debug(
                f"Merge attempt {attempt + 1} failed for PR {owner}/{repo}#{pr_info.number}: {exc}"
            )

        return await self._backoff_before_retry(
            pr_info, owner, repo, error_msg, attempt
        )

    async def _watch_in_progress_merge(
        self, pr_info: PullRequestInfo, owner: str, repo: str, pr_key: str
    ) -> _RetryDecision:
        """Wait out a merge GitHub has already accepted for this PR.

        Typically auto-merge armed earlier in this run, and GitHub is
        completing it.  Dispatching again is pointless and the previous
        short backoff --- 3s then 6s --- expired well before GitHub
        finished, so these were reported as failures despite merging
        moments later: 4 of the 5 such PRs in the run analysed in
        ``docs/BULK_RUN_PERFORMANCE_AUDIT.md`` had merged.  Watch for
        completion instead.
        """
        if await self._await_in_progress_merge(owner, repo, pr_info, pr_key):
            return _RetryDecision.MERGED
        return _RetryDecision.STOP

    async def _decide_after_405(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        error_msg: str,
        attempt: int,
    ) -> _RetryDecision:
        """Sort a 405 into its transient causes and its permanent ones."""
        # Don't log here - will be handled in failure summary
        pr_key = f"{owner}/{repo}#{pr_info.number}"
        if "base branch was modified" in error_msg.lower():
            return await self._retry_after_base_branch_moved(pr_key, attempt)
        if "behind" in error_msg.lower() and self.fix_out_of_date:
            # Allow retry for behind PRs
            return _RetryDecision.BACKOFF
        if pr_info.mergeable_state in ("clean", "unstable"):
            return await self._retry_transient_405(
                pr_info, owner, repo, pr_key, attempt
            )
        if pr_info.mergeable_state == "blocked":
            return await self._retry_blocked_405(pr_info, owner, repo, pr_key, attempt)
        # Don't retry 405 errors unless they're "behind" issues
        return _RetryDecision.STOP

    async def _retry_after_base_branch_moved(
        self, pr_key: str, attempt: int
    ) -> _RetryDecision:
        """Let GitHub recompute against a base a sibling merge advanced.

        Pure concurrency race: in a same-repo batch a sibling PR merged
        and advanced the base branch between GitHub computing this PR's
        merge commit and applying it, so GitHub returns 405 "Base branch
        was modified. Review and try the merge again."  It is always
        transient and unrelated to the PR's own mergeability (no rebase
        or approval is needed), so a short delay lets GitHub recompute
        against the new base head, then we retry.
        """
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

    async def _retry_transient_405(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        pr_key: str,
        attempt: int,
    ) -> _RetryDecision:
        """Retry a 405 on a PR that should have been mergeable.

        This is a transient API error (often follows a 502 during
        GitHub degradation).  Re-fetch state and retry.
        """
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

    async def _retry_blocked_405(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        pr_key: str,
        attempt: int,
    ) -> _RetryDecision:
        """Give a just-granted approval time to reach branch protection.

        If we just approved this PR, the branch protection evaluator may
        not have caught up yet.  Re-fetch the PR state and, if it has
        become "clean", allow a retry instead of giving up immediately.
        """
        if pr_key in self._recently_approved and attempt < self.max_retries:
            if await self._blocked_pr_became_clean(owner, repo, pr_info, pr_key):
                # Approval has propagated — retry the merge
                return _RetryDecision.RETRY
            self._recently_approved.discard(pr_key)
        # Still blocked after re-check (or not recently approved)
        return _RetryDecision.STOP

    async def _backoff_before_retry(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        error_msg: str,
        attempt: int,
    ) -> _RetryDecision:
        """Pause before another attempt, unless the error is permanent."""
        if attempt >= self.max_retries:
            return _RetryDecision.STOP

        # Don't retry certain error types that are unlikely to be transient
        # Exception: Allow retry for 405 errors on "behind" PRs if fix_out_of_date is enabled
        if ("405" in error_msg and "behind" not in error_msg.lower()) or (
            "422" in error_msg and "not mergeable" in error_msg.lower()
        ):
            self.log.info(
                f"Not retrying PR {owner}/{repo}#{pr_info.number} due to permanent error condition"
            )
            return _RetryDecision.STOP
        elif (
            "405" in error_msg
            and "behind" in error_msg.lower()
            and not self.fix_out_of_date
        ):
            self.log.info(
                f"Not retrying PR {owner}/{repo}#{pr_info.number} - behind base branch but --no-fix is set"
            )
            return _RetryDecision.STOP

        # Wait a bit before retrying
        await asyncio.sleep(1.0)
        return _RetryDecision.RETRY
