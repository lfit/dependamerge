# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The merge dispatch retry loop.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio

from ..github_async import PermissionError as GitHubPermissionError
from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._models import (
    _merge_already_in_progress,
)


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
                    should_retry = await self._handle_merge_failure(
                        pr_info, owner, repo
                    )
                    if should_retry:
                        self.log.info(
                            f"Retrying merge for PR {owner}/{repo}#{pr_info.number} (attempt {attempt + 2})"
                        )
                        continue
                    else:
                        self.log.info(
                            f"Not retrying PR {owner}/{repo}#{pr_info.number} - no fixable issues found"
                        )
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

                # Store exception for better error reporting
                pr_key = f"{owner}/{repo}#{pr_info.number}"
                self._last_merge_exception[pr_key] = e
                self._last_merge_exception_head[pr_key] = pr_info.head_sha
                self.log.debug(
                    f"Stored exception for {pr_key}: {type(e).__name__}: {str(e)[:200]}"
                )

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

                # Enhanced error handling with specific status code checks
                if "405" in error_msg and "Method Not Allowed" in error_msg:
                    # Don't log here - will be handled in failure summary
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
                            continue
                        else:
                            break
                    if "behind" in error_msg.lower() and self.fix_out_of_date:
                        # Allow retry for behind PRs
                        pass
                    elif pr_info.mergeable_state in ("clean", "unstable"):
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
                            await self._refresh_pr_mergeable(
                                owner, repo, pr_info, pr_key
                            )
                            continue
                        else:
                            break
                    elif pr_info.mergeable_state == "blocked":
                        # If we just approved this PR, the branch protection
                        # evaluator may not have caught up yet.  Re-fetch the
                        # PR state and, if it has become "clean", allow a retry
                        # instead of giving up immediately.
                        if (
                            pr_key in self._recently_approved
                            and attempt < self.max_retries
                        ):
                            if await self._blocked_pr_became_clean(
                                owner, repo, pr_info, pr_key
                            ):
                                # Approval has propagated — retry the merge
                                continue
                            self._recently_approved.discard(pr_key)
                        # Still blocked after re-check (or not recently approved)
                        break
                    else:
                        # Don't retry 405 errors unless they're "behind" issues
                        break
                elif "403" in error_msg and "Forbidden" in error_msg:
                    break
                elif "422" in error_msg:
                    break
                else:
                    # Only log for debugging purposes
                    self.log.debug(
                        f"Merge attempt {attempt + 1} failed for PR {owner}/{repo}#{pr_info.number}: {e}"
                    )

                if attempt >= self.max_retries:
                    break

                # Don't retry certain error types that are unlikely to be transient
                # Exception: Allow retry for 405 errors on "behind" PRs if fix_out_of_date is enabled
                if ("405" in error_msg and "behind" not in error_msg.lower()) or (
                    "422" in error_msg and "not mergeable" in error_msg.lower()
                ):
                    self.log.info(
                        f"Not retrying PR {owner}/{repo}#{pr_info.number} due to permanent error condition"
                    )
                    break
                elif (
                    "405" in error_msg
                    and "behind" in error_msg.lower()
                    and not self.fix_out_of_date
                ):
                    self.log.info(
                        f"Not retrying PR {owner}/{repo}#{pr_info.number} - behind base branch but --no-fix is set"
                    )
                    break

                # Wait a bit before retrying
                await asyncio.sleep(1.0)

        self.log.debug(
            f"_merge_pr_with_retry returning False for {owner}/{repo}#{pr_info.number} after all retries"
        )
        return False
