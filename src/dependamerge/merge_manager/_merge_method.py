# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Repository merge method, failure handling, and dispatch locking.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio

from ..bot_identity import is_dependabot
from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _MergeMethodMixin(_MergeManagerBase):
    """Repository merge method, failure handling, and dispatch locking."""

    async def _get_merge_method_for_repo(self, owner: str, repo: str) -> str:
        """
        Get the appropriate merge method for a specific repository based on branch protection settings.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Merge method to use: "merge", "squash", or "rebase"
        """
        if not self._github_service:
            self.log.warning("GitHubService not available, using default merge method")
            return self.default_merge_method

        try:
            protection_settings = (
                await self._github_service.get_branch_protection_settings(
                    owner, repo, "main"
                )
            )

            # Determine appropriate merge method
            merge_method = self._github_service.determine_merge_method(
                protection_settings, self.default_merge_method
            )

            if merge_method != self.default_merge_method:
                self.log.debug(
                    f"Repository {owner}/{repo} requires '{merge_method}' merge method "
                    f"(protection: requiresLinearHistory={protection_settings and protection_settings.get('requiresLinearHistory', False)})"
                )

            return merge_method

        except Exception as e:
            self.log.warning(
                f"Failed to determine merge method for {owner}/{repo}, using default '{self.default_merge_method}': {e}"
            )
            return self.default_merge_method

    async def _handle_merge_failure(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """
        Handle a merge failure and determine if we should retry.

        Args:
            pr_info: Pull request information
            owner: Repository owner
            repo: Repository name

        Returns:
            True if we should retry, False otherwise
        """
        if not self._github_client:
            return False

        # Check if the branch is out of date and we can fix it
        if self.fix_out_of_date and pr_info.mergeable_state == "behind":
            if is_dependabot(pr_info.author):
                # Prefer the ``@dependabot rebase`` macro over REST
                # ``update-branch``: the REST endpoint creates an
                # unsigned merge commit that can violate
                # signature-requiring branch protection, while
                # dependabot force-pushes a freshly signed rebase.
                # The macro completes asynchronously (minutes), so an
                # immediate retry is pointless — arm auto-merge so
                # GitHub finishes the merge server-side once the
                # rebase lands and checks pass, and let the caller's
                # not-merged classification report AUTO_MERGE_PENDING.
                self.log.info(
                    f"PR {owner}/{repo}#{pr_info.number} is behind - "
                    "requesting dependabot rebase"
                )
                if self._dependabot_is_rebasing(
                    pr_info.body
                ) or await self._request_dependabot_rebase(pr_info, owner, repo):
                    await self._enable_auto_merge_with_approval(pr_info, owner, repo)
                return False
            try:
                self.log.info(
                    f"PR {owner}/{repo}#{pr_info.number} is behind - updating branch"
                )
                await self._github_client.update_branch(owner, repo, pr_info.number)
                self._record_rebase()
                # Wait a moment for GitHub to process the update
                await asyncio.sleep(min(2.0, self._merge_recheck_interval))
                return True
            except Exception as e:
                self.log.error(
                    f"Failed to update branch for PR {owner}/{repo}#{pr_info.number}: {e}"
                )

        # For other failure types, don't retry
        return False

    async def _get_merge_dispatch_lock(self, owner: str, repo: str) -> asyncio.Lock:
        """Return the ``asyncio.Lock`` that serialises merge dispatch for ``owner/repo``.

        The lock is created lazily on first request and shared by
        every worker targeting the same repository.  Workers
        targeting different repositories receive distinct locks and
        can dispatch in parallel.

        Holding this lock around the actual ``merge_pull_request``
        API call (and its retry loop) prevents back-to-back merges
        on the same repo from racing GitHub's branch-protection
        propagation, while leaving every other phase of the merge
        flow — approve, rebase polling, Step 5.5's auto-merge wait —
        free to run in parallel across workers.
        """
        key = f"{owner}/{repo}"
        async with self._merge_dispatch_locks_lock:
            lock = self._merge_dispatch_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._merge_dispatch_locks[key] = lock
            return lock
