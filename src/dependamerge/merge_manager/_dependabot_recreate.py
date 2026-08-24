# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Asking Dependabot to recreate a pull request from scratch.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..bot_identity import is_dependabot
from ..models import PullRequestInfo
from ._base import _MergeManagerBase


def _has_recreate_comment(comments: Any) -> bool:
    """Report whether a comment listing already asks for a recreate."""
    if not isinstance(comments, list):
        return False
    for c in comments:
        if not isinstance(c, dict):
            continue
        body = c.get("body")
        if isinstance(body, str) and "@dependabot recreate" in body:
            return True
    return False


class _DependabotRecreateMixin(_MergeManagerBase):
    """Driving the ``@dependabot recreate`` command."""

    async def _trigger_dependabot_recreate(
        self, pr_info: PullRequestInfo
    ) -> PullRequestInfo | None:
        """
        Detect an unsigned dependabot commit and ask dependabot to recreate
        the pull request so that the new commit is properly signed.

        When a repository's branch protection requires commit signatures,
        dependabot PRs can end up with unverified commits (e.g. after a
        rebase or force-push by GitHub).  Posting ``@dependabot recreate``
        causes dependabot to close the current PR and open a fresh one
        whose commit is signed by GitHub.

        Args:
            pr_info: Pull request information for the failing PR.

        Returns:
            A new ``PullRequestInfo`` for the recreated PR if the recreate
            was triggered, the old PR was closed, and a replacement was
            found.  Returns ``None`` if the recreate was not applicable or
            did not succeed within the polling window.
        """
        if not self._github_client:
            return None

        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)

        # 1. Only applies to dependabot PRs
        if not is_dependabot(pr_info.author):
            return None

        # 2. Check whether the branch requires signed commits
        if not await self._recreate_branch_requires_signatures(
            repo_owner, repo_name, pr_info
        ):
            return None

        # 3. Check whether any commits are unverified
        unverified_shas = await self._recreate_unverified_shas(
            repo_owner, repo_name, pr_info
        )
        if unverified_shas is None:
            return None

        # 4. Guard against duplicate recreate comments
        if await self._recreate_already_requested(repo_owner, repo_name, pr_info):
            return None

        # 5. Post the recreate comment
        self._pr_status(
            f"🔄 Requesting dependabot recreate: {pr_info.html_url} "
            f"[unverified commits: {', '.join(unverified_shas)}]",
            level="info",
        )

        if not await self._post_recreate_comment(repo_owner, repo_name, pr_info):
            return None

        # 6. Poll for the old PR to close and a replacement to appear.
        return await self._await_dependabot_recreate(repo_owner, repo_name, pr_info)

    async def _recreate_branch_requires_signatures(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> bool:
        """Report whether the base branch mandates signed commits.

        Returns False — suppressing the recreate — when the requirement
        cannot be determined.
        """
        if not self._github_client:
            return False
        try:
            requires_signatures = await self._github_client.requires_commit_signatures(
                repo_owner, repo_name, pr_info.base_branch or "main"
            )
            if not requires_signatures:
                self.log.debug(
                    "Branch %s/%s:%s does not require commit signatures; "
                    "skipping dependabot recreate.",
                    repo_owner,
                    repo_name,
                    pr_info.base_branch or "main",
                )
                return False
        except Exception as e:
            self.log.debug(
                "Could not determine signature requirement for %s: %s",
                pr_info.repository_full_name,
                e,
            )
            return False
        return True

    async def _recreate_unverified_shas(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> list[str] | None:
        """Return the unverified commit SHAs, or None when there is nothing to do.

        None covers both "every commit is verified" and "the signature
        state could not be read"; either way no recreate is warranted.
        """
        if not self._github_client:
            return None
        try:
            (
                all_verified,
                unverified_shas,
            ) = await self._github_client.check_pr_commit_signatures(
                repo_owner, repo_name, pr_info.number
            )
            if all_verified:
                self.log.debug(
                    "All commits on %s#%s are verified; recreate not needed.",
                    pr_info.repository_full_name,
                    pr_info.number,
                )
                return None
        except Exception as e:
            self.log.debug(
                "Could not check commit signatures for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                e,
            )
            return None
        return unverified_shas

    async def _recreate_already_requested(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> bool:
        """Report whether a recreate comment should be suppressed.

        True when one has already been posted, and also when the comment
        listing could not be read: posting blind risks a duplicate.
        """
        if not self._github_client:
            return True
        try:
            comments = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/issues/{pr_info.number}/comments"
                f"?per_page=100&direction=desc"
            )
            if _has_recreate_comment(comments):
                self.log.info(
                    "Found existing @dependabot recreate comment on "
                    "%s#%s; skipping duplicate.",
                    pr_info.repository_full_name,
                    pr_info.number,
                )
                return True
        except Exception as e:
            self.log.warning(
                "Could not list comments for %s#%s to check for existing "
                "@dependabot recreate comment: %s",
                pr_info.repository_full_name,
                pr_info.number,
                e,
            )
            return True
        return False

    async def _post_recreate_comment(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> bool:
        """Post ``@dependabot recreate``, reporting whether it was accepted."""
        if not self._github_client:
            return False
        try:
            await self._github_client.post_issue_comment(
                repo_owner, repo_name, pr_info.number, "@dependabot recreate"
            )
            self._record_retrigger()
        except Exception as e:
            self.log.warning(
                "Failed to post @dependabot recreate comment on %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                e,
            )
            return False
        return True

    async def _await_dependabot_recreate(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> PullRequestInfo | None:
        """Wait for the old PR to close and its replacement to appear.

        Dependabot typically responds within 30-90 seconds.  We poll
        using the centralised merge timeout.  The whole poll (including
        the nested recreated-PR checks wait) is a wait on dependabot +
        CI, so the worker's concurrency slot is released for its
        duration (``parked()``).
        """
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        max_polls = self._merge_poll_max_attempts
        old_pr_closed = False

        async with _mm.parked():
            for attempt in range(max_polls):
                await asyncio.sleep(self._merge_recheck_interval)

                # 6a. Check if the old PR has been closed
                if not old_pr_closed:
                    old_pr_closed = await self._old_pr_is_closed(
                        repo_owner, repo_name, pr_info, attempt
                    )

                # 6b. Once the old PR is closed, look for the replacement
                if old_pr_closed:
                    resolved, new_pr_info = await self._find_recreated_pr(
                        repo_owner, repo_name, pr_info
                    )
                    # Always return after the first wait attempt to avoid
                    # performing multiple long waits for the same PR.
                    if resolved:
                        return new_pr_info

                if attempt % 3 == 2:
                    self.log.debug(
                        "Still waiting for dependabot recreate on %s#%s (%.0fs elapsed, old_pr_closed=%s)",
                        pr_info.repository_full_name,
                        pr_info.number,
                        (attempt + 1) * self._merge_recheck_interval,
                        old_pr_closed,
                    )

        self.log.warning(
            "Timed out waiting for dependabot to recreate %s#%s",
            pr_info.repository_full_name,
            pr_info.number,
        )
        return None

    async def _old_pr_is_closed(
        self,
        repo_owner: str,
        repo_name: str,
        pr_info: PullRequestInfo,
        attempt: int,
    ) -> bool:
        """Report whether dependabot has closed the superseded PR yet."""
        if not self._github_client:
            return False
        try:
            old_pr_data = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/pulls/{pr_info.number}"
            )
            if isinstance(old_pr_data, dict) and old_pr_data.get("state") == "closed":
                self._pr_status(
                    f"✅ Old PR closed by dependabot: "
                    f"{pr_info.html_url} "
                    f"({(attempt + 1) * self._merge_recheck_interval:.0f}s elapsed)",
                    level="info",
                )
                return True
        except Exception as e:
            self.log.debug(
                "Error polling old PR state for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                e,
            )
        return False

    async def _find_recreated_pr(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> tuple[bool, PullRequestInfo | None]:
        """Look for the replacement PR and wait for its checks.

        Returns a ``(resolved, pr_info)`` pair.  ``resolved`` is True once
        a replacement has been found and waited on, whatever the outcome
        of that wait; the caller then stops polling.
        """
        if not self._github_client:
            return False, None
        try:
            # Search for open PRs from dependabot on the same head branch
            prs = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/pulls"
                f"?state=open&head={repo_owner}:{pr_info.head_branch}&per_page=5"
            )
            if isinstance(prs, list):
                for pr_data in prs:
                    new_number = self._recreated_pr_number(pr_data, pr_info)
                    if new_number is None:
                        continue

                    # Found a replacement — now wait for checks to pass
                    new_pr_info = await self._wait_for_recreated_pr_checks(
                        repo_owner, repo_name, new_number, pr_data
                    )
                    return True, new_pr_info
        except Exception as e:
            self.log.debug(
                "Error searching for replacement PR for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                e,
            )
        return False, None

    def _recreated_pr_number(self, pr_data: Any, pr_info: PullRequestInfo) -> Any:
        """Return the PR number when ``pr_data`` is a viable replacement.

        The number is handed back exactly as the API reported it, rather
        than coerced, so the value reaching the checks wait is the one
        the unrefactored code passed on.  None means "not a replacement".
        """
        if not isinstance(pr_data, dict):
            return None

        pr_author = (pr_data.get("user") or {}).get("login", "")
        if not is_dependabot(pr_author):
            return None

        new_number = pr_data.get("number")
        if new_number is None or new_number == pr_info.number:
            return None

        # Verify the replacement targets the same base branch
        new_base = (pr_data.get("base") or {}).get("ref", "")
        if new_base != (pr_info.base_branch or "main"):
            self.log.debug(
                "Skipping candidate PR #%s: targets %s, expected %s",
                new_number,
                new_base,
                pr_info.base_branch or "main",
            )
            return None

        return new_number
