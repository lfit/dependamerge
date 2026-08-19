# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Asking dependabot to recreate a pull request whose checks stuck.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio

import dependamerge.merge_manager as _pkg

from ..bot_identity import is_dependabot
from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _DependabotRecreateMixin(_MergeManagerBase):
    """Asking dependabot to recreate a pull request whose checks stuck."""

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
                return None
        except Exception as e:
            self.log.debug(
                "Could not determine signature requirement for %s: %s",
                pr_info.repository_full_name,
                e,
            )
            return None

        # 3. Check whether any commits are unverified
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

        # 4. Guard against duplicate recreate comments
        try:
            comments = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/issues/{pr_info.number}/comments"
                f"?per_page=100&direction=desc"
            )
            if isinstance(comments, list):
                for c in comments:
                    if not isinstance(c, dict):
                        continue
                    body = c.get("body")
                    if isinstance(body, str) and "@dependabot recreate" in body:
                        self.log.info(
                            "Found existing @dependabot recreate comment on "
                            "%s#%s; skipping duplicate.",
                            pr_info.repository_full_name,
                            pr_info.number,
                        )
                        return None
        except Exception as e:
            self.log.warning(
                "Could not list comments for %s#%s to check for existing "
                "@dependabot recreate comment: %s",
                pr_info.repository_full_name,
                pr_info.number,
                e,
            )
            return None

        # 5. Post the recreate comment
        self._pr_status(
            f"🔄 Requesting dependabot recreate: {pr_info.html_url} "
            f"[unverified commits: {', '.join(unverified_shas)}]",
            level="info",
        )

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
            return None

        # 6. Poll for the old PR to close and a replacement to appear.
        #    Dependabot typically responds within 30-90 seconds.
        #    We poll using the centralised merge timeout.  The whole
        #    poll (including the nested recreated-PR checks wait) is a
        #    wait on dependabot + CI, so the worker's concurrency slot
        #    is released for its duration (``parked()``).
        max_polls = self._merge_poll_max_attempts
        old_pr_closed = False

        async with _pkg.parked():
            for attempt in range(max_polls):
                await asyncio.sleep(self._merge_recheck_interval)

                # 6a. Check if the old PR has been closed
                if not old_pr_closed:
                    try:
                        old_pr_data = await self._github_client.get(
                            f"/repos/{repo_owner}/{repo_name}/pulls/{pr_info.number}"
                        )
                        if isinstance(old_pr_data, dict):
                            if old_pr_data.get("state") == "closed":
                                old_pr_closed = True
                                self._pr_status(
                                    f"✅ Old PR closed by dependabot: "
                                    f"{pr_info.html_url} "
                                    f"({(attempt + 1) * self._merge_recheck_interval:.0f}s elapsed)",
                                    level="info",
                                )
                    except Exception as e:
                        self.log.debug(
                            "Error polling old PR state for %s#%s: %s",
                            pr_info.repository_full_name,
                            pr_info.number,
                            e,
                        )

                # 6b. Once the old PR is closed, look for the replacement
                if old_pr_closed:
                    try:
                        # Search for open PRs from dependabot on the same head branch
                        prs = await self._github_client.get(
                            f"/repos/{repo_owner}/{repo_name}/pulls"
                            f"?state=open&head={repo_owner}:{pr_info.head_branch}&per_page=5"
                        )
                        if isinstance(prs, list):
                            for pr_data in prs:
                                if not isinstance(pr_data, dict):
                                    continue
                                pr_author = (pr_data.get("user") or {}).get("login", "")
                                if not is_dependabot(pr_author):
                                    continue

                                new_number = pr_data.get("number")
                                if new_number is None or new_number == pr_info.number:
                                    continue

                                # Verify the replacement targets the same base branch
                                new_base = (pr_data.get("base") or {}).get("ref", "")
                                if new_base != (pr_info.base_branch or "main"):
                                    self.log.debug(
                                        "Skipping candidate PR #%s: targets %s, "
                                        "expected %s",
                                        new_number,
                                        new_base,
                                        pr_info.base_branch or "main",
                                    )
                                    continue

                                # Found a replacement — now wait for checks to pass
                                new_pr_info = await self._wait_for_recreated_pr_checks(
                                    repo_owner, repo_name, new_number, pr_data
                                )
                                # Always return after the first wait attempt to avoid
                                # performing multiple long waits for the same PR.
                                return new_pr_info
                    except Exception as e:
                        self.log.debug(
                            "Error searching for replacement PR for %s#%s: %s",
                            pr_info.repository_full_name,
                            pr_info.number,
                            e,
                        )

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
