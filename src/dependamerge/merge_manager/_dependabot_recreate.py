# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Asking dependabot to recreate a pull request whose checks stuck.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import dependamerge.merge_manager as _pkg

from ..bot_identity import is_dependabot
from ..models import PullRequestInfo
from ._base import _MergeManagerBase

if TYPE_CHECKING:
    from ..github_async import GitHubAsync


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

        client = self._github_client
        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)

        # 1. Only applies to dependabot PRs
        if not is_dependabot(pr_info.author):
            return None

        # 2. Check whether the branch requires signed commits
        if not await self._recreate_requires_signatures(
            client, pr_info, repo_owner, repo_name
        ):
            return None

        # 3. Check whether any commits are unverified
        unverified_shas = await self._recreate_unverified_shas(
            client, pr_info, repo_owner, repo_name
        )
        if unverified_shas is None:
            return None

        # 4. Guard against duplicate recreate comments
        if await self._recreate_already_requested(
            client, pr_info, repo_owner, repo_name
        ):
            return None

        # 5. Post the recreate comment
        if not await self._post_recreate_comment(
            client, pr_info, repo_owner, repo_name, unverified_shas
        ):
            return None

        # 6. Poll for the old PR to close and a replacement to appear
        return await self._await_recreated_pr(client, pr_info, repo_owner, repo_name)

    async def _recreate_requires_signatures(
        self, client: GitHubAsync, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Report whether the base branch demands signed commits.

        Split out so the trigger reads as a sequence of guards.  An API
        failure and an explicit "no requirement" are treated alike: both
        mean the recreate does not apply, so both answer False.
        """
        try:
            requires_signatures = await client.requires_commit_signatures(
                owner, repo, pr_info.base_branch or "main"
            )
            if not requires_signatures:
                self.log.debug(
                    "Branch %s/%s:%s does not require commit signatures; "
                    "skipping dependabot recreate.",
                    owner,
                    repo,
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
        self, client: GitHubAsync, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> list[str] | None:
        """Collect the SHAs of the PR's unverified commits.

        Returns the unverified SHAs when a recreate is warranted, and
        ``None`` when it is not — every commit is already verified, or the
        check could not be made.  Kept separate so those two "give up"
        cases share one exit while the caller still receives the SHAs its
        status line reports.
        """
        try:
            (
                all_verified,
                unverified_shas,
            ) = await client.check_pr_commit_signatures(owner, repo, pr_info.number)
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
        self, client: GitHubAsync, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Report whether a recreate was already asked for on this PR.

        Failing to list the comments also counts as "already requested":
        a duplicate cannot be ruled out, and asking twice makes dependabot
        churn through two replacement pull requests.
        """
        try:
            comments = await client.get(
                f"/repos/{owner}/{repo}/issues/{pr_info.number}/comments"
                f"?per_page=100&direction=desc"
            )
            if self._comments_contain_recreate(comments):
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

    @staticmethod
    def _comments_contain_recreate(
        comments: dict[str, Any] | list[dict[str, Any]],
    ) -> bool:
        """Report whether a comment payload holds a recreate request.

        Pure inspection of the API response, separated so the scan over
        comments stays clear of the error handling around the fetch.
        """
        if not isinstance(comments, list):
            return False
        for c in comments:
            if not isinstance(c, dict):
                continue
            body = c.get("body")
            if isinstance(body, str) and "@dependabot recreate" in body:
                return True
        return False

    async def _post_recreate_comment(
        self,
        client: GitHubAsync,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        unverified_shas: list[str],
    ) -> bool:
        """Announce and post the ``@dependabot recreate`` comment.

        Returns True once the comment is posted and the retrigger
        recorded, False if it could not be posted — in which case
        dependabot was never asked, so there is nothing to poll for.
        """
        self._pr_status(
            f"🔄 Requesting dependabot recreate: {pr_info.html_url} "
            f"[unverified commits: {', '.join(unverified_shas)}]",
            level="info",
        )

        try:
            await client.post_issue_comment(
                owner, repo, pr_info.number, "@dependabot recreate"
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

    async def _await_recreated_pr(
        self, client: GitHubAsync, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> PullRequestInfo | None:
        """Poll for the old PR to close and a replacement to appear.

        Dependabot typically responds within 30-90 seconds.  We poll
        using the centralised merge timeout.  The whole poll (including
        the nested recreated-PR checks wait) is a wait on dependabot +
        CI, so the worker's concurrency slot is released for its
        duration (``parked()``).
        """
        max_polls = self._merge_poll_max_attempts
        old_pr_closed = False

        async with _pkg.parked():
            for attempt in range(max_polls):
                await asyncio.sleep(self._merge_recheck_interval)

                if not old_pr_closed:
                    old_pr_closed = await self._old_pr_is_closed(
                        client, pr_info, owner, repo, attempt
                    )

                if old_pr_closed:
                    found, new_pr_info = await self._find_recreated_pr(
                        client, pr_info, owner, repo
                    )
                    if found:
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
        client: GitHubAsync,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        attempt: int,
    ) -> bool:
        """Take one poll of the original PR's state.

        Announces the closure, with the elapsed time, at the moment it is
        first observed; the caller only asks while the PR is still open.
        Errors are swallowed so a transient API failure costs one
        interval rather than the whole recreate.
        """
        closed = False
        try:
            old_pr_data = await client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
            )
            if isinstance(old_pr_data, dict):
                if old_pr_data.get("state") == "closed":
                    closed = True
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
        return closed

    async def _find_recreated_pr(
        self, client: GitHubAsync, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> tuple[bool, PullRequestInfo | None]:
        """Search the head branch for the replacement PR and wait on it.

        Returns ``(True, result)`` once a replacement has been found and
        waited on — ``result`` is None when its checks never passed — and
        ``(False, None)`` while there is nothing to act on yet.  The flag
        is what stops the caller starting a second long checks wait for
        the same PR on a later poll.
        """
        try:
            prs = await client.get(
                f"/repos/{owner}/{repo}/pulls"
                f"?state=open&head={owner}:{pr_info.head_branch}&per_page=5"
            )
            if isinstance(prs, list):
                for pr_data in prs:
                    if not isinstance(pr_data, dict):
                        continue
                    new_number = self._recreated_pr_number(pr_data, pr_info)
                    if new_number is None:
                        continue
                    new_pr_info = await self._wait_for_recreated_pr_checks(
                        owner, repo, new_number, pr_data
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

    def _recreated_pr_number(
        self, pr_data: dict[str, Any], pr_info: PullRequestInfo
    ) -> int | None:
        """Return an open PR's number if it is the awaited replacement.

        The head-branch search can return a PR from another author, the
        original PR itself, or one retargeted at a different base branch,
        so candidates are vetted before committing to a long checks wait.
        ``None`` means "not the replacement", not "no number".
        """
        pr_author = (pr_data.get("user") or {}).get("login", "")
        if not is_dependabot(pr_author):
            return None

        new_number: int | None = pr_data.get("number")
        if new_number is None or new_number == pr_info.number:
            return None

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
