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
from ._types import RecreateCause, RecreateResult


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
        self, pr_info: PullRequestInfo, cause: RecreateCause
    ) -> RecreateResult:
        """
        Ask dependabot to recreate a pull request from scratch.

        Two unrelated problems are recoverable this way, and the gates
        differ between them:

        ``UNSIGNED``
            When a repository's branch protection requires commit
            signatures, dependabot PRs can end up with unverified
            commits (e.g. after a rebase or force-push by GitHub).
            Posting ``@dependabot recreate`` produces a fresh PR whose
            commit is signed by GitHub.  The signature checks decide
            whether a recreate would help, so both are applied.

        ``STUCK_CHECK``
            A required check that never reported.  This is a *timing*
            problem; signatures are irrelevant to it.  Applying the
            signature gates here would make the recovery inert on every
            repository that does not enforce signatures --- while still
            telling the user a recreate was requested.

        Args:
            pr_info: Pull request information for the failing PR.
            cause: Why the recreate is being requested.

        Returns:
            A :class:`RecreateResult` describing what became of the
            request.
        """
        if not self._github_client:
            return RecreateResult.none()

        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)

        # Applies to both causes: only dependabot answers the macro.
        if not is_dependabot(pr_info.author):
            return RecreateResult.none()

        detail = ""
        if cause is RecreateCause.UNSIGNED:
            if not await self._recreate_branch_requires_signatures(
                repo_owner, repo_name, pr_info
            ):
                return RecreateResult.none()

            unverified_shas = await self._recreate_unverified_shas(
                repo_owner, repo_name, pr_info
            )
            if unverified_shas is None:
                return RecreateResult.none()
            detail = f" [unverified commits: {', '.join(unverified_shas)}]"

        # Guard against duplicate recreate comments
        if await self._recreate_already_requested(repo_owner, repo_name, pr_info):
            return RecreateResult.none()

        self._pr_status(
            f"🔄 Requesting dependabot recreate: {pr_info.html_url}{detail}",
            level="info",
        )

        if not await self._post_recreate_comment(repo_owner, repo_name, pr_info):
            return RecreateResult.none()

        # Poll for the old PR to close and a replacement to appear.
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
    ) -> RecreateResult:
        """Wait for the old PR to close and its replacement to appear.

        Dependabot typically responds within 30-90 seconds.  The whole
        poll --- including the nested recreated-PR checks wait --- is a
        wait on dependabot and CI, so the worker's concurrency slot is
        released for its duration (``parked()``).

        **One** deadline governs the whole path.  It is established here
        and passed into the nested wait, which previously started a
        fresh budget of its own: finding a replacement near the ceiling
        could then add another complete ``merge_timeout`` on top of
        whatever this loop had already spent, so a single PR could
        consume ``2 x merge_timeout`` beyond ``--max-wait`` while
        holding its slot lease.
        """
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        # ``--max-wait 0`` promises never to block.  The recreate macro
        # has been posted, so dependabot still acts on it --- but note
        # what this does **not** do: returning here means the
        # replacement is never discovered, so nothing approves or arms
        # auto-merge on it.  Dependabot typically takes 30-90 seconds to
        # open the replacement, so there is nothing to find yet, and
        # searching for it would be the very wait this flag forbids.
        #
        # The replacement is therefore left for a later run to collect
        # and merge normally.  Under ``--max-wait 0`` this PR is
        # reported as a failure, which is accurate: nothing will finish
        # it without another run.
        if self._no_wait:
            self.log.debug(
                "Not waiting for dependabot recreate on %s#%s (--max-wait 0)",
                pr_info.repository_full_name,
                pr_info.number,
            )
            return RecreateResult.none()

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._merge_timeout
        if self._run_deadline is not None:
            deadline = min(deadline, self._run_deadline)

        max_polls = self._merge_poll_max_attempts
        old_pr_closed = False

        async with _mm.parked():
            for attempt in range(max_polls):
                # Clamped rather than merely checked: sleeping a whole
                # interval with less than that left would overshoot the
                # ceiling, which is what the other deadline-aware waits
                # avoid.
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(self._merge_recheck_interval, remaining))

                # 6a. Check if the old PR has been closed
                if not old_pr_closed:
                    old_pr_closed = await self._old_pr_is_closed(
                        repo_owner, repo_name, pr_info, attempt
                    )

                # 6b. Once the old PR is closed, look for the replacement
                if old_pr_closed:
                    resolved, recreate = await self._find_recreated_pr(
                        repo_owner, repo_name, pr_info, deadline
                    )
                    # Always return after the first wait attempt to avoid
                    # performing multiple long waits for the same PR.
                    if resolved:
                        return recreate

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
        return RecreateResult.none()

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
        self,
        repo_owner: str,
        repo_name: str,
        pr_info: PullRequestInfo,
        deadline: float | None = None,
    ) -> tuple[bool, RecreateResult]:
        """Look for the replacement PR and wait for its checks.

        Returns a ``(resolved, result)`` pair.  ``resolved`` is True once
        a replacement has been found and waited on, whatever the outcome
        of that wait; the caller then stops polling.

        The guard covers **only** the search request.  Wrapping the wait
        as well would contradict the caller's own comment --- "always
        return after the first wait attempt" --- because a failure
        inside a multi-minute wait would be swallowed as a failed search
        and the caller would repeat the whole wait.
        """
        if not self._github_client:
            return False, RecreateResult.none()
        try:
            # Search for open PRs from dependabot on the same head branch
            prs = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/pulls"
                f"?state=open&head={repo_owner}:{pr_info.head_branch}&per_page=5"
            )
        except Exception as e:
            self.log.debug(
                "Error searching for replacement PR for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                e,
            )
            return False, RecreateResult.none()

        if isinstance(prs, list):
            for pr_data in prs:
                new_number = self._recreated_pr_number(pr_data, pr_info)
                if new_number is None:
                    continue

                # Found a replacement --- now wait for checks to pass.
                # Deliberately outside the guard above: a failure here
                # must propagate rather than trigger a second full wait.
                recreate = await self._wait_for_recreated_pr_checks(
                    repo_owner, repo_name, new_number, pr_data, deadline
                )
                return True, recreate
        return False, RecreateResult.none()

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
