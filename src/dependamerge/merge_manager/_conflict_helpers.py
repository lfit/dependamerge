# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Supporting steps for merge-conflict recovery.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._models import (
    MergeResult,
    MergeStatus,
)


class _ConflictHelpersMixin(_MergeManagerBase):
    """Supporting steps for merge-conflict recovery."""

    async def _request_dependabot_rebase(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Post ``@dependabot rebase`` on a conflicted dependabot PR.

        Dependabot responds by rebasing the PR branch onto the latest
        base, regenerating any lockfiles and re-signing the commit —
        the reliable way to clear a ``uv.lock`` / dependency conflict
        that a plain ``git rebase`` cannot resolve.

        Guards against duplicate comments: when a ``@dependabot
        rebase`` is already present the request is treated as
        in-flight and ``True`` is returned (the caller proceeds to
        wait).  Returns ``False`` only when the comment could not be
        posted.
        """
        if self._github_client is None:
            return False

        # Duplicate guard — don't stack rebase requests if one is
        # already pending from a previous run / trigger.
        try:
            comments = await self._github_client.get(
                f"/repos/{owner}/{repo}/issues/{pr_info.number}/comments"
                f"?per_page=100&direction=desc"
            )
            if isinstance(comments, list):
                for c in comments:
                    if not isinstance(c, dict):
                        continue
                    body = c.get("body")
                    if isinstance(body, str) and "@dependabot rebase" in body:
                        self.log.info(
                            "Existing @dependabot rebase comment on %s#%s; "
                            "treating rebase as already requested.",
                            pr_info.repository_full_name,
                            pr_info.number,
                        )
                        return True
        except Exception as exc:
            # If we can't list comments, fall through and post anyway:
            # a duplicate rebase request is harmless (dependabot just
            # rebases again) and is better than skipping recovery.
            self.log.debug(
                "Could not list comments for %s#%s before rebase request: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )

        try:
            await self._github_client.post_issue_comment(
                owner, repo, pr_info.number, "@dependabot rebase"
            )
            # One macro comment and one rebase request: both totals
            # move.  The duplicate-guard path above deliberately does
            # not count — that rebase was requested by an earlier run.
            self._record_retrigger()
            self._record_rebase()
            return True
        except Exception as exc:
            self.log.warning(
                "Failed to post @dependabot rebase on %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return False

    def _finish_conflict_close(
        self, pr_info: PullRequestInfo, result: MergeResult, merged: bool
    ) -> MergeResult:
        """Finalise a conflict-recovery result when the PR closed mid-wait.

        ``merged`` distinguishes auto-merge success (the rebase landed
        and GitHub merged the PR) from closed-without-merge (a human
        closed it, dependabot superseded it, etc.).
        """
        if merged:
            result.status = MergeStatus.MERGED
            self._pr_status(
                f"✅ Merged (auto-merge): {pr_info.html_url}",
                level="debug",
            )
        else:
            result.status = MergeStatus.CLOSED
            result.error = (
                "PR closed without merging during conflict rebase "
                "(no operator follow-up needed)"
            )
            self._pr_status(
                f"🚪 Closed without merging: {pr_info.html_url}",
                level="warning",
            )
        return result

    def _dependabot_is_rebasing(self, body: str | None) -> bool:
        """Return True when a PR body shows dependabot mid-self-rebase.

        Dependabot writes a notice into the PR body while it rebases the
        branch on its own (after the base moved).  Detecting it lets the
        conflict handler wait for the in-progress rebase instead of
        sending a redundant ``@dependabot rebase`` macro.
        """
        if not body:
            return False
        lowered = body.lower()
        return "dependabot is rebasing" in lowered or "is rebasing this pr" in lowered
