# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Resolution of merge conflicts.

A conflicted automation PR is usually best rebased or recreated by
the bot that raised it; failing that it is closed, since a superseding
PR will follow.
"""

from __future__ import annotations

import asyncio

from ..bot_identity import is_dependabot
from ..models import PullRequestInfo
from ._conflict_wait import _MergeConflictWaitMixin
from ._types import MergeResult, MergeStatus


class _MergeConflictMixin(_MergeConflictWaitMixin):
    """Reacting to a pull request that conflicts with its base."""

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

    def _report_unresolved_conflict(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> MergeResult:
        """Report a conflict no rebase of ours is going to clear."""
        self._pr_status(
            f"🔀 Merge conflict: {pr_info.html_url}",
            level="info",
        )
        result.status = MergeStatus.FAILED
        result.error = "merge conflicts"
        return result

    async def _ensure_rebase_requested(
        self, pr_info: PullRequestInfo, owner: str, repo: str, already_rebasing: bool
    ) -> bool:
        """Get a rebase under way, returning False when none could be.

        Dependabot's rebase regenerates the lockfile and signs the
        commit; when it is already rebasing on its own we say so and
        leave it to finish rather than sending a duplicate macro.
        """
        if already_rebasing:
            self._pr_status(
                f"🔄 Dependabot already rebasing: {pr_info.html_url}",
                level="info",
            )
            return True
        self._pr_status(
            f"🔄 Requesting dependabot rebase: {pr_info.html_url}",
            level="info",
        )
        return await self._request_dependabot_rebase(pr_info, owner, repo)

    async def _handle_merge_conflict(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        result: MergeResult,
    ) -> MergeResult:
        """Recover from (or report) a PR with a real merge conflict.

        A ``dirty`` PR has no merge path of its own.  For a dependabot
        PR we ask dependabot to rebase — which regenerates lockfiles
        and re-signs the commit — then wait (bounded by
        ``merge_timeout``) for the rebase and required checks to land,
        approving the *rebased* commit and enabling auto-merge so
        GitHub completes the merge.  For any other author there is no
        automated way to resolve a content conflict, so we report it
        and fail fast (no wait).

        Must be called *outside* the per-repo dispatch lock: the wait
        can run for the full ``merge_timeout`` and must not block
        sibling merges.  Sets ``result`` and returns it.
        """
        # Non-dependabot authors: no comment macro regenerates a
        # conflicted lockfile, and a blind force-push would only break
        # the approval chain (this org forbids self-merge of pushed
        # commits).  Report the conflict and fail fast.
        if not is_dependabot(pr_info.author):
            return self._report_unresolved_conflict(pr_info, result)

        # Detect whether dependabot is already self-rebasing this PR.
        # When its base branch moves (e.g. a sibling PR merged), it
        # rebases the branch on its own and writes a marker into the PR
        # body while it does.  In that window we must not send a
        # duplicate ``@dependabot rebase`` macro: the in-progress rebase
        # will clear the conflict, so we wait for it rather than poke it.
        already_rebasing = self._dependabot_is_rebasing(pr_info.body)

        if self._no_wait:
            return await self._handle_conflict_no_wait(
                pr_info, owner, repo, result, already_rebasing
            )

        if not await self._ensure_rebase_requested(
            pr_info, owner, repo, already_rebasing
        ):
            return self._report_unresolved_conflict(pr_info, result)

        # Share a single ``merge_timeout`` budget across both wait
        # phases (waiting for the rebase, then for checks).
        deadline = asyncio.get_running_loop().time() + self._merge_timeout

        # Wait for dependabot's rebase to clear the conflict.
        # Keep waiting while still ``dirty`` or while GitHub recomputes
        # mergeability (a transient null is preserved as the prior
        # ``dirty`` by ``_wait_for_auto_merge``).
        self._track_pr_state(pr_info, "rebasing")
        try:
            closed, merged = await self._wait_for_auto_merge(
                pr_info,
                owner,
                repo,
                continue_states=("dirty", "unknown", ""),
                deadline=deadline,
            )
        finally:
            self._track_pr_state(pr_info, None)
        if closed:
            return self._finish_conflict_close(pr_info, result, merged)
        if pr_info.mergeable_state == "dirty":
            # Timed out still conflicting — the rebase did not happen
            # or could not resolve the conflict.
            return self._report_unresolved_conflict(pr_info, result)

        return await self._complete_after_rebase(pr_info, owner, repo, result, deadline)
