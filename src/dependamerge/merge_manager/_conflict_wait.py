# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The waiting phases of conflict recovery.

Clearing a conflict by rebase costs two bounded waits — one for the
rebase itself, then one for the checks that follow it — plus the
terminal steps that turn each outcome into a result.  They live apart
from the conflict handler so that it reads as the sequence it is.
"""

from __future__ import annotations

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._types import MergeResult, MergeStatus


class _MergeConflictWaitMixin(_MergeManagerBase):
    """Waiting out a dependabot rebase and reporting where it ended."""

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

    async def _handle_conflict_no_wait(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        result: MergeResult,
        already_rebasing: bool,
    ) -> MergeResult:
        """Arm the recovery and report it without waiting for the rebase.

        Fire-and-forget (``max_wait == 0``): ask dependabot to rebase
        (unless it already is), arm auto-merge, and report pending
        without blocking this repository's serial worker.  Approval is
        best-effort here — a subsequent dependabot force-push dismisses
        it when the branch enables "dismiss stale reviews", which is the
        documented trade-off of not waiting to approve the rebased head.
        """
        if not already_rebasing:
            await self._request_dependabot_rebase(pr_info, owner, repo)
        try:
            await self._approve_pr(owner, repo, pr_info.number)
        except Exception as exc:
            self.log.debug(
                "no-wait approve failed for %s/%s#%s: %s",
                owner,
                repo,
                pr_info.number,
                exc,
            )
        auto_ok = await self._enable_auto_merge_for_pr(pr_info, owner, repo)
        if auto_ok:
            result.status = MergeStatus.AUTO_MERGE_PENDING
            result.error = "auto-merge pending: conflict rebase requested (no-wait)"
            self._pr_status(
                f"⏳ Auto-merge armed (no-wait): {pr_info.html_url}",
                level="info",
            )
        else:
            # Auto-merge could not be armed (e.g. the repository has
            # the feature disabled), so nothing will merge this PR
            # later.  Report BLOCKED rather than a misleading
            # AUTO_MERGE_PENDING: the PR is left approved and rebased
            # but will not merge on its own.
            result.status = MergeStatus.BLOCKED
            result.error = (
                "auto-merge unavailable (no-wait); PR approved and "
                "rebase requested but not merged"
            )
            self._pr_status(
                f"🛑 Auto-merge unavailable (no-wait): {pr_info.html_url}",
                level="warning",
            )
        return result

    async def _wait_for_rebased_checks(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        deadline: float,
        auto_ok: bool,
    ) -> tuple[bool, bool]:
        """Wait (sharing the deadline) for required checks to land.

        When auto-merge is armed we wait *through* ``clean``
        (``stop_on_clean=False``) so we can observe GitHub actually
        close the PR and report MERGED.  When auto-merge could NOT be
        enabled, waiting through ``clean`` would just spin until the
        deadline (nothing would merge the PR), so we stop on ``clean``
        and the caller merges it itself.
        """
        if auto_ok:
            continue_states: tuple[str, ...] = (
                "clean",
                "blocked",
                "behind",
                "unstable",
                "unknown",
                "",
            )
        else:
            continue_states = ("blocked", "behind", "unstable", "unknown", "")
        # The rebase landed; what remains is a wait on required checks.
        # No counting here: ``_request_dependabot_rebase`` above owns
        # the cumulative "Rebased" total, and deliberately counts
        # nothing when its duplicate guard finds a macro an earlier run
        # already posted.
        self._track_pr_state(pr_info, "waiting")
        try:
            return await self._wait_for_auto_merge(
                pr_info,
                owner,
                repo,
                continue_states=continue_states,
                deadline=deadline,
                stop_on_clean=not auto_ok,
                # Only a measurement of *checks* when the wait stops at
                # ``clean``.  With auto-merge armed it deliberately waits
                # through ``clean`` until GitHub closes the PR, so the
                # duration would also carry merge-queue latency and would
                # oversize a sibling's head start.
                measures_checks=not auto_ok,
            )
        finally:
            self._track_pr_state(pr_info, None)

    async def _complete_after_rebase(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        result: MergeResult,
        deadline: float,
    ) -> MergeResult:
        """Approve the rebased head, then see the merge through.

        Approving happens *now* (not before — approving the pre-rebase
        head would just be dismissed by dependabot's force-push,
        producing the duplicate approvals we want to avoid).  An
        approval failure (permission / API error) is handled here
        rather than left to bubble to the generic catch-all, which
        would lose the conflict-recovery context.
        """
        try:
            await self._approve_pr(owner, repo, pr_info.number)
        except Exception as exc:
            return self._report_rebase_approval_failure(
                pr_info, owner, repo, result, exc
            )
        auto_ok = await self._enable_auto_merge_for_pr(pr_info, owner, repo)
        if auto_ok:
            self._pr_status(
                f"🤖 Auto-merge: {pr_info.html_url}",
                level="debug",
            )

        closed, merged = await self._wait_for_rebased_checks(
            pr_info, owner, repo, deadline, auto_ok
        )
        if closed:
            return self._finish_conflict_close(pr_info, result, merged)

        if auto_ok:
            # Auto-merge is armed: GitHub will complete the merge once
            # the required checks pass (often after our run ends).
            result.status = MergeStatus.AUTO_MERGE_PENDING
            result.error = "auto-merge pending: checks after conflict rebase"
            self._pr_status(
                f"⏳ Waiting: {pr_info.html_url} [auto-merge after rebase]",
                level="debug",
            )
            return result

        return await self._merge_rebased_pr(pr_info, owner, repo, result)

    def _report_rebase_approval_failure(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        result: MergeResult,
        exc: Exception,
    ) -> MergeResult:
        """Report a rebase that cleared the conflict but could not be approved."""
        self.log.warning(
            "Failed to approve %s/%s#%s after rebase: %s",
            owner,
            repo,
            pr_info.number,
            exc,
        )
        result.status = MergeStatus.FAILED
        result.error = f"rebase cleared the conflict but approval failed: {exc}"
        self._pr_status(f"❌ Failed: {pr_info.html_url}", level="error")
        return result

    async def _merge_rebased_pr(
        self, pr_info: PullRequestInfo, owner: str, repo: str, result: MergeResult
    ) -> MergeResult:
        """Merge a rebased PR directly, auto-merge being unavailable.

        If the rebase left the PR mergeable, merge it now; otherwise it
        will not land on its own — report the failure rather than a
        misleading ``AUTO_MERGE_PENDING`` that would never resolve.
        """
        if pr_info.mergeable_state == "clean":
            dispatch_lock = await self._get_merge_dispatch_lock(owner, repo)
            async with dispatch_lock:
                merged = await self._merge_pr_with_retry(pr_info, owner, repo)
            if merged:
                result.status = MergeStatus.MERGED
                self._pr_status(
                    f"✅ Merged: {pr_info.html_url}",
                    level="debug",
                )
                return result

        result.status = MergeStatus.FAILED
        result.error = (
            "rebase cleared the conflict but the PR could not be merged "
            "(auto-merge unavailable)"
        )
        self._pr_status(f"❌ Failed: {pr_info.html_url}", level="error")
        return result
