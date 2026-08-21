# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Recovery from a merge conflict on an automation pull request.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio

from ..bot_identity import is_dependabot
from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._models import (
    MergeResult,
    MergeStatus,
)


class _ConflictsMixin(_MergeManagerBase):
    """Recovery from a merge conflict on an automation pull request."""

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
            return self._fail_with_conflict(pr_info, result)

        # Detect whether dependabot is already self-rebasing this PR.
        # When its base branch moves (e.g. a sibling PR merged), it
        # rebases the branch on its own and writes a marker into the PR
        # body while it does.  In that window we must not send a
        # duplicate ``@dependabot rebase`` macro: the in-progress rebase
        # will clear the conflict, so we wait for it rather than poke it.
        already_rebasing = self._dependabot_is_rebasing(pr_info.body)

        if self._no_wait:
            return await self._arm_conflict_rebase_no_wait(
                pr_info, owner, repo, result, already_rebasing=already_rebasing
            )

        if not await self._start_conflict_rebase(
            pr_info, owner, repo, already_rebasing=already_rebasing
        ):
            return self._fail_with_conflict(pr_info, result)

        # Share a single ``merge_timeout`` budget across both wait
        # phases (waiting for the rebase, then for checks).
        deadline = asyncio.get_running_loop().time() + self._merge_timeout

        # Wait for dependabot's rebase to clear the conflict.
        # Keep waiting while still ``dirty`` or while GitHub recomputes
        # mergeability (a transient null is preserved as the prior
        # ``dirty`` by ``_wait_for_auto_merge``).
        #
        # This wait stays inline: ``test_the_rebase_wait_call_site_is_
        # unmarked`` reads the source of *this* method to prove the
        # rebase turnaround is never marked ``measures_checks``.
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
            return self._fail_with_conflict(pr_info, result)

        approval_failure = await self._approve_rebased_head(
            pr_info, owner, repo, result
        )
        if approval_failure is not None:
            return approval_failure

        auto_ok = await self._enable_auto_merge_for_pr(pr_info, owner, repo)
        if auto_ok:
            self._pr_status(
                f"🤖 Auto-merge: {pr_info.html_url}",
                level="debug",
            )

        closed, merged = await self._wait_for_conflict_checks(
            pr_info, owner, repo, deadline, auto_ok=auto_ok
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

        return await self._merge_conflict_pr_directly(pr_info, owner, repo, result)

    def _fail_with_conflict(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> MergeResult:
        """Report an unresolvable conflict and mark ``result`` failed.

        Three distinct dead ends in the conflict flow — a
        non-dependabot author, a rebase macro we could not post, and a
        rebase that ran out of time still ``dirty`` — all land on the
        same terminal state.  Sharing one helper keeps the reported
        status and error string identical across them, since the
        summary groups PRs by that exact ``error`` text.
        """
        self._pr_status(
            f"🔀 Merge conflict: {pr_info.html_url}",
            level="info",
        )
        result.status = MergeStatus.FAILED
        result.error = "merge conflicts"
        return result

    async def _arm_conflict_rebase_no_wait(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        result: MergeResult,
        *,
        already_rebasing: bool,
    ) -> MergeResult:
        """Request the rebase, arm auto-merge and return without waiting.

        The fire-and-forget path (``max_wait == 0``): ask dependabot to
        rebase (unless it already is), arm auto-merge, and report
        pending without blocking this repository's serial worker.
        Approval is best-effort here — a subsequent dependabot
        force-push dismisses it when the branch enables "dismiss stale
        reviews", which is the documented trade-off of not waiting to
        approve the rebased head.

        Separate from ``_handle_merge_conflict`` because it shares no
        step with the waiting path: it never polls, never approves the
        rebased head, and reaches its own terminal statuses.
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

    async def _start_conflict_rebase(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        *,
        already_rebasing: bool,
    ) -> bool:
        """Ensure a dependabot rebase is under way, reporting progress.

        Requests a rebase (which regenerates the lockfile and signs the
        commit) unless dependabot is already rebasing on its own, in
        which case we only announce it.  Returns whether a rebase is in
        flight; ``False`` means the macro could not be posted and the
        caller has nothing to wait for.

        Split out so the caller reads as a single precondition rather
        than an if/else whose branches both fall through to the wait.
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

    async def _approve_rebased_head(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        result: MergeResult,
    ) -> MergeResult | None:
        """Approve the rebased commit, or return the failed ``result``.

        The conflict has cleared, so we approve *now* — not before.
        Approving the pre-rebase head would just be dismissed by
        dependabot's force-push, producing the duplicate approvals we
        want to avoid.

        Returns ``None`` when the approval succeeds and the caller
        should continue.  An approval failure (permission or API error)
        is handled here rather than left to bubble into the generic
        catch-all, which would lose the conflict-recovery context; in
        that case ``result`` is returned already populated.
        """
        try:
            await self._approve_pr(owner, repo, pr_info.number)
        except Exception as exc:
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
        return None

    async def _wait_for_conflict_checks(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        deadline: float,
        *,
        auto_ok: bool,
    ) -> tuple[bool, bool]:
        """Wait (sharing the deadline) for required checks to land.

        When auto-merge is armed we wait *through* ``clean``
        (``stop_on_clean=False``) so we can observe GitHub actually
        close the PR and report MERGED.  When auto-merge could NOT be
        enabled, waiting through ``clean`` would just spin until the
        deadline (nothing would merge the PR), so we stop on ``clean``
        and let the caller merge it directly.

        The rebase landed; what remains is a wait on required checks.
        No counting here: ``_request_dependabot_rebase`` owns the
        cumulative "Rebased" total, and deliberately counts nothing
        when its duplicate guard finds a macro an earlier run posted.

        Separate from the inline rebase wait above because only this
        one branches on ``auto_ok``, and keeping that choice next to
        the reasoning for it stops the two wait shapes being confused.
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

    async def _merge_conflict_pr_directly(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        result: MergeResult,
    ) -> MergeResult:
        """Merge the rebased PR ourselves when auto-merge is unavailable.

        If the rebase left the PR mergeable, merge it directly now;
        otherwise it will not land on its own — report the failure
        rather than a misleading ``AUTO_MERGE_PENDING`` that would
        never resolve.

        Separate because it is the only part of the conflict flow that
        takes the per-repo dispatch lock, which the rest of
        ``_handle_merge_conflict`` must run outside of.
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
