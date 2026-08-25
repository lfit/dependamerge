# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Turning the Step 6 dispatch into a terminal ``MergeResult``.

Three outcomes arrive here: auto-merge is pending, the merge landed,
or the merge was rejected.  Only the last needs work --- a rejection
can mask a benign race (the PR merged or closed externally), a PR that
auto-merge will still complete, or a dependabot PR that recreation can
rescue.
"""

from __future__ import annotations

from ..models import PullRequestInfo
from ._single_pr_context import _MergeFlow
from ._single_pr_recreate import _SinglePrRecreateMixin
from ._types import MergeResult, MergeStatus, RecreateOutcome


class _SinglePrOutcomeMixin(_SinglePrRecreateMixin):
    """The terminal classification of a Step 6 merge attempt."""

    def _record_auto_merge_pending(self, flow: _MergeFlow) -> None:
        """Record that GitHub will merge the PR asynchronously.

        Tailors the reason to the actual ``mergeable_state`` so the
        end-of-run summary shows what auto-merge is waiting on, rather
        than always "pending checks".
        """
        pr_info = flow.pr_info
        result = flow.result
        result.status = MergeStatus.AUTO_MERGE_PENDING
        if pr_info.mergeable_state == "behind":
            wait_reason = "behind base branch"
        elif pr_info.mergeable_state == "unstable":
            wait_reason = "non-required check failure"
        else:
            # ``blocked`` (the only other state that reaches this
            # branch) routed through ``analyze_block_reason()`` and was
            # classified as pending required checks by
            # ``_block_reason_indicates_pending_checks``.
            wait_reason = "pending checks"
        result.error = f"auto-merge pending: {wait_reason}"
        self._pr_status(
            f"⏳ Waiting: {pr_info.html_url} [{wait_reason}]",
            level="debug",
        )

    def _record_merged(self, flow: _MergeFlow) -> None:
        """Record a successful direct merge."""
        flow.result.status = MergeStatus.MERGED
        self._pr_status(
            f"✅ Merged: {flow.pr_info.html_url}",
            level="debug",
        )

    async def _handle_failed_merge(self, flow: _MergeFlow) -> MergeResult | None:
        """Classify a rejected merge, recovering where recovery exists."""
        early = await self._external_closure_result(flow)
        if early is not None:
            return early
        early = self._behind_auto_merge_result(flow)
        if early is not None:
            return early

        # Compute failure summary once — used for both the recreate
        # decision and the final error reporting.
        failure_reason = await self._get_failure_summary(flow.pr_info)

        recreate = await self._maybe_recreate_dependabot_pr(flow, failure_reason)
        if recreate.outcome is RecreateOutcome.READY and recreate.pr_info is not None:
            await self._merge_recreated_pr(flow, recreate.pr_info)
        elif (
            recreate.outcome is RecreateOutcome.MERGED and recreate.pr_info is not None
        ):
            # Auto-merge is armed on the replacement before the wait
            # begins, so it can complete between polls.  Record the
            # success rather than attempting a merge on a closed PR.
            self._record_recreated_merge(flow, recreate.pr_info)
        elif (
            recreate.outcome is RecreateOutcome.PENDING and recreate.pr_info is not None
        ):
            # We stopped waiting, but the replacement is open, approved
            # and armed, so it is expected to merge on its own.
            # Reporting the original as FAILED would under-report a
            # success, and ``_confirm_failure`` could not correct it:
            # it rechecks the original PR, not the replacement.
            self._record_recreated_pending(flow, recreate.pr_info)
        elif recreate.outcome is RecreateOutcome.ABANDONED:
            self._record_recreated_abandoned(flow, recreate.pr_info)
        else:
            await self._report_merge_failure(
                flow.pr_info,
                flow.repo_owner,
                flow.repo_name,
                flow.result,
                failure_reason,
            )
        return None

    def _record_recreated_merge(
        self, flow: _MergeFlow, recreated_pr: PullRequestInfo
    ) -> None:
        """Record a replacement PR that merged while we were waiting."""
        flow.result.status = MergeStatus.MERGED
        flow.result.pr_info = recreated_pr
        self._pr_status(
            f"✅ Merged (recreated, by auto-merge): {recreated_pr.html_url}",
            level="debug",
        )

    def _record_recreated_pending(
        self, flow: _MergeFlow, recreated_pr: PullRequestInfo
    ) -> None:
        """Record a replacement PR left in flight under auto-merge.

        The same treatment ``--max-wait`` gives any armed-but-unfinished
        merge: report it as pending rather than as a failure, so the
        run's counts describe what is actually going to happen.

        The reason goes on ``error`` because that is the field the
        end-of-run report reads (``cli/_merge_report.py``), and the
        convention every other ``AUTO_MERGE_PENDING`` path already
        follows.  Putting it on ``warning`` would leave the operator
        looking at "no reason reported".
        """
        flow.result.status = MergeStatus.AUTO_MERGE_PENDING
        flow.result.pr_info = recreated_pr
        flow.result.error = (
            "auto-merge pending: dependabot recreated this PR as "
            f"#{recreated_pr.number}, which is approved and armed"
        )
        self._pr_status(
            f"⏳ Auto-merge pending (recreated): {recreated_pr.html_url}",
            level="debug",
        )

    def _record_recreated_abandoned(
        self, flow: _MergeFlow, replacement: PullRequestInfo | None
    ) -> None:
        """Record a replacement that will not merge without help.

        ``BLOCKED`` rather than ``FAILED``, because after a recreate the
        **original** PR is closed unmerged by dependabot --- so a
        ``FAILED`` result is rewritten to ``CLOSED`` by
        ``_confirm_failure``, which documents ``CLOSED`` as "nothing for
        the operator to follow up on".  That is the opposite of what a
        conflicted or closed-unmerged replacement needs.
        ``_confirm_failure`` returns early for any status other than
        ``FAILED``, so ``BLOCKED`` survives it, and it is the status
        already used elsewhere for "will not merge on its own".
        """
        flow.result.status = MergeStatus.BLOCKED
        if replacement is not None:
            flow.result.pr_info = replacement
            flow.result.error = (
                f"dependabot recreated this PR as #{replacement.number}, "
                "which is conflicted or closed unmerged and needs attention"
            )
        else:
            flow.result.error = (
                "dependabot recreated this PR, but the replacement is "
                "conflicted or closed unmerged and needs attention"
            )
        self._pr_status(
            f"🛑 Blocked (recreated): {flow.result.pr_info.html_url}",
            level="debug",
        )

    async def _external_closure_result(self, flow: _MergeFlow) -> MergeResult | None:
        """Recognise a PR that merged or closed outside this run.

        A failed merge attempt can mask two benign races: the PR merged
        externally (a concurrent dependamerge run at org scope, or a
        human admin), or dependabot closed it without merging once
        sibling merges advanced the base.  Neither outcome needs human
        follow-up.
        """
        pr_info = flow.pr_info
        result = flow.result
        ext_state, ext_merged = await self._fetch_pr_state_now(
            pr_info, flow.repo_owner, flow.repo_name
        )
        if ext_state == "closed" and ext_merged:
            result.status = MergeStatus.SKIPPED
            result.error = "already merged externally"
            self._pr_status(
                f"⏭️ Skipped: {pr_info.html_url} [already merged externally]",
                level="info",
            )
            return result
        if ext_state == "closed":
            result.status = MergeStatus.CLOSED
            result.error = (
                "PR closed without merging during the run "
                "(no operator follow-up needed)"
            )
            self._pr_status(
                f"\U0001f6aa Closed without merging: {pr_info.html_url}",
                level="info",
            )
            return result
        return None

    def _behind_auto_merge_result(self, flow: _MergeFlow) -> MergeResult | None:
        """Recognise a ``behind`` PR that auto-merge will still complete.

        A ``behind`` PR whose merge was rejected but that has auto-merge
        armed is not a failure: the reactive recovery in
        ``_handle_merge_failure`` requested a dependabot rebase and armed
        auto-merge, so GitHub completes the merge server-side once the
        rebase lands and required checks pass.
        """
        pr_info = flow.pr_info
        result = flow.result
        if not (
            pr_info.mergeable_state == "behind"
            and flow.pr_key in self._auto_merge_enabled
        ):
            return None
        result.status = MergeStatus.AUTO_MERGE_PENDING
        result.error = "auto-merge pending: behind base branch (rebase requested)"
        self._pr_status(
            f"\u23f3 Waiting: {pr_info.html_url} "
            "[behind base branch; rebase requested]",
            level="debug",
        )
        return result
