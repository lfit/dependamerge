# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Step 5.5: arm auto-merge and wait for required checks.

If the PR is still blocked (e.g. by a pending required status check
such as pre-commit.ci) or unstable (a non-required check failed),
enable auto-merge and wait for required checks to complete.  Skipped
when:

  * preview_mode (no side effects)
  * force_level == "all" (force semantics bypass wait)
  * Step 5 already ran a rebase + wait for this PR (avoid doubling the
    configured merge_timeout)
  * mergeable_state == "blocked" for a reason that cannot resolve on
    its own (e.g. "requires approval", missing code-owner reviews) ---
    waiting would just delay the inevitable failure/merge by up to
    merge_timeout.

``behind`` PRs deliberately do NOT wait here: unless branch protection
enforces the strict up-to-date policy (in which case Step 5 already
refreshed the branch), a behind-but-green PR merges directly, so
parking it in the wait loop --- where the state never advances on its
own --- would just burn the full ``merge_timeout`` before the merge
attempt that was going to succeed anyway.

Exception: when Step 5 just dispatched an *asynchronous* rebase (local
force-push or the ``@dependabot rebase`` macro) and auto-merge could
not be armed, those paths leave ``_rebased_prs`` unset precisely so
this wait can bridge the gap while the rebase lands and GitHub
recomputes mergeability --- the snapshot still reads ``behind``
because neither path refreshes ``pr_info``.  Without the wait, Step 6
would fire a manual merge against the stale state and 405.
``flow.needs_rebase`` captures "Step 5 actually ran" and the ``not
already_rebased`` guard excludes the auto-merge-armed case.

We accept any ``mergeable`` value (including ``False``) when the state
is one of these auto-merge-rescuable states, because GitHub returns
``mergeable=False`` transiently while computing the value or when a
non-required check failed.  The block-reason pre-check still weeds out
genuinely-stuck cases (missing approvals, etc.) so we don't burn
``merge_timeout`` on them.
"""

from __future__ import annotations

from typing import Any

from ._base import _MergeManagerBase
from ._single_pr_context import _MergeFlow
from ._types import MergeResult, MergeStatus


class _SinglePrWaitMixin(_MergeManagerBase):
    """The auto-merge wait and the decision to enter it."""

    def _state_is_waitable(self, flow: _MergeFlow) -> bool:
        """Whether this ``mergeable_state`` can advance while we wait.

        ``unstable`` means a non-required check is failing or pending
        but the PR is otherwise mergeable.  When GitHub also reports
        ``mergeable is True`` the green button is live and a direct
        merge succeeds *now*, so entering the auto-merge wait would be
        pure waste: the state never reaches ``clean`` (the non-required
        check stays red, e.g. an excluded Zizmor scan), so the loop
        burns the full ``merge_timeout``; and
        ``enablePullRequestAutoMerge`` is rejected outright on an
        already-mergeable PR, so the wait isn't even backed by
        auto-merge.  Route those straight to the Step 6 direct merge.
        We still wait on ``unstable`` when ``mergeable`` is not
        literally True (GitHub still computing the value, or a required
        check transiently failing) so a genuinely not-yet-ready PR is
        not merged prematurely.
        """
        pr_info = flow.pr_info
        return (
            pr_info.mergeable_state == "blocked"
            or (pr_info.mergeable_state == "unstable" and pr_info.mergeable is not True)
            or (pr_info.mergeable_state == "behind" and flow.needs_rebase)
        )

    def _should_wait_for_checks(self, flow: _MergeFlow) -> bool:
        """Whether Step 5.5 should run for this PR.

        For ``blocked`` PRs, consult the block-reason analysis
        (computed once by ``_analyze_blocked_state``) before entering
        the wait loop so we don't burn the full merge_timeout on PRs
        blocked for reasons that cannot resolve on their own.
        """
        base_should_wait = (
            not self.preview_mode
            and self._github_client is not None
            and self._state_is_waitable(flow)
            and self.force_level != "all"
            and not flow.already_rebased
        )
        if not base_should_wait or flow.pr_info.mergeable_state != "blocked":
            return base_should_wait

        if not flow.blocked_analysis_ok:
            # Treat analysis failures as 'do not wait' so we don't burn
            # the full ``merge_timeout`` on a PR whose block reason we
            # cannot classify. The PR will fall through to the Step 6
            # skip gate (which re-consults the analysis) and either
            # defer to auto-merge or surface a manual-merge error
            # promptly.
            return False
        if flow.blocked_reason is not None:
            if not self._block_reason_indicates_pending_checks(flow.blocked_reason):
                self.log.debug(
                    "Skipping Step 5.5 wait for %s: block "
                    "reason '%s' will not resolve on its own",
                    flow.pr_key,
                    flow.blocked_reason,
                )
                return False
        return True

    async def _wait_for_checks(self, flow: _MergeFlow) -> MergeResult | None:
        """Run Step 5.5, returning a terminal result if the PR closed."""
        await self._arm_auto_merge_before_wait(flow)
        early = await self._await_auto_merge(flow)
        if early is not None:
            return early
        return await self._recheck_late_precommit(flow)

    async def _arm_auto_merge_before_wait(self, flow: _MergeFlow) -> None:
        """Approve (on demand) and arm auto-merge before waiting."""
        if flow.pr_key in self._auto_merge_enabled:
            return
        # Approve-on-demand: arming auto-merge implies we want the PR to
        # merge once checks pass, so approve the current head first
        # (idempotent) before enabling.
        auto_ok_pre = await self._enable_auto_merge_with_approval(
            flow.pr_info, flow.repo_owner, flow.repo_name
        )
        if auto_ok_pre:
            self._pr_status(
                f"🤖 Auto-merge: {flow.pr_info.html_url}",
                level="debug",
            )

    async def _await_auto_merge(self, flow: _MergeFlow) -> MergeResult | None:
        """Wait for required checks, short-circuiting on a closed PR."""
        pr_info = flow.pr_info
        result = flow.result
        # Wait (bounded by ``merge_timeout``) for required checks to
        # complete and auto-merge to fire.  The continue-states mirror
        # the entry condition in ``_should_wait_for_checks`` (blocked /
        # behind / unstable).
        self._track_pr_state(pr_info, "waiting")
        (
            closed_during_wait,
            merged_during_wait,
        ) = await self._wait_for_auto_merge(
            pr_info,
            flow.repo_owner,
            flow.repo_name,
            continue_states=("blocked", "behind", "unstable"),
            measures_checks=True,
        )
        self._track_pr_state(pr_info, None)

        # If the wait revealed the PR is already closed, short-circuit
        # before attempting a manual merge.  Distinguish auto-merge
        # success from closed-without-merge using the ``merged`` boolean
        # captured from the refresh payload.
        if not closed_during_wait:
            return None
        if merged_during_wait:
            result.status = MergeStatus.MERGED
            self._pr_status(
                f"✅ Merged (auto-merge): {pr_info.html_url}",
                level="debug",
            )
        else:
            result.status = MergeStatus.CLOSED
            result.error = (
                "PR closed without merging during auto-merge wait "
                "(no operator follow-up needed)"
            )
            self._pr_status(
                f"🚪 Closed without merging: {pr_info.html_url}",
                level="warning",
            )
        return result

    async def _recheck_late_precommit(self, flow: _MergeFlow) -> MergeResult | None:
        """Re-trigger a pre-commit.ci run that went stale during the wait.

        The wait can expire with the PR still blocked on a pre-commit.ci
        run that will never finish.  Step 0.5 only retriggers a run that
        was already stale when processing *started*; a run that went
        pending shortly before this run began crosses the stuck
        threshold *during* the wait, so without this re-check the merge
        below fails on the pending check without the recovery macro ever
        being posted.  The helper re-gates on required-check status,
        pending age, and duplicate comments, so this is a no-op unless
        the run is genuinely stuck.
        """
        pr_info = flow.pr_info
        if pr_info.mergeable_state != "blocked" or self._github_client is None:
            return None

        late_precommit_fixed = await self._trigger_stale_precommit_ci(pr_info)
        if not late_precommit_fixed:
            return None

        # pre-commit.ci now reports success.  Auto-merge was armed
        # before the wait, so GitHub may merge the PR the moment the
        # check lands — re-fetch and short-circuit on a closed PR before
        # attempting a manual merge below.
        late_updated: Any = None
        try:
            late_updated = await self._github_client.get(
                f"/repos/{flow.repo_owner}/{flow.repo_name}/pulls/{pr_info.number}"
            )
        except Exception as e:
            self.log.debug(
                "Failed to refresh PR %s state after post-wait pre-commit.ci rerun: %s",
                flow.pr_key,
                e,
            )
        if not isinstance(late_updated, dict):
            return None
        return self._apply_late_refresh(flow, late_updated)

    def _apply_late_refresh(
        self, flow: _MergeFlow, late_updated: dict[str, Any]
    ) -> MergeResult | None:
        """Fold a post-wait PR payload back into the snapshot."""
        pr_info = flow.pr_info
        result = flow.result
        if late_updated.get("state") == "closed":
            pr_info.state = "closed"
            if late_updated.get("merged", False):
                result.status = MergeStatus.MERGED
                self._pr_status(
                    f"✅ Merged (auto-merge): {pr_info.html_url}",
                    level="debug",
                )
            else:
                result.status = MergeStatus.CLOSED
                result.error = (
                    "PR closed without merging during auto-merge wait "
                    "(no operator follow-up needed)"
                )
                self._pr_status(
                    f"🚪 Closed without merging: {pr_info.html_url}",
                    level="warning",
                )
            return result

        # Only accept concrete values: GitHub returns null / "" /
        # "unknown" while it recomputes mergeability right after the
        # check lands, and clobbering the known snapshot with those
        # would change the downstream routing (mirrors the guards in the
        # ``_wait_for_auto_merge`` refresh).
        if late_updated.get("mergeable") is not None:
            pr_info.mergeable = late_updated.get("mergeable")
        late_state = late_updated.get("mergeable_state")
        if late_state not in (None, "", "unknown"):
            pr_info.mergeable_state = late_state
        updated_head = (late_updated.get("head") or {}).get("sha")
        if updated_head:
            pr_info.head_sha = updated_head
        return None
