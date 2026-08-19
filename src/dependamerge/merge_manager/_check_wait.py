# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Step 5.5: waiting for a pull request's required checks to complete.

Decides whether waiting can plausibly change the outcome, arms
auto-merge and runs the bounded wait when it can, and folds whatever
happened during the wait back into the snapshot the merge dispatch will
act on.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._merge_state import _Attempt, _RebaseOutcome, _WaitOutcome
from ._models import (
    MergeResult,
    MergeStatus,
)

if TYPE_CHECKING:
    from ..github_async import GitHubAsync


class _CheckWaitMixin(_MergeManagerBase):
    """Step 5.5: waiting for a pull request's required checks."""

    async def _wait_for_required_checks(
        self, attempt: _Attempt, rebased: _RebaseOutcome
    ) -> _WaitOutcome:
        """Wait for required checks when the decision says it is worth it.

        The decision is returned alongside the result because two later
        gates — the auto-merge skip and the required-workflow retry —
        branch on whether this wait ran and on whether Step 5 had
        already spent the pull request's wait budget.
        """
        should_wait, already_rebased = self._plan_required_check_wait(attempt, rebased)
        result = await self._run_required_check_wait(attempt) if should_wait else None
        return _WaitOutcome(
            result=result,
            should_wait=should_wait,
            already_rebased=already_rebased,
        )

    def _plan_required_check_wait(
        self, attempt: _Attempt, rebased: _RebaseOutcome
    ) -> tuple[bool, bool]:
        """Decide whether waiting for required checks can help.

        If the pull request is still blocked (e.g. by a pending required
        status check such as pre-commit.ci) or unstable (a non-required
        check failed), enabling auto-merge and waiting for required
        checks to complete is worthwhile.  Waiting is skipped when:

        * ``preview_mode`` (no side effects)
        * ``force_level == "all"`` (force semantics bypass the wait)
        * Step 5 already ran a rebase + wait for this pull request
          (avoid doubling the configured ``merge_timeout``)
        * ``mergeable_state == "blocked"`` for a reason that cannot
          resolve on its own (e.g. "requires approval", missing
          code-owner reviews) — waiting would just delay the inevitable
          failure/merge by up to ``merge_timeout``.

        ``behind`` pull requests deliberately do NOT wait here: unless
        branch protection enforces the strict up-to-date policy (in
        which case Step 5 already refreshed the branch), a
        behind-but-green pull request merges directly, so parking it in
        the wait loop — where the state never advances on its own —
        would just burn the full ``merge_timeout`` before the merge
        attempt that was going to succeed anyway.

        Exception: when Step 5 just dispatched an *asynchronous* rebase
        (local force-push or the ``@dependabot rebase`` macro) and
        auto-merge could not be armed, those paths leave ``_rebased_prs``
        unset precisely so this wait can bridge the gap while the rebase
        lands and GitHub recomputes mergeability — the snapshot still
        reads ``behind`` because neither path refreshes ``pr_info``.
        Without the wait, Step 6 would fire a manual merge against the
        stale state and 405.  ``needs_rebase`` captures "Step 5 actually
        ran" and the ``not already_rebased`` guard excludes the
        auto-merge-armed case.

        Any ``mergeable`` value (including ``False``) is accepted when
        the state is one of these auto-merge-rescuable states, because
        GitHub returns ``mergeable=False`` transiently while computing
        the value or when a non-required check failed.  The block-reason
        pre-check below still weeds out genuinely-stuck cases (missing
        approvals, etc.) so we don't burn ``merge_timeout`` on them.

        Returns ``(should_wait, already_rebased)``.  ``already_rebased``
        is sampled here, before the wait runs, because the later
        auto-merge and required-workflow gates must see the state the
        wait decision was made against.
        """
        pr_info = attempt.pr_info
        already_rebased = attempt.pr_key in self._rebased_prs
        # ``unstable`` means a non-required check is failing or
        # pending but the PR is otherwise mergeable.  When GitHub
        # also reports ``mergeable is True`` the green button is
        # live and a direct merge succeeds *now*, so entering the
        # auto-merge wait would be pure waste: the state never
        # reaches ``clean`` (the non-required check stays red, e.g.
        # an excluded Zizmor scan), so the loop burns the full
        # ``merge_timeout``; and ``enablePullRequestAutoMerge``
        # is rejected outright on an already-mergeable PR, so the
        # wait isn't even backed by auto-merge.  Route those
        # straight to the Step 6 direct merge.  We still wait on
        # ``unstable`` when ``mergeable`` is not literally True
        # (GitHub still computing the value, or a required check
        # transiently failing) so a genuinely not-yet-ready PR is
        # not merged prematurely.
        state_is_waitable = (
            pr_info.mergeable_state == "blocked"
            or (pr_info.mergeable_state == "unstable" and pr_info.mergeable is not True)
            or (pr_info.mergeable_state == "behind" and rebased.needs_rebase)
        )
        base_should_wait = (
            not self.preview_mode
            and self._github_client is not None
            and state_is_waitable
            and self.force_level != "all"
            and not already_rebased
        )

        # For ``blocked`` PRs, consult the block-reason analysis
        # (computed once in Step 5) before entering the wait loop so
        # we don't burn the full merge_timeout on PRs blocked for
        # reasons that cannot resolve on their own.
        should_wait = base_should_wait
        if base_should_wait and pr_info.mergeable_state == "blocked":
            if not rebased.blocked_analysis_ok:
                # Treat analysis failures as 'do not wait' so we
                # don't burn the full ``merge_timeout`` on a PR
                # whose block reason we cannot classify. The PR
                # will fall through to the Step 6 skip gate (which
                # re-consults the analysis) and either defer to
                # auto-merge or surface a manual-merge error
                # promptly.
                should_wait = False
            elif rebased.blocked_reason is not None:
                if not self._block_reason_indicates_pending_checks(
                    rebased.blocked_reason
                ):
                    self.log.debug(
                        "Skipping Step 5.5 wait for %s: block "
                        "reason '%s' will not resolve on its own",
                        attempt.pr_key,
                        rebased.blocked_reason,
                    )
                    should_wait = False
        return should_wait, already_rebased

    async def _run_required_check_wait(self, attempt: _Attempt) -> MergeResult | None:
        """Arm auto-merge and wait, bounded by ``merge_timeout``.

        Approve-on-demand: arming auto-merge implies we want the pull
        request to merge once checks pass, so the current head is
        approved first (idempotently) before enabling.  The
        continue-states mirror the entry condition in
        ``_plan_required_check_wait`` (blocked / behind / unstable).

        Returns a terminal ``MergeResult`` when the pull request closed
        during the wait, and ``None`` when the flow should continue to
        the merge dispatch.
        """
        pr_info = attempt.pr_info
        if attempt.pr_key not in self._auto_merge_enabled:
            auto_ok_pre = await self._enable_auto_merge_with_approval(
                pr_info, attempt.owner, attempt.repo
            )
            if auto_ok_pre:
                self._pr_status(
                    f"🤖 Auto-merge: {pr_info.html_url}",
                    level="debug",
                )

        self._track_pr_state(pr_info, "waiting")
        (
            closed_during_wait,
            merged_during_wait,
        ) = await self._wait_for_auto_merge(
            pr_info,
            attempt.owner,
            attempt.repo,
            continue_states=("blocked", "behind", "unstable"),
            measures_checks=True,
        )
        self._track_pr_state(pr_info, None)

        # If the wait revealed the PR is already closed,
        # short-circuit before attempting a manual merge.
        # Distinguish auto-merge success from
        # closed-without-merge using the ``merged`` boolean
        # captured from the refresh payload.
        if closed_during_wait:
            return self._record_wait_close(attempt, merged_during_wait)

        return await self._recheck_late_precommit(attempt)

    def _record_wait_close(self, attempt: _Attempt, merged: bool) -> MergeResult:
        """Record a pull request that closed during the auto-merge wait.

        Both places that observe the close — the wait's own return
        value, and the refresh after a late pre-commit.ci rerun — report
        the same two outcomes in the same words, so the status and the
        error text are produced once here: the end-of-run summary groups
        pull requests by that exact ``error`` string, and two copies of
        it could drift apart.
        """
        result = attempt.result
        if merged:
            result.status = MergeStatus.MERGED
            self._pr_status(
                f"✅ Merged (auto-merge): {attempt.pr_info.html_url}",
                level="debug",
            )
        else:
            result.status = MergeStatus.CLOSED
            result.error = (
                "PR closed without merging during auto-merge wait "
                "(no operator follow-up needed)"
            )
            self._pr_status(
                f"🚪 Closed without merging: {attempt.pr_info.html_url}",
                level="warning",
            )
        return result

    async def _recheck_late_precommit(self, attempt: _Attempt) -> MergeResult | None:
        """Retrigger a pre-commit.ci run that went stuck during the wait.

        The wait can expire with the pull request still blocked on a
        pre-commit.ci run that will never finish.  Step 0.5 only
        retriggers a run that was already stale when processing
        *started*; a run that went pending shortly before this run began
        crosses the stuck threshold *during* the wait, so without this
        re-check the merge below fails on the pending check without the
        recovery macro ever being posted.  The helper re-gates on
        required-check status, pending age, and duplicate comments, so
        this is a no-op unless the run is genuinely stuck.

        When it does fire, auto-merge was armed before the wait, so
        GitHub may merge the pull request the moment the check lands.
        The pull request is therefore re-read, and a closed one
        short-circuits before a manual merge is attempted.
        """
        pr_info = attempt.pr_info
        client = self._github_client
        if pr_info.mergeable_state != "blocked" or client is None:
            return None

        late_precommit_fixed = await self._trigger_stale_precommit_ci(pr_info)
        if not late_precommit_fixed:
            return None

        late_updated = await self._fetch_late_pr_state(attempt, client)
        if not isinstance(late_updated, dict):
            return None
        if late_updated.get("state") == "closed":
            pr_info.state = "closed"
            return self._record_wait_close(
                attempt, bool(late_updated.get("merged", False))
            )
        self._apply_late_pr_refresh(pr_info, late_updated)
        return None

    async def _fetch_late_pr_state(self, attempt: _Attempt, client: GitHubAsync) -> Any:
        """Re-read the pull request after a late pre-commit.ci rerun.

        A failed read is logged and answered with ``None``: the merge
        dispatch can still proceed against the snapshot already held, so
        a transient error here must not end an otherwise healthy
        attempt.
        """
        try:
            return await client.get(
                f"/repos/{attempt.owner}/{attempt.repo}/pulls/{attempt.pr_info.number}"
            )
        except Exception as e:
            self.log.debug(
                "Failed to refresh PR %s state after post-wait pre-commit.ci rerun: %s",
                attempt.pr_key,
                e,
            )
            return None

    @staticmethod
    def _apply_late_pr_refresh(
        pr_info: PullRequestInfo, payload: dict[str, Any]
    ) -> None:
        """Fold a post-rerun refresh into the snapshot, ignoring nulls.

        Only concrete values are accepted: GitHub returns null / "" /
        "unknown" while it recomputes mergeability right after the check
        lands, and clobbering the known snapshot with those would change
        the downstream routing.  Mirrors the guards in the
        ``_wait_for_auto_merge`` refresh.
        """
        if payload.get("mergeable") is not None:
            pr_info.mergeable = payload.get("mergeable")
        late_state = payload.get("mergeable_state")
        if late_state not in (None, "", "unknown"):
            pr_info.mergeable_state = late_state
        updated_head = (payload.get("head") or {}).get("sha")
        if updated_head:
            pr_info.head_sha = updated_head
