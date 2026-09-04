# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""The recreate path's terminal states, gates and wait budget.

Three defects converged on this path, and the corrected return contract
is what ties them together.

**#441** --- auto-merge is armed on the replacement *before* the checks
wait begins, so it can complete between polls.  A closed/merged payload
satisfied neither the ready test nor the ``dirty`` test, so the loop ran
to timeout and the caller reported a failure for a PR that had in fact
merged.  Under-reporting a success is the most damaging direction to be
wrong in, and it cost the full wait window to get there.

**#440** --- a stuck required check set the recreate in motion, but the
trigger applied signature-only gates, so on any repository that does not
enforce commit signatures the recovery was inert.  The user was still
told ``requesting recreate``, so the failure was silent unless someone
read debug logs.

**#434** --- the recreate poll and the nested checks wait each ran their
own full budget, ignoring the run-wide ceiling.  A single PR could
consume ``2 x merge_timeout`` beyond ``--max-wait`` while holding its
slot lease, and ``--max-wait 0`` --- documented as "never block" ---
blocked anyway.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dependamerge.merge_manager import (
    MergeResult,
    MergeStatus,
    RecreateCause,
    RecreateOutcome,
    RecreateResult,
)
from dependamerge.merge_manager._single_pr_context import _MergeFlow
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

REPO = "lfreleng-actions/some-repo"


def _pr(number: int = 1) -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
        title="Chore: Bump a dependency",
        body=None,
        author="dependabot[bot]",
        head_sha="a" * 40,
        base_branch="main",
        head_branch="dependabot/pip/thing-1.2.3",
        state="open",
        mergeable=False,
        mergeable_state="blocked",
        behind_by=None,
        files_changed=[],
        repository_full_name=REPO,
        html_url=f"https://github.com/{REPO}/pull/{number}",
        reviews=[],
        review_comments=[],
    )


def _payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "number": 107,
        "node_id": "PR_kw107",
        "title": "Chore: Bump a dependency",
        "state": "open",
        "user": {"login": "dependabot[bot]"},
        "head": {"sha": "b" * 40, "ref": "dependabot/pip/thing-1.2.3"},
        "base": {"ref": "main"},
        "html_url": f"https://github.com/{REPO}/pull/107",
    }
    base.update(overrides)
    return base


def _mgr(**overrides: Any):
    mgr, client = make_merge_manager(**overrides)
    # The wait approves *and* arms first. Both are stubbed because
    # PENDING requires both --- see TestTheReplacementIsApprovedNotJustArmed.
    mgr._ensure_pr_approved = AsyncMock(return_value=True)  # type: ignore[method-assign]
    mgr._enable_auto_merge_for_pr = AsyncMock(return_value=True)  # type: ignore[method-assign]
    mgr._merge_recheck_interval = 0.001
    mgr._merge_poll_max_attempts = 3
    return mgr, client


class TestAReplacementThatClosesIsTerminal:
    """Closed is an answer, not a reason to keep polling."""

    @pytest.mark.asyncio
    async def test_merged_by_auto_merge_is_a_success(self) -> None:
        """The defect: this used to poll to timeout and report a failure."""
        mgr, client = _mgr()
        client.get = AsyncMock(
            return_value=_payload(state="closed", merged=True, merged_at="2026-08-25")
        )

        result = await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert result.outcome is RecreateOutcome.MERGED
        assert result.pr_info is not None
        assert result.pr_info.number == 107

    @pytest.mark.asyncio
    async def test_it_stops_polling_immediately(self) -> None:
        """Promptness is half the fix: the old path burned the whole window."""
        mgr, client = _mgr()
        client.get = AsyncMock(
            return_value=_payload(state="closed", merged=True, merged_at="2026-08-25")
        )

        await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_closed_unmerged_is_a_resolved_failure(self) -> None:
        mgr, client = _mgr()
        client.get = AsyncMock(
            return_value=_payload(state="closed", merged=False, merged_at=None)
        )

        result = await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert result.outcome is RecreateOutcome.ABANDONED

    @pytest.mark.asyncio
    async def test_an_ambiguous_payload_is_not_claimed_as_merged(self) -> None:
        """Neither ``merged`` nor ``merged_at`` present: do not guess upward."""
        mgr, client = _mgr()
        client.get = AsyncMock(return_value=_payload(state="closed"))

        result = await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert result.outcome is RecreateOutcome.ABANDONED

    @pytest.mark.asyncio
    async def test_a_clean_replacement_is_still_ready(self) -> None:
        """The path that already worked must be unchanged."""
        mgr, client = _mgr()
        client.get = AsyncMock(
            return_value=_payload(mergeable=True, mergeable_state="clean")
        )
        mgr._recreated_pr_files = AsyncMock(return_value=[])  # type: ignore[method-assign]

        result = await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert result.outcome is RecreateOutcome.READY
        assert result.pr_info is not None


class TestAStuckCheckCanActuallyTriggerARecreate:
    """The gates belong to the unsigned-commit cause, not to every cause."""

    @pytest.mark.asyncio
    async def test_a_stuck_check_posts_the_macro_without_signatures(self) -> None:
        """The defect: this returned at the first gate and posted nothing."""
        mgr, client = _mgr()
        # A repository that does not require signed commits at all.
        client.requires_commit_signatures = AsyncMock(return_value=False)
        client.check_pr_commit_signatures = AsyncMock(return_value=(True, []))
        client.get = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock(return_value=None)
        mgr._await_dependabot_recreate = AsyncMock()  # type: ignore[method-assign]

        await mgr._trigger_dependabot_recreate(_pr(), RecreateCause.STUCK_CHECK)

        client.post_issue_comment.assert_awaited_once()
        await_args = client.post_issue_comment.await_args
        assert await_args is not None
        assert "@dependabot recreate" in await_args.args[3]

    @pytest.mark.asyncio
    async def test_a_stuck_check_ignores_verified_commits(self) -> None:
        """Signed commits are the common case, and irrelevant to a stall."""
        mgr, client = _mgr()
        client.requires_commit_signatures = AsyncMock(return_value=True)
        client.check_pr_commit_signatures = AsyncMock(return_value=(True, []))
        client.get = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock(return_value=None)
        mgr._await_dependabot_recreate = AsyncMock()  # type: ignore[method-assign]

        await mgr._trigger_dependabot_recreate(_pr(), RecreateCause.STUCK_CHECK)

        client.post_issue_comment.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_unsigned_cause_still_applies_both_gates(self) -> None:
        mgr, client = _mgr()
        client.requires_commit_signatures = AsyncMock(return_value=False)
        client.post_issue_comment = AsyncMock(return_value=None)

        result = await mgr._trigger_dependabot_recreate(_pr(), RecreateCause.UNSIGNED)

        assert result.outcome is RecreateOutcome.NONE
        client.post_issue_comment.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_non_dependabot_pr_is_rejected_on_both_causes(self) -> None:
        for cause in (RecreateCause.UNSIGNED, RecreateCause.STUCK_CHECK):
            mgr, client = _mgr()
            client.post_issue_comment = AsyncMock(return_value=None)
            human = _pr()
            human.author = "a-human"

            result = await mgr._trigger_dependabot_recreate(human, cause)

            assert result.outcome is RecreateOutcome.NONE, cause
            client.post_issue_comment.assert_not_awaited()


class TestTheRecreatePathHonoursTheWaitCeiling:
    """``--max-wait`` bounds the run; the recreate path used to escape it."""

    @pytest.mark.asyncio
    async def test_no_wait_skips_the_recreate_poll(self) -> None:
        """``--max-wait 0`` promises never to block."""
        mgr, client = _mgr(max_wait=0)
        mgr._no_wait = True
        client.get = AsyncMock(return_value=_payload())

        result = await mgr._await_dependabot_recreate("o", "r", _pr())

        assert result.outcome is RecreateOutcome.NONE
        client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_precommit_sleep_is_clamped_near_the_deadline(self) -> None:
        """The pre-commit loop needs the same proof as the recreate one.

        Only ``_no_wait`` was covered here, so swapping the clamped
        sleep back for a full-interval one would have left the suite
        green while reintroducing the ``--max-wait`` overshoot.
        """
        mgr, client = _mgr()
        mgr._merge_recheck_interval = 5.0
        loop = asyncio.get_running_loop()
        mgr._run_deadline = loop.time() + 0.1
        client.get = AsyncMock(return_value={})
        slept: list[float] = []

        async def _sleep(seconds: float) -> None:
            slept.append(seconds)

        import dependamerge.merge_manager as mod

        original = mod.asyncio.sleep
        mod.asyncio.sleep = _sleep  # type: ignore[assignment]
        try:
            await mgr._await_precommit_ci("o", "r", _pr())
        finally:
            mod.asyncio.sleep = original  # type: ignore[assignment]

        assert slept, "expected at least one sleep"
        assert max(slept) <= 0.1, slept

    @pytest.mark.asyncio
    async def test_no_wait_skips_the_precommit_poll(self) -> None:
        mgr, client = _mgr(max_wait=0)
        mgr._no_wait = True
        client.get = AsyncMock(return_value={})

        assert await mgr._await_precommit_ci("o", "r", _pr()) is False
        client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_expired_run_deadline_stops_the_recreate_poll(self) -> None:
        mgr, client = _mgr()
        loop = asyncio.get_running_loop()
        mgr._run_deadline = loop.time() - 1.0
        client.get = AsyncMock(return_value=_payload())

        result = await mgr._await_dependabot_recreate("o", "r", _pr())

        assert result.outcome is RecreateOutcome.NONE
        client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_nested_wait_shares_the_outer_deadline(self) -> None:
        """The compounding case: no second full budget.

        Passing an already-expired deadline models a recreate that found
        its replacement right at the ceiling. The nested wait must stop
        rather than start a fresh ``merge_timeout`` of its own --- and
        must not poll at all.
        """
        mgr, client = _mgr()
        loop = asyncio.get_running_loop()
        client.get = AsyncMock(return_value=_payload())

        result = await mgr._wait_for_recreated_pr_checks(
            "o", "r", 107, _payload(), loop.time() - 1.0
        )

        # Stopped immediately: no polling happened.
        client.get.assert_not_awaited()
        # But the replacement is armed and in flight, so this is not
        # "nothing found" --- see TestGivingUpIsNotTheSameAsFinding.
        assert result.outcome is RecreateOutcome.PENDING

    @pytest.mark.asyncio
    async def test_a_near_deadline_sleep_is_clamped(self) -> None:
        """Checking the deadline is not enough; the sleep must be clamped.

        With less than one interval of budget left, sleeping a whole
        interval overshoots the ceiling. The established waits clamp to
        the remaining budget, and these now match.
        """
        mgr, client = _mgr()
        mgr._merge_recheck_interval = 5.0
        loop = asyncio.get_running_loop()
        client.get = AsyncMock(return_value=_payload())
        slept: list[float] = []

        async def _sleep(seconds: float) -> None:
            slept.append(seconds)

        import dependamerge.merge_manager as mod

        original = mod.asyncio.sleep
        mod.asyncio.sleep = _sleep  # type: ignore[assignment]
        try:
            await mgr._wait_for_recreated_pr_checks(
                "o", "r", 107, _payload(), loop.time() + 0.1
            )
        finally:
            mod.asyncio.sleep = original  # type: ignore[assignment]

        assert slept, "expected at least one sleep"
        assert max(slept) <= 0.1, slept


class TestAFailingWaitIsNotRetried:
    """The search guard must not swallow a failure inside the wait.

    ``_find_recreated_pr`` previously wrapped the entire multi-minute
    wait in its ``try``, so an exception there was reported as a failed
    search and the outer loop repeated the whole wait --- contradicting
    its own comment, *"always return after the first wait attempt"*.
    """

    @pytest.mark.asyncio
    async def test_an_exception_in_the_wait_propagates(self) -> None:
        mgr, client = _mgr()
        client.get = AsyncMock(return_value=[_payload()])
        mgr._wait_for_recreated_pr_checks = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("wait blew up")
        )

        with pytest.raises(RuntimeError, match="wait blew up"):
            await mgr._find_recreated_pr("o", "r", _pr())

        # One search, one wait --- not a second full wait.
        assert mgr._wait_for_recreated_pr_checks.await_count == 1

    @pytest.mark.asyncio
    async def test_a_failing_search_is_still_tolerated(self) -> None:
        """The guard still covers the request it was written for."""
        mgr, client = _mgr()
        client.get = AsyncMock(side_effect=RuntimeError("search failed"))

        resolved, result = await mgr._find_recreated_pr("o", "r", _pr())

        assert resolved is False
        assert result.outcome is RecreateOutcome.NONE


class TestGivingUpIsNotTheSameAsFinding:
    """An armed replacement left in flight is pending, not a failure.

    Auto-merge is armed on the replacement before the wait begins, so
    once we stop waiting --- ceiling, exhausted budget, or
    ``--max-wait 0`` --- the PR is still open and still expected to
    merge. Reporting the original as FAILED would under-report a
    success in exactly the way the closed/merged case did, and
    ``_confirm_failure`` cannot correct it: it rechecks the *original*
    PR, not the replacement.

    ``NONE`` is therefore reserved for the cases where nothing was
    acted on --- including a replacement that could not be armed, which
    will not merge by itself.
    """

    @pytest.mark.asyncio
    async def test_an_exhausted_budget_is_pending(self) -> None:
        mgr, client = _mgr()
        # Never resolves, so the poll budget runs out.
        client.get = AsyncMock(return_value=_payload(mergeable_state="blocked"))

        result = await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert result.outcome is RecreateOutcome.PENDING
        assert result.pr_info is not None
        assert result.pr_info.number == 107

    @pytest.mark.asyncio
    async def test_no_wait_is_pending_and_does_not_poll(self) -> None:
        """``--max-wait 0``: arm, report pending, move on.

        The guard belongs to this wait too, not only to its callers: a
        direct call with the default ``deadline=None`` would otherwise
        run the whole poll loop under a flag that promises never to
        block.
        """
        mgr, client = _mgr(max_wait=0)
        mgr._no_wait = True
        client.get = AsyncMock(return_value=_payload(mergeable_state="blocked"))

        result = await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert result.outcome is RecreateOutcome.PENDING
        client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_wait_still_arms_auto_merge(self) -> None:
        """Fire-and-forget only works if the fire part still happens."""
        mgr, client = _mgr(max_wait=0)
        mgr._no_wait = True
        client.get = AsyncMock(return_value=_payload())

        await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        mgr._enable_auto_merge_for_pr.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_unarmed_replacement_is_not_pending(self) -> None:
        """Nothing will happen on its own, so this is not in flight."""
        mgr, client = _mgr()
        mgr._enable_auto_merge_for_pr = AsyncMock(return_value=False)  # type: ignore[method-assign]
        client.get = AsyncMock(return_value=_payload(mergeable_state="blocked"))

        result = await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert result.outcome is RecreateOutcome.NONE

    @pytest.mark.asyncio
    async def test_a_replacement_without_a_node_id_is_not_pending(self) -> None:
        """Auto-merge cannot be armed without one, so it cannot be in flight."""
        mgr, client = _mgr()
        payload = _payload(mergeable_state="blocked")
        payload.pop("node_id")
        client.get = AsyncMock(return_value=payload)

        result = await mgr._wait_for_recreated_pr_checks("o", "r", 107, payload)

        assert result.outcome is RecreateOutcome.NONE
        mgr._enable_auto_merge_for_pr.assert_not_awaited()


class TestTheReplacementIsApprovedNotJustArmed:
    """``PENDING`` claims the replacement will merge without us.

    Arming auto-merge commits GitHub to finish the merge once branch
    protection is satisfied, so the head must already carry this run's
    approval, or auto-merge waits forever on a missing review. The
    approval in ``_merge_recreated_pr`` is reached only for ``READY``,
    so a replacement we stop waiting on has no other chance to get one.

    ``_enable_auto_merge_with_approval`` is deliberately not used: it
    swallows non-permission approval failures and returns the *arming*
    result, so its ``True`` evidences only that auto-merge is active.
    That is fine where a real merge follows, but ``PENDING`` is a claim
    about what happens without us, so the two signals are kept apart.
    """

    @pytest.mark.asyncio
    async def test_both_approval_and_arming_are_required(self) -> None:
        mgr, client = _mgr()
        client.get = AsyncMock(return_value=_payload(mergeable_state="blocked"))

        result = await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert result.outcome is RecreateOutcome.PENDING
        mgr._ensure_pr_approved.assert_awaited_once()
        mgr._enable_auto_merge_for_pr.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_already_approved_replacement_is_pending(self) -> None:
        """``False`` means "already sufficiently approved", not "declined".

        ``_ensure_pr_approved`` returns True only when it submits a
        *new* review, and False when the head already carries adequate
        approval --- from us or anyone else. Treating False as "not
        approved" would reject exactly the replacements ready to merge.
        """
        mgr, client = _mgr()
        mgr._ensure_pr_approved = AsyncMock(return_value=False)  # type: ignore[method-assign]
        client.get = AsyncMock(return_value=_payload(mergeable_state="blocked"))

        result = await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert result.outcome is RecreateOutcome.PENDING

    @pytest.mark.asyncio
    async def test_a_swallowed_approval_error_is_not_pending(self) -> None:
        """A transient approval failure must not become a promise."""
        mgr, client = _mgr()
        mgr._ensure_pr_approved = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("transient approval hiccup")
        )
        client.get = AsyncMock(return_value=_payload(mergeable_state="blocked"))

        result = await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert result.outcome is RecreateOutcome.NONE
        # Arming is still attempted: it may help where no review is
        # required, it simply cannot be promised.
        mgr._enable_auto_merge_for_pr.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_permission_error_propagates(self) -> None:
        """Typed permission errors reach the caller's dedicated handler."""
        from dependamerge.github_async import PermissionError as GitHubPermissionError

        mgr, client = _mgr()
        mgr._ensure_pr_approved = AsyncMock(  # type: ignore[method-assign]
            side_effect=GitHubPermissionError(
                operation="approve", message="token lacks the required scope"
            )
        )
        client.get = AsyncMock(return_value=_payload(mergeable_state="blocked"))

        with pytest.raises(GitHubPermissionError):
            await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

    @pytest.mark.asyncio
    async def test_it_is_approved_against_the_replacement(self) -> None:
        """Not the original PR: approval must land on the new head."""
        mgr, client = _mgr()
        client.get = AsyncMock(return_value=_payload(mergeable_state="blocked"))

        await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        approved = mgr._ensure_pr_approved.await_args.args[0]
        assert approved.number == 107
        assert approved.node_id == "PR_kw107"


class TestTheFlowActsOnTheOutcome:
    """The contract is only worth having if the caller honours it."""

    def _flow(self, mgr) -> _MergeFlow:
        pr = _pr()
        return _MergeFlow(
            pr_info=pr,
            repo_owner="lfreleng-actions",
            repo_name="some-repo",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )

    @pytest.mark.asyncio
    async def test_a_merged_replacement_is_not_merged_again(self) -> None:
        """``MERGED`` records the success without a second merge call."""
        mgr, _ = _mgr()
        replacement = _pr(number=107)
        mgr._get_failure_summary = AsyncMock(return_value=("blocked", True))  # type: ignore[method-assign]
        mgr._external_closure_result = AsyncMock(return_value=None)  # type: ignore[method-assign]
        mgr._behind_auto_merge_result = lambda flow: None  # type: ignore[method-assign]
        mgr._maybe_recreate_dependabot_pr = AsyncMock(  # type: ignore[method-assign]
            return_value=RecreateResult(RecreateOutcome.MERGED, replacement)
        )
        mgr._merge_recreated_pr = AsyncMock()  # type: ignore[method-assign]
        mgr._report_merge_failure = AsyncMock()  # type: ignore[method-assign]

        flow = self._flow(mgr)
        await mgr._handle_failed_merge(flow)

        mgr._merge_recreated_pr.assert_not_awaited()
        mgr._report_merge_failure.assert_not_awaited()
        assert flow.result.status is MergeStatus.MERGED
        assert flow.result.pr_info is replacement

    @pytest.mark.asyncio
    async def test_a_ready_replacement_is_merged(self) -> None:
        mgr, _ = _mgr()
        replacement = _pr(number=107)
        mgr._get_failure_summary = AsyncMock(return_value=("blocked", True))  # type: ignore[method-assign]
        mgr._external_closure_result = AsyncMock(return_value=None)  # type: ignore[method-assign]
        mgr._behind_auto_merge_result = lambda flow: None  # type: ignore[method-assign]
        mgr._maybe_recreate_dependabot_pr = AsyncMock(  # type: ignore[method-assign]
            return_value=RecreateResult(RecreateOutcome.READY, replacement)
        )
        mgr._merge_recreated_pr = AsyncMock()  # type: ignore[method-assign]
        mgr._report_merge_failure = AsyncMock()  # type: ignore[method-assign]

        await mgr._handle_failed_merge(self._flow(mgr))

        mgr._merge_recreated_pr.assert_awaited_once()
        await_args = mgr._merge_recreated_pr.await_args
        assert await_args is not None
        assert await_args.args[1] is replacement

    @pytest.mark.asyncio
    async def test_an_abandoned_replacement_is_blocked_not_failed(self) -> None:
        """``BLOCKED`` survives ``_confirm_failure``; ``FAILED`` would not.

        After a recreate the *original* PR is closed unmerged by
        dependabot, so a ``FAILED`` result is rewritten to ``CLOSED``
        --- documented as "nothing for the operator to follow up on",
        the opposite of what a conflicted replacement needs.
        """
        mgr, _ = _mgr()
        replacement = _pr(number=107)
        mgr._get_failure_summary = AsyncMock(return_value=("blocked", True))  # type: ignore[method-assign]
        mgr._external_closure_result = AsyncMock(return_value=None)  # type: ignore[method-assign]
        mgr._behind_auto_merge_result = lambda flow: None  # type: ignore[method-assign]
        mgr._maybe_recreate_dependabot_pr = AsyncMock(  # type: ignore[method-assign]
            return_value=RecreateResult(RecreateOutcome.ABANDONED, replacement)
        )
        mgr._merge_recreated_pr = AsyncMock()  # type: ignore[method-assign]
        mgr._report_merge_failure = AsyncMock()  # type: ignore[method-assign]

        flow = self._flow(mgr)
        await mgr._handle_failed_merge(flow)

        mgr._merge_recreated_pr.assert_not_awaited()
        mgr._report_merge_failure.assert_not_awaited()
        assert flow.result.status is MergeStatus.BLOCKED
        assert flow.result.pr_info is replacement
        assert flow.result.error and "needs attention" in flow.result.error

    @pytest.mark.asyncio
    async def test_an_abandoned_outcome_survives_confirm_failure(self) -> None:
        """End to end through the confirmation step, not just the branch.

        Drives the real ``_confirm_failure`` against an original PR that
        is closed unmerged --- the state dependabot leaves it in --- and
        asserts the outcome is not rewritten to ``CLOSED``.
        """
        mgr, client = _mgr()
        replacement = _pr(number=107)
        mgr._get_failure_summary = AsyncMock(return_value=("blocked", True))  # type: ignore[method-assign]
        mgr._external_closure_result = AsyncMock(return_value=None)  # type: ignore[method-assign]
        mgr._behind_auto_merge_result = lambda flow: None  # type: ignore[method-assign]
        mgr._maybe_recreate_dependabot_pr = AsyncMock(  # type: ignore[method-assign]
            return_value=RecreateResult(RecreateOutcome.ABANDONED, replacement)
        )
        # The original PR as dependabot leaves it: closed, unmerged.
        client.get = AsyncMock(
            return_value={"state": "closed", "merged": False, "merged_at": None}
        )

        flow = self._flow(mgr)
        await mgr._handle_failed_merge(flow)
        confirmed = await mgr._confirm_failure(flow.pr_info, flow.result)

        assert confirmed.status is MergeStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_a_failed_outcome_would_have_been_rewritten(self) -> None:
        """The control that makes the test above meaningful.

        Confirms ``_confirm_failure`` really does rewrite a ``FAILED``
        result to ``CLOSED`` in this exact situation --- so the previous
        test is pinning a real hazard, not a hypothetical one.
        """
        mgr, client = _mgr()
        pr = _pr()
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="boom")
        client.get = AsyncMock(
            return_value={"state": "closed", "merged": False, "merged_at": None}
        )

        confirmed = await mgr._confirm_failure(pr, result)

        assert confirmed.status is MergeStatus.CLOSED

    @pytest.mark.asyncio
    async def test_a_pending_replacement_is_reported_as_pending(self) -> None:
        """Not merged again, and not reported as a failure either."""
        mgr, _ = _mgr()
        replacement = _pr(number=107)
        mgr._get_failure_summary = AsyncMock(return_value=("blocked", True))  # type: ignore[method-assign]
        mgr._external_closure_result = AsyncMock(return_value=None)  # type: ignore[method-assign]
        mgr._behind_auto_merge_result = lambda flow: None  # type: ignore[method-assign]
        mgr._maybe_recreate_dependabot_pr = AsyncMock(  # type: ignore[method-assign]
            return_value=RecreateResult(RecreateOutcome.PENDING, replacement)
        )
        mgr._merge_recreated_pr = AsyncMock()  # type: ignore[method-assign]
        mgr._report_merge_failure = AsyncMock()  # type: ignore[method-assign]

        flow = self._flow(mgr)
        await mgr._handle_failed_merge(flow)

        mgr._merge_recreated_pr.assert_not_awaited()
        mgr._report_merge_failure.assert_not_awaited()
        assert flow.result.status is MergeStatus.AUTO_MERGE_PENDING
        assert flow.result.pr_info is replacement
        # The reason must land on ``error``: that is the field the
        # end-of-run report reads, so ``warning`` would show the
        # operator "no reason reported".
        assert flow.result.error and "auto-merge pending" in flow.result.error


class TestTheCauseIsSelectedNotAssumed:
    """The gate fix only helps if the right cause reaches the trigger."""

    def _flow(self, pr: PullRequestInfo) -> _MergeFlow:
        return _MergeFlow(
            pr_info=pr,
            repo_owner="lfreleng-actions",
            repo_name="some-repo",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )

    @pytest.mark.asyncio
    async def test_a_stuck_check_selects_the_stuck_cause(self) -> None:
        """Returning ``UNSIGNED`` here would leave recovery inert again."""
        mgr, _ = _mgr()
        pr = _pr()
        mgr._detect_stuck_required_check = AsyncMock(  # type: ignore[method-assign]
            return_value=(True, "dco", 900.0)
        )
        mgr._trigger_dependabot_recreate = AsyncMock(  # type: ignore[method-assign]
            return_value=RecreateResult.none()
        )

        await mgr._maybe_recreate_dependabot_pr(self._flow(pr), "blocked by checks")

        mgr._trigger_dependabot_recreate.assert_awaited_once_with(
            pr, RecreateCause.STUCK_CHECK
        )

    @pytest.mark.asyncio
    async def test_branch_protection_selects_the_unsigned_cause(self) -> None:
        mgr, _ = _mgr()
        pr = _pr()
        mgr._trigger_dependabot_recreate = AsyncMock(  # type: ignore[method-assign]
            return_value=RecreateResult.none()
        )

        await mgr._maybe_recreate_dependabot_pr(
            self._flow(pr), "branch protection rules prevent merge"
        )

        mgr._trigger_dependabot_recreate.assert_awaited_once_with(
            pr, RecreateCause.UNSIGNED
        )

    @pytest.mark.asyncio
    async def test_no_recognised_cause_triggers_nothing(self) -> None:
        mgr, _ = _mgr()
        mgr._detect_stuck_required_check = AsyncMock(  # type: ignore[method-assign]
            return_value=(False, None, 0.0)
        )
        mgr._trigger_dependabot_recreate = AsyncMock()  # type: ignore[method-assign]

        result = await mgr._maybe_recreate_dependabot_pr(
            self._flow(_pr()), "some other failure"
        )

        assert result.outcome is RecreateOutcome.NONE
        mgr._trigger_dependabot_recreate.assert_not_awaited()


class TestThePendingReasonReachesTheOperator:
    """The end-of-run report is the only place reasons appear.

    ``_print_failed_pr_details`` covers ``AUTO_MERGE_PENDING`` and reads
    ``r.error``, falling back to "no reason reported". Nothing in the
    reporting path reads ``warning``, so a reason recorded there would
    be invisible --- which is why the recreate-pending reason lives on
    ``error``, as every other auto-merge-pending path already does.
    """

    def test_the_reason_is_rendered(self, capsys) -> None:
        from dependamerge.cli._merge_report import _print_failed_pr_details

        replacement = _pr(number=107)
        result = MergeResult(
            pr_info=replacement,
            status=MergeStatus.AUTO_MERGE_PENDING,
            error=(
                "auto-merge pending: dependabot recreated this PR as "
                "#107, which is approved and armed"
            ),
        )

        _print_failed_pr_details([result])

        out = capsys.readouterr().out
        assert "Auto-merge pending PRs" in out
        assert "no reason reported" not in out
        assert "dependabot recreated this PR" in out

    def test_a_reason_on_warning_would_be_invisible(self, capsys) -> None:
        """Pins *why* the reason must not live on ``warning``.

        If a future change moves it, this fails rather than silently
        degrading the report to "no reason reported".
        """
        from dependamerge.cli._merge_report import _print_failed_pr_details

        result = MergeResult(
            pr_info=_pr(number=107),
            status=MergeStatus.AUTO_MERGE_PENDING,
            warning="this text is never read by the reporter",
        )

        _print_failed_pr_details([result])

        out = capsys.readouterr().out
        assert "no reason reported" in out
        assert "never read by the reporter" not in out


class TestAConflictedReplacementNamesItself:
    """The report must point at the PR that needs attention.

    A conflicted replacement previously returned ``ABANDONED`` with no
    ``pr_info``, so the BLOCKED result kept pointing at the *original*
    PR --- which dependabot has already closed. The operator was sent
    to the wrong PR, and the one actually needing work went unnamed.
    """

    @pytest.mark.asyncio
    async def test_a_dirty_replacement_carries_its_pr(self) -> None:
        mgr, client = _mgr()
        client.get = AsyncMock(return_value=_payload(mergeable_state="dirty"))

        result = await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert result.outcome is RecreateOutcome.ABANDONED
        assert result.pr_info is not None
        assert result.pr_info.number == 107
        assert result.pr_info.html_url.endswith("/pull/107")


class TestTheRecreatedMergeConfirmsBeforeFailing:
    """Auto-merge can win the race against our manual dispatch.

    It is armed on the replacement before the checks wait begins, so it
    can start between the wait's last poll and the dispatch. GitHub then
    answers "merge already in progress", ``_dispatch_recreated_merge``
    swallows it and returns False --- and the outer ``_confirm_failure``
    cannot correct the result, because it rechecks the *original* PR,
    which dependabot has closed. That is the success-under-reporting
    race #441 exists to close, reappearing on the READY path.
    """

    def _flow(self, mgr) -> _MergeFlow:
        pr = _pr()
        return _MergeFlow(
            pr_info=pr,
            repo_owner="lfreleng-actions",
            repo_name="some-repo",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )

    @pytest.mark.asyncio
    async def test_a_replacement_that_merged_meanwhile_is_a_success(self) -> None:
        mgr, client = _mgr()
        replacement = _pr(number=107)
        mgr._approve_pr = AsyncMock(return_value=True)  # type: ignore[method-assign]
        # GitHub refused the dispatch because auto-merge already started.
        mgr._dispatch_recreated_merge = AsyncMock(return_value=False)  # type: ignore[method-assign]
        # The replacement is in fact merged.
        client.get = AsyncMock(
            return_value={"state": "closed", "merged": True, "merged_at": "2026-08-25"}
        )

        flow = self._flow(mgr)
        await mgr._merge_recreated_pr(flow, replacement)

        assert flow.result.status is MergeStatus.MERGED

    @pytest.mark.asyncio
    async def test_a_genuinely_unmerged_replacement_is_blocked(self) -> None:
        """BLOCKED, so the outer confirmation cannot rewrite it to CLOSED."""
        mgr, client = _mgr()
        replacement = _pr(number=107)
        mgr._approve_pr = AsyncMock(return_value=True)  # type: ignore[method-assign]
        mgr._dispatch_recreated_merge = AsyncMock(return_value=False)  # type: ignore[method-assign]
        # The replacement is still open: the merge really did fail.
        client.get = AsyncMock(
            return_value={"state": "open", "merged": False, "merged_at": None}
        )

        flow = self._flow(mgr)
        await mgr._merge_recreated_pr(flow, replacement)

        assert flow.result.status is MergeStatus.BLOCKED
        assert flow.result.pr_info is replacement

        # And it survives the confirmation the run applies afterwards,
        # which rechecks the *original* (closed) PR.
        client.get = AsyncMock(
            return_value={"state": "closed", "merged": False, "merged_at": None}
        )
        confirmed = await mgr._confirm_failure(flow.pr_info, flow.result)
        assert confirmed.status is MergeStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_a_successful_dispatch_is_unchanged(self) -> None:
        mgr, _ = _mgr()
        replacement = _pr(number=107)
        mgr._approve_pr = AsyncMock(return_value=True)  # type: ignore[method-assign]
        mgr._dispatch_recreated_merge = AsyncMock(return_value=True)  # type: ignore[method-assign]

        flow = self._flow(mgr)
        await mgr._merge_recreated_pr(flow, replacement)

        assert flow.result.status is MergeStatus.MERGED
        assert flow.result.pr_info is replacement


class TestTheReplacementApprovalKeyNeverLeaks:
    """Cleared where it is added, so no exit path can strand it.

    ``_ensure_pr_approved`` registers the replacement in
    ``_recently_approved``, but ``_merge_single_pr_impl``'s cleanup
    knows only about the *original* PR. Clearing it in the caller's
    outcome handling would leave it behind whenever an exception or
    cancellation skipped that handling --- while arming, polling, or
    merging the replacement.

    A stale entry makes ``_approve_and_retry_if_review_required`` treat
    a later run as having already approved this PR, silently disabling
    approval recovery for a new head.
    """

    def _key(self, number: int = 107) -> str:
        return f"o/r#{number}"

    @pytest.mark.asyncio
    async def test_it_is_cleared_on_the_normal_path(self) -> None:
        mgr, client = _mgr()
        client.get = AsyncMock(return_value=_payload(mergeable_state="blocked"))

        async def _approve(pr, owner, repo, **_kw):
            mgr._recently_approved.add(f"{owner}/{repo}#{pr.number}")
            return True

        mgr._ensure_pr_approved = _approve  # type: ignore[method-assign]

        await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert self._key() not in mgr._recently_approved

    @pytest.mark.asyncio
    async def test_it_is_cleared_when_arming_raises(self) -> None:
        mgr, client = _mgr()
        client.get = AsyncMock(return_value=_payload(mergeable_state="blocked"))

        async def _approve(pr, owner, repo, **_kw):
            mgr._recently_approved.add(f"{owner}/{repo}#{pr.number}")
            return True

        mgr._ensure_pr_approved = _approve  # type: ignore[method-assign]
        mgr._enable_auto_merge_for_pr = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("arming blew up")
        )

        with pytest.raises(RuntimeError):
            await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert self._key() not in mgr._recently_approved

    @pytest.mark.asyncio
    async def test_it_is_cleared_when_the_wait_is_cancelled(self) -> None:
        """Cancellation must not strand it either."""
        mgr, client = _mgr()

        async def _approve(pr, owner, repo, **_kw):
            mgr._recently_approved.add(f"{owner}/{repo}#{pr.number}")
            return True

        mgr._ensure_pr_approved = _approve  # type: ignore[method-assign]
        mgr._enable_auto_merge_for_pr = AsyncMock(  # type: ignore[method-assign]
            side_effect=asyncio.CancelledError
        )

        with pytest.raises(asyncio.CancelledError):
            await mgr._wait_for_recreated_pr_checks("o", "r", 107, _payload())

        assert self._key() not in mgr._recently_approved
