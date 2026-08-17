# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for bounding waits on required workflows that never started.

A ruleset can require a workflow that GitHub queues but never
dispatches --- typically one hosted in the org's ``.github`` repository.
The PR then reports "Required workflows … are not satisfied" forever.

Two costs follow, both observed in the 503-PR run analysed in
``docs/BULK_RUN_PERFORMANCE_AUDIT.md``:

- the wait loop spends its entire ``merge_timeout`` learning nothing;
- because the striped scheduler serialises a repository's PRs, siblings
  repeat the discovery one after another. ``workflows-template#29``
  burned 300 s and failed, then ``#30`` began its own fresh 300 s wait
  for the identical reason.

These tests pin the detection and the per-head isolation that replaced
the propagation this item originally proposed --- absence is a fact
about one commit, so a sibling's finding is never reused --- and, as
importantly, that an *ambiguous* answer still waits.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from dependamerge.models import PullRequestInfo
from dependamerge.rule_violations import (
    is_rule_violation,
    required_status_check_names,
    required_workflow_names,
    violation_verb,
    workflow_name_fragments,
)
from tests.conftest import make_merge_manager

REPO = "lfreleng-actions/workflows-template"
OWNER, NAME = REPO.split("/")

VIOLATION = (
    "Failed to merge PR #29 in lfreleng-actions/workflows-template. "
    "GitHub: Repository rule violations found "
    "Required workflows 'AI Slop Scan 🧹, Zizmor Scan 🌈' are not satisfied"
)


@pytest.fixture(autouse=True)
def _no_confirm_delay(monkeypatch):
    """Remove the confirmation pause; its behaviour is tested explicitly."""
    monkeypatch.setattr(
        "dependamerge.merge_manager.UNDISPATCHED_CONFIRM_DELAY_SECONDS", 0.0
    )


HEAD_SHA = "f00dcafe" * 5


def _make(head: str = HEAD_SHA):
    """A manager whose client reports *head* as the PR's current commit.

    The confirmation re-reads the head before its second observation, so
    a client that cannot answer would make every test look like a
    force-push. Tests covering that race override ``client.get``.
    """
    mgr, client = make_merge_manager()
    client.get = AsyncMock(return_value={"head": {"sha": head}})
    return mgr, client


def _pr(number: int = 29) -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
        title="CI(actions): bump something",
        body=None,
        author="dependabot[bot]",
        head_sha=HEAD_SHA,
        base_branch="main",
        head_branch="dependabot/x",
        state="open",
        mergeable=True,
        mergeable_state="blocked",
        behind_by=None,
        files_changed=[],
        repository_full_name=REPO,
        html_url=f"https://github.com/{REPO}/pull/{number}",
        reviews=[],
        review_comments=[],
    )


# --------------------------------------------------------------------------
# Parsing the violation string
# --------------------------------------------------------------------------


class TestRuleViolationParsing:
    def test_extracts_workflow_names(self) -> None:
        assert required_workflow_names(VIOLATION) == [
            "AI Slop Scan 🧹",
            "Zizmor Scan 🌈",
        ]

    def test_collapses_repeated_names(self) -> None:
        reason = "Required workflows 'A, B, A' are not satisfied"
        assert required_workflow_names(reason) == ["A", "B"]

    def test_extracts_status_check_names(self) -> None:
        reason = (
            'Repository rule violations found Required status check "DCO" is failing.'
        )
        assert required_status_check_names(reason) == ["DCO"]

    def test_distinguishes_failed_from_unsatisfied(self) -> None:
        assert violation_verb("Required workflows 'A' are not satisfied") == (
            "not satisfied"
        )
        assert violation_verb("Required workflows 'A' failed") == "failed"

    def test_enclosing_exception_text_does_not_poison_the_verb(self) -> None:
        """Regression: the wrapper always begins "Failed to merge PR ...".

        Scanning the whole string finds that "Fail" and reports every
        rejection as ``failed``, including the "not satisfied" case this
        work depends on distinguishing.
        """
        assert violation_verb(VIOLATION) == "not satisfied"

    def test_a_workflow_named_with_fail_does_not_poison_the_verb(self) -> None:
        reason = (
            "Failed to merge PR #1. Repository rule violations found "
            "Required workflows 'Fail Fast Lint' are not satisfied"
        )
        assert violation_verb(reason) == "not satisfied"

    def test_genuine_failure_is_still_reported(self) -> None:
        reason = (
            "Failed to merge PR #1. Repository rule violations found "
            "Required workflows 'Semantic Pull Request' failed"
        )
        assert violation_verb(reason) == "failed"

    def test_recognises_a_rule_violation(self) -> None:
        assert is_rule_violation(VIOLATION)
        assert not is_rule_violation("Merge already in progress")

    @pytest.mark.parametrize("reason", ["", "something else entirely"])
    def test_returns_nothing_for_unrelated_text(self, reason: str) -> None:
        assert required_workflow_names(reason) == []
        assert required_status_check_names(reason) == []


class TestNamesMayContainAnApostrophe:
    """The quote GitHub wraps the list in is not escaped.

    A workflow name is an arbitrary Actions ``name:`` value, so one
    called ``Don't Fail`` puts an apostrophe inside the quoted list.
    Taking the first apostrophe as the closing delimiter reported the
    name as ``Don`` and left ``t Fail' are not satisfied`` as the
    outcome --- read as "failed", which skips the recovery path for a
    workflow that had merely not started.
    """

    APOSTROPHE = (
        "Failed to merge PR #1. Repository rule violations found "
        "Required workflows 'Don't Fail' are not satisfied"
    )

    def test_the_whole_name_survives(self) -> None:
        assert required_workflow_names(self.APOSTROPHE) == ["Don't Fail"]

    def test_the_outcome_is_read_correctly(self) -> None:
        assert violation_verb(self.APOSTROPHE) == "not satisfied"

    def test_a_genuine_failure_is_still_terminal(self) -> None:
        reason = (
            "Repository rule violations found Required workflows 'Don't Fail' failed"
        )
        assert violation_verb(reason) == "failed"
        assert required_workflow_names(reason) == ["Don't Fail"]

    def test_an_apostrophe_in_trailing_prose_is_passed_over(self) -> None:
        """Only an apostrophe an *outcome* follows can be the delimiter."""
        reason = (
            "Repository rule violations found Required workflows 'CI' are "
            "not satisfied (PR state: open) [blocked by GitHub's ruleset]"
        )
        assert required_workflow_names(reason) == ["CI"]
        assert violation_verb(reason) == "not satisfied"

    def test_several_names_with_one_apostrophe_between_them(self) -> None:
        reason = "Required workflows 'Don't Fail, Zizmor Scan 🌈' are not satisfied"
        assert required_workflow_names(reason) == ["Don't Fail", "Zizmor Scan 🌈"]
        assert violation_verb(reason) == "not satisfied"

    def test_ordinary_names_are_unaffected(self) -> None:
        assert required_workflow_names(VIOLATION) == [
            "AI Slop Scan 🧹",
            "Zizmor Scan 🌈",
        ]
        assert violation_verb(VIOLATION) == "not satisfied"

    def test_a_quoted_phrase_inside_a_name_survives(self) -> None:
        """The closing quote is the *last* an outcome follows, not the first.

        ``'CI 'Fail Fast'' are not satisfied`` puts a quoted phrase
        inside the name. The inner quote is followed by ``Fail Fast``,
        which the outcome pattern matches, so stopping at the first
        qualifying apostrophe cut the name to ``CI`` and read the
        remainder as a failure.
        """
        reason = (
            "Repository rule violations found "
            "Required workflows 'CI 'Fail Fast'' are not satisfied"
        )
        assert required_workflow_names(reason) == ["CI 'Fail Fast'"]
        assert violation_verb(reason) == "not satisfied"

    def test_fragments_keep_duplicates_that_names_collapse(self) -> None:
        """Reconciliation needs the sequence exactly as GitHub wrote it.

        One workflow called ``Build, Build`` splits into two identical
        pieces. Collapsing them leaves ``['Build']``, which no span can
        rejoin into the observed run name --- so the workflow would read
        as never dispatched and the wait would stop on one that ran.
        """
        reason = (
            "Repository rule violations found "
            "Required workflows 'Build, Build' are not satisfied"
        )

        assert workflow_name_fragments(reason) == ["Build", "Build"]
        # The CLI still renders one bullet.
        assert required_workflow_names(reason) == ["Build"]


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


class TestWorkflowsNeverDispatched:
    @pytest.mark.asyncio
    async def test_missing_workflow_is_detected(self) -> None:
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )

        missing = await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION)

        assert missing == ["AI Slop Scan 🧹", "Zizmor Scan 🌈"]

    @pytest.mark.asyncio
    async def test_dispatched_workflow_is_not_reported(self) -> None:
        """If it started, waiting is worthwhile."""
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"AI Slop Scan 🧹", "Zizmor Scan 🌈", "DCO"}
        )

        assert (
            await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION) == []
        )

    @pytest.mark.asyncio
    async def test_no_runs_at_all_is_treated_as_unknown(self) -> None:
        """An empty answer may mean the lookup failed; on doubt, wait."""
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(return_value=set())

        assert (
            await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION) == []
        )

    @pytest.mark.asyncio
    async def test_lookup_failure_falls_back_to_waiting(self) -> None:
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        assert (
            await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION) == []
        )

    @pytest.mark.asyncio
    async def test_unparseable_error_is_ignored(self) -> None:
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock()

        assert (
            await mgr._workflows_never_dispatched(
                _pr(), OWNER, NAME, "Merge already in progress"
            )
            == []
        )
        client.get_workflow_run_names_for_sha.assert_not_called()


# --------------------------------------------------------------------------
# Per-head isolation (in place of propagation to siblings)
# --------------------------------------------------------------------------


class TestPerHeadShaJudgement:
    @pytest.mark.asyncio
    async def test_each_pr_is_judged_on_its_own_head_sha(self) -> None:
        """Absence is a fact about a commit, not about a repository.

        Reusing a sibling's finding would skip a wait that could have
        succeeded: workflow A missing from #29's head says nothing about
        #30's, which may have been pushed later and dispatched fine. The
        lookup it would save is one request, against a wait worth five
        minutes.
        """
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            side_effect=[
                {"DCO", "CodeQL"},  # #29 absent
                {"DCO", "CodeQL"},  # #29 confirmed
                {"DCO", "AI Slop Scan 🧹", "Zizmor Scan 🌈"},  # #30 present
            ]
        )

        first = await mgr._workflows_never_dispatched(_pr(29), OWNER, NAME, VIOLATION)
        assert first == ["AI Slop Scan 🧹", "Zizmor Scan 🌈"]

        # #30's head *did* get the workflows, so it must still be waited on.
        second = await mgr._workflows_never_dispatched(_pr(30), OWNER, NAME, VIOLATION)

        assert second == []
        assert client.get_workflow_run_names_for_sha.await_count == 3

    @pytest.mark.asyncio
    async def test_a_different_workflow_is_still_checked(self) -> None:
        """Reuse must not swallow a name nobody has looked up yet."""
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "Zizmor Scan 🌈"}
        )

        await mgr._workflows_never_dispatched(_pr(29), OWNER, NAME, VIOLATION)
        before = client.get_workflow_run_names_for_sha.await_count

        other = "Required workflows 'Semantic Pull Request 🛠️' are not satisfied"
        await mgr._workflows_never_dispatched(_pr(30), OWNER, NAME, other)

        assert client.get_workflow_run_names_for_sha.await_count > before


# --------------------------------------------------------------------------
# The wait itself
# --------------------------------------------------------------------------


class TestWaitIsBounded:
    @pytest.mark.asyncio
    async def test_wait_returns_immediately_when_nothing_was_dispatched(self) -> None:
        mgr, client = _make()
        pr = _pr()
        mgr._last_merge_exception[f"{OWNER}/{NAME}#29"] = Exception(VIOLATION)
        mgr._last_merge_exception_head[f"{OWNER}/{NAME}#29"] = pr.head_sha
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )
        # Would be consulted only if the method proceeded to wait.
        mgr._wait_for_auto_merge = AsyncMock(  # type: ignore[method-assign]
            return_value=(False, False)
        )

        merged = await mgr._wait_for_required_workflows_and_retry(pr, OWNER, NAME)

        assert merged is False
        mgr._wait_for_auto_merge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_rejection_from_another_head_does_not_stop_the_wait(self) -> None:
        """A stale rejection must not cut short a wait on a new commit."""
        mgr, client = _make()
        pr = _pr()
        mgr._last_merge_exception[f"{OWNER}/{NAME}#29"] = Exception(VIOLATION)
        mgr._last_merge_exception_head[f"{OWNER}/{NAME}#29"] = "beefbeef" * 5
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )
        mgr._wait_for_auto_merge = AsyncMock(  # type: ignore[method-assign]
            return_value=(False, False)
        )

        await mgr._wait_for_required_workflows_and_retry(pr, OWNER, NAME)

        mgr._wait_for_auto_merge.assert_awaited()
        # No verdict was attempted, so no lookup was spent on one.
        client.get_workflow_run_names_for_sha.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_question_is_reasked_once_the_head_is_judgeable(self) -> None:
        """Skipping the stale rejection must not mean never asking.

        After a force-push the opening verdict is skipped, and the loop
        would otherwise wait out the entire timeout. The retry inside
        the loop produces a rejection *for the current head*, and that
        is the moment the question becomes answerable.
        """
        mgr, client = _make()
        pr = _pr()
        key = f"{OWNER}/{NAME}#29"
        # Evidence from before a rebase: not usable for this head.
        mgr._last_merge_exception[key] = Exception(VIOLATION)
        mgr._last_merge_exception_head[key] = "beefbeef" * 5
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )
        client.merge_pull_request = AsyncMock(side_effect=Exception(VIOLATION))
        mgr._wait_for_auto_merge = AsyncMock(  # type: ignore[method-assign]
            return_value=(False, False)
        )

        merged = await mgr._wait_for_required_workflows_and_retry(pr, OWNER, NAME)

        assert merged is False
        # The retry's rejection belongs to this head, so the detector
        # ran and stopped the wait instead of burning the timeout.
        assert client.get_workflow_run_names_for_sha.await_count >= 1
        assert mgr._wait_for_auto_merge.await_count == 1

    @pytest.mark.asyncio
    async def test_the_question_is_not_reasked_for_the_same_head(self) -> None:
        """Once per commit, not once per retry."""
        mgr, client = _make()
        pr = _pr()
        key = f"{OWNER}/{NAME}#29"
        mgr._last_merge_exception[key] = Exception(VIOLATION)
        mgr._last_merge_exception_head[key] = pr.head_sha
        # Dispatched, so the opening verdict says "keep waiting".
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"AI Slop Scan 🧹", "Zizmor Scan 🌈"}
        )
        client.merge_pull_request = AsyncMock(side_effect=Exception(VIOLATION))
        mgr._wait_for_auto_merge = AsyncMock(  # type: ignore[method-assign]
            return_value=(False, False)
        )
        # A short but non-zero budget: several retries happen, then the
        # deadline ends the loop.  Zero would leave nothing for the
        # opening question either, which is not what this test is about.
        mgr._merge_timeout = 0.5
        mgr._merge_recheck_interval = 0.05

        await mgr._wait_for_required_workflows_and_retry(pr, OWNER, NAME)

        # Only the opening question; every retry saw the same head.
        assert client.get_workflow_run_names_for_sha.await_count == 1


class TestAbsenceIsConfirmed:
    """One snapshot cannot prove a workflow will never dispatch.

    A workflow triggered moments earlier may not be visible yet.
    Concluding "never" from a single look turns an ordinary dispatch
    delay into a terminal merge failure, so absence must be seen twice.
    """

    @pytest.mark.asyncio
    async def test_a_workflow_that_appears_on_the_second_look_is_waited_for(
        self,
    ) -> None:
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            side_effect=[
                {"DCO"},  # neither present yet
                {"DCO", "AI Slop Scan 🧹", "Zizmor Scan 🌈"},  # both arrived
            ]
        )

        assert (
            await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION) == []
        )
        assert client.get_workflow_run_names_for_sha.await_count == 2

    @pytest.mark.asyncio
    async def test_only_the_still_absent_names_are_reported(self) -> None:
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            side_effect=[
                {"DCO"},
                {"DCO", "Zizmor Scan 🌈"},  # one arrived, one did not
            ]
        )

        assert await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION) == [
            "AI Slop Scan 🧹"
        ]

    @pytest.mark.asyncio
    async def test_a_failed_confirmation_falls_back_to_waiting(self) -> None:
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            side_effect=[{"DCO"}, RuntimeError("boom")]
        )

        assert (
            await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION) == []
        )


class TestConfirmationRespectsTheBudget:
    """The confirmation pause must live inside the run's ceiling.

    Sleeping past the advertised ``--merge-timeout`` or a nearly
    exhausted owner-wide ``max_wait`` would break the guarantee the
    deadline exists to give --- and doing it while holding a worker slot
    would starve runnable PRs as well.
    """

    @pytest.mark.asyncio
    async def test_no_confirmation_when_the_budget_cannot_fit_it(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "dependamerge.merge_manager.UNDISPATCHED_CONFIRM_DELAY_SECONDS", 10.0
        )
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )
        loop = asyncio.get_running_loop()

        result = await mgr._workflows_never_dispatched(
            _pr(), OWNER, NAME, VIOLATION, deadline=loop.time() + 1.0
        )

        # Unknown rather than "never runs": the safe direction.
        assert result == []
        assert client.get_workflow_run_names_for_sha.await_count == 1

    @pytest.mark.asyncio
    async def test_confirmation_proceeds_when_the_budget_allows(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "dependamerge.merge_manager.UNDISPATCHED_CONFIRM_DELAY_SECONDS", 0.0
        )
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )
        loop = asyncio.get_running_loop()

        result = await mgr._workflows_never_dispatched(
            _pr(), OWNER, NAME, VIOLATION, deadline=loop.time() + 600.0
        )

        assert result == ["AI Slop Scan 🧹", "Zizmor Scan 🌈"]
        assert client.get_workflow_run_names_for_sha.await_count == 2

    @pytest.mark.asyncio
    async def test_absent_deadline_still_confirms(self, monkeypatch) -> None:
        """Repository-scoped runs pass no deadline; they must still confirm."""
        monkeypatch.setattr(
            "dependamerge.merge_manager.UNDISPATCHED_CONFIRM_DELAY_SECONDS", 0.0
        )
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )

        result = await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION)

        assert result == ["AI Slop Scan 🧹", "Zizmor Scan 🌈"]
        assert client.get_workflow_run_names_for_sha.await_count == 2

    @pytest.mark.asyncio
    async def test_the_second_lookup_is_budgeted_too(self, monkeypatch) -> None:
        """Room for the pause alone is not room enough.

        The lookup after the pause retries and paginates. A budget that
        covers the sleep but little more leaves nothing for it, so the
        request would start against an all-but-expired deadline and hold
        its worker slot past the run's ceiling.
        """
        monkeypatch.setattr(
            "dependamerge.merge_manager.UNDISPATCHED_CONFIRM_DELAY_SECONDS", 10.0
        )
        monkeypatch.setattr(
            "dependamerge.merge_manager.UNDISPATCHED_CONFIRM_LOOKUP_SECONDS", 5.0
        )
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )
        loop = asyncio.get_running_loop()

        # Twelve seconds fits the pause but not the lookup behind it, so
        # nothing should sleep and no second request should be made.
        result = await mgr._workflows_never_dispatched(
            _pr(), OWNER, NAME, VIOLATION, deadline=loop.time() + 12.0
        )

        assert result == []
        assert client.get_workflow_run_names_for_sha.await_count == 1

    @pytest.mark.asyncio
    async def test_a_lookup_that_outlasts_the_budget_reads_as_unknown(
        self, monkeypatch
    ) -> None:
        """A slow confirmation is abandoned rather than allowed to overrun."""
        monkeypatch.setattr(
            "dependamerge.merge_manager.UNDISPATCHED_CONFIRM_DELAY_SECONDS", 0.0
        )
        monkeypatch.setattr(
            "dependamerge.merge_manager.UNDISPATCHED_CONFIRM_LOOKUP_SECONDS", 0.0
        )
        mgr, client = _make()

        async def _first_fast_then_hang(*_args: object, **_kwargs: object):
            if client.get_workflow_run_names_for_sha.await_count > 1:
                await asyncio.sleep(30.0)
            return {"DCO", "CodeQL"}

        client.get_workflow_run_names_for_sha = AsyncMock(
            side_effect=_first_fast_then_hang
        )
        loop = asyncio.get_running_loop()

        result = await mgr._workflows_never_dispatched(
            _pr(), OWNER, NAME, VIOLATION, deadline=loop.time() + 0.05
        )

        assert result == []
        assert client.get_workflow_run_names_for_sha.await_count == 2

    @pytest.mark.asyncio
    async def test_the_first_lookup_is_bounded_too(self) -> None:
        """The confirmation is not the only request that can overrun.

        The deadline is created before the PR refresh, so under a nearly
        exhausted ``max_wait`` even the *first* paginated lookup can
        start with almost nothing left.
        """
        mgr, client = _make()

        async def _hang(*_args: object, **_kwargs: object):
            await asyncio.sleep(30.0)
            return {"DCO"}

        client.get_workflow_run_names_for_sha = AsyncMock(side_effect=_hang)
        loop = asyncio.get_running_loop()

        result = await mgr._workflows_never_dispatched(
            _pr(), OWNER, NAME, VIOLATION, deadline=loop.time() + 0.05
        )

        assert result == []
        assert client.get_workflow_run_names_for_sha.await_count == 1

    @pytest.mark.asyncio
    async def test_an_expired_deadline_skips_the_lookup_entirely(self) -> None:
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )
        loop = asyncio.get_running_loop()

        result = await mgr._workflows_never_dispatched(
            _pr(), OWNER, NAME, VIOLATION, deadline=loop.time() - 1.0
        )

        assert result == []
        client.get_workflow_run_names_for_sha.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_head_re_read_is_bounded_too(self, monkeypatch) -> None:
        """The lookup before it may have spent the whole budget.

        The head check is the last request on the path, and it runs on
        a slot this task has already re-acquired --- so letting it start
        past the ceiling holds that slot beyond ``max_wait``.
        """
        monkeypatch.setattr(
            "dependamerge.merge_manager.UNDISPATCHED_CONFIRM_LOOKUP_SECONDS", 0.0
        )
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )

        async def _hang(*_args: object, **_kwargs: object):
            await asyncio.sleep(30.0)
            return {"head": {"sha": HEAD_SHA}}

        client.get = AsyncMock(side_effect=_hang)
        loop = asyncio.get_running_loop()

        result = await mgr._workflows_never_dispatched(
            _pr(), OWNER, NAME, VIOLATION, deadline=loop.time() + 0.05
        )

        # Unknown rather than terminal: the head could not be confirmed.
        assert result == []


class TestTheHeadMayMoveDuringConfirmation:
    """Both observations must describe the same commit.

    Dependabot force-pushes when it rebases, and the confirmation pause
    is long enough to straddle one. Judging the abandoned head would
    report "never dispatched" while the live head runs its workflows
    perfectly --- the exact mistake the second observation exists to
    prevent.
    """

    @pytest.mark.asyncio
    async def test_a_force_push_makes_the_answer_unknown(self) -> None:
        mgr, client = _make(head="beefbeef" * 5)
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )

        result = await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION)

        assert result == []
        # Both observations ran; the head check behind them is what
        # withdrew the verdict.
        assert client.get_workflow_run_names_for_sha.await_count == 2

    @pytest.mark.asyncio
    async def test_an_unreadable_head_makes_the_answer_unknown(self) -> None:
        """Doubt resolves towards waiting, as everywhere else on this path."""
        mgr, client = _make()
        client.get = AsyncMock(side_effect=RuntimeError("boom"))
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )

        result = await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION)

        assert result == []

    @pytest.mark.asyncio
    async def test_a_malformed_response_makes_the_answer_unknown(self) -> None:
        mgr, client = _make()
        client.get = AsyncMock(return_value={"head": {}})
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )

        result = await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION)

        assert result == []

    @pytest.mark.asyncio
    async def test_a_steady_head_still_confirms(self) -> None:
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )

        result = await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION)

        assert result == ["AI Slop Scan 🧹", "Zizmor Scan 🌈"]
        assert client.get_workflow_run_names_for_sha.await_count == 2

    @pytest.mark.asyncio
    async def test_the_head_is_checked_after_the_lookup(self) -> None:
        """One request, covering the pause *and* the lookup behind it.

        Checking before the lookup left a force-push during the lookup
        itself unnoticed. Checking after costs the same one request and
        covers the whole window.
        """
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )
        calls: list[str] = []

        async def _runs(*_args: object, **_kwargs: object):
            calls.append("runs")
            return {"DCO", "CodeQL"}

        async def _get(*_args: object, **_kwargs: object):
            calls.append("head")
            return {"head": {"sha": HEAD_SHA}}

        client.get_workflow_run_names_for_sha = AsyncMock(side_effect=_runs)
        client.get = AsyncMock(side_effect=_get)

        await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION)

        assert calls == ["runs", "runs", "head"]

    @pytest.mark.asyncio
    async def test_no_head_check_when_nothing_is_missing(self) -> None:
        """ "Keep waiting" is the answer whatever the head is doing."""
        mgr, client = _make()
        # Absent on the first look, dispatched by the second.
        client.get_workflow_run_names_for_sha = AsyncMock(
            side_effect=[{"DCO"}, {"AI Slop Scan 🧹", "Zizmor Scan 🌈"}]
        )

        result = await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION)

        assert result == []
        client.get.assert_not_called()


class TestACommaMayBelongToAName:
    """``'Build, Test'`` is one workflow or two, and the string cannot say.

    GitHub joins required names with a comma and a workflow name may
    contain one, so splitting can invent names no run will ever match.
    Since an unmatched name reads as "never dispatched", the ambiguity
    could report a terminal failure against a workflow that ran
    perfectly. The observed runs settle it without another request.
    """

    COMMA_VIOLATION = (
        "Repository rule violations found "
        "Required workflows 'Build, Test' are not satisfied"
    )

    @pytest.mark.asyncio
    async def test_a_dispatched_comma_name_is_not_reported_missing(self) -> None:
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"Build, Test", "DCO"}
        )

        result = await mgr._workflows_never_dispatched(
            _pr(), OWNER, NAME, self.COMMA_VIOLATION
        )

        # Waiting is still worthwhile: the workflow did dispatch.
        assert result == []

    @pytest.mark.asyncio
    async def test_the_unspaced_separator_is_recognised_too(self) -> None:
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(return_value={"Build,Test"})

        result = await mgr._workflows_never_dispatched(
            _pr(), OWNER, NAME, self.COMMA_VIOLATION
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_two_genuine_workflows_still_report_individually(self) -> None:
        """The guard must not swallow an ordinary two-name violation."""
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )

        result = await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION)

        assert result == ["AI Slop Scan 🧹", "Zizmor Scan 🌈"]

    @pytest.mark.asyncio
    async def test_one_of_two_dispatched_still_reports_the_other(self) -> None:
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"AI Slop Scan 🧹"}
        )

        result = await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION)

        assert result == ["Zizmor Scan 🌈"]

    @pytest.mark.asyncio
    async def test_a_mixed_list_is_matched_span_by_span(self) -> None:
        """``'Build, Test, Lint'`` can be two workflows, not one or three.

        Only rejoining the *whole* list would miss this: no run is
        called ``Build, Test, Lint``, so all three fragments would be
        reported missing even though every workflow dispatched.
        """
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"Build, Test", "Lint"}
        )
        reason = (
            "Repository rule violations found "
            "Required workflows 'Build, Test, Lint' are not satisfied"
        )

        result = await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, reason)

        assert result == []

    @pytest.mark.asyncio
    async def test_a_span_match_still_reports_an_unmatched_neighbour(self) -> None:
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"Build, Test", "DCO"}
        )
        reason = (
            "Repository rule violations found "
            "Required workflows 'Build, Test, Lint' are not satisfied"
        )

        result = await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, reason)

        assert result == ["Lint"]

    def test_longer_spans_win_over_single_fragments(self) -> None:
        """A real ``Build, Test`` beats a coincidental lone ``Build``."""
        mgr, _ = _make()

        assert mgr._unmatched_names(["Build", "Test"], {"Build", "Build, Test"}) == []

    def test_overlapping_spans_do_not_strand_a_fragment(self) -> None:
        """No tie-break may reject a partition that does explain the list.

        ``A|B|C`` against runs ``A``, ``A, B`` and ``B, C`` is wholly
        explained by ``A`` + ``B, C``. Committing to ``A, B`` first and
        skipping every span that overlapped it would leave ``C`` looking
        undispatched, reporting a terminal failure on workflows that all
        ran.
        """
        mgr, _ = _make()

        assert mgr._unmatched_names(["A", "B", "C"], {"A", "A, B", "B, C"}) == []

    def test_a_fragment_no_span_explains_is_still_reported(self) -> None:
        """Accounting generously must not become accounting for everything."""
        mgr, _ = _make()

        assert mgr._unmatched_names(["A", "B", "C"], {"A, B"}) == ["C"]

    @pytest.mark.asyncio
    async def test_a_repeated_fragment_can_still_be_rejoined(self) -> None:
        """``'Build, Build'`` is one dispatched workflow, not one missing."""
        mgr, client = _make()
        client.get_workflow_run_names_for_sha = AsyncMock(return_value={"Build, Build"})
        reason = (
            "Repository rule violations found "
            "Required workflows 'Build, Build' are not satisfied"
        )

        result = await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, reason)

        assert result == []


class TestTheRejectionMustDescribeTheCurrentHead:
    """A rejection is evidence about the commit that was rejected.

    ``_wait_for_required_workflows_and_retry`` refreshes the PR before
    asking whether waiting can help, so the head it judges may no longer
    be the head the merge error came from. Applying the old rejection to
    a new commit is doubly wrong: the evidence is stale, and a
    just-force-pushed head is exactly when workflows are most likely to
    be legitimately absent for a few seconds.
    """

    @staticmethod
    def _manager():
        mgr, client = _make()
        client.get = AsyncMock(return_value={"head": {"sha": HEAD_SHA}})
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )
        return mgr, client

    @pytest.mark.asyncio
    async def test_a_matching_head_uses_the_rejection(self) -> None:
        mgr, client = self._manager()
        pr = _pr()
        key = f"{OWNER}/{NAME}#{pr.number}"
        mgr._last_merge_exception[key] = RuntimeError(VIOLATION)
        mgr._last_merge_exception_head[key] = pr.head_sha

        last_error = ""
        if mgr._last_merge_exception_head.get(key) == pr.head_sha:
            last_error = str(mgr._last_merge_exception.get(key) or "")
        result = await mgr._workflows_never_dispatched(pr, OWNER, NAME, last_error)

        assert result == ["AI Slop Scan 🧹", "Zizmor Scan 🌈"]

    @pytest.mark.asyncio
    async def test_a_moved_head_discards_the_rejection(self) -> None:
        mgr, client = self._manager()
        pr = _pr()
        key = f"{OWNER}/{NAME}#{pr.number}"
        mgr._last_merge_exception[key] = RuntimeError(VIOLATION)
        # The rejection was raised against the commit before a rebase.
        mgr._last_merge_exception_head[key] = "beefbeef" * 5

        last_error = ""
        if mgr._last_merge_exception_head.get(key) == pr.head_sha:
            last_error = str(mgr._last_merge_exception.get(key) or "")

        assert last_error == ""
        # With no violation text there are no names, so nothing is
        # judged and no request is spent.
        result = await mgr._workflows_never_dispatched(pr, OWNER, NAME, last_error)
        assert result == []
        client.get_workflow_run_names_for_sha.assert_not_called()

    def test_the_call_site_gates_on_the_recorded_head(self) -> None:
        """Guards the wiring, not just the bookkeeping."""
        import inspect

        from dependamerge.merge_manager import AsyncMergeManager

        source = inspect.getsource(
            AsyncMergeManager._wait_for_required_workflows_and_retry
        )
        assert "_last_merge_exception_head.get(pr_key) == pr_info.head_sha" in source


class TestPendingGateUsesTheSharedParser:
    """The gate and the parser must agree on what "failed" means.

    ``_merge_error_indicates_pending_workflows`` decides whether the
    recovery path runs at all. It previously matched "fail" against the
    whole clause, including the workflow *names*, so a pending violation
    naming a workflow like "Fail Fast Lint" was classified terminal and
    never reached the detector --- even though ``violation_verb`` handles
    that case correctly. Two parsers reading one string is the drift
    ``rule_violations`` exists to prevent.
    """

    @staticmethod
    def _gate(text: str) -> bool:
        from dependamerge.merge_manager import AsyncMergeManager

        return AsyncMergeManager._merge_error_indicates_pending_workflows(text)

    def test_a_workflow_named_with_fail_still_reaches_recovery(self) -> None:
        reason = (
            "Failed to merge PR #1. Repository rule violations found "
            "Required workflows 'Fail Fast Lint' are not satisfied"
        )
        assert self._gate(reason) is True

    def test_a_genuine_failure_remains_terminal(self) -> None:
        reason = (
            "Failed to merge PR #1. Repository rule violations found "
            "Required workflows 'Semantic Pull Request' failed"
        )
        assert self._gate(reason) is False

    def test_an_ordinary_pending_violation_still_matches(self) -> None:
        assert self._gate(VIOLATION) is True

    def test_the_pr_state_suffix_is_trimmed_before_parsing(self) -> None:
        """Both parsers must ignore the context we append ourselves.

        ``_validate_merge_result`` appends the PR's state after GitHub's
        detail, and that context can read "blocked by failing checks".
        Handing it to the outcome parser would read our own annotation
        as the workflows' verdict and skip the recovery.
        """
        reason = (
            "Failed to merge PR #39 in org/repo. GitHub: Repository rule "
            "violations found Required workflows 'Fail Fast Lint' are not "
            "satisfied (PR state: open, mergeable: False, "
            "mergeable_state: blocked) [blocked by failing checks]"
        )
        assert self._gate(reason) is True

    def test_unrelated_text_does_not_match(self) -> None:
        assert self._gate("Merge already in progress") is False
