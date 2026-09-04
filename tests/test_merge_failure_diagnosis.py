# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the diagnosis a failed merge reports.

A 157-PR owner-wide run reported 17 failures.  Sampling four of them
against the API showed the stated cause was wrong in every case, and
that two of the four were fully mergeable by the time the run printed
them as failed (lfreleng-actions/dependamerge#482).

The cause is that a merge rejection is a *snapshot*.  GitHub states what
was unsatisfied at the instant the merge was attempted, which routinely
names required checks that had merely not finished, and nothing re-read
that before the summary was printed.  The four sampled pull requests are
used here as fixtures so the reported outcome is driven by the state
they were actually in:

===========================  ================  ====================
Pull request                 Live state        Expected outcome
===========================  ================  ====================
test-python-project#396      ``clean``         unsettled
docker-workflows#70          ``clean``         unsettled
python-workflows#84          ``blocked``       failed
workflows-template#55        ``blocked``       failed
===========================  ================  ====================
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dependamerge.cli._merge_report import (
    _print_failed_pr_details,
    _print_final_merge_summary,
)
from dependamerge.merge_manager import MergeResult, MergeStatus
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

# The rejection python-workflows#84 was reported under.  Every workflow
# it names had passed by the time the run printed it; the actual blocker
# was the ``pre-commit.ci - pr`` status context, which the message does
# not mention at all.
SAMPLED_REJECTION = (
    "Repository rule violations found Required workflows "
    "'Package Hardening Audit, SHA Pinned Actions 📌, "
    "Audit GitHub Actions 📌' are not satisfied"
)


def _pr(repo: str, number: int) -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
        title="Chore: Bump something",
        body="bump",
        author="dependabot[bot]",
        head_sha="c0ffee11" * 5,
        base_branch="main",
        head_branch="dependabot/x",
        state="open",
        mergeable=True,
        mergeable_state="blocked",
        behind_by=None,
        files_changed=[],
        repository_full_name=f"lfreleng-actions/{repo}",
        html_url=f"https://github.com/lfreleng-actions/{repo}/pull/{number}",
        reviews=[],
        review_comments=[],
    )


async def _confirmed(payload: dict[str, object]) -> MergeResult:
    """Drive the real confirmation step over a refreshed PR payload."""
    mgr, client = make_merge_manager()
    pr = _pr("python-workflows", 84)
    client.get = AsyncMock(return_value=payload)
    result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error=SAMPLED_REJECTION)
    return await mgr._confirm_failure(pr, result)


class TestAPullRequestThatBecameMergeableIsNotAFailure:
    """The headline correction: two of the four sampled PRs were clean.

    Both had merged nothing and needed nothing.  The run judged them
    before their required checks settled, so the only accurate report is
    that they did not finish --- not that they failed.
    """

    @pytest.mark.parametrize(
        ("repo", "number"),
        [("test-python-project", 396), ("docker-workflows", 70)],
    )
    @pytest.mark.asyncio
    async def test_a_clean_pull_request_is_unsettled(
        self, repo: str, number: int
    ) -> None:
        mgr, client = make_merge_manager()
        pr = _pr(repo, number)
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "merged_at": None,
                "mergeable": True,
                "mergeable_state": "clean",
            }
        )
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, error=SAMPLED_REJECTION
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.status is MergeStatus.UNSETTLED

    @pytest.mark.asyncio
    async def test_the_expired_reason_does_not_stay_the_cause(self) -> None:
        """The rejection described a state that has stopped holding.

        Leaving it on ``error`` would keep the summary naming workflows
        that pass, which is the misdiagnosis this fixes.  It is kept as
        a note instead, matching how a stale reason is handled on a PR
        that turned out to have merged.
        """
        out = await _confirmed(
            {
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": "clean",
            }
        )

        assert out.error is not None
        assert SAMPLED_REJECTION not in out.error
        assert out.warning is not None
        assert SAMPLED_REJECTION in out.warning

    @pytest.mark.parametrize("state", ["clean", "unstable", "has_hooks"])
    @pytest.mark.asyncio
    async def test_every_state_github_would_merge_counts(self, state: str) -> None:
        """``unstable`` and ``has_hooks`` do not block a merge either.

        Only optional checks are failing under ``unstable``, and
        pre-receive hooks do not block under ``has_hooks``.  Both are
        already treated as worth attempting by ``_should_attempt_merge``,
        so reading them as blocking here would contradict the gate that
        decides whether to try at all.
        """
        out = await _confirmed(
            {"state": "open", "merged": False, "mergeable_state": state}
        )

        assert out.status is MergeStatus.UNSETTLED

    @pytest.mark.asyncio
    async def test_no_extra_request_is_made(self) -> None:
        """Reconciliation must not cost a request per failed PR.

        A 157-PR run has a finite API budget, and the payload the
        confirmation step already fetches carries ``mergeable_state``.
        """
        mgr, client = make_merge_manager()
        pr = _pr("docker-workflows", 70)
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable_state": "clean",
            }
        )
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="stale")

        await mgr._confirm_failure(pr, result)

        assert client.get.await_count == 1


class TestAStillBlockedPullRequestStaysFailed:
    """The control that stops the correction from swallowing real failures."""

    @pytest.mark.parametrize(
        ("repo", "number"),
        [("python-workflows", 84), ("workflows-template", 55)],
    )
    @pytest.mark.asyncio
    async def test_a_blocked_pull_request_is_still_a_failure(
        self, repo: str, number: int
    ) -> None:
        mgr, client = make_merge_manager()
        pr = _pr(repo, number)
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": "blocked",
            }
        )
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, error=SAMPLED_REJECTION
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.status is MergeStatus.FAILED

    @pytest.mark.asyncio
    async def test_an_unknown_state_stays_a_failure(self) -> None:
        """``unknown`` is an absence of evidence, not evidence of health.

        GitHub computes mergeability in the background and reports
        ``unknown`` until it has.  That is equally not evidence the PR
        would merge, and only that would justify withdrawing a failure.
        """
        out = await _confirmed(
            {"state": "open", "merged": False, "mergeable_state": "unknown"}
        )

        assert out.status is MergeStatus.FAILED

    @pytest.mark.asyncio
    async def test_a_contradictory_payload_stays_a_failure(self) -> None:
        """``clean`` and ``mergeable: false`` cannot both be true.

        Under-reporting a success costs a re-run; withdrawing a real
        failure loses it silently.  The asymmetry decides the tie.
        """
        out = await _confirmed(
            {
                "state": "open",
                "merged": False,
                "mergeable": False,
                "mergeable_state": "clean",
            }
        )

        assert out.status is MergeStatus.FAILED

    @pytest.mark.asyncio
    async def test_a_payload_without_a_state_stays_a_failure(self) -> None:
        """Absent is not the same as clean."""
        out = await _confirmed({"state": "open", "merged": False})

        assert out.status is MergeStatus.FAILED


class TestTheRealBlockerIsNamed:
    """python-workflows#84: the rejection named everything but the cause.

    All three workflows the message listed had passed.  The sole blocker
    was the ``pre-commit.ci - pr`` status context, which reports through
    the commit status API and is therefore invisible to any view built
    from check runs alone.
    """

    @staticmethod
    def _blocked_manager(
        *,
        required: list[dict[str, str]] | None = None,
        failing_contexts: list[str] | None = None,
        check_runs: list[dict[str, object]] | None = None,
    ):
        mgr, client = make_merge_manager()
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": "blocked",
            }
        )
        client.get_required_status_checks = AsyncMock(return_value=required or [])
        client.get_failing_status_contexts = AsyncMock(
            return_value=failing_contexts or []
        )
        client.get_check_runs_for_ref = AsyncMock(return_value=check_runs or [])
        return mgr, client

    @pytest.mark.asyncio
    async def test_a_failing_required_context_is_named(self) -> None:
        mgr, _ = self._blocked_manager(
            required=[{"context": "DCO"}, {"context": "pre-commit.ci - pr"}],
            failing_contexts=["pre-commit.ci - pr"],
        )
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, error=SAMPLED_REJECTION
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.status is MergeStatus.FAILED
        assert out.error == "blocked by required status check: pre-commit.ci - pr"

    @pytest.mark.asyncio
    async def test_the_passing_workflows_stop_being_the_cause(self) -> None:
        """The misdiagnosis itself: naming conditions that pass."""
        mgr, _ = self._blocked_manager(
            required=[{"context": "pre-commit.ci - pr"}],
            failing_contexts=["pre-commit.ci - pr"],
        )
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, error=SAMPLED_REJECTION
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error is not None
        assert "Package Hardening Audit" not in out.error
        assert out.warning is not None
        assert SAMPLED_REJECTION in out.warning

    @pytest.mark.asyncio
    async def test_an_advisory_context_is_not_presented_as_the_reason(self) -> None:
        """A failing context the branch does not require blocks nothing."""
        mgr, _ = self._blocked_manager(
            required=[{"context": "DCO"}],
            failing_contexts=["some-advisory-bot"],
        )
        pr = _pr("python-workflows", 84)
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="original")

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "original"

    @pytest.mark.asyncio
    async def test_a_failing_check_run_is_named_too(self) -> None:
        """Ruleset-required workflows report as check runs, not contexts.

        They never appear among the required status contexts, so
        filtering check runs on that list would discard the very names a
        ruleset rejection is about.
        """
        mgr, _ = self._blocked_manager(
            check_runs=[
                {"name": "Zizmor Scan 🌈", "conclusion": "failure"},
                {"name": "AI Slop Scan 🧹", "conclusion": "success"},
            ],
        )
        pr = _pr("workflows-template", 55)
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="original")

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "blocked by failing check: Zizmor Scan 🌈"

    @pytest.mark.asyncio
    async def test_a_superseded_run_is_not_a_blocker(self) -> None:
        """A cancelled run beside a successful one of the same name."""
        mgr, _ = self._blocked_manager(
            check_runs=[
                {
                    "name": "Package Hardening Audit",
                    "conclusion": "cancelled",
                    "completed_at": "2026-09-03T10:00:00Z",
                },
                {
                    "name": "Package Hardening Audit",
                    "conclusion": "success",
                    "completed_at": "2026-09-03T10:05:00Z",
                },
            ],
        )
        pr = _pr("python-workflows", 84)
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="original")

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "original"

    @pytest.mark.asyncio
    async def test_both_kinds_are_reported_together(self) -> None:
        mgr, _ = self._blocked_manager(
            required=[{"context": "pre-commit.ci - pr"}],
            failing_contexts=["pre-commit.ci - pr"],
            check_runs=[{"name": "Zizmor Scan 🌈", "conclusion": "failure"}],
        )
        pr = _pr("python-workflows", 84)
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="original")

        out = await mgr._confirm_failure(pr, result)

        assert out.error == (
            "blocked by required status check: pre-commit.ci - pr; "
            "failing check: Zizmor Scan 🌈"
        )

    @pytest.mark.asyncio
    async def test_nothing_established_keeps_the_original_reason(self) -> None:
        """An empty reading is not an all-clear.

        The token may be unable to read rulesets, or the requests may
        have failed.  Discarding the recorded reason would leave the
        operator with less than they had before.
        """
        mgr, client = self._blocked_manager()
        client.get_required_status_checks = AsyncMock(side_effect=RuntimeError("403"))
        client.get_failing_status_contexts = AsyncMock(side_effect=RuntimeError("403"))
        client.get_check_runs_for_ref = AsyncMock(side_effect=RuntimeError("403"))
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, error=SAMPLED_REJECTION
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.status is MergeStatus.FAILED
        assert out.error == SAMPLED_REJECTION

    @pytest.mark.asyncio
    async def test_a_conflicted_pr_is_not_re_examined(self) -> None:
        """Its state already describes it; reading checks adds only cost."""
        mgr, client = self._blocked_manager()
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable_state": "dirty",
            }
        )
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, error="merge conflicts"
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "merge conflicts"
        client.get_check_runs_for_ref.assert_not_called()


class TestUnsettledIsReportedApartFromFailed:
    """A re-runnable outcome and one needing a human are different news."""

    def test_the_counts_separate_the_two(self, capsys) -> None:
        results = [
            MergeResult(
                pr_info=_pr("test-python-project", 396),
                status=MergeStatus.UNSETTLED,
                error="not settled during the run; now clean",
            ),
            MergeResult(
                pr_info=_pr("python-workflows", 84),
                status=MergeStatus.FAILED,
                error=SAMPLED_REJECTION,
            ),
        ]

        _print_final_merge_summary(results)
        out = capsys.readouterr().out

        assert "1 failed" in out
        assert "1 unsettled" in out

    def test_the_unsettled_prs_are_listed_with_their_reason(self, capsys) -> None:
        """The end-of-run report is the only place reasons appear."""
        result = MergeResult(
            pr_info=_pr("docker-workflows", 70),
            status=MergeStatus.UNSETTLED,
            error="not settled during the run; now clean",
        )

        _print_failed_pr_details([result])
        out = capsys.readouterr().out

        assert "https://github.com/lfreleng-actions/docker-workflows/pull/70" in out
        assert "not settled during the run" in out
        assert "no reason reported" not in out

    def test_an_unsettled_pr_is_not_counted_as_failed(self, capsys) -> None:
        """The regression this closes: everything unmerged read as failed."""
        result = MergeResult(
            pr_info=_pr("test-python-project", 396),
            status=MergeStatus.UNSETTLED,
            error="not settled during the run; now clean",
        )

        _print_final_merge_summary([result])
        out = capsys.readouterr().out

        assert "0 failed" in out
        assert "❌ Failed PRs:" not in out


class TestTheTrackerCountsUnsettledOnce:
    """Every PR ends in exactly one counter, or the display stops adding up."""

    def test_the_dedicated_counter_is_used(self) -> None:
        from dependamerge.progress_tracker import MergeProgressTracker

        tracker = MergeProgressTracker("lfreleng-actions")
        tracker.set_total_prs(1)
        mgr, _ = make_merge_manager(progress_tracker=tracker)

        mgr._record_terminal_outcome(_pr("docker-workflows", 70), MergeStatus.UNSETTLED)

        assert tracker.prs_unsettled == 1
        assert tracker.prs_failed == 0
        assert tracker.prs_pending == 0
        assert tracker.completed_prs == 1
