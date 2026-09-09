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

from dependamerge.check_runs import failing_check_names
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
    result = MergeResult(
        pr_info=pr,
        status=MergeStatus.FAILED,
        merge_refused=True,
        error=SAMPLED_REJECTION,
    )
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
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
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
            {
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": state,
            }
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
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, merge_refused=True, error="stale"
        )

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
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
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
        workflow_runs: list[str] | None = None,
        head_sha: str | None = None,
        base_ref: str | None = None,
    ):
        mgr, client = make_merge_manager()
        payload: dict[str, object] = {
            "state": "open",
            "merged": False,
            "mergeable": True,
            "mergeable_state": "blocked",
        }
        if head_sha is not None:
            payload["head"] = {"sha": head_sha}
        if base_ref is not None:
            payload["base"] = {"ref": base_ref}
        client.get = AsyncMock(return_value=payload)
        client.get_required_status_checks_reliable = AsyncMock(
            return_value=(required or [], True)
        )
        client.get_failing_status_contexts = AsyncMock(
            return_value=failing_contexts or []
        )
        client.get_check_runs_for_ref = AsyncMock(return_value=check_runs or [])
        client.get_failing_workflow_run_names_for_sha = AsyncMock(
            return_value=workflow_runs or []
        )
        return mgr, client

    @pytest.mark.asyncio
    async def test_a_failing_required_context_is_named(self) -> None:
        mgr, _ = self._blocked_manager(
            required=[{"context": "DCO"}, {"context": "pre-commit.ci - pr"}],
            failing_contexts=["pre-commit.ci - pr"],
        )
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
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
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error is not None
        assert "Package Hardening Audit" not in out.error
        assert out.warning is not None
        assert SAMPLED_REJECTION in out.warning

    @pytest.mark.asyncio
    async def test_an_advisory_context_is_named_but_not_blamed(self) -> None:
        """A failing context the branch does not require blocks nothing.

        It is still worth naming -- it is failing -- but it must not be
        presented as the reason the merge was refused.
        """
        mgr, _ = self._blocked_manager(
            required=[{"context": "DCO"}],
            failing_contexts=["some-advisory-bot"],
        )
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, merge_refused=True, error="original"
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "failing checks: some-advisory-bot"

    @pytest.mark.asyncio
    async def test_a_workflow_the_rejection_named_is_a_proven_blocker(self) -> None:
        """GitHub quoting a workflow as required settles that it is.

        The check-runs API does not say whether a check is required, and
        a ruleset-required workflow never appears among the required
        status contexts.  The rejection message does name it, which is
        why ``rule_violations`` already treats that message as the more
        dependable source for those names.
        """
        rejection = (
            "Repository rule violations found Required workflows "
            "'Zizmor Scan 🌈' are not satisfied"
        )
        mgr, _ = self._blocked_manager(workflow_runs=["Zizmor Scan 🌈"])
        pr = _pr("workflows-template", 55)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, merge_refused=True, error=rejection
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "blocked by required workflow: Zizmor Scan 🌈"

    @pytest.mark.asyncio
    async def test_a_job_name_is_not_matched_against_a_workflow_name(self) -> None:
        """A rejection quotes the workflow; a check run carries the job.

        ``.github/workflows/codeql.yml`` declares the workflow ``CodeQL``
        and the job ``Audit Repository``, so comparing the two
        vocabularies would either miss the match or invent one.  The
        workflow claim is made against workflow *runs*; the job name is
        reported as failing without being blamed.
        """
        rejection = (
            "Repository rule violations found Required workflows "
            "'CodeQL' are not satisfied"
        )
        mgr, _ = self._blocked_manager(
            check_runs=[{"name": "Audit Repository", "conclusion": "failure"}],
            workflow_runs=["CodeQL"],
        )
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, merge_refused=True, error=rejection
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == (
            "blocked by required workflow: CodeQL; also failing: Audit Repository"
        )

    @pytest.mark.asyncio
    async def test_an_unproven_check_is_named_without_being_blamed(self) -> None:
        """Nothing here shows the check is required, so nothing claims it.

        A PR can be blocked on a review while an optional workflow
        fails.  Naming that workflow as the blocker would repeat the
        misdiagnosis this change exists to correct, in the opposite
        direction.
        """
        mgr, _ = self._blocked_manager(
            check_runs=[
                {"name": "Zizmor Scan 🌈", "conclusion": "failure"},
                {"name": "AI Slop Scan 🧹", "conclusion": "success"},
            ],
        )
        pr = _pr("workflows-template", 55)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, merge_refused=True, error="original"
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "failing checks: Zizmor Scan 🌈"

    @pytest.mark.parametrize(
        "conclusion",
        ["failure", "cancelled", "timed_out", "action_required", "startup_failure"],
    )
    @pytest.mark.asyncio
    async def test_every_blocking_conclusion_counts(self, conclusion: str) -> None:
        """``action_required`` and ``startup_failure`` are terminal too.

        Each is a check that will report nothing further and, when
        required, holds the merge exactly as a failure does.
        """
        mgr, _ = self._blocked_manager(
            check_runs=[{"name": "Zizmor Scan 🌈", "conclusion": conclusion}],
        )
        pr = _pr("workflows-template", 55)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, merge_refused=True, error="original"
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "failing checks: Zizmor Scan 🌈"

    @pytest.mark.asyncio
    async def test_a_stale_run_is_not_a_failure(self) -> None:
        """``stale`` says the result no longer describes the head.

        Reading it as a failure would let a superseded run block a
        commit it never examined.
        """
        mgr, _ = self._blocked_manager(
            check_runs=[{"name": "Zizmor Scan 🌈", "conclusion": "stale"}],
        )
        pr = _pr("workflows-template", 55)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, merge_refused=True, error="original"
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "original"

    @pytest.mark.asyncio
    async def test_the_live_head_is_read_not_the_snapshot(self) -> None:
        """A dependabot rebase moves the head, and this tool requests one.

        Reading the snapshot's commit would report conditions belonging
        to a commit nobody is trying to merge.
        """
        rebased = "feedface" * 5
        mgr, client = self._blocked_manager(head_sha=rebased, base_ref="release/v2")
        pr = _pr("docker-workflows", 70)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, merge_refused=True, error="original"
        )

        await mgr._confirm_failure(pr, result)

        assert pr.head_sha != rebased  # the snapshot really is stale
        assert client.get_check_runs_for_ref.await_args.args[2] == rebased
        assert client.get_failing_status_contexts.await_args.args[2] == rebased
        assert client.get_required_status_checks_reliable.await_args.args[2] == (
            "release/v2"
        )

    @pytest.mark.asyncio
    async def test_the_snapshot_is_used_when_the_payload_omits_the_head(self) -> None:
        mgr, client = self._blocked_manager()
        pr = _pr("docker-workflows", 70)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, merge_refused=True, error="original"
        )

        await mgr._confirm_failure(pr, result)

        assert client.get_check_runs_for_ref.await_args.args[2] == pr.head_sha

    @pytest.mark.asyncio
    async def test_both_kinds_are_reported_together(self) -> None:
        mgr, _ = self._blocked_manager(
            required=[{"context": "pre-commit.ci - pr"}],
            failing_contexts=["pre-commit.ci - pr"],
            check_runs=[{"name": "Zizmor Scan 🌈", "conclusion": "failure"}],
        )
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, merge_refused=True, error="original"
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == (
            "blocked by required status check: pre-commit.ci - pr; "
            "also failing: Zizmor Scan 🌈"
        )

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
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, merge_refused=True, error="original"
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "original"

    @pytest.mark.asyncio
    async def test_nothing_established_keeps_the_original_reason(self) -> None:
        """An empty reading is not an all-clear.

        The token may be unable to read rulesets, or the requests may
        have failed.  Discarding the recorded reason would leave the
        operator with less than they had before.
        """
        mgr, client = self._blocked_manager()
        client.get_required_status_checks_reliable = AsyncMock(
            side_effect=RuntimeError("403")
        )
        client.get_failing_status_contexts = AsyncMock(side_effect=RuntimeError("403"))
        client.get_check_runs_for_ref = AsyncMock(side_effect=RuntimeError("403"))
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
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
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error="merge conflicts",
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


class TestOnlyARefusalMayBeWithdrawn:
    """The run reports ``FAILED`` for its own troubles too.

    An unhandled exception (``_single_pr.py``) and a rebase that did not
    complete (``_single_pr_rebase.py``) both produce ``FAILED`` without
    GitHub having said anything about mergeability.  Rewriting one of
    those because the PR happens to read ``clean`` would bury the
    actionable error in a note the summary never prints, and advise a
    re-run that would fail in exactly the same way.
    """

    @staticmethod
    async def _confirm_clean(result: MergeResult) -> MergeResult:
        mgr, client = make_merge_manager()
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": "clean",
            }
        )
        return await mgr._confirm_failure(result.pr_info, result)

    @pytest.mark.asyncio
    async def test_a_run_side_failure_is_preserved(self) -> None:
        result = MergeResult(
            pr_info=_pr("docker-workflows", 70),
            status=MergeStatus.FAILED,
            error="local rebase failed: could not push to fork",
        )

        out = await self._confirm_clean(result)

        assert out.status is MergeStatus.FAILED
        assert out.error == "local rebase failed: could not push to fork"
        assert out.warning is None

    @pytest.mark.asyncio
    async def test_a_refusal_on_the_same_state_is_withdrawn(self) -> None:
        """The control that makes the test above meaningful.

        Identical PR state, identical code path; only the origin of the
        failure differs.
        """
        result = MergeResult(
            pr_info=_pr("docker-workflows", 70),
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
        )

        out = await self._confirm_clean(result)

        assert out.status is MergeStatus.UNSETTLED

    @pytest.mark.asyncio
    async def test_a_run_side_failure_is_not_re_diagnosed_either(self) -> None:
        """A blocked PR whose failure came from the run keeps its reason."""
        mgr, client = make_merge_manager()
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable_state": "blocked",
            }
        )
        client.get_required_status_checks_reliable = AsyncMock(
            return_value=([{"context": "pre-commit.ci - pr"}], True)
        )
        client.get_failing_status_contexts = AsyncMock(
            return_value=["pre-commit.ci - pr"]
        )
        client.get_check_runs_for_ref = AsyncMock(return_value=[])
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, error="token lacks 'workflow' scope"
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "token lacks 'workflow' scope"

    def test_the_flag_defaults_to_off(self) -> None:
        """A new failure path must opt in, not inherit the correction."""
        result = MergeResult(
            pr_info=_pr("docker-workflows", 70), status=MergeStatus.FAILED
        )

        assert result.merge_refused is False


class TestOnlyAStateBasedRefusalCounts:
    """Not every ``FAILED`` from the merge path is about the PR's state.

    ``_report_merge_failure`` is also reached after a 502, a 403 or a
    missing token scope. None of those says anything about mergeability,
    so a later ``clean`` reading must not withdraw them -- the message
    that explains what to fix is the whole value of the report.
    """

    @staticmethod
    def _manager_raising(message: str):
        mgr, client = make_merge_manager()
        pr = _pr("python-workflows", 84)
        mgr._last_merge_exception[f"{pr.repository_full_name}#{pr.number}"] = (
            RuntimeError(message)
        )
        return mgr, pr

    @pytest.mark.parametrize(
        ("message", "refused"),
        [
            # The real shape, from ``tests/test_transient_merge_errors.py``:
            # the merge layer embeds the status alongside GitHub's body.
            (
                "Failed to merge PR #84. Client error '405 Method Not Allowed' "
                "for url '.../merge'. GitHub: Required workflows 'X' are not "
                "satisfied",
                True,
            ),
            ("Client error '502 Bad Gateway' for url '.../merge'", False),
            ("Client error '403 Forbidden' for url '.../merge'", False),
            ("Missing 'workflow' scope on the token", False),
            # A body with no status at all: a wrapped or transport
            # failure that still quoted GitHub says nothing about
            # mergeability, so it cannot be withdrawn.
            ("Failed to merge PR #84. GitHub: Required workflows 'X'", False),
        ],
    )
    @pytest.mark.asyncio
    async def test_the_origin_is_classified(self, message: str, refused: bool) -> None:
        mgr, pr = self._manager_raising(message)

        _reason, is_refusal = await mgr._get_failure_summary(pr)

        assert is_refusal is refused

    @pytest.mark.asyncio
    async def test_a_transport_failure_survives_a_clean_reading(self) -> None:
        """End to end: the 502 must still be what the operator sees."""
        mgr, client = make_merge_manager()
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": "clean",
            }
        )
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            error="GitHub API returned 502 Bad Gateway",
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.status is MergeStatus.FAILED
        assert out.error == "GitHub API returned 502 Bad Gateway"

    @pytest.mark.asyncio
    async def test_a_state_derived_reason_is_a_refusal(self) -> None:
        """With no exception, every reason is read off the PR's state."""
        mgr, _client = make_merge_manager()
        pr = _pr("python-workflows", 84)
        pr.mergeable_state = "dirty"

        reason, is_refusal = await mgr._get_failure_summary(pr)

        assert reason == "merge conflicts"
        assert is_refusal is True


class TestOnlyAffirmativeMergeabilityWithdraws:
    """``mergeable: null`` means GitHub is still working it out.

    That is the same absence of evidence which keeps ``unknown`` a
    failure, and the reason ``_state_is_waitable`` waits unless it sees
    ``True``.
    """

    @pytest.mark.parametrize("mergeable", [None, False, "yes"])
    @pytest.mark.asyncio
    async def test_a_non_true_mergeable_keeps_the_failure(self, mergeable) -> None:
        payload = {"state": "open", "merged": False, "mergeable_state": "unstable"}
        if mergeable is not None:
            payload["mergeable"] = mergeable

        out = await _confirmed(payload)

        assert out.status is MergeStatus.FAILED

    @pytest.mark.asyncio
    async def test_an_affirmative_mergeable_withdraws_it(self) -> None:
        """The control: identical state, mergeability asserted."""
        out = await _confirmed(
            {
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": "unstable",
            }
        )

        assert out.status is MergeStatus.UNSETTLED


class TestTheLiveReadingIsNotTruncated:
    """A partial reading composes a confident, wrong diagnosis.

    Both probes feed a message that *replaces* the recorded reason, and
    both reason about absence. The combined-status endpoint returns 30
    contexts a page by default, so the required context that is actually
    blocking has no reason to be among the first thirty.
    """

    @pytest.mark.asyncio
    async def test_status_contexts_are_paginated(self) -> None:
        from dependamerge.github_async import GitHubAsync

        pages = [
            {"statuses": [{"context": f"check-{i}", "state": "success"}]}
            for i in range(30)
        ] + [{"statuses": [{"context": "pre-commit.ci - pr", "state": "failure"}]}]

        async with GitHubAsync(token="t") as api:

            async def _paginate(path, **kwargs):
                for page in pages:
                    yield page

            api.get_paginated = _paginate  # type: ignore[method-assign]
            failing = await api.get_failing_status_contexts("o", "r", "sha")

        assert failing == ["pre-commit.ci - pr"]

    @pytest.mark.asyncio
    async def test_check_runs_are_paginated(self) -> None:
        from dependamerge.github_async import GitHubAsync

        pages = [
            {"check_runs": [{"name": f"job-{i}", "conclusion": "success"}]}
            for i in range(100)
        ] + [{"check_runs": [{"name": "Zizmor Scan 🌈", "conclusion": "failure"}]}]

        async with GitHubAsync(token="t") as api:

            async def _paginate(path, **kwargs):
                for page in pages:
                    yield page

            api.get_paginated = _paginate  # type: ignore[method-assign]
            runs = await api.get_check_runs_for_ref("o", "r", "sha")

        assert len(runs) == 101
        assert failing_check_names(runs) == ["Zizmor Scan 🌈"]

    @pytest.mark.asyncio
    async def test_only_completed_workflow_runs_are_failing(self) -> None:
        """A queued run has no conclusion and has failed at nothing."""
        from dependamerge.github_async import GitHubAsync

        page = {
            "workflow_runs": [
                {"name": "CodeQL", "conclusion": "failure"},
                {"name": "Zizmor Scan 🌈", "conclusion": None},
                {"name": "AI Slop Scan 🧹", "conclusion": "success"},
            ]
        }

        async with GitHubAsync(token="t") as api:

            async def _paginate(path, **kwargs):
                yield page

            api.get_paginated = _paginate  # type: ignore[method-assign]
            names = await api.get_failing_workflow_run_names_for_sha("o", "r", "sha")

        assert names == ["CodeQL"]


class TestABodyIsNotAlwaysAVerdict:
    """GitHub attaches a response body to transport failures too.

    ``Client error '502 Bad Gateway' … GitHub: Server Error`` carries a
    body, so the body's presence says nothing about what it is *about*.
    The status does.
    """

    @staticmethod
    async def _classify(message: str) -> bool:
        mgr, _client = make_merge_manager()
        pr = _pr("python-workflows", 84)
        mgr._last_merge_exception[f"{pr.repository_full_name}#{pr.number}"] = (
            RuntimeError(message)
        )
        _reason, refused = await mgr._get_failure_summary(pr)
        return refused

    @pytest.mark.parametrize(
        ("status", "refused"),
        [
            ("405 Method Not Allowed", True),  # GitHub's "not mergeable"
            ("502 Bad Gateway", False),
            ("500 Internal Server Error", False),
            ("403 Forbidden", False),
            ("401 Unauthorized", False),
            ("429 Too Many Requests", False),
        ],
    )
    @pytest.mark.asyncio
    async def test_the_status_decides_not_the_body(
        self, status: str, refused: bool
    ) -> None:
        message = (
            f"Failed to merge PR #84. Client error '{status}' for url "
            "'https://api.github.com/repos/o/r/pulls/84/merge'. "
            "GitHub: Something GitHub said"
        )

        assert await self._classify(message) is refused

    @pytest.mark.asyncio
    async def test_the_body_is_still_the_reported_text(self) -> None:
        """Downgrading the claim must not discard the explanation."""
        mgr, _client = make_merge_manager()
        pr = _pr("python-workflows", 84)
        mgr._last_merge_exception[f"{pr.repository_full_name}#{pr.number}"] = (
            RuntimeError(
                "Client error '502 Bad Gateway' for url '.../merge'. "
                "GitHub: Server Error"
            )
        )

        reason, refused = await mgr._get_failure_summary(pr)

        assert reason == "Server Error"
        assert refused is False


class TestASupersededWorkflowRunIsNotFailing:
    """A re-run leaves the old run attached to the same commit.

    Blaming the superseded failure is the stale-snapshot mistake this
    whole change exists to remove, arriving by another route.
    """

    @staticmethod
    async def _failing(runs: list[dict[str, object]]) -> list[str]:
        from dependamerge.github_async import GitHubAsync

        async with GitHubAsync(token="t") as api:

            async def _paginate(path, **kwargs):
                yield {"workflow_runs": runs}

            api.get_paginated = _paginate  # type: ignore[method-assign]
            return await api.get_failing_workflow_run_names_for_sha("o", "r", "sha")

    @pytest.mark.asyncio
    async def test_a_newer_success_supersedes_an_older_failure(self) -> None:
        assert (
            await self._failing(
                [
                    {
                        "name": "CodeQL",
                        "conclusion": "failure",
                        "updated_at": "2026-09-04T10:00:00Z",
                    },
                    {
                        "name": "CodeQL",
                        "conclusion": "success",
                        "updated_at": "2026-09-04T10:30:00Z",
                    },
                ]
            )
            == []
        )

    @pytest.mark.asyncio
    async def test_a_newer_failure_is_still_reported(self) -> None:
        """The control: ordering must decide, not merely 'a success exists'."""
        assert await self._failing(
            [
                {
                    "name": "CodeQL",
                    "conclusion": "success",
                    "updated_at": "2026-09-04T10:00:00Z",
                },
                {
                    "name": "CodeQL",
                    "conclusion": "failure",
                    "updated_at": "2026-09-04T10:30:00Z",
                },
            ]
        ) == ["CodeQL"]


class TestAPartialReadingDoesNotReplaceTheReason:
    """A failed probe is silent, not loud.

    The 503-PR audit recorded ``GET /orgs/{org}/rulesets`` returning 403.
    With the required-context lookup unavailable, a failing status
    context cannot be proven required -- so a reading that saw only an
    optional check would compose a confident "failing checks: …" from
    half the evidence and discard the real rejection.
    """

    @staticmethod
    def _manager(*, required_fails: bool):
        mgr, client = make_merge_manager()
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable_state": "blocked",
            }
        )
        if required_fails:
            client.get_required_status_checks_reliable = AsyncMock(
                side_effect=RuntimeError("403 Forbidden")
            )
        else:
            client.get_required_status_checks_reliable = AsyncMock(
                return_value=([], True)
            )
        client.get_failing_status_contexts = AsyncMock(return_value=[])
        client.get_check_runs_for_ref = AsyncMock(
            return_value=[{"name": "Some Optional Check", "conclusion": "failure"}]
        )
        client.get_failing_workflow_run_names_for_sha = AsyncMock(return_value=[])
        return mgr, client

    @pytest.mark.asyncio
    async def test_an_unproven_reading_needs_every_probe(self) -> None:
        mgr, _ = self._manager(required_fails=True)
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == SAMPLED_REJECTION
        assert out.warning is None

    @pytest.mark.asyncio
    async def test_a_complete_reading_may_report_it(self) -> None:
        """The control: same failing check, every probe answering."""
        mgr, _ = self._manager(required_fails=False)
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "failing checks: Some Optional Check"

    @pytest.mark.asyncio
    async def test_a_proven_blocker_stands_on_its_own_evidence(self) -> None:
        """A probe that did answer proves what it proves.

        The required-context lookup succeeded here; it is the *workflow*
        lookup that failed, and nothing in the claim depends on it.
        """
        mgr, client = self._manager(required_fails=False)
        client.get_required_status_checks_reliable = AsyncMock(
            return_value=([{"context": "pre-commit.ci - pr"}], True)
        )
        client.get_failing_status_contexts = AsyncMock(
            return_value=["pre-commit.ci - pr"]
        )
        client.get_check_runs_for_ref = AsyncMock(return_value=[])
        client.get_failing_workflow_run_names_for_sha = AsyncMock(
            side_effect=RuntimeError("403 Forbidden")
        )
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "blocked by required status check: pre-commit.ci - pr"


class TestAnUnrecognisedStatusKeepsItsMessage:
    """The classification is an allowlist, so novelty is safe.

    Naming the statuses that describe the *attempt* meant every status
    nobody had thought of -- a 404 on the merge endpoint, whatever
    GitHub adds next -- defaulted to being read as a verdict on the pull
    request, and so became withdrawable on a later clean reading.
    """

    @staticmethod
    async def _classify(status: str) -> bool:
        mgr, _client = make_merge_manager()
        pr = _pr("python-workflows", 84)
        mgr._last_merge_exception[f"{pr.repository_full_name}#{pr.number}"] = (
            RuntimeError(
                f"Failed to merge PR #84. Client error '{status}' for url "
                "'https://api.github.com/repos/o/r/pulls/84/merge'. "
                "GitHub: Something GitHub said"
            )
        )
        _reason, refused = await mgr._get_failure_summary(pr)
        return refused

    @pytest.mark.parametrize(
        "status",
        [
            "404 Not Found",
            "410 Gone",
            "451 Unavailable For Legal Reasons",
            # Validation Failed: a malformed request, such as a
            # merge_method the repository does not allow.  Persistent,
            # so a clean reading must not advise a re-run.
            "422 Unprocessable Entity",
        ],
    )
    @pytest.mark.asyncio
    async def test_an_unlisted_status_is_not_a_verdict(self, status: str) -> None:
        assert await self._classify(status) is False

    @pytest.mark.parametrize(
        "status",
        ["405 Method Not Allowed", "409 Conflict"],
    )
    @pytest.mark.asyncio
    async def test_the_listed_statuses_still_are(self, status: str) -> None:
        assert await self._classify(status) is True


class TestAnUnreadableRequiredCheckListIsNotEmpty:
    """A 403 and a branch with no requirements return the same list.

    ``get_required_status_checks`` swallowed the difference, reporting
    the empty list either way and using its own reliability signal only
    to decide caching. The completeness guard therefore read an
    unreadable ruleset as a confident "nothing is required".
    """

    @pytest.mark.asyncio
    async def test_the_reliability_signal_reaches_the_caller(self) -> None:
        from dependamerge.github_async import GitHubAsync

        async with GitHubAsync(token="t") as api:
            api.get = AsyncMock(side_effect=RuntimeError("403 Forbidden"))  # type: ignore[method-assign]
            checks, reliable = await api.get_required_status_checks_reliable(
                "o", "r", "main"
            )

        assert checks == []
        assert reliable is False

    @pytest.mark.asyncio
    async def test_an_unreliable_read_blocks_an_unproven_report(self) -> None:
        mgr, client = make_merge_manager()
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable_state": "blocked",
            }
        )
        client.get_required_status_checks_reliable = AsyncMock(return_value=([], False))
        client.get_failing_status_contexts = AsyncMock(return_value=[])
        client.get_check_runs_for_ref = AsyncMock(
            return_value=[{"name": "Some Optional Check", "conclusion": "failure"}]
        )
        client.get_failing_workflow_run_names_for_sha = AsyncMock(return_value=[])
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == SAMPLED_REJECTION


class TestRequiredNamesAreMatchedExactly:
    """GitHub matches required contexts case-sensitively, so this must.

    A branch requiring ``Build`` is not satisfied by a check called
    ``build``. Folding case would let an optional check be proven
    required by a name it merely resembles -- the false diagnosis this
    reading exists to prevent, reintroduced by a call to ``.lower()``.
    """

    @staticmethod
    def _manager(*, required: list[str], failing: list[str]):
        mgr, client = make_merge_manager()
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable_state": "blocked",
            }
        )
        client.get_required_status_checks_reliable = AsyncMock(
            return_value=([{"context": c} for c in required], True)
        )
        client.get_failing_status_contexts = AsyncMock(return_value=failing)
        client.get_check_runs_for_ref = AsyncMock(return_value=[])
        client.get_failing_workflow_run_names_for_sha = AsyncMock(return_value=[])
        return mgr, client

    @staticmethod
    async def _reason(mgr) -> str | None:
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
        )
        confirmed: MergeResult = await mgr._confirm_failure(pr, result)
        return confirmed.error

    @pytest.mark.asyncio
    async def test_a_differently_cased_name_is_not_the_required_one(self) -> None:
        mgr, _ = self._manager(required=["Build"], failing=["build"])

        # Named, since it is failing -- but not called the blocker.
        assert await self._reason(mgr) == "failing checks: build"

    @pytest.mark.asyncio
    async def test_the_exact_name_is(self) -> None:
        """The control: only case differs between this and the last."""
        mgr, _ = self._manager(required=["Build"], failing=["Build"])

        assert await self._reason(mgr) == "blocked by required status check: Build"


class TestTheSummaryExposesTheThirdOutcome:
    """A programmatic consumer must be able to see it.

    ``UNSETTLED`` counts toward ``total``, so leaving it out of the
    summary makes the numbers fail to add up and gives a caller no way
    to tell a re-runnable PR from one needing a person.
    """

    def test_unsettled_is_counted(self) -> None:
        mgr, _client = make_merge_manager()
        mgr._results = [
            MergeResult(pr_info=_pr("a", 1), status=MergeStatus.UNSETTLED),
            MergeResult(pr_info=_pr("b", 2), status=MergeStatus.FAILED),
            MergeResult(pr_info=_pr("c", 3), status=MergeStatus.MERGED),
        ]

        summary = mgr.get_results_summary()

        assert summary["unsettled"] == 1
        assert summary["failed"] == 1
        assert summary["total"] == 3

    def test_the_empty_shape_carries_the_key(self) -> None:
        """Otherwise a consumer's key lookup raises on an empty run."""
        mgr, _client = make_merge_manager()
        mgr._results = []

        assert mgr.get_results_summary()["unsettled"] == 0


class TestAnUnresolvedDefaultBranchIsUnreliable:
    """``~DEFAULT_BRANCH`` rulesets cannot be filtered without it.

    They are then conservatively treated as applying to every branch, so
    on a non-default base the collected checks may belong to a branch
    the pull request is not targeting -- which could prove a failing
    context required when it is not.
    """

    @pytest.mark.asyncio
    async def test_it_is_not_reported_as_reliable(self) -> None:
        from dependamerge.github_async import GitHubAsync

        async def _router(path, **kwargs):
            if path == "/repos/o/r":
                raise RuntimeError("500 Internal Server Error")
            if "rulesets" in path:
                return []
            raise RuntimeError("404 Not Found")

        async with GitHubAsync(token="t") as api:
            api.get = AsyncMock(side_effect=_router)  # type: ignore[method-assign]
            _checks, reliable = await api.get_required_status_checks_reliable(
                "o", "r", "release/v2"
            )

        assert reliable is False


class TestAClosedPullRequestIsNeverUnsettled:
    """Re-running cannot merge a PR that is closed.

    ``_confirm_failure`` reaches the settling step for a PR closed with
    merged-ness unknown -- a trimmed payload carrying neither ``merged``
    nor ``merged_at`` -- and such a payload can still report ``clean``.
    """

    @pytest.mark.asyncio
    async def test_a_closed_pr_with_unknown_mergedness_stays_failed(self) -> None:
        mgr, client = make_merge_manager()
        pr = _pr("docker-workflows", 70)
        client.get = AsyncMock(
            return_value={
                "state": "closed",
                "mergeable": True,
                "mergeable_state": "clean",
            }
        )
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.status is MergeStatus.FAILED
        assert out.error == SAMPLED_REJECTION

    @pytest.mark.asyncio
    async def test_the_same_payload_open_is_unsettled(self) -> None:
        """The control: only the state differs."""
        mgr, client = make_merge_manager()
        pr = _pr("docker-workflows", 70)
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "mergeable": True,
                "mergeable_state": "clean",
            }
        )
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.status is MergeStatus.UNSETTLED


class TestAnUnexplainedFailureIsNotAVerdict:
    """An exception this code cannot classify says nothing about the PR.

    A retry-exhausted network timeout carries no ``GitHub:`` body and no
    recognisable status, so the classifier declines it -- and the
    state-based fallback then claimed every such failure as a verdict,
    letting a later clean read rewrite the timeout as ``UNSETTLED``.
    """

    @staticmethod
    async def _classify(message: str | None):
        mgr, _client = make_merge_manager()
        pr = _pr("python-workflows", 84)
        pr.mergeable_state = "blocked"
        if message is not None:
            mgr._last_merge_exception[f"{pr.repository_full_name}#{pr.number}"] = (
                RuntimeError(message)
            )
        return await mgr._get_failure_summary(pr)

    @pytest.mark.asyncio
    async def test_an_unrecognised_exception_is_not_withdrawable(self) -> None:
        _reason, refused = await self._classify(
            "ReadTimeout: the read operation timed out"
        )

        assert refused is False

    @pytest.mark.asyncio
    async def test_a_verdict_status_without_a_body_still_qualifies(self) -> None:
        """A 405 says GitHub judged the PR even with nothing to quote."""
        _reason, refused = await self._classify(
            "Client error '405 Method Not Allowed' for url '.../merge'"
        )

        assert refused is True

    @pytest.mark.asyncio
    async def test_no_exception_at_all_is_a_verdict(self) -> None:
        """With nothing else to go on, the reason *is* the PR's state."""
        _reason, refused = await self._classify(None)

        assert refused is True


class TestFailuresAreRejudgedBeforeTheSummary:
    """A PR can go green while its siblings are still being merged.

    The per-PR confirmation runs the moment that PR's task completes,
    which in an owner-wide run is long before anything is printed. One
    refused at minute two and clean by minute ten was still reported as
    failed -- the exact case the issue's acceptance criterion names.
    """

    @pytest.mark.asyncio
    async def test_a_failure_that_cleared_is_corrected(self) -> None:
        mgr, client = make_merge_manager()
        pr = _pr("test-python-project", 396)
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": "clean",
            }
        )
        results = [
            MergeResult(
                pr_info=pr,
                status=MergeStatus.FAILED,
                merge_refused=True,
                error=SAMPLED_REJECTION,
            )
        ]

        out = await mgr._reconcile_reported_failures(results)

        assert out[0].status is MergeStatus.UNSETTLED

    @pytest.mark.asyncio
    async def test_a_still_failing_pr_costs_one_request(self) -> None:
        mgr, client = make_merge_manager()
        pr = _pr("python-workflows", 84)
        client.get = AsyncMock(
            return_value={"state": "open", "merged": False, "mergeable_state": "dirty"}
        )
        results = [
            MergeResult(
                pr_info=pr,
                status=MergeStatus.FAILED,
                merge_refused=True,
                error="merge conflicts",
            )
        ]

        out = await mgr._reconcile_reported_failures(results)

        assert out[0].status is MergeStatus.FAILED
        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_outcomes_already_settled_are_not_re_read(self) -> None:
        """Only failures are re-judged; everything else costs nothing."""
        mgr, client = make_merge_manager()
        client.get = AsyncMock()
        results = [
            MergeResult(pr_info=_pr("a", 1), status=MergeStatus.MERGED),
            MergeResult(pr_info=_pr("b", 2), status=MergeStatus.UNSETTLED),
            MergeResult(pr_info=_pr("c", 3), status=MergeStatus.BLOCKED),
        ]

        await mgr._reconcile_reported_failures(results)

        client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_tracker_tally_is_corrected_too(self) -> None:
        """Otherwise the live counts and the closing summary disagree."""
        from dependamerge.progress_tracker import MergeProgressTracker

        tracker = MergeProgressTracker("lfreleng-actions")
        tracker.set_total_prs(1)
        mgr, client = make_merge_manager(progress_tracker=tracker)
        pr = _pr("docker-workflows", 70)
        mgr._record_terminal_outcome(pr, MergeStatus.FAILED)
        assert tracker.prs_failed == 1

        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": "clean",
            }
        )
        await mgr._reconcile_reported_failures(
            [
                MergeResult(
                    pr_info=pr,
                    status=MergeStatus.FAILED,
                    merge_refused=True,
                    error=SAMPLED_REJECTION,
                )
            ]
        )

        assert tracker.prs_failed == 0
        assert tracker.prs_unsettled == 1
        # The PR finished once; only which counter holds it changed.
        assert tracker.completed_prs == 1


class TestAFailingWorkflowIsAlwaysNamed:
    """A workflow can fail with no check run to surface it.

    ``startup_failure`` means it never got far enough to report a job,
    so if the rejection also did not name it, dropping it would lose the
    only mention of that failure anywhere in the report.
    """

    @pytest.mark.asyncio
    async def test_an_unnamed_failing_workflow_still_appears(self) -> None:
        mgr, client = make_merge_manager()
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable_state": "blocked",
            }
        )
        client.get_required_status_checks_reliable = AsyncMock(return_value=([], True))
        client.get_failing_status_contexts = AsyncMock(return_value=[])
        client.get_check_runs_for_ref = AsyncMock(return_value=[])
        client.get_failing_workflow_run_names_for_sha = AsyncMock(
            return_value=["Zizmor Scan 🌈"]
        )
        pr = _pr("workflows-template", 55)
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            # Names a different workflow entirely.
            error=SAMPLED_REJECTION,
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "failing checks: Zizmor Scan 🌈"

    @pytest.mark.asyncio
    async def test_a_promoted_workflow_is_not_listed_twice(self) -> None:
        """The control: naming it proves it, and it appears once."""
        rejection = (
            "Repository rule violations found Required workflows "
            "'Zizmor Scan 🌈' are not satisfied"
        )
        mgr, client = make_merge_manager()
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable_state": "blocked",
            }
        )
        client.get_required_status_checks_reliable = AsyncMock(return_value=([], True))
        client.get_failing_status_contexts = AsyncMock(return_value=[])
        client.get_check_runs_for_ref = AsyncMock(return_value=[])
        client.get_failing_workflow_run_names_for_sha = AsyncMock(
            return_value=["Zizmor Scan 🌈"]
        )
        pr = _pr("workflows-template", 55)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, merge_refused=True, error=rejection
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "blocked by required workflow: Zizmor Scan 🌈"


class TestTheRejectionNoteSurvivesASecondPass:
    """The end-of-run reconciliation re-reads still-failed PRs.

    By then ``error`` holds the live blocker from the first pass, so
    re-noting it would overwrite the rejection the note exists to keep.
    """

    @pytest.mark.asyncio
    async def test_the_original_rejection_is_still_the_note(self) -> None:
        mgr, client = make_merge_manager()
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable_state": "blocked",
            }
        )
        client.get_required_status_checks_reliable = AsyncMock(
            return_value=([{"context": "pre-commit.ci - pr"}], True)
        )
        client.get_failing_status_contexts = AsyncMock(
            return_value=["pre-commit.ci - pr"]
        )
        client.get_check_runs_for_ref = AsyncMock(return_value=[])
        client.get_failing_workflow_run_names_for_sha = AsyncMock(return_value=[])
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
        )

        # First pass, as each PR's task completes.
        await mgr._confirm_failure(pr, result)
        # Second pass, after every task has finished.
        await mgr._reconcile_reported_failures([result])

        assert result.error == "blocked by required status check: pre-commit.ci - pr"
        assert result.warning is not None
        assert SAMPLED_REJECTION in result.warning


class TestNothingIsBothBlockerAndAlsoFailing:
    """A required context often has a same-named check run beside it.

    Listing it twice reads as two problems where there is one.
    """

    @pytest.mark.asyncio
    async def test_a_promoted_context_is_not_repeated(self) -> None:
        mgr, client = make_merge_manager()
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable_state": "blocked",
            }
        )
        client.get_required_status_checks_reliable = AsyncMock(
            return_value=([{"context": "lint"}], True)
        )
        client.get_failing_status_contexts = AsyncMock(return_value=["lint"])
        client.get_check_runs_for_ref = AsyncMock(
            return_value=[{"name": "lint", "conclusion": "failure"}]
        )
        client.get_failing_workflow_run_names_for_sha = AsyncMock(return_value=[])
        pr = _pr("python-workflows", 84)
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            merge_refused=True,
            error=SAMPLED_REJECTION,
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.error == "blocked by required status check: lint"
