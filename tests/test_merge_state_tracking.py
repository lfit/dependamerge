# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for merge progress tracker state transitions and counters.

The live merge display moved from per-PR console lines to counter-based
reporting.  Three families of counter feed it: transitory per-PR states
(rebasing → waiting) rendered live on the stats line, cumulative
activity totals (rebases triggered, comment macros issued) that survive
the PR reaching a terminal outcome, and the terminal counters themselves
(merged / pending / closed / failed / skipped / blocked).
``AsyncMergeManager._record_terminal_outcome`` is the single accounting
point mapping ``MergeStatus`` onto the terminal counters.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dependamerge import rebase as rebase_module
from dependamerge.merge_manager import MergeStatus
from dependamerge.models import PullRequestInfo
from dependamerge.progress_tracker import (
    DummyProgressTracker,
    MergeProgressTracker,
)
from tests.conftest import make_merge_manager


def _make_pr(**overrides: Any) -> PullRequestInfo:
    defaults: dict[str, Any] = {
        "number": 7,
        "title": "CI: Bump foo from 1 to 2",
        "body": "Dependabot PR",
        "author": "dependabot[bot]",
        "head_sha": "abc123",
        "base_branch": "main",
        "head_branch": "dependabot/foo",
        "state": "open",
        "mergeable": True,
        "mergeable_state": "clean",
        "behind_by": 0,
        "files_changed": [],
        "repository_full_name": "org/repo",
        "html_url": "https://github.com/org/repo/pull/7",
    }
    defaults.update(overrides)
    return PullRequestInfo(**defaults)


class TestTransitoryStates:
    """PRs move between transitory display states, one state at a time."""

    def test_track_pr_state_moves_between_states(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.track_pr_state("org/repo#1", "rebasing")
        assert tracker._pr_states == {"org/repo#1": "rebasing"}

        # Transition: the PR occupies exactly one state at a time.
        tracker.track_pr_state("org/repo#1", "recreating")
        assert tracker._pr_states == {"org/repo#1": "recreating"}

        tracker.track_pr_state("org/repo#1", "waiting")
        assert tracker._pr_states == {"org/repo#1": "waiting"}

    def test_track_pr_state_none_clears(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.track_pr_state("org/repo#1", "waiting")
        tracker.track_pr_state("org/repo#1", None)
        assert tracker._pr_states == {}

    def test_clear_unknown_key_is_noop(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.track_pr_state("org/repo#1", None)
        assert tracker._pr_states == {}

    def test_terminal_outcome_clears_transitory_state(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.set_total_prs(2)
        tracker.track_pr_state("org/repo#1", "rebasing")
        tracker.track_pr_state("org/repo#2", "waiting")

        tracker.merge_success("org/repo#1")
        tracker.merge_failure("org/repo#2")

        assert tracker._pr_states == {}
        assert tracker.prs_merged == 1
        assert tracker.prs_failed == 1
        assert tracker.completed_prs == 2

    def test_terminal_outcome_without_key_keeps_states(self) -> None:
        """Legacy no-arg calls still count but cannot clear a state."""
        tracker = MergeProgressTracker("org")
        tracker.track_pr_state("org/repo#1", "rebasing")
        tracker.merge_success()
        assert tracker.prs_merged == 1
        assert tracker._pr_states == {"org/repo#1": "rebasing"}


class TestTerminalCounters:
    """New pending/blocked counters and completion accounting."""

    def test_merge_pending_counts_and_completes(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.set_total_prs(1)
        tracker.merge_pending("org/repo#1")
        assert tracker.prs_pending == 1
        assert tracker.completed_prs == 1

    def test_merge_blocked_counts_and_completes(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.set_total_prs(1)
        tracker.merge_blocked("org/repo#1")
        assert tracker.prs_blocked == 1
        assert tracker.completed_prs == 1

    def test_summary_includes_new_counters(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.merge_pending()
        tracker.merge_blocked()
        summary = tracker.get_summary()
        assert summary["prs_pending"] == 1
        assert summary["prs_blocked"] == 1

    def test_dummy_tracker_mirrors_surface(self) -> None:
        """DummyProgressTracker accepts the whole new surface."""
        dummy = DummyProgressTracker()
        dummy.track_pr_state("org/repo#1", "rebasing")
        dummy.merge_success("org/repo#1")
        dummy.merge_failure("org/repo#1")
        dummy.merge_skipped("org/repo#1")
        dummy.merge_blocked("org/repo#1")
        dummy.merge_pending("org/repo#1")
        dummy.increment_closed("org/repo#1")
        dummy.record_rebase()
        dummy.record_retrigger()

    def test_concurrent_outcomes_lose_no_increments(self) -> None:
        """Counter updates from worker threads are never lost.

        The Gerrit submit manager records outcomes from a
        ThreadPoolExecutor, so the tracker must serialise its
        mutations: each thread walks a PR through a transitory state
        and a terminal outcome, and every increment must survive.
        """
        from concurrent.futures import ThreadPoolExecutor

        tracker = MergeProgressTracker("org")
        total = 64
        tracker.set_total_prs(total)

        def _submit_one(index: int) -> None:
            key = f"org/repo#{index}"
            tracker.track_pr_state(key, "submitting")
            if index % 2 == 0:
                tracker.merge_success(key)
            else:
                tracker.merge_failure(key)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(_submit_one, range(total)))

        assert tracker.prs_merged == total // 2
        assert tracker.prs_failed == total // 2
        assert tracker.completed_prs == total
        assert tracker._pr_states == {}


class TestActivityCounters:
    """Cumulative rebase / re-trigger totals outlive the PRs they came from."""

    def test_record_rebase_accumulates(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.record_rebase()
        tracker.record_rebase()
        assert tracker.rebases_triggered == 2

    def test_record_retrigger_accumulates(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.record_retrigger()
        tracker.record_retrigger(2)
        assert tracker.retriggers_issued == 3

    def test_non_positive_counts_are_ignored(self) -> None:
        """The totals are monotonic whatever a caller passes.

        A display counter must never abort a merge run, so a bad
        argument is dropped rather than raised — but it must also
        never walk the total backwards.
        """
        tracker = MergeProgressTracker("org")
        tracker.record_rebase(3)
        tracker.record_retrigger(3)

        tracker.record_rebase(0)
        tracker.record_rebase(-5)
        tracker.record_retrigger(0)
        tracker.record_retrigger(-5)

        assert tracker.rebases_triggered == 3
        assert tracker.retriggers_issued == 3

    def test_totals_survive_terminal_outcomes(self) -> None:
        """The whole point: the counts stay put once the PR lands.

        Previously "rebased" was a per-PR transitory state, so the
        moment a rebased PR merged or failed the display lost all
        trace that a rebase had ever been requested.
        """
        tracker = MergeProgressTracker("org")
        tracker.rich_available = True
        tracker.set_total_prs(2)

        tracker.track_pr_state("org/repo#1", "rebasing")
        tracker.record_rebase()
        tracker.record_retrigger()
        tracker.merge_success("org/repo#1")

        tracker.track_pr_state("org/repo#2", "rebasing")
        tracker.record_rebase()
        tracker.record_retrigger()
        tracker.merge_failure("org/repo#2")

        assert tracker._pr_states == {}
        assert tracker.rebases_triggered == 2
        assert tracker.retriggers_issued == 2
        plain = tracker._generate_display_text().plain
        assert "⬆️ Rebased: 2" in plain
        assert "📣 Retriggered: 2" in plain
        assert "✅ Merged: 1" in plain
        assert "❌ Failed: 1" in plain

    def test_summary_includes_activity_counters(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.record_rebase()
        tracker.record_retrigger()
        summary = tracker.get_summary()
        assert summary["rebases_triggered"] == 1
        assert summary["retriggers_issued"] == 1

    def test_concurrent_activity_increments_are_serialised(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        tracker = MergeProgressTracker("org")

        def _record(_index: int) -> None:
            tracker.record_rebase()
            tracker.record_retrigger()

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(_record, range(64)))

        assert tracker.rebases_triggered == 64
        assert tracker.retriggers_issued == 64


class TestDisplayRendering:
    """Stats line renders transitory states then terminal counters."""

    def test_states_render_in_pipeline_order(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.rich_available = True
        tracker.set_total_prs(7)
        tracker.track_pr_state("org/repo#1", "waiting")
        tracker.track_pr_state("org/repo#2", "rebasing")
        tracker.track_pr_state("org/repo#3", "recreating")
        tracker.track_pr_state("org/repo#7", "submitting")
        tracker.record_rebase()
        tracker.record_retrigger()
        tracker.merge_success("org/repo#4")
        tracker.merge_pending("org/repo#5")
        tracker.merge_failure("org/repo#6")

        plain = tracker._generate_display_text().plain
        assert "🔄 Rebasing: 1" in plain
        assert "♻️ Recreating: 1" in plain
        assert "⏳ Waiting: 1" in plain
        assert "📤 Submitting: 1" in plain
        assert "⬆️ Rebased: 1" in plain
        assert "📣 Retriggered: 1" in plain
        assert "✅ Merged: 1" in plain
        assert "🤖 Pending: 1" in plain
        assert "❌ Failed: 1" in plain
        # Pipeline order: transitory states, then cumulative activity
        # totals, then terminal counters.
        assert plain.index("Rebasing") < plain.index("Recreating")
        assert plain.index("Recreating") < plain.index("Waiting")
        assert plain.index("Waiting") < plain.index("Submitting")
        assert plain.index("Submitting") < plain.index("Rebased")
        assert plain.index("Rebased") < plain.index("Retriggered")
        assert plain.index("Retriggered") < plain.index("Merged")

    def test_zero_counters_do_not_render(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.rich_available = True
        tracker.set_total_prs(1)
        tracker.merge_success("org/repo#1")
        plain = tracker._generate_display_text().plain
        assert "Pending" not in plain
        assert "Blocked" not in plain
        assert "Rebasing" not in plain
        assert "Rebased" not in plain
        assert "Retriggered" not in plain

    def test_unit_label_defaults_to_prs(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.rich_available = True
        tracker.set_total_prs(2)
        plain = tracker._generate_display_text().plain
        assert "(0/2 PRs, " in plain

    def test_unit_label_renders_custom_noun(self) -> None:
        """Gerrit runs label the progress fraction with 'changes'."""
        tracker = MergeProgressTracker(
            "gerrit.example.org",
            operation_label="Submitting changes",
            operation_icon="\u25b6\ufe0f",
            unit_label="changes",
        )
        tracker.rich_available = True
        tracker.set_total_prs(3)
        tracker.merge_success("proj#1")
        plain = tracker._generate_display_text().plain
        assert "Submitting changes in gerrit.example.org" in plain
        assert "(1/3 changes, " in plain
        assert "PRs" not in plain

    def test_unknown_state_rendered_defensively(self) -> None:
        tracker = MergeProgressTracker("org")
        tracker.rich_available = True
        tracker.track_pr_state("org/repo#1", "polishing")
        plain = tracker._generate_display_text().plain
        assert "Polishing: 1" in plain

    def test_preview_counters_use_evaluation_labels(self) -> None:
        """Preview runs merge nothing — counters must say so.

        A preview tracker showing "✅ Merged: 42" misleads the
        operator into thinking merges happened; the run only judged
        the PRs mergeable.  Failure counts are likewise predictions,
        not events.
        """
        tracker = MergeProgressTracker(
            "org",
            operation_label="Evaluating PRs",
            operation_icon="\U0001f50d",
            preview=True,
        )
        tracker.rich_available = True
        tracker.set_total_prs(3)
        tracker.merge_success("org/repo#1")
        tracker.merge_failure("org/repo#2")
        tracker.merge_blocked("org/repo#3")

        plain = tracker._generate_display_text().plain
        assert "\U0001f50d Evaluating PRs in" in plain
        assert "\u2705 Mergeable: 1" in plain
        assert "\u274c Would fail: 1" in plain
        # Neutral labels stay unchanged.
        assert "\U0001f6d1 Blocked: 1" in plain
        # Execution labels must not appear anywhere in preview.
        assert "Merged:" not in plain
        assert "Failed:" not in plain

    def test_execute_counters_keep_execution_labels(self) -> None:
        tracker = MergeProgressTracker(
            "org",
            operation_label="Merging PRs",
            operation_icon="\u25b6\ufe0f",
        )
        tracker.rich_available = True
        tracker.set_total_prs(2)
        tracker.merge_success("org/repo#1")
        tracker.merge_failure("org/repo#2")

        plain = tracker._generate_display_text().plain
        assert "\u2705 Merged: 1" in plain
        assert "\u274c Failed: 1" in plain
        assert "Mergeable" not in plain
        assert "Would fail" not in plain


class TestRecordTerminalOutcome:
    """_record_terminal_outcome maps MergeStatus onto tracker methods."""

    @pytest.mark.parametrize(
        ("status", "method"),
        [
            (MergeStatus.MERGED, "merge_success"),
            (MergeStatus.FAILED, "merge_failure"),
            (MergeStatus.SKIPPED, "merge_skipped"),
            (MergeStatus.BLOCKED, "merge_blocked"),
            (MergeStatus.CLOSED, "increment_closed"),
            (MergeStatus.AUTO_MERGE_PENDING, "merge_pending"),
        ],
    )
    def test_status_maps_to_counter(self, status: MergeStatus, method: str) -> None:
        tracker = MagicMock()
        mgr, _client = make_merge_manager(progress_tracker=tracker)
        pr = _make_pr()

        mgr._record_terminal_outcome(pr, status)

        getattr(tracker, method).assert_called_once_with("org/repo#7")
        # Exactly one terminal method fires per outcome.
        all_methods = {
            "merge_success",
            "merge_failure",
            "merge_skipped",
            "merge_blocked",
            "merge_pending",
            "increment_closed",
            "pr_completed",
        }
        for other in all_methods - {method}:
            getattr(tracker, other).assert_not_called()

    def test_unexpected_status_falls_back_to_pr_completed(self) -> None:
        tracker = MagicMock()
        mgr, _client = make_merge_manager(progress_tracker=tracker)
        pr = _make_pr()

        mgr._record_terminal_outcome(pr, MergeStatus.PENDING)

        tracker.pr_completed.assert_called_once()
        tracker.merge_success.assert_not_called()
        tracker.merge_failure.assert_not_called()

    def test_no_tracker_is_noop(self) -> None:
        mgr, _client = make_merge_manager(progress_tracker=None)
        # Must not raise.
        mgr._record_terminal_outcome(_make_pr(), MergeStatus.MERGED)

    def test_track_pr_state_delegates_with_pr_key(self) -> None:
        tracker = MagicMock()
        mgr, _client = make_merge_manager(progress_tracker=tracker)
        pr = _make_pr()

        mgr._track_pr_state(pr, "rebasing")
        tracker.track_pr_state.assert_called_once_with("org/repo#7", "rebasing")

        tracker.track_pr_state.reset_mock()
        mgr._track_pr_state(pr, None)
        tracker.track_pr_state.assert_called_once_with("org/repo#7", None)

    def test_record_helpers_delegate_to_tracker(self) -> None:
        tracker = MagicMock()
        mgr, _client = make_merge_manager(progress_tracker=tracker)

        mgr._record_rebase()
        mgr._record_retrigger()

        tracker.record_rebase.assert_called_once_with()
        tracker.record_retrigger.assert_called_once_with()

    def test_record_helpers_without_tracker_are_noops(self) -> None:
        mgr, _client = make_merge_manager(progress_tracker=None)
        # Must not raise.
        mgr._record_rebase()
        mgr._record_retrigger()


class TestMacroSitesRecordActivity:
    """Every comment macro and rebase feeds the cumulative counters."""

    @pytest.mark.asyncio
    async def test_dependabot_rebase_counts_rebase_and_retrigger(self) -> None:
        """``@dependabot rebase`` is one macro *and* one rebase."""
        tracker = MagicMock()
        mgr, client = make_merge_manager(progress_tracker=tracker)
        client.get = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock()

        assert await mgr._request_dependabot_rebase(_make_pr(), "org", "repo")

        tracker.record_rebase.assert_called_once_with()
        tracker.record_retrigger.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_existing_rebase_comment_counts_nothing(self) -> None:
        """A rebase requested by an earlier run is not ours to claim."""
        tracker = MagicMock()
        mgr, client = make_merge_manager(progress_tracker=tracker)
        client.get = AsyncMock(return_value=[{"body": "@dependabot rebase"}])
        client.post_issue_comment = AsyncMock()

        assert await mgr._request_dependabot_rebase(_make_pr(), "org", "repo")

        tracker.record_rebase.assert_not_called()
        tracker.record_retrigger.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_post_counts_nothing(self) -> None:
        tracker = MagicMock()
        mgr, client = make_merge_manager(progress_tracker=tracker)
        client.get = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock(side_effect=RuntimeError("nope"))

        assert not await mgr._request_dependabot_rebase(_make_pr(), "org", "repo")

        tracker.record_rebase.assert_not_called()
        tracker.record_retrigger.assert_not_called()


class TestRebaseModuleRecordsRebases:
    """The rebase orchestrator counts the rebases it performs itself."""

    def _make_ctx(
        self, client: Any, record: Any, track: Any = None
    ) -> rebase_module.RebaseContext:
        return rebase_module.RebaseContext(
            github_client=client,
            token="t",
            rebase_local=False,
            preview_mode=False,
            merge_recheck_interval=0.0,
            merge_poll_max_attempts=1,
            log=logging.getLogger("test"),
            console=MagicMock(),
            rebased_prs=set(),
            enable_auto_merge=AsyncMock(return_value=True),
            track_pr_state=track,
            record_rebase=record,
        )

    @pytest.mark.asyncio
    async def test_local_path_counts_successful_rebase(self) -> None:
        record = MagicMock()
        ctx = self._make_ctx(AsyncMock(), record)

        with patch.object(
            rebase_module.local_rebase,
            "local_rebase_pr",
            new=AsyncMock(return_value=True),
        ):
            await rebase_module._run_local_path(
                ctx=ctx,
                pr_info=_make_pr(),
                owner="org",
                repo="repo",
                local_reason="signatures required",
            )

        record.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_local_path_failure_counts_nothing(self) -> None:
        record = MagicMock()
        ctx = self._make_ctx(AsyncMock(), record)

        with patch.object(
            rebase_module.local_rebase,
            "local_rebase_pr",
            new=AsyncMock(return_value=False),
        ):
            await rebase_module._run_local_path(
                ctx=ctx,
                pr_info=_make_pr(),
                owner="org",
                repo="repo",
                local_reason="signatures required",
            )

        record.assert_not_called()

    @pytest.mark.parametrize("rebase_ok", [True, False])
    @pytest.mark.asyncio
    async def test_local_path_always_clears_rebasing_state(
        self, rebase_ok: bool
    ) -> None:
        """A failed local rebase must not strand the PR in "Rebasing".

        ``perform_step5_rebase`` sets "rebasing" before dispatch, and
        this path defers to auto-merge rather than reaching a terminal
        outcome, so nothing else would clear it.
        """
        track = MagicMock()
        ctx = self._make_ctx(AsyncMock(), MagicMock(), track)
        pr = _make_pr()

        with patch.object(
            rebase_module.local_rebase,
            "local_rebase_pr",
            new=AsyncMock(return_value=rebase_ok),
        ):
            await rebase_module._run_local_path(
                ctx=ctx,
                pr_info=pr,
                owner="org",
                repo="repo",
                local_reason="signatures required",
            )

        track.assert_called_once_with(pr, None)

    @pytest.mark.asyncio
    async def test_rest_path_counts_update_branch(self) -> None:
        record = MagicMock()
        client = AsyncMock()
        client.update_branch = AsyncMock()
        ctx = self._make_ctx(client, record)

        with patch.object(
            rebase_module.polling,
            "_poll_post_rebase",
            new=AsyncMock(return_value=(True, "clean")),
        ):
            outcome = await rebase_module._run_rest_path(
                ctx=ctx, pr_info=_make_pr(), owner="org", repo="repo"
            )

        assert not outcome.failed
        record.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_dependabot_macro_path_does_not_double_count(self) -> None:
        """The macro path delegates to the manager, which already counts."""
        record = MagicMock()
        ctx = self._make_ctx(AsyncMock(), record)
        ctx.request_dependabot_rebase = AsyncMock(return_value=True)

        handled = await rebase_module._run_dependabot_macro_path(
            ctx=ctx,
            pr_info=_make_pr(),
            owner="org",
            repo="repo",
            local_reason="signatures required",
        )

        assert handled is True
        record.assert_not_called()
