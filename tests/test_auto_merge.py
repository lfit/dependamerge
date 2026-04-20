# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Unit tests for auto-merge functionality in AsyncMergeManager.

Covers:
- Enabling auto-merge via GraphQL (success, missing node_id, idempotent,
  graceful failure).
- Merge-flow behaviour when auto-merge is active (blocked vs clean).
- Centralised timing defaults and custom merge-timeout computation.
- Invalid merge-timeout fallback to the default.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dependamerge.github2gerrit_detector import GitHub2GerritDetectionResult
from dependamerge.merge_manager import MergeStatus
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

# ---------------------------------------------------------------------------
# Module-level default PullRequestInfo instance
# ---------------------------------------------------------------------------

_DEFAULT_PR = PullRequestInfo(
    number=42,
    node_id="PR_kwDOTestNode42",
    title="Bump foo from 1.0 to 2.0",
    body="Dependabot PR",
    author="dependabot[bot]",
    head_sha="abc123def456",
    base_branch="main",
    head_branch="dependabot/pip/foo-2.0",
    state="open",
    mergeable=True,
    mergeable_state="blocked",
    behind_by=0,
    files_changed=[],
    repository_full_name="owner/repo",
    html_url="https://github.com/owner/repo/pull/42",
    reviews=[],
    review_comments=[],
)


# ---------------------------------------------------------------------------
# 1. _enable_auto_merge_for_pr - success path
# ---------------------------------------------------------------------------


class TestEnableAutoMergeSuccess:
    """Verify that _enable_auto_merge_for_pr enables auto-merge."""

    @pytest.mark.asyncio
    async def test_enable_auto_merge_success(self) -> None:
        """Enable auto-merge returns True and tracks the PR key."""
        mgr, client = make_merge_manager()
        pr = _DEFAULT_PR.model_copy()

        client.enable_auto_merge = AsyncMock(return_value=True)

        result = await mgr._enable_auto_merge_for_pr(pr, "owner", "repo")

        assert result is True
        assert "owner/repo#42" in mgr._auto_merge_enabled
        client.enable_auto_merge.assert_called_once_with("PR_kwDOTestNode42", "merge")


# ---------------------------------------------------------------------------
# 2. _enable_auto_merge_for_pr - missing node_id
# ---------------------------------------------------------------------------


class TestEnableAutoMergeNoNodeId:
    """When node_id is None, auto-merge cannot be enabled."""

    @pytest.mark.asyncio
    async def test_enable_auto_merge_no_node_id(self) -> None:
        """Return False when the PR has no node_id."""
        mgr, client = make_merge_manager()
        pr = _DEFAULT_PR.model_copy(update={"node_id": None})

        result = await mgr._enable_auto_merge_for_pr(pr, "owner", "repo")

        assert result is False
        assert len(mgr._auto_merge_enabled) == 0


# ---------------------------------------------------------------------------
# 3. _enable_auto_merge_for_pr - idempotent
# ---------------------------------------------------------------------------


class TestEnableAutoMergeIdempotent:
    """Calling enable twice should only hit the API once."""

    @pytest.mark.asyncio
    async def test_enable_auto_merge_idempotent(self) -> None:
        """Second call returns True without calling the API again."""
        mgr, client = make_merge_manager()
        pr = _DEFAULT_PR.model_copy()

        # Pre-populate the tracking set
        mgr._auto_merge_enabled.add("owner/repo#42")

        client.enable_auto_merge = AsyncMock(return_value=True)

        result = await mgr._enable_auto_merge_for_pr(pr, "owner", "repo")

        assert result is True
        client.enable_auto_merge.assert_not_called()


# ---------------------------------------------------------------------------
# 4. _enable_auto_merge_for_pr - GraphQL failure
# ---------------------------------------------------------------------------


class TestEnableAutoMergeFailureGraceful:
    """When the GraphQL call fails, return False without raising."""

    @pytest.mark.asyncio
    async def test_enable_auto_merge_failure_graceful(self) -> None:
        """Return False and do not track the PR when enable fails."""
        mgr, client = make_merge_manager()
        pr = _DEFAULT_PR.model_copy()

        client.enable_auto_merge = AsyncMock(return_value=False)

        result = await mgr._enable_auto_merge_for_pr(pr, "owner", "repo")

        assert result is False
        assert "owner/repo#42" not in mgr._auto_merge_enabled


# ---------------------------------------------------------------------------
# 5. Merge skipped when auto-merge active and blocked
# ---------------------------------------------------------------------------


class TestMergeSkippedWhenAutoMergeActiveAndBlocked:
    """When auto-merge is enabled and PR is blocked, skip manual merge."""

    @pytest.mark.asyncio
    async def test_merge_skipped_when_auto_merge_active_and_blocked(
        self,
    ) -> None:
        """Auto-merge pending: _merge_pr_with_retry is NOT called."""
        mgr, client = make_merge_manager(preview_mode=False)
        pr = _DEFAULT_PR.model_copy(
            update={
                "mergeable_state": "blocked",
                "mergeable": True,
                "state": "open",
            }
        )

        # Pre-populate auto-merge tracking
        mgr._auto_merge_enabled.add("owner/repo#42")

        # Stub GitHub client methods used during the flow
        client.get = AsyncMock(return_value={})
        client.get_required_status_checks = AsyncMock(return_value=[])
        # The skip gate consults analyze_block_reason; return a reason
        # that indicates pending required checks so the skip fires.
        client.analyze_block_reason = AsyncMock(
            return_value="Blocked by pending required check: pre-commit.ci"
        )

        # Patch the manager methods that _merge_single_pr calls
        # before reaching the auto-merge gate.
        no_g2g = GitHub2GerritDetectionResult()
        with (
            patch.object(
                mgr,
                "_detect_github2gerrit",
                new_callable=AsyncMock,
                return_value=no_g2g,
            ),
            patch.object(
                mgr,
                "_get_merge_method_for_repo",
                new_callable=AsyncMock,
                return_value="merge",
            ),
            patch.object(
                mgr,
                "_trigger_stale_precommit_ci",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                mgr,
                "_check_merge_requirements",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                mgr,
                "_approve_pr",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                mgr,
                "_merge_pr_with_retry",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_merge_retry,
        ):
            result = await mgr._merge_single_pr(pr)

        assert result.status == MergeStatus.AUTO_MERGE_PENDING
        mock_merge_retry.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Merge proceeds when auto-merge active and state is clean
# ---------------------------------------------------------------------------


class TestMergeProceedsWhenAutoMergeActiveAndClean:
    """When auto-merge is active but PR is clean, merge proceeds."""

    @pytest.mark.asyncio
    async def test_merge_proceeds_when_auto_merge_active_and_clean(
        self,
    ) -> None:
        """Manual merge still happens when mergeable_state is clean."""
        mgr, client = make_merge_manager(preview_mode=False)
        pr = _DEFAULT_PR.model_copy(
            update={
                "mergeable_state": "clean",
                "mergeable": True,
                "state": "open",
            }
        )

        # Pre-populate auto-merge tracking
        mgr._auto_merge_enabled.add("owner/repo#42")

        client.get = AsyncMock(return_value={})
        client.get_required_status_checks = AsyncMock(return_value=[])

        no_g2g = GitHub2GerritDetectionResult()
        with (
            patch.object(
                mgr,
                "_detect_github2gerrit",
                new_callable=AsyncMock,
                return_value=no_g2g,
            ),
            patch.object(
                mgr,
                "_get_merge_method_for_repo",
                new_callable=AsyncMock,
                return_value="merge",
            ),
            patch.object(
                mgr,
                "_trigger_stale_precommit_ci",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                mgr,
                "_check_merge_requirements",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                mgr,
                "_approve_pr",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                mgr,
                "_merge_pr_with_retry",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_merge_retry,
        ):
            result = await mgr._merge_single_pr(pr)

        assert result.status == MergeStatus.MERGED
        mock_merge_retry.assert_called_once()


# ---------------------------------------------------------------------------
# 7. Centralised timing - defaults
# ---------------------------------------------------------------------------


class TestCentralisedTimingDefaults:
    """Verify timing constants are correctly computed from defaults."""

    def test_centralised_timing_defaults(self) -> None:
        """Default timeout, recheck interval, and poll max are correct."""
        mgr, _client = make_merge_manager()

        assert mgr._merge_timeout == 300.0
        assert mgr._merge_recheck_interval == 10.0
        assert mgr._merge_poll_max_attempts == 30


# ---------------------------------------------------------------------------
# 8. Custom merge timeout
# ---------------------------------------------------------------------------


class TestCustomMergeTimeout:
    """Verify a custom merge_timeout correctly computes poll max."""

    def test_custom_merge_timeout(self) -> None:
        """600s timeout with 10s interval yields 60 poll attempts."""
        mgr, _client = make_merge_manager(merge_timeout=600.0)

        assert mgr._merge_timeout == 600.0
        assert mgr._merge_poll_max_attempts == 60


# ---------------------------------------------------------------------------
# 9. Invalid merge-timeout fallback
# ---------------------------------------------------------------------------


class TestInvalidMergeTimeoutFallback:
    """Invalid merge_timeout values must fall back to the default."""

    def test_negative_merge_timeout_fallback(self) -> None:
        """Negative timeout falls back to 300.0."""
        mgr, _client = make_merge_manager(merge_timeout=-1.0)

        assert mgr._merge_timeout == 300.0

    def test_inf_merge_timeout_fallback(self) -> None:
        """Infinity falls back to 300.0."""
        mgr, _client = make_merge_manager(merge_timeout=float("inf"))

        assert mgr._merge_timeout == 300.0

    def test_nan_merge_timeout_fallback(self) -> None:
        """NaN falls back to 300.0."""
        mgr, _client = make_merge_manager(merge_timeout=float("nan"))

        assert mgr._merge_timeout == 300.0


# ---------------------------------------------------------------------------
# 10. Auto-merge skip gate: do NOT skip when mergeable is False
# ---------------------------------------------------------------------------


class TestAutoMergeSkipGateMergeableFalse:
    """Blocked + mergeable=False (failing checks) must NOT skip manual merge."""

    @pytest.mark.asyncio
    async def test_manual_merge_runs_when_blocked_by_failing_checks(
        self,
    ) -> None:
        """Failing checks: auto-merge gate must not override; retry is called."""
        mgr, client = make_merge_manager(preview_mode=False)
        pr = _DEFAULT_PR.model_copy(
            update={
                "mergeable_state": "blocked",
                "mergeable": False,
                "state": "open",
            }
        )

        mgr._auto_merge_enabled.add("owner/repo#42")

        client.get = AsyncMock(return_value={})
        client.get_required_status_checks = AsyncMock(return_value=[])

        no_g2g = GitHub2GerritDetectionResult()
        with (
            patch.object(
                mgr,
                "_detect_github2gerrit",
                new_callable=AsyncMock,
                return_value=no_g2g,
            ),
            patch.object(
                mgr,
                "_get_merge_method_for_repo",
                new_callable=AsyncMock,
                return_value="merge",
            ),
            patch.object(
                mgr,
                "_trigger_stale_precommit_ci",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                mgr,
                "_check_merge_requirements",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                mgr,
                "_approve_pr",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                mgr,
                "_merge_pr_with_retry",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_merge_retry,
        ):
            result = await mgr._merge_single_pr(pr)

        assert result.status == MergeStatus.MERGED
        mock_merge_retry.assert_called_once()


# ---------------------------------------------------------------------------
# 11. Auto-merge skip gate: force=all must proceed with manual merge
# ---------------------------------------------------------------------------


class TestAutoMergeSkipGateForceAll:
    """--force=all must proceed with manual merge even with auto-merge active."""

    @pytest.mark.asyncio
    async def test_manual_merge_runs_with_force_all(self) -> None:
        """force_level='all': auto-merge gate must not override."""
        mgr, client = make_merge_manager(preview_mode=False, force_level="all")
        pr = _DEFAULT_PR.model_copy(
            update={
                "mergeable_state": "blocked",
                "mergeable": True,
                "state": "open",
            }
        )

        mgr._auto_merge_enabled.add("owner/repo#42")

        client.get = AsyncMock(return_value={})
        client.get_required_status_checks = AsyncMock(return_value=[])

        no_g2g = GitHub2GerritDetectionResult()
        with (
            patch.object(
                mgr,
                "_detect_github2gerrit",
                new_callable=AsyncMock,
                return_value=no_g2g,
            ),
            patch.object(
                mgr,
                "_get_merge_method_for_repo",
                new_callable=AsyncMock,
                return_value="merge",
            ),
            patch.object(
                mgr,
                "_trigger_stale_precommit_ci",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                mgr,
                "_check_merge_requirements",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                mgr,
                "_approve_pr",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                mgr,
                "_merge_pr_with_retry",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_merge_retry,
        ):
            result = await mgr._merge_single_pr(pr)

        assert result.status == MergeStatus.MERGED
        mock_merge_retry.assert_called_once()


# ---------------------------------------------------------------------------
# 12. Poll-max ceiling: non-multiple timeouts round up, not down
# ---------------------------------------------------------------------------


class TestMergePollMaxAttemptsCeiling:
    """Non-multiple merge_timeout must round UP via math.ceil, not truncate."""

    def test_poll_max_rounds_up_for_non_multiple_timeout(self) -> None:
        """301s/10s must yield 31 attempts (not 30 via truncation)."""
        mgr, _client = make_merge_manager(merge_timeout=301.0)

        assert mgr._merge_timeout == 301.0
        assert mgr._merge_poll_max_attempts == 31

    def test_poll_max_exact_multiple_unchanged(self) -> None:
        """Exact multiple (300s/10s) still yields 30 attempts."""
        mgr, _client = make_merge_manager(merge_timeout=300.0)

        assert mgr._merge_poll_max_attempts == 30


# ---------------------------------------------------------------------------
# 13. Recheck-interval clamping for small merge_timeout values
# ---------------------------------------------------------------------------


class TestRecheckIntervalClamping:
    """merge_timeout < default interval must clamp the per-iteration sleep."""

    def test_recheck_interval_clamped_for_small_timeout(self) -> None:
        """merge_timeout=3s clamps interval to 3s so loops don't oversleep."""
        mgr, _client = make_merge_manager(merge_timeout=3.0)

        assert mgr._merge_timeout == 3.0
        assert mgr._merge_recheck_interval == 3.0
        # ceil(3/3) = 1 attempt — total wait still equals merge_timeout
        assert mgr._merge_poll_max_attempts == 1

    def test_recheck_interval_unchanged_for_large_timeout(self) -> None:
        """merge_timeout >= default interval keeps the default 10s cadence."""
        mgr, _client = make_merge_manager(merge_timeout=600.0)

        assert mgr._merge_recheck_interval == 10.0
        assert mgr._merge_poll_max_attempts == 60


# ---------------------------------------------------------------------------
# 14. AUTO_MERGE_PENDING triggers progress tracker pr_completed()
# ---------------------------------------------------------------------------


class TestAutoMergePendingCompletesProgress:
    """AUTO_MERGE_PENDING status must bump PR-level progress to completed."""

    @pytest.mark.asyncio
    async def test_auto_merge_pending_calls_pr_completed(self) -> None:
        """pr_completed() is called so PR progress reaches 100%."""
        tracker = MagicMock()
        mgr, client = make_merge_manager(preview_mode=False, progress_tracker=tracker)
        pr = _DEFAULT_PR.model_copy(
            update={
                "mergeable_state": "blocked",
                "mergeable": True,
                "state": "open",
            }
        )

        mgr._auto_merge_enabled.add("owner/repo#42")

        client.get = AsyncMock(return_value={})
        client.get_required_status_checks = AsyncMock(return_value=[])
        # The skip gate consults analyze_block_reason; return a reason
        # that indicates pending required checks so the skip fires.
        client.analyze_block_reason = AsyncMock(
            return_value="Blocked by pending required check: pre-commit.ci"
        )

        no_g2g = GitHub2GerritDetectionResult()
        with (
            patch.object(
                mgr,
                "_detect_github2gerrit",
                new_callable=AsyncMock,
                return_value=no_g2g,
            ),
            patch.object(
                mgr,
                "_get_merge_method_for_repo",
                new_callable=AsyncMock,
                return_value="merge",
            ),
            patch.object(
                mgr,
                "_trigger_stale_precommit_ci",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                mgr,
                "_check_merge_requirements",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                mgr,
                "_approve_pr",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                mgr,
                "_merge_pr_with_retry",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await mgr._merge_single_pr_with_semaphore(pr)

        assert result.status == MergeStatus.AUTO_MERGE_PENDING
        tracker.pr_completed.assert_called_once()
        tracker.merge_success.assert_not_called()
        tracker.merge_failure.assert_not_called()


# ---------------------------------------------------------------------------
# 15. Auto-merge skip gate: only skip on pending required checks
# ---------------------------------------------------------------------------


class TestAutoMergeSkipGateBlockReason:
    """Skip gate must consult analyze_block_reason and only skip on pending checks."""

    @pytest.mark.asyncio
    async def test_manual_merge_runs_when_blocked_by_missing_approvals(
        self,
    ) -> None:
        """Block reason = 'requires approval': manual merge must still run."""
        mgr, client = make_merge_manager(preview_mode=False)
        pr = _DEFAULT_PR.model_copy(
            update={
                "mergeable_state": "blocked",
                "mergeable": True,
                "state": "open",
            }
        )

        mgr._auto_merge_enabled.add("owner/repo#42")

        client.get = AsyncMock(return_value={})
        client.get_required_status_checks = AsyncMock(return_value=[])
        # Block reason is NOT pending required checks — skip must not fire.
        client.analyze_block_reason = AsyncMock(
            return_value="Blocked by branch protection (requires approval)"
        )

        no_g2g = GitHub2GerritDetectionResult()
        with (
            patch.object(
                mgr,
                "_detect_github2gerrit",
                new_callable=AsyncMock,
                return_value=no_g2g,
            ),
            patch.object(
                mgr,
                "_get_merge_method_for_repo",
                new_callable=AsyncMock,
                return_value="merge",
            ),
            patch.object(
                mgr,
                "_trigger_stale_precommit_ci",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                mgr,
                "_check_merge_requirements",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                mgr,
                "_approve_pr",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                mgr,
                "_merge_pr_with_retry",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_merge_retry,
        ):
            result = await mgr._merge_single_pr(pr)

        assert result.status == MergeStatus.MERGED
        mock_merge_retry.assert_called_once()
        # analyze_block_reason was consulted before deciding to proceed
        client.analyze_block_reason.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_manual_merge_runs_when_analyze_block_reason_fails(
        self,
    ) -> None:
        """If analyze_block_reason raises, fall back to manual merge attempt."""
        mgr, client = make_merge_manager(preview_mode=False)
        pr = _DEFAULT_PR.model_copy(
            update={
                "mergeable_state": "blocked",
                "mergeable": True,
                "state": "open",
            }
        )

        mgr._auto_merge_enabled.add("owner/repo#42")

        client.get = AsyncMock(return_value={})
        client.get_required_status_checks = AsyncMock(return_value=[])
        client.analyze_block_reason = AsyncMock(side_effect=RuntimeError("boom"))

        no_g2g = GitHub2GerritDetectionResult()
        with (
            patch.object(
                mgr,
                "_detect_github2gerrit",
                new_callable=AsyncMock,
                return_value=no_g2g,
            ),
            patch.object(
                mgr,
                "_get_merge_method_for_repo",
                new_callable=AsyncMock,
                return_value="merge",
            ),
            patch.object(
                mgr,
                "_trigger_stale_precommit_ci",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                mgr,
                "_check_merge_requirements",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                mgr,
                "_approve_pr",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                mgr,
                "_merge_pr_with_retry",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_merge_retry,
        ):
            result = await mgr._merge_single_pr(pr)

        # Defensive fallback: proceed with manual merge rather than
        # silently marking as AUTO_MERGE_PENDING.
        assert result.status == MergeStatus.MERGED
        mock_merge_retry.assert_called_once()
