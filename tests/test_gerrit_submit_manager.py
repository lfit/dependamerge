# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Tests for Gerrit submit manager.

This module tests the GerritSubmitManager class for reviewing and
submitting Gerrit changes, including parallel operations, dry-run mode,
and error handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from dependamerge.gerrit.models import (
    GerritChangeInfo,
    GerritFileChange,
    GerritSubmitResult,
)
from dependamerge.gerrit.submit_manager import (
    GerritSubmitManager,
    SubmitStatus,
    create_submit_manager,
)


@pytest.fixture
def mock_client():
    """Create a mock Gerrit REST client."""
    client = MagicMock()
    client.is_authenticated = True
    client.post.return_value = {}
    return client


@pytest.fixture
def sample_change():
    """Create a sample open Gerrit change."""
    return GerritChangeInfo(
        number=12345,
        change_id="I1234567890abcdef",
        project="my-project",
        subject="Chore: Bump actions/checkout from 4.1.0 to 4.2.0",
        owner="dependabot",
        branch="main",
        status="NEW",
        submittable=True,
        work_in_progress=False,
        files_changed=[
            GerritFileChange(
                filename=".github/workflows/ci.yml",
                status="M",
                lines_inserted=1,
                lines_deleted=1,
            )
        ],
    )


@pytest.fixture
def wip_change():
    """Create a work-in-progress change."""
    return GerritChangeInfo(
        number=12346,
        change_id="I9876543210fedcba",
        project="my-project",
        subject="WIP: Work in progress",
        owner="developer",
        branch="main",
        status="NEW",
        work_in_progress=True,
    )


@pytest.fixture
def merged_change():
    """Create an already merged change."""
    return GerritChangeInfo(
        number=12347,
        change_id="Iabcdef1234567890",
        project="my-project",
        subject="Already merged",
        owner="developer",
        branch="main",
        status="MERGED",
    )


class TestSubmitStatus:
    """Tests for SubmitStatus enum."""

    def test_pending_status(self):
        """Test PENDING status value."""
        assert SubmitStatus.PENDING.value == "pending"

    def test_submitted_status(self):
        """Test SUBMITTED status value."""
        assert SubmitStatus.SUBMITTED.value == "submitted"

    def test_failed_status(self):
        """Test FAILED status value."""
        assert SubmitStatus.FAILED.value == "failed"

    def test_blocked_status(self):
        """Test BLOCKED status value."""
        assert SubmitStatus.BLOCKED.value == "blocked"


class TestGerritSubmitManagerInit:
    """Tests for GerritSubmitManager initialization."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_basic_init(self, mock_build_client, mock_client):
        """Test basic manager initialization."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        assert manager.host == "gerrit.example.org"
        assert manager.is_authenticated is True

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_init_with_base_path(self, mock_build_client, mock_client):
        """Test initialization with base path."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            base_path="infra",
        )

        assert manager.base_path == "infra"

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_init_without_auth(self, mock_build_client):
        """Test initialization without authentication."""
        mock_client = MagicMock()
        mock_client.is_authenticated = False
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(host="gerrit.example.org")

        assert manager.is_authenticated is False


class TestReviewChange:
    """Tests for the _review_change method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_review_success(self, mock_build_client, mock_client):
        """Test successful review operation."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._review_change(12345, {"Code-Review": 2})

        assert result is True
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/changes/12345/revisions/current/review" in call_args[0][0]

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_review_failure(self, mock_build_client, mock_client):
        """Test failed review operation."""
        from dependamerge.gerrit.client import GerritRestError

        mock_build_client.return_value = mock_client
        mock_client.post.side_effect = GerritRestError("Failed", 500)

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._review_change(12345, {"Code-Review": 2})

        assert result is False


class TestSubmitChange:
    """Tests for the _submit_change method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_success(self, mock_build_client, mock_client):
        """Test successful submit operation."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._submit_change(12345)

        assert result is True
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/changes/12345/submit" in call_args[0][0]

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_failure(self, mock_build_client, mock_client):
        """Test failed submit operation."""
        from dependamerge.gerrit.client import GerritRestError

        mock_build_client.return_value = mock_client
        mock_client.post.side_effect = GerritRestError("Not submittable", 409)

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._submit_change(12345)

        assert result is False


class TestSubmitSingleChange:
    """Tests for the _submit_single_change method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_open_change(self, mock_build_client, mock_client, sample_change):
        """Test submitting an open change."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._submit_single_change(
            sample_change, {"Code-Review": 2}, dry_run=False
        )

        assert result.success is True
        assert result.reviewed is True
        assert result.submitted is True
        assert result.change_number == 12345
        assert result.project == "my-project"

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_wip_change(self, mock_build_client, mock_client, wip_change):
        """Test submitting a WIP change fails."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._submit_single_change(
            wip_change, {"Code-Review": 2}, dry_run=False
        )

        assert result.success is False
        assert "Work In Progress" in result.error

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_merged_change(self, mock_build_client, mock_client, merged_change):
        """Test submitting an already merged change fails."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._submit_single_change(
            merged_change, {"Code-Review": 2}, dry_run=False
        )

        assert result.success is False
        assert "not open" in result.error

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_dry_run(self, mock_build_client, mock_client, sample_change):
        """Test dry run mode."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._submit_single_change(
            sample_change, {"Code-Review": 2}, dry_run=True
        )

        assert result.success is True
        assert result.reviewed is True
        assert result.submitted is True
        # No actual API calls should be made
        mock_client.post.assert_not_called()


class TestSubmitChanges:
    """Tests for the submit_changes method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_multiple_changes(self, mock_build_client, mock_client):
        """Test submitting multiple changes sequentially."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        changes = [
            (
                GerritChangeInfo(
                    number=i,
                    change_id=f"I{i:040d}",
                    project="proj",
                    subject="Test",
                    owner="bot",
                    branch="main",
                    status="NEW",
                ),
                None,
            )
            for i in range(3)
        ]

        results = manager.submit_changes(changes)

        assert len(results) == 3
        assert all(r.success for r in results)

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_empty_list(self, mock_build_client, mock_client):
        """Test submitting empty list."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        results = manager.submit_changes([])

        assert results == []


class TestSubmitChangesParallel:
    """Tests for the submit_changes_parallel method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_parallel_submit(self, mock_build_client, mock_client):
        """Test parallel submission of changes."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
            max_workers=3,
        )

        changes = [
            (
                GerritChangeInfo(
                    number=i,
                    change_id=f"I{i:040d}",
                    project="proj",
                    subject="Test",
                    owner="bot",
                    branch="main",
                    status="NEW",
                ),
                None,
            )
            for i in range(5)
        ]

        results = manager.submit_changes_parallel(changes)

        assert len(results) == 5
        # All should succeed
        assert all(r.success for r in results)

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_parallel_empty_list(self, mock_build_client, mock_client):
        """Test parallel submit with empty list."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        results = manager.submit_changes_parallel([])

        assert results == []


class TestReviewOnly:
    """Tests for the review_only method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_review_only_success(self, mock_build_client, mock_client, sample_change):
        """Test review-only operation."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        results = manager.review_only([sample_change])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].reviewed is True
        assert results[0].submitted is False

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_review_only_dry_run(self, mock_build_client, mock_client, sample_change):
        """Test review-only with dry run."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        results = manager.review_only([sample_change], dry_run=True)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].reviewed is True
        assert results[0].submitted is False
        mock_client.post.assert_not_called()


class TestGetSubmitSummary:
    """Tests for the get_submit_summary method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_summary_all_success(self, mock_build_client, mock_client):
        """Test summary with all successful results."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        results = [
            GerritSubmitResult.success_result(
                change_number=i,
                project="proj",
                reviewed=True,
                submitted=True,
                duration=1.0,
            )
            for i in range(5)
        ]

        summary = manager.get_submit_summary(results)

        assert summary["total"] == 5
        assert summary["successful"] == 5
        assert summary["failed"] == 0
        assert summary["reviewed"] == 5
        assert summary["submitted"] == 5
        assert summary["total_duration_seconds"] == 5.0
        assert summary["average_duration_seconds"] == 1.0

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_summary_mixed_results(self, mock_build_client, mock_client):
        """Test summary with mixed success/failure results."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        results = [
            GerritSubmitResult.success_result(
                change_number=1,
                project="proj",
                reviewed=True,
                submitted=True,
                duration=1.0,
            ),
            GerritSubmitResult.failure_result(
                change_number=2,
                project="proj",
                error="Failed",
                reviewed=True,
                duration=0.5,
            ),
            GerritSubmitResult.failure_result(
                change_number=3,
                project="proj",
                error="Failed",
                duration=0.5,
            ),
        ]

        summary = manager.get_submit_summary(results)

        assert summary["total"] == 3
        assert summary["successful"] == 1
        assert summary["failed"] == 2
        assert summary["reviewed"] == 2
        assert summary["submitted"] == 1

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_summary_empty_results(self, mock_build_client, mock_client):
        """Test summary with empty results."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        summary = manager.get_submit_summary([])

        assert summary["total"] == 0
        assert summary["successful"] == 0
        assert summary["average_duration_seconds"] == 0.0


class TestCreateSubmitManager:
    """Tests for the create_submit_manager factory function."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_create_with_defaults(self, mock_build_client, mock_client):
        """Test factory with default parameters."""
        mock_build_client.return_value = mock_client

        manager = create_submit_manager(host="gerrit.example.org")

        assert isinstance(manager, GerritSubmitManager)
        assert manager.host == "gerrit.example.org"

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_create_with_all_params(self, mock_build_client, mock_client):
        """Test factory with all parameters."""
        mock_build_client.return_value = mock_client

        manager = create_submit_manager(
            host="gerrit.example.org",
            base_path="infra",
            username="user",
            password="pass",
            max_workers=10,
        )

        assert isinstance(manager, GerritSubmitManager)
        assert manager.host == "gerrit.example.org"
        assert manager.base_path == "infra"
