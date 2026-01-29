# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Tests for Gerrit data models.

This module tests the Pydantic models for Gerrit changes, file changes,
labels, and comparison results.
"""

from dependamerge.gerrit.models import (
    GerritChangeInfo,
    GerritChangeStatus,
    GerritComparisonResult,
    GerritFileChange,
    GerritFileStatus,
    GerritLabelInfo,
    GerritSubmitResult,
)


class TestGerritChangeStatus:
    """Tests for GerritChangeStatus enum."""

    def test_new_status(self):
        """Test NEW status value."""
        assert GerritChangeStatus.NEW.value == "NEW"

    def test_merged_status(self):
        """Test MERGED status value."""
        assert GerritChangeStatus.MERGED.value == "MERGED"

    def test_abandoned_status(self):
        """Test ABANDONED status value."""
        assert GerritChangeStatus.ABANDONED.value == "ABANDONED"


class TestGerritFileStatus:
    """Tests for GerritFileStatus enum."""

    def test_added_status(self):
        """Test ADDED status value."""
        assert GerritFileStatus.ADDED.value == "A"

    def test_modified_status(self):
        """Test MODIFIED status value."""
        assert GerritFileStatus.MODIFIED.value == "M"

    def test_deleted_status(self):
        """Test DELETED status value."""
        assert GerritFileStatus.DELETED.value == "D"

    def test_renamed_status(self):
        """Test RENAMED status value."""
        assert GerritFileStatus.RENAMED.value == "R"


class TestGerritFileChange:
    """Tests for GerritFileChange model."""

    def test_basic_construction(self):
        """Test basic model construction."""
        file_change = GerritFileChange(
            filename="src/main.py",
            status="M",
            lines_inserted=10,
            lines_deleted=5,
        )

        assert file_change.filename == "src/main.py"
        assert file_change.status == "M"
        assert file_change.lines_inserted == 10
        assert file_change.lines_deleted == 5

    def test_default_values(self):
        """Test default values for optional fields."""
        file_change = GerritFileChange(filename="README.md")

        assert file_change.status == "M"
        assert file_change.lines_inserted == 0
        assert file_change.lines_deleted == 0
        assert file_change.size_delta == 0
        assert file_change.old_path is None

    def test_from_api_response(self):
        """Test creation from API response data."""
        api_data = {
            "status": "A",
            "lines_inserted": 50,
            "lines_deleted": 0,
            "size_delta": 1500,
        }

        file_change = GerritFileChange.from_api_response("new_file.py", api_data)

        assert file_change.filename == "new_file.py"
        assert file_change.status == "A"
        assert file_change.lines_inserted == 50
        assert file_change.lines_deleted == 0
        assert file_change.size_delta == 1500

    def test_from_api_response_with_rename(self):
        """Test creation from API response with rename."""
        api_data = {
            "status": "R",
            "old_path": "old_name.py",
            "lines_inserted": 0,
            "lines_deleted": 0,
        }

        file_change = GerritFileChange.from_api_response("new_name.py", api_data)

        assert file_change.filename == "new_name.py"
        assert file_change.status == "R"
        assert file_change.old_path == "old_name.py"

    def test_from_api_response_missing_fields(self):
        """Test creation from API response with missing fields."""
        api_data = {}

        file_change = GerritFileChange.from_api_response("file.txt", api_data)

        assert file_change.filename == "file.txt"
        assert file_change.status == "M"
        assert file_change.lines_inserted == 0


class TestGerritLabelInfo:
    """Tests for GerritLabelInfo model."""

    def test_basic_construction(self):
        """Test basic model construction."""
        label = GerritLabelInfo(
            name="Code-Review",
            approved=True,
            value=2,
        )

        assert label.name == "Code-Review"
        assert label.approved is True
        assert label.value == 2

    def test_default_values(self):
        """Test default values."""
        label = GerritLabelInfo(name="Verified")

        assert label.approved is False
        assert label.rejected is False
        assert label.value is None
        assert label.blocking is False

    def test_from_api_response_approved(self):
        """Test creation from approved label API response."""
        api_data = {
            "approved": {"_account_id": 1000},
            "value": 2,
        }

        label = GerritLabelInfo.from_api_response("Code-Review", api_data)

        assert label.name == "Code-Review"
        assert label.approved is True
        assert label.rejected is False
        assert label.value == 2

    def test_from_api_response_rejected(self):
        """Test creation from rejected label API response."""
        api_data = {
            "rejected": {"_account_id": 1000},
        }

        label = GerritLabelInfo.from_api_response("Code-Review", api_data)

        assert label.name == "Code-Review"
        assert label.approved is False
        assert label.rejected is True
        assert label.value == -2

    def test_from_api_response_empty(self):
        """Test creation from empty label API response."""
        api_data = {}

        label = GerritLabelInfo.from_api_response("Verified", api_data)

        assert label.name == "Verified"
        assert label.approved is False
        assert label.rejected is False
        assert label.value is None


class TestGerritChangeInfo:
    """Tests for GerritChangeInfo model."""

    def test_basic_construction(self):
        """Test basic model construction."""
        change = GerritChangeInfo(
            number=12345,
            change_id="I1234567890abcdef",
            project="my-project",
            subject="Fix something",
            owner="testuser",
            branch="main",
            status="NEW",
        )

        assert change.number == 12345
        assert change.change_id == "I1234567890abcdef"
        assert change.project == "my-project"
        assert change.subject == "Fix something"
        assert change.owner == "testuser"
        assert change.branch == "main"
        assert change.status == "NEW"

    def test_default_values(self):
        """Test default values for optional fields."""
        change = GerritChangeInfo(
            number=1,
            change_id="I123",
            project="proj",
            subject="Test",
            owner="user",
            branch="main",
            status="NEW",
        )

        assert change.message is None
        assert change.topic is None
        assert change.owner_email is None
        assert change.current_revision == ""
        assert change.submittable is False
        assert change.mergeable is None
        assert change.work_in_progress is False
        assert change.files_changed == []
        assert change.labels == []
        assert change.url == ""

    def test_is_open_property(self):
        """Test is_open property."""
        change = GerritChangeInfo(
            number=1,
            change_id="I123",
            project="proj",
            subject="Test",
            owner="user",
            branch="main",
            status="NEW",
        )

        assert change.is_open is True
        assert change.is_merged is False
        assert change.is_abandoned is False

    def test_is_merged_property(self):
        """Test is_merged property."""
        change = GerritChangeInfo(
            number=1,
            change_id="I123",
            project="proj",
            subject="Test",
            owner="user",
            branch="main",
            status="MERGED",
        )

        assert change.is_open is False
        assert change.is_merged is True
        assert change.is_abandoned is False

    def test_is_abandoned_property(self):
        """Test is_abandoned property."""
        change = GerritChangeInfo(
            number=1,
            change_id="I123",
            project="proj",
            subject="Test",
            owner="user",
            branch="main",
            status="ABANDONED",
        )

        assert change.is_open is False
        assert change.is_merged is False
        assert change.is_abandoned is True

    def test_can_submit_property(self):
        """Test can_submit property."""
        change = GerritChangeInfo(
            number=1,
            change_id="I123",
            project="proj",
            subject="Test",
            owner="user",
            branch="main",
            status="NEW",
            submittable=True,
            submit_requirements_met=True,
            work_in_progress=False,
        )

        assert change.can_submit is True

    def test_can_submit_false_when_wip(self):
        """Test can_submit is False when work in progress."""
        change = GerritChangeInfo(
            number=1,
            change_id="I123",
            project="proj",
            subject="Test",
            owner="user",
            branch="main",
            status="NEW",
            submittable=True,
            work_in_progress=True,
        )

        assert change.can_submit is False

    def test_can_submit_false_when_not_submittable(self):
        """Test can_submit is False when not submittable."""
        change = GerritChangeInfo(
            number=1,
            change_id="I123",
            project="proj",
            subject="Test",
            owner="user",
            branch="main",
            status="NEW",
            submittable=False,
        )

        assert change.can_submit is False

    def test_file_count_property(self):
        """Test file_count property."""
        change = GerritChangeInfo(
            number=1,
            change_id="I123",
            project="proj",
            subject="Test",
            owner="user",
            branch="main",
            status="NEW",
            files_changed=[
                GerritFileChange(filename="a.py"),
                GerritFileChange(filename="b.py"),
                GerritFileChange(filename="c.py"),
            ],
        )

        assert change.file_count == 3

    def test_total_lines_changed_property(self):
        """Test total_lines_changed property."""
        change = GerritChangeInfo(
            number=1,
            change_id="I123",
            project="proj",
            subject="Test",
            owner="user",
            branch="main",
            status="NEW",
            files_changed=[
                GerritFileChange(filename="a.py", lines_inserted=10, lines_deleted=5),
                GerritFileChange(filename="b.py", lines_inserted=20, lines_deleted=3),
            ],
        )

        assert change.total_lines_changed == 38

    def test_get_label_value(self):
        """Test get_label_value method."""
        change = GerritChangeInfo(
            number=1,
            change_id="I123",
            project="proj",
            subject="Test",
            owner="user",
            branch="main",
            status="NEW",
            labels=[
                GerritLabelInfo(name="Code-Review", value=2),
                GerritLabelInfo(name="Verified", value=1),
            ],
        )

        assert change.get_label_value("Code-Review") == 2
        assert change.get_label_value("Verified") == 1
        assert change.get_label_value("Unknown") is None

    def test_is_label_approved(self):
        """Test is_label_approved method."""
        change = GerritChangeInfo(
            number=1,
            change_id="I123",
            project="proj",
            subject="Test",
            owner="user",
            branch="main",
            status="NEW",
            labels=[
                GerritLabelInfo(name="Code-Review", approved=True),
                GerritLabelInfo(name="Verified", approved=False),
            ],
        )

        assert change.is_label_approved("Code-Review") is True
        assert change.is_label_approved("Verified") is False
        assert change.is_label_approved("Unknown") is False

    def test_from_api_response_basic(self):
        """Test from_api_response with basic data."""
        api_data = {
            "_number": 74080,
            "change_id": "I1234567890abcdef1234567890abcdef12345678",
            "project": "releng/project",
            "subject": "Chore: Bump actions/checkout from 4.1.0 to 4.2.0",
            "branch": "main",
            "status": "NEW",
            "owner": {"username": "dependabot", "email": "bot@example.com"},
            "created": "2024-01-15 10:00:00.000000000",
            "updated": "2024-01-15 12:00:00.000000000",
        }

        change = GerritChangeInfo.from_api_response(api_data)

        assert change.number == 74080
        assert change.change_id == "I1234567890abcdef1234567890abcdef12345678"
        assert change.project == "releng/project"
        assert change.subject == "Chore: Bump actions/checkout from 4.1.0 to 4.2.0"
        assert change.owner == "dependabot"
        assert change.owner_email == "bot@example.com"
        assert change.branch == "main"
        assert change.status == "NEW"

    def test_from_api_response_with_url(self):
        """Test from_api_response URL construction."""
        api_data = {
            "_number": 12345,
            "change_id": "I123",
            "project": "my-project",
            "subject": "Test",
            "branch": "main",
            "status": "NEW",
            "owner": {"username": "user"},
        }

        change = GerritChangeInfo.from_api_response(api_data, host="gerrit.example.org")

        assert change.url == "https://gerrit.example.org/c/my-project/+/12345"

    def test_from_api_response_with_base_path(self):
        """Test from_api_response URL construction with base path."""
        api_data = {
            "_number": 12345,
            "change_id": "I123",
            "project": "my-project",
            "subject": "Test",
            "branch": "main",
            "status": "NEW",
            "owner": {"username": "user"},
        }

        change = GerritChangeInfo.from_api_response(
            api_data, host="gerrit.example.org", base_path="infra"
        )

        assert change.url == "https://gerrit.example.org/infra/c/my-project/+/12345"

    def test_from_api_response_with_files(self):
        """Test from_api_response with file changes."""
        api_data = {
            "_number": 1,
            "change_id": "I123",
            "project": "proj",
            "subject": "Test",
            "branch": "main",
            "status": "NEW",
            "owner": {"username": "user"},
            "current_revision": "abc123",
            "revisions": {
                "abc123": {
                    "files": {
                        "/COMMIT_MSG": {"lines_inserted": 5},
                        "src/main.py": {
                            "status": "M",
                            "lines_inserted": 10,
                            "lines_deleted": 5,
                        },
                        "src/test.py": {
                            "status": "A",
                            "lines_inserted": 50,
                        },
                    }
                }
            },
        }

        change = GerritChangeInfo.from_api_response(api_data)

        # Should skip /COMMIT_MSG
        assert len(change.files_changed) == 2
        filenames = [f.filename for f in change.files_changed]
        assert "src/main.py" in filenames
        assert "src/test.py" in filenames
        assert "/COMMIT_MSG" not in filenames

    def test_from_api_response_with_labels(self):
        """Test from_api_response with label info."""
        api_data = {
            "_number": 1,
            "change_id": "I123",
            "project": "proj",
            "subject": "Test",
            "branch": "main",
            "status": "NEW",
            "owner": {"username": "user"},
            "labels": {
                "Code-Review": {"approved": {"_account_id": 1}},
                "Verified": {"value": 1},
            },
        }

        change = GerritChangeInfo.from_api_response(api_data)

        assert len(change.labels) == 2
        label_names = [label.name for label in change.labels]
        assert "Code-Review" in label_names
        assert "Verified" in label_names

    def test_from_api_response_with_commit_message(self):
        """Test from_api_response extracts commit message."""
        api_data = {
            "_number": 1,
            "change_id": "I123",
            "project": "proj",
            "subject": "Test subject",
            "branch": "main",
            "status": "NEW",
            "owner": {"username": "user"},
            "current_revision": "abc123",
            "revisions": {
                "abc123": {
                    "commit": {
                        "message": "Test subject\n\nFull commit message body.",
                    }
                }
            },
        }

        change = GerritChangeInfo.from_api_response(api_data)

        assert change.message == "Test subject\n\nFull commit message body."

    def test_from_api_response_owner_fallback_to_name(self):
        """Test owner extraction falls back to name field."""
        api_data = {
            "_number": 1,
            "change_id": "I123",
            "project": "proj",
            "subject": "Test",
            "branch": "main",
            "status": "NEW",
            "owner": {"name": "John Doe"},  # No username
        }

        change = GerritChangeInfo.from_api_response(api_data)

        assert change.owner == "John Doe"


class TestGerritComparisonResult:
    """Tests for GerritComparisonResult model."""

    def test_basic_construction(self):
        """Test basic model construction."""
        result = GerritComparisonResult(
            is_similar=True,
            confidence_score=0.95,
            reasons=["Same author", "Similar subject"],
        )

        assert result.is_similar is True
        assert result.confidence_score == 0.95
        assert len(result.reasons) == 2

    def test_not_similar_factory(self):
        """Test not_similar factory method."""
        result = GerritComparisonResult.not_similar("Different packages")

        assert result.is_similar is False
        assert result.confidence_score == 0.0
        assert "Different packages" in result.reasons

    def test_not_similar_factory_no_reason(self):
        """Test not_similar factory with no reason."""
        result = GerritComparisonResult.not_similar()

        assert result.is_similar is False
        assert result.confidence_score == 0.0
        assert result.reasons == []

    def test_similar_factory(self):
        """Test similar factory method."""
        result = GerritComparisonResult.similar(0.85, ["Same author", "Similar files"])

        assert result.is_similar is True
        assert result.confidence_score == 0.85
        assert len(result.reasons) == 2

    def test_similar_factory_no_reasons(self):
        """Test similar factory with no reasons."""
        result = GerritComparisonResult.similar(0.9)

        assert result.is_similar is True
        assert result.confidence_score == 0.9
        assert result.reasons == []


class TestGerritSubmitResult:
    """Tests for GerritSubmitResult model."""

    def test_basic_construction(self):
        """Test basic model construction."""
        result = GerritSubmitResult(
            change_number=12345,
            project="my-project",
            success=True,
            reviewed=True,
            submitted=True,
        )

        assert result.change_number == 12345
        assert result.project == "my-project"
        assert result.success is True
        assert result.reviewed is True
        assert result.submitted is True

    def test_success_result_factory(self):
        """Test success_result factory method."""
        result = GerritSubmitResult.success_result(
            change_number=12345,
            project="my-project",
            reviewed=True,
            submitted=True,
            duration=1.5,
        )

        assert result.success is True
        assert result.reviewed is True
        assert result.submitted is True
        assert result.error is None
        assert result.duration_seconds == 1.5

    def test_failure_result_factory(self):
        """Test failure_result factory method."""
        result = GerritSubmitResult.failure_result(
            change_number=12345,
            project="my-project",
            error="Merge conflict",
            reviewed=True,
            duration=0.5,
        )

        assert result.success is False
        assert result.reviewed is True
        assert result.submitted is False
        assert result.error == "Merge conflict"
        assert result.duration_seconds == 0.5

    def test_default_values(self):
        """Test default values for optional fields."""
        result = GerritSubmitResult(
            change_number=1,
            project="proj",
            success=True,
        )

        assert result.reviewed is False
        assert result.submitted is False
        assert result.error is None
        assert result.duration_seconds == 0.0
