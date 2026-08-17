# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Value types for Gerrit changes: statuses, files, labels and results.

These are the leaf models that describe the parts of a Gerrit change
(status enums, per-file changes, label votes) together with the small
records reporting the outcome of comparison and submit operations.
They carry no dependency on :class:`GerritChangeInfo`, which composes
them.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GerritChangeStatus(str, Enum):
    """Gerrit change status values."""

    NEW = "NEW"
    MERGED = "MERGED"
    ABANDONED = "ABANDONED"


class GerritFileStatus(str, Enum):
    """Status of a file in a Gerrit change."""

    ADDED = "A"
    MODIFIED = "M"
    DELETED = "D"
    RENAMED = "R"
    COPIED = "C"
    REWRITE = "W"


class GerritFileChange(BaseModel):
    """
    Represents a file change in a Gerrit change.

    This parallels the FileChange model used for GitHub PRs.
    """

    filename: str
    status: str = "M"  # Default to modified
    lines_inserted: int = 0
    lines_deleted: int = 0
    size_delta: int = 0
    old_path: str | None = None  # For renames/copies

    @classmethod
    def from_api_response(
        cls, filename: str, file_data: dict[str, Any]
    ) -> GerritFileChange:
        """
        Create a GerritFileChange from Gerrit API file info.

        Args:
            filename: The file path.
            file_data: The file info dict from Gerrit API.

        Returns:
            A GerritFileChange instance.
        """
        return cls(
            filename=filename,
            status=file_data.get("status", "M"),
            lines_inserted=file_data.get("lines_inserted", 0),
            lines_deleted=file_data.get("lines_deleted", 0),
            size_delta=file_data.get("size_delta", 0),
            old_path=file_data.get("old_path"),
        )


class GerritLabelInfo(BaseModel):
    """
    Represents label (vote) information for a Gerrit change.

    Labels like Code-Review, Verified, etc.
    """

    name: str
    approved: bool = False
    rejected: bool = False
    value: int | None = None
    blocking: bool = False

    @classmethod
    def from_api_response(
        cls, name: str, label_data: dict[str, Any]
    ) -> GerritLabelInfo:
        """
        Create a GerritLabelInfo from Gerrit API label info.

        Args:
            name: The label name (e.g., "Code-Review").
            label_data: The label info dict from Gerrit API.

        Returns:
            A GerritLabelInfo instance.
        """
        # Gerrit uses "approved" and "rejected" sub-objects
        approved = "approved" in label_data
        rejected = "rejected" in label_data

        # Get the current vote value if present
        value = None
        if "value" in label_data:
            value = label_data["value"]
        elif approved:
            # If approved, typically means max positive vote
            value = 2
        elif rejected:
            # If rejected, typically means max negative vote
            value = -2

        return cls(
            name=name,
            approved=approved,
            rejected=rejected,
            value=value,
            blocking=label_data.get("blocking", False),
        )


class GerritComparisonResult(BaseModel):
    """
    Result of comparing two Gerrit changes for similarity.

    This parallels the ComparisonResult model used for GitHub PRs.
    """

    is_similar: bool = Field(
        ..., description="Whether the changes are considered similar"
    )
    confidence_score: float = Field(
        ..., description="Similarity confidence score (0.0 to 1.0)"
    )
    reasons: list[str] = Field(
        default_factory=list, description="Reasons for the similarity assessment"
    )

    @classmethod
    def not_similar(cls, reason: str = "") -> GerritComparisonResult:
        """Create a result indicating changes are not similar."""
        reasons = [reason] if reason else []
        return cls(is_similar=False, confidence_score=0.0, reasons=reasons)

    @classmethod
    def similar(
        cls, score: float, reasons: list[str] | None = None
    ) -> GerritComparisonResult:
        """Create a result indicating changes are similar."""
        return cls(
            is_similar=True,
            confidence_score=score,
            reasons=reasons or [],
        )


class GerritSubmitResult(BaseModel):
    """
    Result of attempting to submit a Gerrit change.

    This is used by the submit manager to track operation outcomes.
    """

    change_number: int = Field(..., description="The change number")
    project: str = Field(..., description="The project name")
    success: bool = Field(..., description="Whether submission succeeded")
    reviewed: bool = Field(default=False, description="Whether review was applied")
    submitted: bool = Field(default=False, description="Whether change was submitted")
    error: str | None = Field(default=None, description="Error message if failed")
    duration_seconds: float = Field(default=0.0, description="Operation duration")

    @classmethod
    def success_result(
        cls,
        change_number: int,
        project: str,
        reviewed: bool = True,
        submitted: bool = True,
        duration: float = 0.0,
    ) -> GerritSubmitResult:
        """Create a successful submit result."""
        return cls(
            change_number=change_number,
            project=project,
            success=True,
            reviewed=reviewed,
            submitted=submitted,
            error=None,
            duration_seconds=duration,
        )

    @classmethod
    def failure_result(
        cls,
        change_number: int,
        project: str,
        error: str,
        reviewed: bool = False,
        duration: float = 0.0,
    ) -> GerritSubmitResult:
        """Create a failed submit result."""
        return cls(
            change_number=change_number,
            project=project,
            success=False,
            reviewed=reviewed,
            submitted=False,
            error=error,
            duration_seconds=duration,
        )
