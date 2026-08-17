# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The GerritChangeInfo model: a whole Gerrit change.

This module holds the aggregate model built from a Gerrit REST API
change response, along with the status, label and permission
accessors used when deciding whether a change can be reviewed and
submitted.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .values import GerritChangeStatus, GerritFileChange, GerritLabelInfo


class GerritChangeInfo(BaseModel):
    """
    Represents a Gerrit change (parallels PullRequestInfo for GitHub).

    This model contains all the information needed to compare changes
    and perform operations like review and submit.
    """

    # Core identifiers
    number: int = Field(..., description="Gerrit change number")
    change_id: str = Field(..., description="Gerrit Change-Id (I-prefixed)")
    project: str = Field(..., description="Gerrit project name")

    # Content
    subject: str = Field(..., description="First line of commit message")
    message: str | None = Field(default=None, description="Full commit message")
    topic: str | None = Field(default=None, description="Change topic (if set)")

    # Author/owner
    owner: str = Field(..., description="Change owner username")
    owner_email: str | None = Field(default=None, description="Change owner email")

    # Branch info
    branch: str = Field(..., description="Target branch")
    current_revision: str = Field(default="", description="Current revision SHA")

    # Status
    status: str = Field(..., description="Change status (NEW, MERGED, ABANDONED)")
    submittable: bool = Field(
        default=False, description="Whether change can be submitted"
    )
    mergeable: bool | None = Field(
        default=None, description="Whether change is mergeable"
    )
    work_in_progress: bool = Field(default=False, description="Whether change is WIP")

    # Files
    files_changed: list[GerritFileChange] = Field(
        default_factory=list, description="List of changed files"
    )

    # Labels
    labels: list[GerritLabelInfo] = Field(
        default_factory=list, description="Label/vote information"
    )

    # URLs
    url: str = Field(default="", description="Web URL for the change")

    # Timestamps
    created: str = Field(default="", description="Creation timestamp")
    updated: str = Field(default="", description="Last update timestamp")

    # Submit requirements (Gerrit 3.x+)
    submit_requirements_met: bool = Field(
        default=True, description="Whether all submit requirements are satisfied"
    )

    # Permissions - for checking what the current user can do
    permitted_labels: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map of label names to permitted voting values for current user",
    )
    actions: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Union of change-level actions (CHANGE_ACTIONS) and "
            "current-revision actions (CURRENT_ACTIONS) the caller can "
            "perform. Revision-level actions include 'submit'."
        ),
    )

    @classmethod
    def from_api_response(
        cls,
        data: dict[str, Any],
        host: str = "",
        base_path: str | None = None,
    ) -> GerritChangeInfo:
        """
        Create a GerritChangeInfo from Gerrit REST API response.

        Args:
            data: The change info dict from Gerrit API.
            host: Optional host for constructing URLs.
            base_path: Optional base path for the Gerrit server.

        Returns:
            A GerritChangeInfo instance.
        """
        number = data.get("_number", 0)
        change_id = data.get("change_id", "")
        project = data.get("project", "")
        subject = data.get("subject", "")
        branch = data.get("branch", "")
        status = data.get("status", "NEW")
        topic = data.get("topic")

        owner_data = data.get("owner", {})
        owner = owner_data.get("username") or owner_data.get("name", "unknown")
        owner_email = owner_data.get("email")

        current_revision = data.get("current_revision", "")

        message = None
        if current_revision and "revisions" in data:
            revision_data = data["revisions"].get(current_revision, {})
            commit_data = revision_data.get("commit", {})
            message = commit_data.get("message")

        files_changed: list[GerritFileChange] = []
        if current_revision and "revisions" in data:
            revision_data = data["revisions"].get(current_revision, {})
            files_data = revision_data.get("files", {})
            for filename, file_info in files_data.items():
                # Skip the special /COMMIT_MSG file
                if filename == "/COMMIT_MSG":
                    continue
                files_changed.append(
                    GerritFileChange.from_api_response(filename, file_info)
                )

        labels: list[GerritLabelInfo] = []
        labels_data = data.get("labels", {})
        for label_name, label_info in labels_data.items():
            labels.append(GerritLabelInfo.from_api_response(label_name, label_info))

        # Submittable and mergeable
        submittable = data.get("submittable", False)
        mergeable = data.get("mergeable")
        work_in_progress = data.get("work_in_progress", False)

        # Extract permitted labels (what the current user can vote on)
        permitted_labels: dict[str, list[str]] = data.get("permitted_labels", {})

        # Gerrit splits caller actions across two locations: change-level
        # actions (populated by the CHANGE_ACTIONS query option) and
        # revision-level actions such as 'submit', 'rebase', and
        # 'cherrypick' (populated by CURRENT_ACTIONS under
        # revisions[<rev>].actions). Merge both so permission checks see
        # the complete picture regardless of which options were requested.
        actions: dict[str, dict[str, Any]] = dict(data.get("actions") or {})
        if current_revision and "revisions" in data:
            revision_data = data["revisions"].get(current_revision, {})
            actions.update(revision_data.get("actions") or {})

        # Check submit requirements (Gerrit 3.x+)
        submit_requirements_met = True
        submit_records = data.get("submit_records", [])
        for record in submit_records:
            if record.get("status") != "OK":
                submit_requirements_met = False
                break

        # Construct URL via the centralised builder to ensure base_path
        # is handled consistently (see GerritUrlBuilder).
        url = ""
        if host:
            from dependamerge.gerrit.urls import GerritUrlBuilder

            builder = GerritUrlBuilder(
                host=host, base_path=base_path, auto_discover=False
            )
            url = builder.change_url(project, number)

        # Timestamps
        created = data.get("created", "")
        updated = data.get("updated", "")

        return cls(
            number=number,
            change_id=change_id,
            project=project,
            subject=subject,
            message=message,
            topic=topic,
            owner=owner,
            owner_email=owner_email,
            branch=branch,
            current_revision=current_revision,
            status=status,
            submittable=submittable,
            mergeable=mergeable,
            work_in_progress=work_in_progress,
            files_changed=files_changed,
            labels=labels,
            url=url,
            created=created,
            updated=updated,
            submit_requirements_met=submit_requirements_met,
            permitted_labels=permitted_labels,
            actions=actions,
        )

    @property
    def is_open(self) -> bool:
        """Check if the change is open (NEW status)."""
        return self.status == GerritChangeStatus.NEW.value

    @property
    def is_merged(self) -> bool:
        """Check if the change has been merged."""
        return self.status == GerritChangeStatus.MERGED.value

    @property
    def is_abandoned(self) -> bool:
        """Check if the change has been abandoned."""
        return self.status == GerritChangeStatus.ABANDONED.value

    @property
    def can_submit(self) -> bool:
        """Check if the change can be submitted."""
        return (
            self.is_open
            and self.submittable
            and self.submit_requirements_met
            and not self.work_in_progress
        )

    @property
    def file_count(self) -> int:
        """Get the number of files changed."""
        return len(self.files_changed)

    @property
    def total_lines_changed(self) -> int:
        """Get the total number of lines changed (inserted + deleted)."""
        return sum(f.lines_inserted + f.lines_deleted for f in self.files_changed)

    def get_label_value(self, label_name: str) -> int | None:
        """
        Get the current vote value for a label.

        Args:
            label_name: The label name (e.g., "Code-Review").

        Returns:
            The vote value, or None if not found.
        """
        for label in self.labels:
            if label.name == label_name:
                return label.value
        return None

    def is_label_approved(self, label_name: str) -> bool:
        """
        Check if a label has been approved.

        Args:
            label_name: The label name to check.

        Returns:
            True if the label is approved.
        """
        for label in self.labels:
            if label.name == label_name:
                return label.approved
        return False

    def can_vote_label(self, label_name: str, value: int) -> bool:
        """
        Check if the current user can vote a specific value on a label.

        Args:
            label_name: The label name (e.g., "Code-Review").
            value: The vote value to check (e.g., 2 for +2).

        Returns:
            True if the user can vote this value on the label.
        """
        if label_name not in self.permitted_labels:
            return False
        permitted_values = self.permitted_labels[label_name]
        # Values are strings like "-2", "-1", "0", "+1", "+2"
        value_str = f"+{value}" if value > 0 else str(value)
        return value_str in permitted_values

    def can_code_review_plus_two(self) -> bool:
        """
        Check if the current user can give +2 Code-Review.

        Returns:
            True if the user can vote +2 on Code-Review.
        """
        return self.can_vote_label("Code-Review", 2)

    def can_submit_action(self) -> bool:
        """
        Check if the current user has the submit action available.

        The submit action is a revision-level action that Gerrit returns
        under revisions[<rev>].actions when CURRENT_ACTIONS is requested
        (merged into ``actions`` by ``from_api_response``). Gerrit only
        exposes the submit action once a change is submittable, and only
        to callers with submit permission.

        Returns:
            True if the submit action is available.
        """
        return "submit" in self.actions

    def get_permission_warnings(self) -> list[str]:
        """
        Get a list of permission warnings for operations on this change.

        Returns:
            List of warning messages about missing permissions.
        """
        warnings = []

        if not self.can_code_review_plus_two():
            warnings.append(
                "You may not have permission to give +2 Code-Review on this change"
            )

        if self.submittable and not self.can_submit_action():
            warnings.append("You may not have permission to submit this change")

        return warnings

    def has_required_permissions(self) -> bool:
        """
        Check if the user has all required permissions for merge operations.

        This checks +2 Code-Review permissions. Gerrit only exposes the
        submit action once a change is already submittable, so submit
        permission is required only for submittable changes.

        Returns:
            True if the user has all required permissions.
        """
        if not self.can_code_review_plus_two():
            return False

        return not self.submittable or self.can_submit_action()
