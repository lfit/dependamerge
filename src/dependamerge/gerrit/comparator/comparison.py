# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Comparison entry points for Gerrit changes.

This module holds the comparator itself: automation detection and the
owner, subject, message and file comparisons that combine into an
overall similarity score, plus the factory used to construct it.

The text normalization these comparisons rely on lives alongside them
in :mod:`dependamerge.gerrit.comparator.helpers`.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from dependamerge.gerrit.models import (
    GerritChangeInfo,
    GerritComparisonResult,
)

from .helpers import GerritComparatorHelpers

# Automation tool indicators
AUTOMATION_INDICATORS: tuple[str, ...] = (
    "dependabot",
    "pre-commit",
    "renovate",
    "github-actions",
    "auto-update",
    "automated",
    "bot",
    "pre-commit-ci",
    "greenkeeper",
    "snyk",
)


class GerritChangeComparator(GerritComparatorHelpers):
    """
    Compare Gerrit changes to determine similarity.

    This class provides methods to compare two Gerrit changes based on:
    - Owner (author) matching
    - Subject (title) similarity
    - Commit message similarity
    - File change patterns
    - Automation detection
    """

    def __init__(self, similarity_threshold: float = 0.8) -> None:
        """
        Initialize the comparator.

        Args:
            similarity_threshold: Minimum confidence score for changes
                                 to be considered similar (0.0 to 1.0).
        """
        self.similarity_threshold = similarity_threshold

    def compare_gerrit_changes(
        self,
        source_change: GerritChangeInfo,
        target_change: GerritChangeInfo,
        only_automation: bool = True,
    ) -> GerritComparisonResult:
        """
        Compare two Gerrit changes for similarity.

        Args:
            source_change: The reference change to compare against.
            target_change: The change to evaluate for similarity.
            only_automation: If True, only match automation changes.

        Returns:
            A GerritComparisonResult indicating similarity.
        """
        reasons: list[str] = []
        scores: list[float] = []

        if only_automation:
            source_is_auto = self.is_automation_change(source_change)
            target_is_auto = self.is_automation_change(target_change)

            if not source_is_auto or not target_is_auto:
                return GerritComparisonResult.not_similar(
                    "One or both changes are not from automation tools"
                )
        elif self._normalize_owner_identity(
            source_change.owner
        ) != self._normalize_owner_identity(target_change.owner):
            return GerritComparisonResult.not_similar(
                "Change owner does not match source owner"
            )

        # Compare owners (authors)
        owner_score = self._compare_owners(source_change, target_change)
        scores.append(owner_score)
        if owner_score == 1.0:
            reasons.append("Same automation author")

        # Compare subjects (commit titles)
        subject_score = self._compare_subjects(
            source_change.subject, target_change.subject
        )
        scores.append(subject_score)
        if subject_score > 0.7:
            reasons.append(f"Similar subjects (score: {subject_score:.2f})")

        # Compare commit messages
        message_score = self._compare_messages(
            source_change.message, target_change.message
        )
        scores.append(message_score)
        if message_score > 0.6:
            reasons.append(f"Similar commit messages (score: {message_score:.2f})")

        # Compare file changes
        files_score = self._compare_files(source_change, target_change)
        scores.append(files_score)
        if files_score > 0.5:
            reasons.append(f"Similar file changes (score: {files_score:.2f})")

        # Calculate overall confidence score
        confidence_score = sum(scores) / len(scores) if scores else 0.0
        is_similar = confidence_score >= self.similarity_threshold

        if is_similar:
            return GerritComparisonResult(
                is_similar=True,
                confidence_score=confidence_score,
                reasons=reasons,
            )

        return GerritComparisonResult.not_similar()

    def is_automation_change(self, change: GerritChangeInfo) -> bool:
        """
        Check if a change is from an automation tool.

        Args:
            change: The change to check.

        Returns:
            True if the change appears to be from automation.
        """
        # Combine relevant fields for checking
        text = f"{change.subject} {change.message or ''} {change.owner}".lower()

        for indicator in AUTOMATION_INDICATORS:
            if indicator in text:
                return True

        automation_patterns = [
            r"^chore\(deps\):",
            r"^build\(deps\):",
            r"^chore: bump",
            r"^chore: update",
            r"\[bot\]$",
        ]

        subject_lower = change.subject.lower()
        for pattern in automation_patterns:
            if re.search(pattern, subject_lower, re.IGNORECASE):
                return True

        return False

    def _is_automation_change(self, change: GerritChangeInfo) -> bool:
        """Backward-compatible alias for automation detection."""
        return self.is_automation_change(change)

    def _compare_owners(
        self,
        source: GerritChangeInfo,
        target: GerritChangeInfo,
    ) -> float:
        """
        Compare change owners for matching.

        Returns 1.0 if owners match (normalized), 0.0 otherwise.
        """
        source_owner = self._normalize_owner(source.owner)
        target_owner = self._normalize_owner(target.owner)

        if source_owner == target_owner:
            return 1.0
        return 0.0

    def _compare_subjects(self, subject1: str, subject2: str) -> float:
        """
        Compare change subjects for similarity.

        For dependency updates, extracts and compares package names.
        """
        package1 = self._extract_package_name(subject1)
        package2 = self._extract_package_name(subject2)

        # If both are dependency updates for the same package
        if package1 and package2:
            if package1 == package2:
                return 1.0  # Same package update
            return 0.0  # Different packages

        # Fall back to text similarity
        norm1 = self._normalize_subject(subject1)
        norm2 = self._normalize_subject(subject2)

        return SequenceMatcher(None, norm1, norm2).ratio()

    def _compare_messages(self, message1: str | None, message2: str | None) -> float:
        """
        Compare commit messages for similarity.
        """
        if not message1 or not message2:
            return 0.0

        # Normalize messages
        norm1 = self._normalize_message(message1)
        norm2 = self._normalize_message(message2)

        # For very short messages, use exact matching
        if len(norm1) < 50 or len(norm2) < 50:
            return 1.0 if norm1 == norm2 else 0.0

        pattern_score = self._compare_automation_patterns(message1, message2)
        if pattern_score > 0:
            return pattern_score

        # Fall back to sequence matching
        return SequenceMatcher(None, norm1, norm2).ratio()

    def _compare_files(
        self,
        source: GerritChangeInfo,
        target: GerritChangeInfo,
    ) -> float:
        """
        Compare file changes between two changes.
        """
        if not source.files_changed or not target.files_changed:
            return 0.0

        source_files = {
            self._normalize_filename(f.filename) for f in source.files_changed
        }
        target_files = {
            self._normalize_filename(f.filename) for f in target.files_changed
        }

        # Calculate Jaccard similarity
        intersection = len(source_files & target_files)
        union = len(source_files | target_files)

        if union == 0:
            return 0.0

        base_score = intersection / union

        # Boost score for workflow files
        source_workflows = {f for f in source_files if ".github/workflows/" in f}
        target_workflows = {f for f in target_files if ".github/workflows/" in f}

        if source_workflows and target_workflows:
            # Both modify workflow files - consider partial match
            return max(base_score, 0.5)

        return base_score


def create_gerrit_comparator(
    similarity_threshold: float = 0.8,
) -> GerritChangeComparator:
    """
    Factory function to create a GerritChangeComparator.

    Args:
        similarity_threshold: Minimum confidence for similarity matching.

    Returns:
        Configured GerritChangeComparator instance.
    """
    return GerritChangeComparator(similarity_threshold=similarity_threshold)
