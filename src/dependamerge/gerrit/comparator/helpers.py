# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Normalization and automation-pattern helpers for change comparison.

These are pure text operations over owners, subjects, commit messages
and filenames: they strip versions, hashes and dates so that changes
differing only in those details compare equal, and they recognize the
message shapes produced by Dependabot and pre-commit.

They are mixed into :class:`GerritChangeComparator`, which supplies
the comparison entry points that call them.
"""

from __future__ import annotations

import re


class GerritComparatorHelpers:
    """
    Text normalization and automation-pattern detection helpers.

    This is a mixin: it holds the helper methods of
    :class:`GerritChangeComparator` and is not instantiated directly.
    """

    def _normalize_owner(self, owner: str) -> str:
        """
        Normalize owner name for comparison.

        Handles variations like 'dependabot[bot]' vs 'dependabot'.
        """
        if not owner:
            return ""

        normalized = owner.lower().strip()

        # Remove [bot] suffix
        if normalized.endswith("[bot]"):
            normalized = normalized[:-5]

        for suffix in ("-bot", "_bot", ".bot"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break

        return normalized

    def _normalize_owner_identity(self, owner: str) -> str:
        """
        Normalize owner identity for hard same-owner gates.

        Unlike similarity scoring, this preserves bot suffixes so distinct
        Gerrit accounts like "alice" and "alice-bot" do not collapse together.
        """
        return owner.lower().strip() if owner else ""

    def _normalize_subject(self, subject: str) -> str:
        """
        Normalize subject by removing version-specific information.
        """
        subject = re.sub(r"v?\d+\.\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.-]+)?", "", subject)
        subject = re.sub(r"\b[a-f0-9]{7,40}\b", "", subject)
        subject = re.sub(r"\d{4}-\d{2}-\d{2}", "", subject)
        subject = " ".join(subject.split())

        return subject.lower()

    def _extract_package_name(self, subject: str) -> str:
        """
        Extract package name from dependency update subjects.

        Handles common patterns like:
        - "Bump package from X to Y"
        - "Chore: Bump package from X to Y"
        - "Update package from X to Y"
        """
        subject_lower = subject.lower()

        patterns = [
            r"(?:chore:\s*)?bump\s+([^\s]+)\s+from\s+",
            r"(?:chore:\s*)?update\s+([^\s]+)\s+from\s+",
            r"(?:chore:\s*)?upgrade\s+([^\s]+)\s+from\s+",
            r"(?:build\(deps\):\s*)?bump\s+([^\s]+)\s+from\s+",
            r"(?:build\(deps-dev\):\s*)?bump\s+([^\s]+)\s+from\s+",
        ]

        for pattern in patterns:
            match = re.search(pattern, subject_lower)
            if match:
                package = match.group(1).strip()
                package = re.sub(r'^["\']|["\']$', "", package)
                return package

        return ""

    def _normalize_message(self, message: str) -> str:
        """
        Normalize commit message for comparison.
        """
        # Convert to lowercase
        message = message.lower()

        message = re.sub(r"https?://[^\s]+", "", message)
        message = re.sub(
            r"v?\d+\.\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.-]+)?", "VERSION", message
        )
        message = re.sub(r"\b[a-f0-9]{7,40}\b", "COMMIT", message)
        message = re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", message)
        message = re.sub(r"\s+", " ", message).strip()

        return message

    def _compare_automation_patterns(self, message1: str, message2: str) -> float:
        """
        Compare messages for specific automation tool patterns.
        """
        # Dependabot patterns
        if self._is_dependabot_message(message1) and self._is_dependabot_message(
            message2
        ):
            package1 = self._extract_dependabot_package(message1)
            package2 = self._extract_dependabot_package(message2)

            if package1 and package2 and package1 == package2:
                return 0.95  # Same package
            if package1 and package2:
                return 0.1  # Different packages

        # Pre-commit patterns
        if self._is_precommit_message(message1) and self._is_precommit_message(
            message2
        ):
            return 0.9

        return 0.0

    def _is_dependabot_message(self, message: str) -> bool:
        """Check if message has Dependabot-specific patterns."""
        indicators = [
            "dependabot",
            "bumps",
            "from .* to",
            "release notes",
            "changelog",
            "dependency-name:",
        ]

        message_lower = message.lower()
        matches = sum(1 for ind in indicators if ind in message_lower)
        return matches >= 2

    def _extract_dependabot_package(self, message: str) -> str:
        """Extract package name from Dependabot commit message."""
        # Look for "dependency-name: package" pattern
        yaml_match = re.search(r"dependency-name:\s*([^\s\n]+)", message, re.IGNORECASE)
        if yaml_match:
            return yaml_match.group(1).strip()

        # Look for "Bumps [package]" pattern
        bump_match = re.search(r"bumps\s+\[([^\]]+)\]", message, re.IGNORECASE)
        if bump_match:
            return bump_match.group(1).strip()

        return ""

    def _is_precommit_message(self, message: str) -> bool:
        """Check if message has pre-commit specific patterns."""
        indicators = [
            "pre-commit",
            "autoupdate",
            "hooks",
            ".pre-commit-config.yaml",
        ]

        message_lower = message.lower()
        return any(ind in message_lower for ind in indicators)

    def _normalize_filename(self, filename: str) -> str:
        """
        Normalize filename for comparison.
        """
        filename = re.sub(r"v?\d+\.\d+\.\d+(?:\.\d+)?", "", filename)
        return filename.lower()
