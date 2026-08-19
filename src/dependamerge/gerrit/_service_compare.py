# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The fallback change comparison behind ``GerritService``.

:class:`_GerritCompareMixin` carries the scoring used when
``find_similar_changes`` is given no external comparator: the automation
check, the owner normalisation rules, the subject and file-set scores,
and the weighted verdict that combines them.

It lives here rather than in ``dependamerge.gerrit.service`` purely to
keep that module reviewable.  Nothing in here references
``create_url_builder`` or ``build_client``: those names are only
resolved in ``service``'s own namespace, so that patching them there
stays effective.  Every attribute this mixin reads is established by
``GerritService.__init__``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from dependamerge.gerrit.models import (
    GerritChangeInfo,
    GerritComparisonResult,
)


class _GerritCompareMixin:
    """Fallback comparison behaviour shared into ``GerritService``."""

    # Established by GerritService.__init__.
    _similarity_threshold: float

    def _basic_compare(
        self,
        source: GerritChangeInfo,
        target: GerritChangeInfo,
        only_automation: bool,
    ) -> GerritComparisonResult:
        """
        Perform basic comparison between two changes.

        This is a fallback when no external comparator is provided.
        Uses the similarity_threshold configured at initialization
        (default: 0.8) to determine if changes are similar.
        """
        reasons: list[str] = []
        scores: list[float] = []

        # Check automation if required
        if only_automation:
            if not self._is_automation_change(source) or not self._is_automation_change(
                target
            ):
                return GerritComparisonResult.not_similar(
                    "One or both changes are not from automation"
                )
        elif self._owners_differ(source, target):
            return GerritComparisonResult.not_similar(
                "Change owner does not match source owner"
            )

        # Compare owners
        if self._normalize_owner(source.owner) == self._normalize_owner(target.owner):
            scores.append(1.0)
            reasons.append("Same author")
        else:
            scores.append(0.0)

        # Compare subjects (titles)
        subject_score = self._compare_subjects(source.subject, target.subject)
        scores.append(subject_score)
        if subject_score > 0.7:
            reasons.append(f"Similar subjects (score: {subject_score:.2f})")

        # Compare files
        files_score = self._compare_files(source, target)
        scores.append(files_score)
        if files_score > 0.5:
            reasons.append(f"Similar files (score: {files_score:.2f})")

        # Calculate overall score
        confidence = sum(scores) / len(scores) if scores else 0.0
        is_similar = confidence >= self._similarity_threshold

        if is_similar:
            return GerritComparisonResult.similar(confidence, reasons)
        return GerritComparisonResult.not_similar()

    def _is_automation_change(self, change: GerritChangeInfo) -> bool:
        """Check if a change is from automation."""
        automation_indicators = [
            "dependabot",
            "pre-commit",
            "renovate",
            "github-actions",
            "auto-update",
            "automated",
            "bot",
        ]

        text = f"{change.subject} {change.message or ''} {change.owner}".lower()
        return any(indicator in text for indicator in automation_indicators)

    def _owners_differ(
        self,
        source: GerritChangeInfo,
        target: GerritChangeInfo,
    ) -> bool:
        """Return True when Gerrit change owner identities differ."""
        return self._normalize_owner_identity(source.owner) != (
            self._normalize_owner_identity(target.owner)
        )

    def _normalize_owner_identity(self, owner: str) -> str:
        """
        Normalize owner identity for hard same-owner gates.

        This preserves bot suffixes so distinct Gerrit accounts like "alice"
        and "alice-bot" do not collapse together.
        """
        return owner.lower().strip() if owner else ""

    def _normalize_owner(self, owner: str) -> str:
        """Normalize owner name using the Gerrit comparator rules."""
        if not owner:
            return ""

        normalized = owner.lower().strip()

        if normalized.endswith("[bot]"):
            normalized = normalized[:-5]

        for suffix in ("-bot", "_bot", ".bot"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break

        return normalized

    def _compare_subjects(self, subject1: str, subject2: str) -> float:
        """Compare two change subjects for similarity."""
        # Normalize subjects
        s1 = subject1.lower().strip()
        s2 = subject2.lower().strip()

        if s1 == s2:
            return 1.0

        patterns = [
            "bump",
            "update",
            "upgrade",
            "chore:",
            "build(deps):",
        ]

        s1_pattern = None
        s2_pattern = None

        for pattern in patterns:
            if pattern in s1:
                s1_pattern = pattern
            if pattern in s2:
                s2_pattern = pattern

        if s1_pattern and s2_pattern and s1_pattern == s2_pattern:
            return 0.8

        return 0.3

    def _compare_files(
        self,
        source: GerritChangeInfo,
        target: GerritChangeInfo,
    ) -> float:
        """Compare file changes between two changes."""
        if not source.files_changed or not target.files_changed:
            return 0.0

        source_files = {f.filename for f in source.files_changed}
        target_files = {f.filename for f in target.files_changed}

        intersection = len(source_files & target_files)
        union = len(source_files | target_files)

        if union == 0:
            return 0.0

        return intersection / union
