# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Result types describing GitHub2Gerrit mappings found on pull requests.

The detection and comment-building helpers that produce and consume these
live alongside them in :mod:`dependamerge.github2gerrit_detector`; this
module holds only the mode enum and result types so that every consumer
can share them without importing one another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GitHub2GerritMode(str, Enum):
    """GitHub2Gerrit submission mode."""

    SQUASH = "squash"
    MULTI_COMMIT = "multi-commit"


@dataclass(frozen=True)
class GitHub2GerritMapping:
    """
    Parsed GitHub2Gerrit mapping extracted from a PR comment.

    Attributes:
        pr_url: The GitHub PR URL recorded in the mapping comment.
        mode: The submission mode (squash or multi-commit).
        topic: The Gerrit topic name (e.g., ``GH-repo-41``).
        change_ids: Ordered list of Gerrit Change-IDs (I-prefixed SHA-1).
        github_hash: The GitHub-Hash trailer value used for verification.
        raw_comment_body: The full comment body the mapping was extracted from.
    """

    pr_url: str
    mode: str
    topic: str
    change_ids: tuple[str, ...]
    github_hash: str = ""
    raw_comment_body: str = ""

    @property
    def primary_change_id(self) -> str:
        """Return the first (primary) Change-ID."""
        return self.change_ids[0] if self.change_ids else ""

    @property
    def is_valid(self) -> bool:
        """Check whether the mapping has the minimum required fields."""
        return bool(self.topic and self.change_ids and self.mode)


@dataclass
class GitHub2GerritDetectionResult:
    """
    Result of scanning a pull request for GitHub2Gerrit mapping comments.

    Attributes:
        has_mapping: True if at least one valid mapping comment was found.
        mapping: The latest valid mapping (if any).
        comment_indices: Indices into the comment list that contained mappings.
        detection_source: How the mapping was detected ("marker" or "heuristic").
    """

    has_mapping: bool = False
    mapping: GitHub2GerritMapping | None = None
    comment_indices: list[int] = field(default_factory=list)
    detection_source: str = ""
