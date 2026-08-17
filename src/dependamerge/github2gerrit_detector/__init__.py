# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
GitHub2Gerrit detection utilities for dependamerge.

This module detects GitHub pull requests that have corresponding Gerrit changes
created by the GitHub2Gerrit workflow. It parses the structured mapping comments
that GitHub2Gerrit posts on PRs to extract Change-IDs, topics, and other
metadata needed to locate and submit the corresponding Gerrit change.

``.gitreview`` parsing and fetching are delegated to :mod:`dependamerge.gitreview`.
This module re-exports :class:`~dependamerge.gitreview.GitReviewInfo`,
:func:`~dependamerge.gitreview.parse_gitreview_text`, and
:func:`~dependamerge.gitreview.fetch_gitreview_from_github` so that existing
callers continue to work without import changes.

The mapping comment format is defined by the github2gerrit-action project and
uses HTML markers for reliable parsing:

    <!-- github2gerrit:change-id-map v1 -->
    PR: https://github.com/owner/repo/pull/41
    Mode: squash
    Topic: GH-repo-41
    Change-Ids:
      I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048
    GitHub-Hash: 41b89b8d5055be4e
    ...
    <!-- end github2gerrit:change-id-map -->
"""

from __future__ import annotations

# Re-export .gitreview symbols so existing callers don't need to change
# their imports.  The canonical implementation lives in gitreview.py.
# NOTE: fetch_gitreview_from_github is re-exported here from
# dependamerge.gitreview — no inline implementation needed.
from ..gitreview import (
    GitReviewInfo,
    fetch_gitreview_from_github,
    parse_gitreview_text,
)
from .detection import (
    _CHANGE_ID_PATTERN,
    _END_MARKER,
    _GITHUB_HASH_PATTERN,
    _MODE_PATTERN,
    _START_MARKER,
    _TOPIC_PATTERN,
    GITHUB2GERRIT_BOT_AUTHORS,
    _detect_via_heuristic,
    _detect_via_markers,
    _extract_author,
    _extract_body,
    _looks_like_mapping,
    _parse_block_lines,
    _parse_heuristic,
    _parse_marker_block,
    detect_github2gerrit_comments,
    detect_github2gerrit_from_graphql_comments,
    has_github2gerrit_comments,
    log,
)
from .messages import (
    build_gerrit_change_url_from_mapping,
    build_gerrit_skip_message,
    build_gerrit_submission_comment,
)
from .models import (
    GitHub2GerritDetectionResult,
    GitHub2GerritMapping,
    GitHub2GerritMode,
)

__all__ = [
    # Re-exported .gitreview symbols (backward-compatible API).
    "GitReviewInfo",
    "fetch_gitreview_from_github",
    "parse_gitreview_text",
    # Public API defined in this module.
    "GITHUB2GERRIT_BOT_AUTHORS",
    "GitHub2GerritMode",
    "GitHub2GerritMapping",
    "GitHub2GerritDetectionResult",
    "detect_github2gerrit_comments",
    "detect_github2gerrit_from_graphql_comments",
    "has_github2gerrit_comments",
    "build_gerrit_change_url_from_mapping",
    "build_gerrit_submission_comment",
    "build_gerrit_skip_message",
]
