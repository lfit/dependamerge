# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
GraphQL query strings for retrieving repositories in an organization and their
open pull requests, including status check rollups and basic file/comment data.

These queries are designed to batch-read as much as possible to reduce the
number of HTTP round-trips compared to multiple REST calls per PR.

Notes:
- The mergeable field is an enum: MERGEABLE | CONFLICTING | UNKNOWN
- The mergeStateStatus field includes states like CLEAN, DIRTY, BLOCKED, BEHIND, DRAFT, UNKNOWN
- statusCheckRollup provides both CheckRun and StatusContext results for the latest commit

The queries themselves are grouped by subject in the sibling modules
:mod:`.repos` (repository listing), :mod:`.pull_requests` (open PR
payloads), :mod:`.reviews` (review threads) and :mod:`.merge_policy`
(branch protection and auto-merge), and re-exported here so every
constant stays reachable as ``dependamerge.github_graphql.<NAME>``.
"""

from __future__ import annotations

from .merge_policy import (
    ENABLE_AUTO_MERGE,
    GET_BRANCH_PROTECTION,
)
from .pull_requests import (
    ORG_REPOS_WITH_OPEN_PRS,
    REPO_OPEN_PRS_PAGE,
)
from .repos import (
    ORG_REPOS_ONLY,
    USER_REPOS_ONLY,
)
from .reviews import (
    GET_PR_REVIEW_THREADS,
    RESOLVE_REVIEW_THREAD,
)

__all__ = [
    "ORG_REPOS_ONLY",
    "USER_REPOS_ONLY",
    "ORG_REPOS_WITH_OPEN_PRS",
    "REPO_OPEN_PRS_PAGE",
    "ENABLE_AUTO_MERGE",
    "GET_BRANCH_PROTECTION",
    "GET_PR_REVIEW_THREADS",
    "RESOLVE_REVIEW_THREAD",
]
