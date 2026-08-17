# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Asynchronous GitHub service for dependamerge.

``GitHubService`` paginates organization repositories and their open
pull requests via GraphQL, maps PR nodes onto the project's Pydantic
models, detects unmergeable reasons, finds similar PRs across an owner,
resolves branch-protection-aware merge methods, and gathers per-repository
status reports.

The implementation is split across private sibling modules purely to
keep each one reviewable; every name this package exposed as a single
module is still reachable as ``dependamerge.github_service.<name>``.
"""

from __future__ import annotations

from ._constants import (
    AUTOMATION_TOOLS,
    DEFAULT_COMMENTS_PAGE_SIZE,
    DEFAULT_CONTEXTS_PAGE_SIZE,
    DEFAULT_FILES_PAGE_SIZE,
    DEFAULT_PRS_PAGE_SIZE,
)
from ._helpers import (
    _bool_or_none,
    _CallbackChain,
    _chain_callbacks,
    _clone_url_with_git_suffix,
    _str_or_none,
    _unchain_callback,
)
from ._service import GitHubService

__all__ = [
    "AUTOMATION_TOOLS",
    "DEFAULT_COMMENTS_PAGE_SIZE",
    "DEFAULT_CONTEXTS_PAGE_SIZE",
    "DEFAULT_FILES_PAGE_SIZE",
    "DEFAULT_PRS_PAGE_SIZE",
    "GitHubService",
]
