# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Synchronous GitHub API client for managing pull requests.

``GitHubClient`` is the blocking front door onto the async HTTP layer:
every method here runs an event loop for the duration of one call, which
is what lets the CLI stay synchronous.  Its construction and URL parsing
live in :mod:`.client`; the API-backed operations are split by intent
into :mod:`.queries` (reads), :mod:`.actions` (writes) and
:mod:`.status` (mergeability interpretation), and mixed back into the
class so its method surface is unchanged.

The names re-exported below have always been reachable as
``dependamerge.github_client.<name>``, so they stay reachable here even
where only the submodules still reference them.
"""

from __future__ import annotations

from ..bot_identity import is_automation_author
from ..models import (
    FileChange,
    OrganizationScanResult,
    PullRequestInfo,
    ReviewInfo,
)
from ..url_parser import _host_matches
from .client import (
    GitHubClient,
    logger,
)

__all__ = [
    "FileChange",
    "GitHubClient",
    "OrganizationScanResult",
    "PullRequestInfo",
    "ReviewInfo",
    "_host_matches",
    "is_automation_author",
    "logger",
]
