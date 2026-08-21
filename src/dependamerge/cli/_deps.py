# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Canonical home for the collaborators the CLI substitutes under test.

Sibling modules reach these through the module object --- ``_deps.X``
rather than ``from ._deps import X`` --- so that patching
``dependamerge.cli._deps.X`` is observed by every call site.
"""

from __future__ import annotations

from ..close_manager import AsyncCloseManager
from ..gerrit.comparator import create_gerrit_comparator
from ..gerrit.service import create_gerrit_service
from ..gerrit.submit_manager import create_submit_manager
from ..github_client import GitHubClient
from ..netrc import resolve_gerrit_credentials
from ..pr_comparator import PRComparator

__all__ = [
    "AsyncCloseManager",
    "GitHubClient",
    "PRComparator",
    "create_gerrit_comparator",
    "create_gerrit_service",
    "create_submit_manager",
    "resolve_gerrit_credentials",
]
