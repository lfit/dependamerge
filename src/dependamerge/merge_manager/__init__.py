# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Parallel approval and merging of automation pull requests.

``AsyncMergeManager`` drives the whole merge lifecycle: approval,
rebase recovery, auto-merge, waiting on required checks, conflict
handling, and failure reporting.

The implementation is split across private sibling modules purely to
keep each one reviewable; every name this package exposed as a single
module is still reachable as ``dependamerge.merge_manager.<name>``.
"""

from __future__ import annotations

# ``asyncio`` is re-exported deliberately. Tests reach the stdlib module
# through this package to substitute ``asyncio.sleep``, and the module
# this package replaced carried the same attribute. Every other
# incidental stdlib import stays out of the re-export list.
import asyncio

from .. import rebase
from ..gerrit import (
    create_gerrit_service,
    create_submit_manager,
)
from ..github_async import GitHubAsync
from ..netrc import resolve_gerrit_credentials
from ..slot_lease import parked
from ._constants import (
    _MERGEABILITY_ICON_AND_STYLE,
    DEFAULT_MERGE_RECHECK_INTERVAL,
    DEFAULT_MERGE_TIMEOUT,
    MERGE_IN_PROGRESS_FIRST_POLL_SECONDS,
    MERGE_IN_PROGRESS_POLL_SECONDS,
    MERGE_IN_PROGRESS_TIMEOUT_SECONDS,
    MERGE_WAIT_FIRST_POLL_SECONDS,
    MERGEABILITY_REFRESH_TIMEOUT_SECONDS,
    PRECOMMIT_CI_STUCK_PENDING_SECONDS,
    STUCK_CHECK_THRESHOLD_SECONDS,
    UNDISPATCHED_CONFIRM_DELAY_SECONDS,
    UNDISPATCHED_CONFIRM_LOOKUP_SECONDS,
)
from ._manager import (
    AsyncMergeManager,
)
from ._models import (
    MergeResult,
    MergeStatus,
    _merge_already_in_progress,
    _merged_from_payload,
)

__all__ = [
    "AsyncMergeManager",
    "MergeResult",
    "MergeStatus",
    "DEFAULT_MERGE_TIMEOUT",
    "DEFAULT_MERGE_RECHECK_INTERVAL",
    "MERGEABILITY_REFRESH_TIMEOUT_SECONDS",
    "MERGE_WAIT_FIRST_POLL_SECONDS",
    "MERGE_IN_PROGRESS_TIMEOUT_SECONDS",
    "MERGE_IN_PROGRESS_POLL_SECONDS",
    "UNDISPATCHED_CONFIRM_DELAY_SECONDS",
    "UNDISPATCHED_CONFIRM_LOOKUP_SECONDS",
    "MERGE_IN_PROGRESS_FIRST_POLL_SECONDS",
    "STUCK_CHECK_THRESHOLD_SECONDS",
    "PRECOMMIT_CI_STUCK_PENDING_SECONDS",
    "_MERGEABILITY_ICON_AND_STYLE",
    "GitHubAsync",
    "asyncio",
    "create_gerrit_service",
    "create_submit_manager",
    "parked",
    "rebase",
    "resolve_gerrit_credentials",
    "_merge_already_in_progress",
    "_merged_from_payload",
]
