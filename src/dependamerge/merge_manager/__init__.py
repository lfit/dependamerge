# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Parallel approval and merging of pull requests.

``AsyncMergeManager`` is assembled here from mixins, one per module,
so that method resolution, ``self`` and every substitution target are
exactly what they were when this was a single module.  The whole
top-level surface of that module — private helpers included — is
re-exported, so every existing import and every substitution target
still resolves against ``dependamerge.merge_manager``.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

from rich.console import Console

from .. import rebase
from ..bot_identity import is_automation_author, is_dependabot
from ..check_runs import failing_check_names
from ..copilot_handler import CopilotCommentHandler
from ..gerrit import (
    GerritAuthError,
    GerritRestError,
    create_gerrit_service,
    create_submit_manager,
)
from ..github2gerrit_detector import (
    GitHub2GerritDetectionResult,
    GitHub2GerritMapping,
    build_gerrit_change_url_from_mapping,
    build_gerrit_skip_message,
    build_gerrit_submission_comment,
    detect_github2gerrit_comments,
    fetch_gitreview_from_github,
)
from ..github_async import GitHubAsync
from ..github_async import PermissionError as GitHubPermissionError
from ..github_service import GitHubService
from ..models import ComparisonResult, PullRequestInfo
from ..netrc import NetrcParseError, resolve_gerrit_credentials
from ..output_utils import log_and_print
from ..pr_poller import PullRequestStatePoller
from ..progress_tracker import MergeProgressTracker
from ..rule_violations import violation_verb, workflow_name_fragments
from ..semantic_title import (
    describe_title_change,
    is_semantic_check_name,
    single_commit_subject,
    version_fragment_removed,
)
from ..slot_lease import holding_slot, parked
from ._approval import _ApprovalMixin
from ._auto_merge import _AutoMergeEnableMixin
from ._conflict import _MergeConflictMixin
from ._constants import (
    _MERGEABILITY_ICON_AND_STYLE,  # noqa: F401  (see __all__ note below)
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
from ._dependabot_recreate import _DependabotRecreateMixin
from ._failure import _FailureReportingMixin
from ._gerrit_submit import _GerritSubmitMixin
from ._lifecycle import _LifecycleMixin
from ._mergeability import _MergeabilityMixin
from ._mergeability_refresh import _MergeabilityRefreshMixin
from ._not_mergeable import _NotMergeableMixin
from ._orchestration import _OrchestrationMixin
from ._org_settings import _OrgSettingsMixin
from ._outcomes import _OutcomeTrackingMixin
from ._pr_state import _PullRequestStateMixin
from ._precommit_ci import _PrecommitCiMixin
from ._prediction import _PredictionMixin
from ._recreated_pr import _RecreatedPullRequestMixin
from ._required_workflows import _RequiredWorkflowWaitMixin
from ._requirements import _MergeRequirementsMixin
from ._retry import _MergeRetryMixin
from ._rulesets import _RulesetMixin
from ._semantic_title import _SemanticTitleMixin
from ._single_pr import _SinglePullRequestMixin
from ._stuck_checks import _StuckCheckMixin
from ._ticker import _StatusTickerMixin
from ._types import (
    MergeResult,
    MergeStatus,
    _merge_already_in_progress,  # noqa: F401  (see __all__ note below)
    _merged_from_payload,  # noqa: F401  (see __all__ note below)
)
from ._undispatched import _UndispatchedWorkflowMixin
from ._wait import _AutoMergeWaitMixin


class AsyncMergeManager(
    _LifecycleMixin,
    _OrchestrationMixin,
    _OutcomeTrackingMixin,
    _StatusTickerMixin,
    _GerritSubmitMixin,
    _SinglePullRequestMixin,
    _NotMergeableMixin,
    _MergeabilityMixin,
    _AutoMergeEnableMixin,
    _ApprovalMixin,
    _MergeRequirementsMixin,
    _SemanticTitleMixin,
    _PrecommitCiMixin,
    _StuckCheckMixin,
    _DependabotRecreateMixin,
    _RecreatedPullRequestMixin,
    _PullRequestStateMixin,
    _MergeabilityRefreshMixin,
    _MergeRetryMixin,
    _UndispatchedWorkflowMixin,
    _RequiredWorkflowWaitMixin,
    _AutoMergeWaitMixin,
    _MergeConflictMixin,
    _FailureReportingMixin,
    _OrgSettingsMixin,
    _RulesetMixin,
    _PredictionMixin,
):
    """
    Manages parallel approval and merging of pull requests.

    This class handles:
    - Concurrent approval of PRs
    - Concurrent merging with retry logic
    - Progress tracking and error handling
    - Rate limit-aware processing
    """


__all__ = [
    "Any",
    "AsyncMergeManager",
    "ComparisonResult",
    "Console",
    "CopilotCommentHandler",
    "DEFAULT_MERGE_RECHECK_INTERVAL",
    "DEFAULT_MERGE_TIMEOUT",
    "Enum",
    "GerritAuthError",
    "GerritRestError",
    "GitHub2GerritDetectionResult",
    "GitHub2GerritMapping",
    "GitHubAsync",
    "GitHubPermissionError",
    "GitHubService",
    "MERGEABILITY_REFRESH_TIMEOUT_SECONDS",
    "MERGE_IN_PROGRESS_FIRST_POLL_SECONDS",
    "MERGE_IN_PROGRESS_POLL_SECONDS",
    "MERGE_IN_PROGRESS_TIMEOUT_SECONDS",
    "MERGE_WAIT_FIRST_POLL_SECONDS",
    "MergeProgressTracker",
    "MergeResult",
    "MergeStatus",
    "NetrcParseError",
    "PRECOMMIT_CI_STUCK_PENDING_SECONDS",
    "Path",
    "PullRequestInfo",
    "PullRequestStatePoller",
    "STUCK_CHECK_THRESHOLD_SECONDS",
    "UNDISPATCHED_CONFIRM_DELAY_SECONDS",
    "UNDISPATCHED_CONFIRM_LOOKUP_SECONDS",
    # The underscore-prefixed re-exports above are deliberately absent from
    # __all__.  The original module declared no __all__, so
    # ``from dependamerge.merge_manager import *`` never picked up private
    # names; listing them here would widen that surface.  They remain
    # importable explicitly, and resolvable as module attributes, which is
    # what the patch targets in the test suite rely on.
    "asyncio",
    "build_gerrit_change_url_from_mapping",
    "build_gerrit_skip_message",
    "build_gerrit_submission_comment",
    "cast",
    "create_gerrit_service",
    "create_submit_manager",
    "dataclass",
    "describe_title_change",
    "detect_github2gerrit_comments",
    "failing_check_names",
    "fetch_gitreview_from_github",
    "fnmatch",
    "holding_slot",
    "is_automation_author",
    "is_dependabot",
    "is_semantic_check_name",
    "log_and_print",
    "logging",
    "math",
    "os",
    "parked",
    "quote",
    "re",
    "rebase",
    "resolve_gerrit_credentials",
    "single_commit_subject",
    "time",
    "version_fragment_removed",
    "violation_verb",
    "workflow_name_fragments",
]
