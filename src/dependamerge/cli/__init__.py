# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Command line interface for dependamerge.

This package re-exports the entire top-level surface of the module it
replaced, private helpers included, so every existing import and every
substitution target still resolves against ``dependamerge.cli``.
"""

import asyncio
import hashlib
import logging
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import typer
import urllib3.exceptions
from rich.console import Console
from rich.table import Table

from .._version import __version__
from ..bot_identity import is_automation_author
from ..close_manager import AsyncCloseManager, CloseResult
from ..error_codes import (
    DependamergeError,
    ExitCode,
    convert_git_error,
    convert_github_api_error,
    convert_network_error,
    exit_for_configuration_error,
    exit_for_github_api_error,
    exit_for_pr_state_error,
    exit_with_error,
    is_github_api_permission_error,
    is_network_error,
)
from ..gerrit import (
    GerritAuthError,
    GerritChangeComparator,
    GerritChangeInfo,
    GerritComparisonResult,
    GerritRestError,
    GerritService,
    GerritSubmitResult,
    create_gerrit_comparator,
    create_gerrit_service,
    create_submit_manager,
)
from ..git_ops import GitError
from ..github_async import (
    GitHubAsync,
    GraphQLError,
    RateLimitError,
    SecondaryRateLimitError,
)
from ..github_async import (
    PermissionError as GitHubPermissionError,
)
from ..github_client import GitHubClient
from ..github_service import AUTOMATION_TOOLS
from ..merge_manager import (
    DEFAULT_MERGE_TIMEOUT,
    AsyncMergeManager,
    MergeResult,
)
from ..models import ComparisonResult, PullRequestInfo
from ..netrc import (
    GerritCredentials,
    NetrcParseError,
    resolve_gerrit_credentials,
)
from ..pr_comparator import PRComparator
from ..progress_tracker import MergeProgressTracker, ProgressTracker
from ..resolve_conflicts import FixOptions, FixOrchestrator, PRSelection
from ..rule_violations import (
    RULE_VIOLATION_MARKER,
    is_rule_violation,
    required_status_check_names,
    required_workflow_names,
    violation_verb,
)
from ..system_utils import get_default_workers
from ..url_parser import (
    ParsedGerritTopicUrl,
    ParsedOrgUrl,
    ParsedRepoUrl,
    ParsedUrl,
    UrlParseError,
    parse_change_url,
    parse_gerrit_topic_url,
    parse_org_url,
    parse_owner_arg,
    parse_repo_url,
)
from ._app import (
    DEFAULT_MAX_WAIT,
    MAX_RETRIES,
    CustomTyper,
    app,
    console,
    main,
    version_callback,
)
from ._close import (
    _CloseContext,
    _find_similar_prs_for_close,
    _print_close_analysis_summary,
    _print_close_debug_matching,
    _run_close_dry_run,
    _run_close_parallel,
    _run_immediate_close,
    _run_interactive_close,
    _validate_close_authorization,
)
from ._context import (
    _MergeContext,
)
from ._gerrit_merge import (
    _find_and_print_similar_changes,
    _handle_gerrit_merge,
    _preview_gerrit_submission,
    _run_gerrit_submission,
)
from ._gerrit_setup import (
    _maybe_rebase_gerrit_change,
    _print_gerrit_final_summary,
    _resolve_gerrit_candidates,
    _resolve_gerrit_credentials_or_exit,
    _resolve_gerrit_only_automation,
    _resolve_gerrit_source_change,
)
from ._merge_authors import (
    _validate_automation_author,
)
from ._merge_inputs import (
    _fetch_and_validate_source_pr,
    _init_github_merge,
    _source_pr_modifies_workflows,
    _validate_merge_inputs,
)
from ._merge_order import (
    _owner_merge_order,
    _print_prs_grouped_by_repo,
    _repo_merge_order,
)
from ._merge_permissions import (
    _check_merge_permissions,
    _maybe_check_merge_permissions,
    _report_missing_permissions,
)
from ._merge_report import (
    _display_merge_results,
    _format_failure_reason,
    _print_failed_pr_details,
    _print_final_merge_summary,
)
from ._merge_scan import (
    _execute_confirmed_merge,
    _handle_preview_confirmation,
    _restart_merge_progress_tracker,
    _run_parallel_merge,
    _scan_and_find_similar,
)
from ._org_confirm import (
    _execute_org_confirmed_merge,
    _handle_org_preview_confirmation,
)
from ._org_merge import (
    _handle_org_merge,
)
from ._pr_display import (
    _display_pr_info,
    _print_debug_matching,
)
from ._repo_confirm import (
    _execute_repo_confirmed_merge,
    _handle_repo_preview_confirmation,
)
from ._repo_merge import (
    _handle_repo_merge,
)
from ._reports import (
    _display_blocked_results,
    _display_status_results,
)
from ._sha import (
    _generate_continue_sha,
    _generate_gerrit_continue_sha,
    _generate_gerrit_override_sha,
    _generate_override_sha,
    _validate_override_sha,
)
from ._similarity import (
    _confirm_gerrit_submission,
    _display_change_info,
    _format_condensed_similarity,
    _format_gerrit_similarity,
)

# The order these four are imported is the order Typer lists them in
# ``--help``, so it is fixed here rather than left to import sorting.
# isort: off
from ._merge_cmd import (
    merge,
)
from ._close_cmd import (
    close,
)
from ._status_cmd import (
    status,
)
from ._blocked_cmd import (
    blocked,
)
# isort: on

# Every name the predecessor module bound at the top level is
# re-exported here, private helpers included, so existing imports and
# substitution targets keep resolving against ``dependamerge.cli``.
__all__ = [
    "AUTOMATION_TOOLS",
    "Any",
    "AsyncCloseManager",
    "AsyncMergeManager",
    "CloseResult",
    "ComparisonResult",
    "Console",
    "CustomTyper",
    "DEFAULT_MAX_WAIT",
    "DEFAULT_MERGE_TIMEOUT",
    "DependamergeError",
    "ExitCode",
    "FixOptions",
    "FixOrchestrator",
    "GerritAuthError",
    "GerritChangeComparator",
    "GerritChangeInfo",
    "GerritComparisonResult",
    "GerritCredentials",
    "GerritRestError",
    "GerritService",
    "GerritSubmitResult",
    "GitError",
    "GitHubAsync",
    "GitHubClient",
    "GitHubPermissionError",
    "GraphQLError",
    "MAX_RETRIES",
    "MergeProgressTracker",
    "MergeResult",
    "NetrcParseError",
    "OrderedDict",
    "PRComparator",
    "PRSelection",
    "ParsedGerritTopicUrl",
    "ParsedOrgUrl",
    "ParsedRepoUrl",
    "ParsedUrl",
    "Path",
    "ProgressTracker",
    "PullRequestInfo",
    "RULE_VIOLATION_MARKER",
    "RateLimitError",
    "SecondaryRateLimitError",
    "Table",
    "UrlParseError",
    "_CloseContext",
    "_MergeContext",
    "__version__",
    "_check_merge_permissions",
    "_confirm_gerrit_submission",
    "_display_blocked_results",
    "_display_change_info",
    "_display_merge_results",
    "_display_pr_info",
    "_display_status_results",
    "_execute_confirmed_merge",
    "_execute_org_confirmed_merge",
    "_execute_repo_confirmed_merge",
    "_fetch_and_validate_source_pr",
    "_find_and_print_similar_changes",
    "_find_similar_prs_for_close",
    "_format_condensed_similarity",
    "_format_failure_reason",
    "_format_gerrit_similarity",
    "_generate_continue_sha",
    "_generate_gerrit_continue_sha",
    "_generate_gerrit_override_sha",
    "_generate_override_sha",
    "_handle_gerrit_merge",
    "_handle_org_merge",
    "_handle_org_preview_confirmation",
    "_handle_preview_confirmation",
    "_handle_repo_merge",
    "_handle_repo_preview_confirmation",
    "_init_github_merge",
    "_maybe_check_merge_permissions",
    "_maybe_rebase_gerrit_change",
    "_owner_merge_order",
    "_preview_gerrit_submission",
    "_print_close_analysis_summary",
    "_print_close_debug_matching",
    "_print_debug_matching",
    "_print_failed_pr_details",
    "_print_final_merge_summary",
    "_print_gerrit_final_summary",
    "_print_prs_grouped_by_repo",
    "_repo_merge_order",
    "_report_missing_permissions",
    "_resolve_gerrit_candidates",
    "_resolve_gerrit_credentials_or_exit",
    "_resolve_gerrit_only_automation",
    "_resolve_gerrit_source_change",
    "_restart_merge_progress_tracker",
    "_run_close_dry_run",
    "_run_close_parallel",
    "_run_gerrit_submission",
    "_run_immediate_close",
    "_run_interactive_close",
    "_run_parallel_merge",
    "_scan_and_find_similar",
    "_source_pr_modifies_workflows",
    "_validate_automation_author",
    "_validate_close_authorization",
    "_validate_merge_inputs",
    "_validate_override_sha",
    "app",
    "asyncio",
    "blocked",
    "close",
    "console",
    "convert_git_error",
    "convert_github_api_error",
    "convert_network_error",
    "create_gerrit_comparator",
    "create_gerrit_service",
    "create_submit_manager",
    "dataclass",
    "exit_for_configuration_error",
    "exit_for_github_api_error",
    "exit_for_pr_state_error",
    "exit_with_error",
    "field",
    "get_default_workers",
    "hashlib",
    "is_automation_author",
    "is_github_api_permission_error",
    "is_network_error",
    "is_rule_violation",
    "logging",
    "main",
    "merge",
    "os",
    "parse_change_url",
    "parse_gerrit_topic_url",
    "parse_org_url",
    "parse_owner_arg",
    "parse_repo_url",
    "requests",
    "required_status_check_names",
    "required_workflow_names",
    "resolve_gerrit_credentials",
    "status",
    "sys",
    "typer",
    "urllib3",
    "urlparse",
    "version_callback",
    "violation_verb",
]
