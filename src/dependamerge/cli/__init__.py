# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Command line interface for dependamerge.

Builds the Typer application exposed as the ``dependamerge`` console
script: bulk merge of GitHub pull requests and Gerrit changes, plus the
``close``, ``status``, and ``blocked`` reports.

The implementation is split across private sibling modules purely to
keep each one reviewable; every name this package exposed as a single
module is still reachable as ``dependamerge.cli.<name>``.  Collaborators
that tests substitute live in :mod:`dependamerge.cli._deps` and are
reached through the module object, so patching them is observed by every
call site.
"""

from __future__ import annotations

from .._version import __version__
from ..bot_identity import is_automation_author
from ..close_manager import CloseResult
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
from ..github_service import AUTOMATION_TOOLS
from ..merge_manager import (
    DEFAULT_MERGE_TIMEOUT,
    AsyncMergeManager,
    MergeResult,
)
from ..models import (
    ComparisonResult,
    PullRequestInfo,
)
from ..netrc import (
    GerritCredentials,
    NetrcParseError,
)
from ..progress_tracker import (
    MergeProgressTracker,
    ProgressTracker,
)
from ..resolve_conflicts import (
    FixOptions,
    FixOrchestrator,
    PRSelection,
)
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
from ._blocked import (
    _display_blocked_results,
    blocked,
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
from ._close_cmd import close
from ._context import (
    _fetch_and_validate_source_pr,
    _init_github_merge,
    _MergeContext,
    _print_debug_matching,
    _source_pr_modifies_workflows,
    _validate_merge_inputs,
)
from ._deps import (
    AsyncCloseManager,
    GitHubClient,
    PRComparator,
    create_gerrit_comparator,
    create_gerrit_service,
    create_submit_manager,
    resolve_gerrit_credentials,
)
from ._display import (
    _display_change_info,
    _display_pr_info,
    _format_condensed_similarity,
    _format_gerrit_similarity,
)
from ._gerrit_merge import (
    _confirm_gerrit_submission,
    _handle_gerrit_merge,
    _preview_gerrit_submission,
    _print_gerrit_final_summary,
    _run_gerrit_submission,
)
from ._gerrit_resolve import (
    _find_and_print_similar_changes,
    _maybe_rebase_gerrit_change,
    _resolve_gerrit_candidates,
    _resolve_gerrit_credentials_or_exit,
    _resolve_gerrit_only_automation,
    _resolve_gerrit_source_change,
)
from ._merge_cmd import merge
from ._org_merge import (
    _execute_org_confirmed_merge,
    _handle_org_merge,
    _handle_org_preview_confirmation,
)
from ._parallel import (
    _execute_confirmed_merge,
    _handle_preview_confirmation,
    _restart_merge_progress_tracker,
    _run_parallel_merge,
)
from ._permissions import (
    _check_merge_permissions,
    _maybe_check_merge_permissions,
    _report_missing_permissions,
)
from ._repo_merge import (
    _execute_repo_confirmed_merge,
    _handle_repo_merge,
    _handle_repo_preview_confirmation,
)
from ._results import (
    _display_merge_results,
    _format_failure_reason,
    _owner_merge_order,
    _print_failed_pr_details,
    _print_final_merge_summary,
    _print_prs_grouped_by_repo,
    _repo_merge_order,
)
from ._scan import (
    _scan_and_find_similar,
    _validate_automation_author,
)
from ._sha import (
    _generate_continue_sha,
    _generate_gerrit_continue_sha,
    _generate_gerrit_override_sha,
    _generate_override_sha,
    _validate_override_sha,
)
from ._status import (
    _display_status_results,
    status,
)

__all__ = [
    "app",
    "AsyncCloseManager",
    "AsyncMergeManager",
    "AUTOMATION_TOOLS",
    "blocked",
    "_check_merge_permissions",
    "close",
    "_CloseContext",
    "CloseResult",
    "ComparisonResult",
    "_confirm_gerrit_submission",
    "console",
    "convert_git_error",
    "convert_github_api_error",
    "convert_network_error",
    "create_gerrit_comparator",
    "create_gerrit_service",
    "create_submit_manager",
    "CustomTyper",
    "DEFAULT_MAX_WAIT",
    "DEFAULT_MERGE_TIMEOUT",
    "DependamergeError",
    "_display_blocked_results",
    "_display_change_info",
    "_display_merge_results",
    "_display_pr_info",
    "_display_status_results",
    "_execute_confirmed_merge",
    "_execute_org_confirmed_merge",
    "_execute_repo_confirmed_merge",
    "exit_for_configuration_error",
    "exit_for_github_api_error",
    "exit_for_pr_state_error",
    "exit_with_error",
    "ExitCode",
    "_fetch_and_validate_source_pr",
    "_find_and_print_similar_changes",
    "_find_similar_prs_for_close",
    "FixOptions",
    "FixOrchestrator",
    "_format_condensed_similarity",
    "_format_failure_reason",
    "_format_gerrit_similarity",
    "_generate_continue_sha",
    "_generate_gerrit_continue_sha",
    "_generate_gerrit_override_sha",
    "_generate_override_sha",
    "GerritAuthError",
    "GerritChangeComparator",
    "GerritChangeInfo",
    "GerritComparisonResult",
    "GerritCredentials",
    "GerritRestError",
    "GerritService",
    "GerritSubmitResult",
    "get_default_workers",
    "GitError",
    "GitHubAsync",
    "GitHubClient",
    "GitHubPermissionError",
    "GraphQLError",
    "_handle_gerrit_merge",
    "_handle_org_merge",
    "_handle_org_preview_confirmation",
    "_handle_preview_confirmation",
    "_handle_repo_merge",
    "_handle_repo_preview_confirmation",
    "_init_github_merge",
    "is_automation_author",
    "is_github_api_permission_error",
    "is_network_error",
    "is_rule_violation",
    "main",
    "MAX_RETRIES",
    "_maybe_check_merge_permissions",
    "_maybe_rebase_gerrit_change",
    "merge",
    "_MergeContext",
    "MergeProgressTracker",
    "MergeResult",
    "NetrcParseError",
    "_owner_merge_order",
    "parse_change_url",
    "parse_gerrit_topic_url",
    "parse_org_url",
    "parse_owner_arg",
    "parse_repo_url",
    "ParsedGerritTopicUrl",
    "ParsedOrgUrl",
    "ParsedRepoUrl",
    "ParsedUrl",
    "PRComparator",
    "_preview_gerrit_submission",
    "_print_close_analysis_summary",
    "_print_close_debug_matching",
    "_print_debug_matching",
    "_print_failed_pr_details",
    "_print_final_merge_summary",
    "_print_gerrit_final_summary",
    "_print_prs_grouped_by_repo",
    "ProgressTracker",
    "PRSelection",
    "PullRequestInfo",
    "RateLimitError",
    "_repo_merge_order",
    "_report_missing_permissions",
    "required_status_check_names",
    "required_workflow_names",
    "_resolve_gerrit_candidates",
    "resolve_gerrit_credentials",
    "_resolve_gerrit_credentials_or_exit",
    "_resolve_gerrit_only_automation",
    "_resolve_gerrit_source_change",
    "_restart_merge_progress_tracker",
    "RULE_VIOLATION_MARKER",
    "_run_close_dry_run",
    "_run_close_parallel",
    "_run_gerrit_submission",
    "_run_immediate_close",
    "_run_interactive_close",
    "_run_parallel_merge",
    "_scan_and_find_similar",
    "SecondaryRateLimitError",
    "_source_pr_modifies_workflows",
    "status",
    "UrlParseError",
    "_validate_automation_author",
    "_validate_close_authorization",
    "_validate_merge_inputs",
    "_validate_override_sha",
    "__version__",
    "version_callback",
    "violation_verb",
]
