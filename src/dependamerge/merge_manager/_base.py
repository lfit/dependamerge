# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Shared attribute and method declarations for the merge-manager mixins.

``AsyncMergeManager`` is far too large for one reviewable module, so
its methods live in ``_XxxMixin`` classes that are mixed back together
in ``dependamerge.merge_manager._manager``.  Each mixin reads state
that ``AsyncMergeManager.__init__`` establishes and calls methods
implemented by its siblings; this base declares both, and declares
nothing at runtime, so the real implementations always win.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from ..copilot_handler import CopilotCommentHandler
    from ..github2gerrit_detector import (
        GitHub2GerritDetectionResult,
        GitHub2GerritMapping,
    )
    from ..github_async import GitHubAsync
    from ..github_service import GitHubService
    from ..models import PullRequestInfo
    from ..pr_poller import PullRequestStatePoller
    from ..progress_tracker import MergeProgressTracker
    from ._merge_state import _Attempt, _DispatchOutcome, _RebaseOutcome, _WaitOutcome
    from ._models import MergeResult, MergeStatus


class _MergeManagerBase:
    """Declarations shared by the ``AsyncMergeManager`` mixins."""

    # Established by AsyncMergeManager.__init__.
    token: str
    default_merge_method: str
    max_retries: int
    concurrency: int
    fix_out_of_date: bool
    fix_semantic_title: bool
    progress_tracker: MergeProgressTracker | None
    preview_mode: bool
    dismiss_copilot: bool
    force_level: str
    github2gerrit_mode: str
    no_netrc: bool
    netrc_file: Path | None
    rebase_local: bool
    _repo_scoped: bool
    _max_wait: float | None
    _run_deadline: float | None
    _no_wait: bool
    log: logging.Logger
    _merge_recheck_interval: float
    _merge_poll_max_attempts: int
    _merge_semaphore: asyncio.Semaphore
    _results: list[MergeResult]
    _github_client: GitHubAsync | None
    _pr_poller: PullRequestStatePoller | None
    _semantic_title_aligned: set[str]
    _repo_wait_seconds: dict[str, list[float]]
    _github_service: GitHubService | None
    _copilot_handler: CopilotCommentHandler | None
    _console: Console
    _pr_merge_methods: dict[str, str]
    _org_settings_cache: dict[str, dict[str, Any] | None]
    _org_settings_locks: dict[str, asyncio.Lock]
    _org_settings_locks_lock: asyncio.Lock
    _branch_approval_cache: dict[str, bool]
    _branch_approval_locks: dict[str, asyncio.Lock]
    _branch_approval_locks_lock: asyncio.Lock
    _org_approval_cache: dict[str, list[dict[str, Any]] | None]
    _org_approval_locks: dict[str, asyncio.Lock]
    _org_approval_locks_lock: asyncio.Lock
    _last_merge_exception: dict[str, Exception]
    _last_merge_exception_head: dict[str, str]
    _recently_approved: set[str]
    _permission_failed_repos: set[str]
    _auto_merge_enabled: set[str]
    _rebased_prs: set[str]
    _waiting_prs: dict[str, float]
    _waiting_lock: asyncio.Lock
    _merge_dispatch_locks: dict[str, asyncio.Lock]
    _merge_dispatch_locks_lock: asyncio.Lock
    _merge_timeout: float
    _post_approval_delay: float

    # Implemented by sibling mixins; declared here for type checking
    # only, so nothing is defined at runtime.
    if TYPE_CHECKING:

        def _record_terminal_outcome(
            self, pr_info: PullRequestInfo, status: MergeStatus
        ) -> None: ...

        def _track_pr_state(
            self, pr_info: PullRequestInfo, state: str | None
        ) -> None: ...

        def _record_rebase(self) -> None: ...

        def _record_retrigger(self) -> None: ...

        def _pr_status(self, message: str, *, level: str = "info") -> None: ...

        async def _merge_single_pr_with_semaphore(
            self, pr_info: PullRequestInfo
        ) -> MergeResult: ...

        async def _wait_status_ticker(self) -> None: ...

        async def _detect_github2gerrit(
            self, repo_owner: str, repo_name: str, pr_number: int
        ) -> GitHub2GerritDetectionResult: ...

        async def _submit_gerrit_change(
            self,
            mapping: GitHub2GerritMapping,
            pr_info: PullRequestInfo,
            repo_owner: str,
            repo_name: str,
        ) -> bool: ...

        async def _resolve_gerrit_host(
            self, mapping: GitHub2GerritMapping, repo_owner: str, repo_name: str
        ) -> tuple[str | None, str | None]: ...

        async def _merge_single_pr(self, pr_info: PullRequestInfo) -> MergeResult: ...

        async def _merge_single_pr_impl(
            self, pr_info: PullRequestInfo
        ) -> MergeResult: ...

        async def _route_to_gerrit(self, attempt: _Attempt) -> MergeResult | None: ...

        async def _check_merge_eligibility(
            self, attempt: _Attempt
        ) -> MergeResult | None: ...

        async def _process_copilot_feedback(
            self, attempt: _Attempt
        ) -> MergeResult | None: ...

        async def _rebase_if_required(self, attempt: _Attempt) -> _RebaseOutcome: ...

        async def _wait_for_required_checks(
            self, attempt: _Attempt, rebased: _RebaseOutcome
        ) -> _WaitOutcome: ...

        async def _dispatch_merge(
            self, attempt: _Attempt, rebased: _RebaseOutcome, waited: _WaitOutcome
        ) -> _DispatchOutcome: ...

        async def _report_merge_outcome(
            self, attempt: _Attempt, merged: bool | None
        ) -> None: ...

        async def _handle_not_mergeable_pr(
            self, pr_info: PullRequestInfo, result: MergeResult
        ) -> MergeResult: ...

        def _simulate_preview_merge(
            self, pr_info: PullRequestInfo, result: MergeResult
        ) -> None: ...

        @staticmethod
        def _block_reason_indicates_pending_checks(
            block_reason: str | None,
        ) -> bool: ...

        @staticmethod
        def _block_reason_indicates_check_blockage(
            block_reason: str | None,
        ) -> bool: ...

        async def _behind_pr_requires_rebase(
            self, pr_info: PullRequestInfo, repo_owner: str, repo_name: str
        ) -> bool: ...

        async def _blocked_pr_needs_rebase(
            self,
            pr_info: PullRequestInfo,
            repo_owner: str,
            repo_name: str,
            block_reason: str | None,
        ) -> bool: ...

        def _is_pr_mergeable(self, pr_info: PullRequestInfo) -> bool: ...

        def _has_blocking_reviews(self, pr_info: PullRequestInfo) -> bool: ...

        async def _post_pr_comment_with_retry(
            self, owner: str, repo: str, pr_number: int, html_url: str, body: str
        ) -> bool: ...

        async def _enable_auto_merge_for_pr(
            self, pr_info: PullRequestInfo, owner: str, repo: str
        ) -> bool: ...

        async def _ensure_pr_approved(
            self,
            pr_info: PullRequestInfo,
            owner: str,
            repo: str,
            *,
            propagation_delay: bool = True,
        ) -> bool: ...

        async def _enable_auto_merge_with_approval(
            self, pr_info: PullRequestInfo, owner: str, repo: str
        ) -> bool: ...

        async def _approve_and_retry_if_review_required(
            self, pr_info: PullRequestInfo, owner: str, repo: str
        ) -> bool: ...

        @staticmethod
        def _merge_error_indicates_pending_workflows(error_text: str) -> bool: ...

        async def _check_merge_requirements(
            self, pr_info: PullRequestInfo
        ) -> tuple[bool, str]: ...

        async def _align_semantic_title(self, pr_info: PullRequestInfo) -> bool: ...

        async def _trigger_stale_precommit_ci(
            self, pr_info: PullRequestInfo
        ) -> bool: ...

        async def _detect_stuck_required_check(
            self, pr_info: PullRequestInfo
        ) -> tuple[bool, str | None, float]: ...

        async def _trigger_dependabot_recreate(
            self, pr_info: PullRequestInfo
        ) -> PullRequestInfo | None: ...

        async def _wait_for_recreated_pr_checks(
            self,
            repo_owner: str,
            repo_name: str,
            new_number: int,
            pr_data: dict[str, Any],
        ) -> PullRequestInfo | None: ...

        async def _approve_pr(self, owner: str, repo: str, pr_number: int) -> bool: ...

        async def _recheck_pr_before_retry(
            self, owner: str, repo: str, pr_info: PullRequestInfo, attempt: int
        ) -> bool | None: ...

        async def _fetch_pr_state(
            self, owner: str, repo: str, number: int
        ) -> dict[str, Any] | list[dict[str, Any]] | None: ...

        async def _refresh_pr_mergeable(
            self, owner: str, repo: str, pr_info: PullRequestInfo, pr_key: str
        ) -> None: ...

        async def _await_in_progress_merge(
            self, owner: str, repo: str, pr_info: PullRequestInfo, pr_key: str
        ) -> bool: ...

        async def _blocked_pr_became_clean(
            self, owner: str, repo: str, pr_info: PullRequestInfo, pr_key: str
        ) -> bool: ...

        async def _merge_pr_with_retry(
            self, pr_info: PullRequestInfo, owner: str, repo: str
        ) -> bool: ...

        async def _workflows_never_dispatched(
            self,
            pr_info: PullRequestInfo,
            owner: str,
            repo: str,
            error_text: str,
            deadline: float | None = None,
        ) -> list[str]: ...

        async def _wait_for_required_workflows_and_retry(
            self, pr_info: PullRequestInfo, owner: str, repo: str
        ) -> bool: ...

        async def _is_pr_already_merged(
            self, pr_info: PullRequestInfo, owner: str, repo: str
        ) -> bool: ...

        async def _fetch_pr_state_now(
            self, pr_info: PullRequestInfo, owner: str, repo: str
        ) -> tuple[str | None, bool | None]: ...

        async def _is_pr_dirty_now(
            self, pr_info: PullRequestInfo, owner: str, repo: str
        ) -> bool: ...

        async def _refresh_pr_mergeability(
            self, pr_info: PullRequestInfo, owner: str, repo: str
        ) -> None: ...

        def _record_wait_duration(
            self, repo_full_name: str, seconds: float
        ) -> None: ...

        async def _apply_wait_head_start(
            self,
            pr_info: PullRequestInfo,
            pr_key: str,
            remaining: float,
            continue_states: tuple[str, ...],
            stop_on_clean: bool,
            measures_checks: bool,
        ) -> None: ...

        async def _wait_for_auto_merge(
            self,
            pr_info: PullRequestInfo,
            owner: str,
            repo: str,
            *,
            continue_states: tuple[str, ...],
            deadline: float | None = None,
            stop_on_clean: bool = True,
            measures_checks: bool = False,
        ) -> tuple[bool, bool]: ...

        async def _request_dependabot_rebase(
            self, pr_info: PullRequestInfo, owner: str, repo: str
        ) -> bool: ...

        def _finish_conflict_close(
            self, pr_info: PullRequestInfo, result: MergeResult, merged: bool
        ) -> MergeResult: ...

        def _dependabot_is_rebasing(self, body: str | None) -> bool: ...

        async def _handle_merge_conflict(
            self, pr_info: PullRequestInfo, owner: str, repo: str, result: MergeResult
        ) -> MergeResult: ...

        async def _report_merge_failure(
            self,
            pr_info: PullRequestInfo,
            owner: str,
            repo: str,
            result: MergeResult,
            failure_reason: str,
        ) -> MergeResult: ...

        async def _get_failure_summary(self, pr_info: PullRequestInfo) -> str: ...

        async def _get_merge_method_for_repo(self, owner: str, repo: str) -> str: ...

        async def _handle_merge_failure(
            self, pr_info: PullRequestInfo, owner: str, repo: str
        ) -> bool: ...

        async def _get_merge_dispatch_lock(
            self, owner: str, repo: str
        ) -> asyncio.Lock: ...

        async def _get_org_settings(self, owner: str) -> dict[str, Any] | None: ...

        @staticmethod
        def _rules_require_approval(rules: Any) -> bool: ...

        async def _approve_if_review_mandated(
            self, pr_info: PullRequestInfo, owner: str, repo: str, pr_key: str
        ) -> None: ...

        async def _org_approval_rulesets(
            self, org: str
        ) -> list[dict[str, Any]] | None: ...

        def _ruleset_condition_applies(
            self, conditions: Any, repo: str, branch: str
        ) -> bool | None: ...

        async def _branch_requires_approval(
            self, owner: str, repo: str, branch: str
        ) -> bool: ...

        async def _predict_merge_outcome(
            self, owner: str, repo: str, pr_number: int, merge_method: str
        ) -> tuple[bool, str]: ...
