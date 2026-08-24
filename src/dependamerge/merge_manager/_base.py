# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The declaration point shared by the AsyncMergeManager mixins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Nothing here runs, so nothing here should be bound at run time: a
# module-level binding of a name the tests substitute on the package
# would shadow that substitution (see tests/test_patch_targets.py).
if TYPE_CHECKING:
    import asyncio
    import logging
    from pathlib import Path
    from typing import Any

    from ..copilot_handler import CopilotCommentHandler
    from ..github2gerrit_detector import (
        GitHub2GerritDetectionResult,
        GitHub2GerritMapping,
    )
    from ..github_async import GitHubAsync
    from ..github_service import GitHubService
    from ..models import ComparisonResult, PullRequestInfo
    from ..pr_poller import PullRequestStatePoller
    from ..progress_tracker import MergeProgressTracker
    from ._types import MergeResult, MergeStatus


class _MergeManagerBase:
    """The state and cross-module methods every mixin may rely on.

    ``AsyncMergeManager`` is assembled from mixins that call freely into
    one another, and those calls are mutually recursive across module
    boundaries.  A type checker needs one acyclic place where the shared
    surface is declared, which is what this is: the instance attributes
    the constructor establishes, and the signature of every method that
    is called from a module other than the one defining it.

    Nothing here is ever executed.  Each declaration is overridden by
    the mixin that implements it, which sits ahead of this class in the
    method resolution order; the bodies exist only so the declarations
    are well-formed.  Declaring them keeps the checker able to verify
    that the implementations stay compatible with how they are called.
    """

    _auto_merge_enabled: set[str]
    _branch_approval_cache: dict[str, bool]
    _branch_approval_locks: dict[str, asyncio.Lock]
    _branch_approval_locks_lock: asyncio.Lock
    _console: Any
    _copilot_handler: CopilotCommentHandler | None
    _github_client: GitHubAsync | None
    _github_service: GitHubService | None
    _last_merge_exception: dict[str, Exception]
    _last_merge_exception_head: dict[str, str]
    _max_wait: float | None
    _merge_dispatch_locks: dict[str, asyncio.Lock]
    _merge_dispatch_locks_lock: asyncio.Lock
    _merge_poll_max_attempts: int
    _merge_recheck_interval: float
    _merge_semaphore: asyncio.Semaphore
    _merge_timeout: float
    _no_wait: bool
    _org_approval_cache: dict[str, list[dict[str, Any]] | None]
    _org_approval_locks: dict[str, asyncio.Lock]
    _org_approval_locks_lock: asyncio.Lock
    _org_settings_cache: dict[str, dict[str, Any] | None]
    _org_settings_locks: dict[str, asyncio.Lock]
    _org_settings_locks_lock: asyncio.Lock
    _permission_failed_repos: set[str]
    _post_approval_delay: float
    _pr_merge_methods: dict[str, str]
    _pr_poller: PullRequestStatePoller | None
    _rebased_prs: set[str]
    _recently_approved: set[str]
    _repo_scoped: bool
    _repo_wait_seconds: dict[str, list[float]]
    _results: list[MergeResult]
    _run_deadline: float | None
    _semantic_title_aligned: set[str]
    _waiting_lock: asyncio.Lock
    _waiting_prs: dict[str, float]
    concurrency: int
    default_merge_method: str
    dismiss_copilot: bool
    fix_out_of_date: bool
    fix_semantic_title: bool
    force_level: str
    github2gerrit_mode: str
    log: logging.Logger
    max_retries: int
    netrc_file: Path | None
    no_netrc: bool
    preview_mode: bool
    progress_tracker: MergeProgressTracker | None
    rebase_local: bool
    token: str

    async def _align_semantic_title(self, pr_info: PullRequestInfo) -> bool:
        raise NotImplementedError

    async def _approve_and_retry_if_review_required(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        raise NotImplementedError

    async def _approve_if_review_mandated(
        self, pr_info: PullRequestInfo, owner: str, repo: str, pr_key: str
    ) -> None:
        raise NotImplementedError

    async def _approve_pr(self, owner: str, repo: str, pr_number: int) -> bool:
        raise NotImplementedError

    async def _await_in_progress_merge(
        self, owner: str, repo: str, pr_info: PullRequestInfo, pr_key: str
    ) -> bool:
        raise NotImplementedError

    async def _behind_pr_requires_rebase(
        self, pr_info: PullRequestInfo, repo_owner: str, repo_name: str
    ) -> bool:
        raise NotImplementedError

    @staticmethod
    def _block_reason_indicates_pending_checks(block_reason: str | None) -> bool:
        raise NotImplementedError

    async def _blocked_pr_became_clean(
        self, owner: str, repo: str, pr_info: PullRequestInfo, pr_key: str
    ) -> bool:
        raise NotImplementedError

    async def _blocked_pr_needs_rebase(
        self,
        pr_info: PullRequestInfo,
        repo_owner: str,
        repo_name: str,
        block_reason: str | None,
    ) -> bool:
        raise NotImplementedError

    async def _check_merge_requirements(
        self, pr_info: PullRequestInfo
    ) -> tuple[bool, str]:
        raise NotImplementedError

    def _collect_results(
        self,
        pr_list: list[tuple[PullRequestInfo, ComparisonResult | None]],
        results: list[Any],
    ) -> list[MergeResult]:
        raise NotImplementedError

    async def _confirm_failure(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> MergeResult:
        raise NotImplementedError

    def _dependabot_is_rebasing(self, body: str | None) -> bool:
        raise NotImplementedError

    async def _detect_github2gerrit(
        self, repo_owner: str, repo_name: str, pr_number: int
    ) -> GitHub2GerritDetectionResult:
        raise NotImplementedError

    async def _detect_stuck_required_check(
        self, pr_info: PullRequestInfo
    ) -> tuple[bool, str | None, float]:
        raise NotImplementedError

    async def _enable_auto_merge_for_pr(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        raise NotImplementedError

    async def _enable_auto_merge_with_approval(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        raise NotImplementedError

    async def _ensure_pr_approved(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        *,
        propagation_delay: bool = True,
    ) -> bool:
        raise NotImplementedError

    async def _fetch_pr_state(
        self, owner: str, repo: str, number: int
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        raise NotImplementedError

    async def _fetch_pr_state_now(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> tuple[str | None, bool | None]:
        raise NotImplementedError

    async def _get_failure_summary(self, pr_info: PullRequestInfo) -> str:
        raise NotImplementedError

    async def _get_merge_dispatch_lock(self, owner: str, repo: str) -> asyncio.Lock:
        raise NotImplementedError

    async def _get_merge_method_for_repo(self, owner: str, repo: str) -> str:
        raise NotImplementedError

    async def _get_org_settings(self, owner: str) -> dict[str, Any] | None:
        raise NotImplementedError

    async def _handle_merge_conflict(
        self, pr_info: PullRequestInfo, owner: str, repo: str, result: MergeResult
    ) -> MergeResult:
        raise NotImplementedError

    async def _handle_merge_failure(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        raise NotImplementedError

    async def _handle_not_mergeable_pr(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> MergeResult:
        raise NotImplementedError

    def _has_blocking_reviews(self, pr_info: PullRequestInfo) -> bool:
        raise NotImplementedError

    async def _is_pr_already_merged(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        raise NotImplementedError

    async def _is_pr_dirty_now(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        raise NotImplementedError

    def _is_pr_mergeable(self, pr_info: PullRequestInfo) -> bool:
        raise NotImplementedError

    @staticmethod
    def _merge_error_indicates_pending_workflows(error_text: str) -> bool:
        raise NotImplementedError

    async def _merge_pr_with_retry(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        raise NotImplementedError

    async def _merge_single_pr_impl(self, pr_info: PullRequestInfo) -> MergeResult:
        raise NotImplementedError

    async def _org_approval_rulesets(self, org: str) -> list[dict[str, Any]] | None:
        raise NotImplementedError

    def _pr_status(self, message: str, *, level: str = "info") -> None:
        raise NotImplementedError

    async def _predict_merge_outcome(
        self, owner: str, repo: str, pr_number: int, merge_method: str
    ) -> tuple[bool, str]:
        raise NotImplementedError

    async def _recheck_pr_before_retry(
        self, owner: str, repo: str, pr_info: PullRequestInfo, attempt: int
    ) -> bool | None:
        raise NotImplementedError

    def _record_rebase(self) -> None:
        raise NotImplementedError

    def _record_retrigger(self) -> None:
        raise NotImplementedError

    def _record_terminal_outcome(
        self, pr_info: PullRequestInfo, status: MergeStatus
    ) -> None:
        raise NotImplementedError

    async def _refresh_pr_mergeability(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> None:
        raise NotImplementedError

    async def _refresh_pr_mergeable(
        self, owner: str, repo: str, pr_info: PullRequestInfo, pr_key: str
    ) -> None:
        raise NotImplementedError

    async def _report_merge_failure(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        result: MergeResult,
        failure_reason: str,
    ) -> MergeResult:
        raise NotImplementedError

    async def _request_dependabot_rebase(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        raise NotImplementedError

    @staticmethod
    def _rules_require_approval(rules: Any) -> bool:
        raise NotImplementedError

    def _ruleset_condition_applies(
        self, conditions: Any, repo: str, branch: str
    ) -> bool | None:
        raise NotImplementedError

    def _simulate_preview_merge(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> None:
        raise NotImplementedError

    async def _stop_for_undispatched_workflows(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        error_text: str,
        deadline: float | None,
    ) -> bool:
        raise NotImplementedError

    async def _submit_gerrit_change(
        self,
        mapping: GitHub2GerritMapping,
        pr_info: PullRequestInfo,
        repo_owner: str,
        repo_name: str,
    ) -> bool:
        raise NotImplementedError

    def _track_pr_state(self, pr_info: PullRequestInfo, state: str | None) -> None:
        raise NotImplementedError

    async def _trigger_dependabot_recreate(
        self, pr_info: PullRequestInfo
    ) -> PullRequestInfo | None:
        raise NotImplementedError

    async def _trigger_stale_precommit_ci(self, pr_info: PullRequestInfo) -> bool:
        raise NotImplementedError

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
    ) -> tuple[bool, bool]:
        raise NotImplementedError

    async def _wait_for_recreated_pr_checks(
        self, repo_owner: str, repo_name: str, new_number: int, pr_data: dict[str, Any]
    ) -> PullRequestInfo | None:
        raise NotImplementedError

    async def _wait_for_required_workflows_and_retry(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        raise NotImplementedError

    async def _wait_status_ticker(self) -> None:
        raise NotImplementedError
