# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The concrete asynchronous merge manager.

Assembles the mixins that hold the implementation, and owns the
constructor, the async context manager, and the state every mixin
reads.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from pathlib import Path
from typing import Any

from rich.console import Console

import dependamerge.merge_manager as _pkg

from ..copilot_handler import CopilotCommentHandler
from ..github_service import GitHubService
from ..pr_poller import PullRequestStatePoller
from ..progress_tracker import MergeProgressTracker
from ._approvals import _ApprovalsMixin
from ._auto_merge_arm import _AutoMergeArmMixin
from ._auto_merge_wait import _AutoMergeWaitMixin
from ._branch_approval import _BranchApprovalMixin
from ._comments import _CommentsMixin
from ._conflict_helpers import _ConflictHelpersMixin
from ._conflicts import _ConflictsMixin
from ._constants import (
    DEFAULT_MERGE_RECHECK_INTERVAL,
    DEFAULT_MERGE_TIMEOUT,
)
from ._dependabot_recreate import _DependabotRecreateMixin
from ._failure_reporting import _FailureReportingMixin
from ._gerrit_host import _GerritHostMixin
from ._gerrit_submit import _GerritSubmitMixin
from ._lifecycle import _LifecycleMixin
from ._merge_flow import _MergeFlowMixin
from ._merge_method import _MergeMethodMixin
from ._mergeability import _MergeabilityMixin
from ._models import MergeResult
from ._orchestration import _OrchestrationMixin
from ._org_settings import _OrgSettingsMixin
from ._pr_probe import _PrProbeMixin
from ._pr_state import _PrStateMixin
from ._precommit_ci import _PrecommitCiMixin
from ._prediction import _PredictionMixin
from ._rebase_checks import _RebaseChecksMixin
from ._recreated_pr_wait import _RecreatedPrWaitMixin
from ._required_workflows import _RequiredWorkflowsMixin
from ._requirements import _RequirementsMixin
from ._results import _ResultsMixin
from ._retry import _RetryMixin
from ._review_required import _ReviewRequiredMixin
from ._rulesets import _RulesetsMixin
from ._semantic_title import _SemanticTitleMixin
from ._single_pr import _SinglePrMixin
from ._status_ticker import _StatusTickerMixin
from ._stuck_checks import _StuckChecksMixin
from ._wait_budget import _WaitBudgetMixin
from ._workflow_dispatch import _WorkflowDispatchMixin


class AsyncMergeManager(
    _LifecycleMixin,
    _OrchestrationMixin,
    _ResultsMixin,
    _StatusTickerMixin,
    _GerritSubmitMixin,
    _GerritHostMixin,
    _SinglePrMixin,
    _MergeFlowMixin,
    _MergeabilityMixin,
    _RebaseChecksMixin,
    _CommentsMixin,
    _AutoMergeArmMixin,
    _ReviewRequiredMixin,
    _RequirementsMixin,
    _SemanticTitleMixin,
    _PrecommitCiMixin,
    _StuckChecksMixin,
    _DependabotRecreateMixin,
    _RecreatedPrWaitMixin,
    _ApprovalsMixin,
    _PrStateMixin,
    _RetryMixin,
    _WorkflowDispatchMixin,
    _RequiredWorkflowsMixin,
    _PrProbeMixin,
    _WaitBudgetMixin,
    _AutoMergeWaitMixin,
    _ConflictHelpersMixin,
    _ConflictsMixin,
    _FailureReportingMixin,
    _MergeMethodMixin,
    _OrgSettingsMixin,
    _RulesetsMixin,
    _BranchApprovalMixin,
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

    def __init__(
        self,
        token: str,
        merge_method: str = "merge",
        max_retries: int = 2,
        concurrency: int = 5,
        fix_out_of_date: bool = False,
        merge_timeout: float = DEFAULT_MERGE_TIMEOUT,
        progress_tracker: MergeProgressTracker | None = None,
        preview_mode: bool = False,
        dismiss_copilot: bool = False,
        force_level: str = "code-owners",
        github2gerrit_mode: str = "submit",
        no_netrc: bool = False,
        netrc_file: Path | None = None,
        rebase_local: bool = True,
        repo_scoped: bool = False,
        max_wait: float | None = None,
        fix_semantic_title: bool = True,
    ):
        self.token = token
        self.default_merge_method = merge_method
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.fix_out_of_date = fix_out_of_date
        self.fix_semantic_title = fix_semantic_title
        self.progress_tracker = progress_tracker
        self.preview_mode = preview_mode
        self.dismiss_copilot = dismiss_copilot
        self.force_level = force_level
        self.github2gerrit_mode = github2gerrit_mode
        self.no_netrc = no_netrc
        self.netrc_file = netrc_file
        # When True (the default), Step 5's rebase path uses a local
        # ``git`` clone + rebase + force-push-with-lease workflow
        # for PRs whose verification status would otherwise be lost
        # by the GitHub REST ``update-branch`` endpoint. The local
        # workflow inherits the user's ``~/.gitconfig`` and so
        # respects ``commit.gpgsign`` / ``gpg.format`` /
        # ``user.signingkey`` automatically. Set to False to force
        # the legacy REST-only path (simpler but loses signature
        # verification on signed branches).
        self.rebase_local = rebase_local
        # When True, a worker refreshes its PR's live merge state just
        # before dispatching, because a sibling merge can make a PR
        # ``dirty`` / ``behind`` between the up-front fetch and this
        # worker's dispatch (see ``_refresh_pr_mergeability``).  Enabled
        # both for single-repository batches and for owner-wide striped
        # runs, where each repository's PRs are serialised so an earlier
        # merge can invalidate a later sibling.  Left False only for
        # similar-PR runs spread across unrelated repositories, where our
        # own merges do not invalidate the snapshot.
        self._repo_scoped = repo_scoped
        # Owner-wide global wait ceiling (seconds), or ``None`` for
        # repository / similar-PR runs which keep the legacy per-PR
        # ``merge_timeout`` behaviour with no overall cap.  Semantics:
        #   * ``None``  — no global ceiling (per-PR ``merge_timeout``
        #                 governs each wait independently).
        #   * ``> 0``   — a wall-clock ceiling for the whole run; every
        #                 per-PR wait deadline is clamped to it, so the
        #                 run cannot block past this bound.  Anything
        #                 still in flight when it elapses keeps auto-merge
        #                 armed and is reported AUTO_MERGE_PENDING.
        #   * ``0``     — fire-and-forget: never block.  Approve, arm
        #                 auto-merge, report AUTO_MERGE_PENDING, move on.
        # ``_run_deadline`` (the resolved monotonic ceiling) and
        # ``_no_wait`` (the ``0`` case) are set when a run starts.
        self._max_wait = max_wait
        self._run_deadline: float | None = None
        self._no_wait: bool = max_wait is not None and max_wait <= 0
        self.log = logging.getLogger(__name__)

        # Centralised merge-operation timing
        # Coerce merge_timeout to float and validate, guarding against Typer
        # OptionInfo objects that leak through when the CLI function is called
        # directly (e.g. from tests) without the Typer argument parser.
        try:
            _mt = float(merge_timeout)
            if not math.isfinite(_mt) or _mt <= 0:
                raise ValueError(f"out of range: {_mt}")
            self._merge_timeout = _mt
        except (TypeError, ValueError):
            self.log.warning(
                "Invalid merge_timeout=%r; falling back to default of %.0f seconds",
                merge_timeout,
                DEFAULT_MERGE_TIMEOUT,
            )
            self._merge_timeout = DEFAULT_MERGE_TIMEOUT
        # Clamp the per-iteration sleep so a small ``merge_timeout``
        # (< DEFAULT_MERGE_RECHECK_INTERVAL) does not over-sleep and
        # blow past the user-specified total timeout. For typical
        # values (>= 10s), this is a no-op and keeps the default
        # 10s polling cadence.
        self._merge_recheck_interval = min(
            DEFAULT_MERGE_RECHECK_INTERVAL, self._merge_timeout
        )
        # Use math.ceil so the effective poll window is at least
        # the configured ``merge_timeout`` — plain ``int()`` would
        # truncate (e.g. 301/10 -> 30 attempts -> only 300s).
        self._merge_poll_max_attempts = max(
            1, math.ceil(self._merge_timeout / self._merge_recheck_interval)
        )

        # Track merge operations
        self._merge_semaphore = asyncio.Semaphore(concurrency)
        self._results: list[MergeResult] = []
        self._github_client: _pkg.GitHubAsync | None = None
        self._pr_poller: PullRequestStatePoller | None = None
        # PRs whose title has already been aligned this run.  One attempt
        # only: a semantic check that keeps failing after the fix must
        # report rather than drive a rewrite loop.
        self._semantic_title_aligned: set[str] = set()
        # Observed check-resolution latency per repository, used to give
        # sibling PRs a head start (see ``_wait_head_start``).
        self._repo_wait_seconds: dict[str, list[float]] = {}
        self._github_service: GitHubService | None = None
        self._copilot_handler: CopilotCommentHandler | None = None
        # Reuse the progress tracker's Rich Console (when one is
        # provided) so per-PR ✅/❌ lines emitted during a merge run
        # interleave cleanly with the Live progress display.  Using a
        # separate Console() instance causes Rich's Live re-draw to
        # garble or eat those messages because the two consoles share
        # the terminal but coordinate independently.
        tracker_console = getattr(progress_tracker, "console", None)
        self._console = tracker_console if tracker_console is not None else Console()

        # Track merge methods per repository
        self._pr_merge_methods: dict[str, str] = {}

        # Cache for organization-level settings to avoid repeated API calls
        # Key: org name, Value: org settings dict (or None on failure)
        self._org_settings_cache: dict[str, dict[str, Any] | None] = {}
        self._org_settings_locks: dict[str, asyncio.Lock] = {}
        self._org_settings_locks_lock = asyncio.Lock()

        # Cache for "does this branch mandate an approving review before
        # any merge" detection, keyed by "owner/repo@branch".  The answer
        # is fixed for the lifetime of a run (rulesets don't change
        # mid-merge), so the resolved verdict is reused for every PR
        # targeting the same repo+branch.
        self._branch_approval_cache: dict[str, bool] = {}
        self._branch_approval_locks: dict[str, asyncio.Lock] = {}
        self._branch_approval_locks_lock = asyncio.Lock()

        # Cache for the organization-level approval requirement, enumerated
        # once per org from its repository rulesets.  Value is the list of
        # approval-mandating rulesets (each ``{"name", "conditions"}``),
        # ``[]`` when the org mandates none, or ``None`` when enumeration
        # failed (e.g. the token cannot read org rulesets) so callers know
        # to consult the authoritative per-repo endpoint instead.
        self._org_approval_cache: dict[str, list[dict[str, Any]] | None] = {}
        self._org_approval_locks: dict[str, asyncio.Lock] = {}
        self._org_approval_locks_lock = asyncio.Lock()

        # Track last merge exception per PR for better error reporting
        self._last_merge_exception: dict[str, Exception] = {}
        # The head SHA each stored exception was raised against.  A
        # rejection is evidence about the commit that was rejected, so
        # a force-push in between invalidates it; see
        # ``_wait_for_required_workflows_and_retry``.
        self._last_merge_exception_head: dict[str, str] = {}

        # Track PRs that were just approved (for post-approval merge retry)
        self._recently_approved: set[str] = set()

        # Track repositories where the token has already failed a
        # permission check during this run.  Subsequent PRs in the
        # same repository are short-circuited with a clean skip
        # message rather than triggering another round-trip to
        # GitHub that will fail identically and emit another
        # screenful of guidance.
        self._permission_failed_repos: set[str] = set()

        # Track PRs where auto-merge has been enabled so that
        # post-timeout merge attempts can be skipped gracefully.
        self._auto_merge_enabled: set[str] = set()

        # Track PRs that have already gone through Step 5's
        # rebase + poll path so Step 5.5 can skip them and avoid
        # doubling the configured ``merge_timeout``. Set after
        # Step 5 completes its wait, regardless of whether the
        # final state is ``clean``, ``blocked``, or ``behind``.
        self._rebased_prs: set[str] = set()

        # Track PRs currently waiting for required checks to complete.
        # Maps ``pr_key`` -> deadline (monotonic seconds) so the
        # parallel merge ticker can render an aggregate countdown
        # without poking inside individual worker tasks.
        self._waiting_prs: dict[str, float] = {}
        self._waiting_lock = asyncio.Lock()

        # Per-repo locks that serialise the actual ``merge_pull_request``
        # API call.  Multiple workers can run in parallel through approve,
        # rebase polling, and Step 5.5's auto-merge wait loop — only the
        # final dispatch is serialised, and only between PRs that target
        # the same repository.  This avoids the head-of-line blocking we
        # used to get from forcing ``concurrency=1`` for repo-scoped
        # runs (where a single PR parked in the wait loop could block
        # every other PR in the batch for the full ``merge_timeout``)
        # while still preventing back-to-back merges on the same repo
        # from racing GitHub's branch-protection propagation.
        self._merge_dispatch_locks: dict[str, asyncio.Lock] = {}
        self._merge_dispatch_locks_lock = asyncio.Lock()

        # Delay (seconds) after submitting a new approval before attempting merge.
        # GitHub needs time to propagate the approval to branch-protection evaluation.
        default_post_approval_delay = 3.0
        env_post_approval_delay = os.getenv(
            "DEPENDAMERGE_POST_APPROVAL_DELAY",
            str(default_post_approval_delay),
        )
        try:
            parsed_delay = float(env_post_approval_delay)
            if not math.isfinite(parsed_delay) or parsed_delay < 0:
                raise ValueError(f"out of range: {parsed_delay}")
            self._post_approval_delay = parsed_delay
        except ValueError:
            self.log.warning(
                "Invalid DEPENDAMERGE_POST_APPROVAL_DELAY=%r; "
                "falling back to default of %.1f seconds",
                env_post_approval_delay,
                default_post_approval_delay,
            )
            self._post_approval_delay = default_post_approval_delay
