# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The state one merge attempt threads between its phases.

``_merge_single_pr_impl`` runs a single pull request through a sequence
of phases that live in sibling mixins.  Each phase reads what the
earlier ones established and hands on what the later ones need, and
these records carry that state explicitly: the manager instance is
shared by every concurrent pull-request worker, so an attribute there
would be a race between unrelated pull requests.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import PullRequestInfo
from ._models import MergeResult


@dataclass(frozen=True)
class _Attempt:
    """The pull request under attempt and the record it reports into.

    ``pr_key`` is the ``owner/repo#number`` key the manager's per-run
    tracking sets use.  It is derived once so the phases that consult
    ``_auto_merge_enabled``, ``_rebased_prs`` and
    ``_last_merge_exception`` cannot drift apart in how they spell it.

    ``result`` is the single mutable object in here: every phase records
    its outcome on the same ``MergeResult`` the orchestrator ultimately
    returns, which is why the frozen container can be passed around
    freely.
    """

    pr_info: PullRequestInfo
    owner: str
    repo: str
    pr_key: str
    result: MergeResult


@dataclass(frozen=True)
class _RebaseOutcome:
    """What the rebase step decided, and the analysis it decided on.

    ``result`` is non-``None`` only when the rebase failed, in which case
    the attempt is over.

    The block-reason analysis travels with the decision because it costs
    roughly four API requests and three consumers want the same
    snapshot: the rebase decision here, the wait pre-check, and the
    auto-merge skip gate.  ``blocked_analysis_ok`` distinguishes an
    analysis that failed from one that returned no conclusive reason;
    the wait pre-check treats those differently.
    """

    result: MergeResult | None
    needs_rebase: bool
    blocked_reason: str | None
    blocked_analysis_ok: bool


@dataclass(frozen=True)
class _WaitOutcome:
    """Whether the attempt waited for required checks, and what happened.

    ``result`` is non-``None`` only when the pull request closed during
    the wait.

    ``should_wait`` and ``already_rebased`` are carried forward because
    two later gates — the auto-merge skip and the required-workflow
    retry — must see the state the wait decision was made against, not
    the state the wait left behind.
    """

    result: MergeResult | None
    should_wait: bool
    already_rebased: bool


@dataclass(frozen=True)
class _DispatchOutcome:
    """How the merge dispatch ended.

    ``merged`` is tri-state: ``None`` means auto-merge has been left to
    complete the merge server-side, so the attempt is neither a success
    nor a failure.  ``conflicted`` means the pull request turned
    ``dirty`` around the dispatch and belongs in conflict recovery
    rather than the failure classifier.
    """

    merged: bool | None
    conflicted: bool
