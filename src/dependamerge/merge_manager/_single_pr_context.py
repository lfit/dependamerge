# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The state one pull request carries through the merge sequence.

``_merge_single_pr_impl`` runs as a chain of steps, and the later
ones consult what the earlier ones learned: the block-reason analysis
is shared by three separate gates, and the Step 6 merge path needs to
know whether Step 5 rebased and whether Step 5.5 waited.  ``_MergeFlow``
is that shared scratchpad, passed by reference so a step can record its
findings without widening every signature.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import PullRequestInfo
from ._types import MergeResult


@dataclass
class _MergeFlow:
    """One pull request's progress through ``_merge_single_pr_impl``."""

    pr_info: PullRequestInfo
    repo_owner: str
    repo_name: str
    result: MergeResult

    # Outcome of the single ``analyze_block_reason()`` call made near
    # the top of the flow.  ``blocked_analysis_ok`` records whether the
    # analysis itself succeeded, which is distinct from it returning
    # ``None`` (inconclusive): a *failure* means "do not wait".
    blocked_reason: str | None = None
    blocked_analysis_ok: bool = False

    # Whether Step 5 decided a rebase was required, whether this PR was
    # already rebased (and had auto-merge armed) before Step 5.5, and
    # whether Step 5.5 entered its wait loop.  Step 6 consults all
    # three to decide how much recovery budget is left.
    needs_rebase: bool = False
    already_rebased: bool = False
    should_wait: bool = False

    @property
    def pr_key(self) -> str:
        """The ``owner/repo#number`` key used by the per-PR state sets."""
        return f"{self.repo_owner}/{self.repo_name}#{self.pr_info.number}"
