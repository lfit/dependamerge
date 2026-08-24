# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The merge outcome vocabulary.

``MergeStatus`` enumerates every terminal state a pull request can
reach and ``MergeResult`` records one, together with the two helpers
that read GitHub's merge responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..models import PullRequestInfo


class MergeStatus(Enum):
    """Status of a PR merge operation."""

    PENDING = "pending"
    APPROVING = "approving"
    APPROVED = "approved"
    MERGING = "merging"
    MERGED = "merged"
    AUTO_MERGE_PENDING = "auto_merge_pending"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    # Terminal: the PR was closed without merging (dependabot decided
    # the update is no longer needed after sibling merges, the PR was
    # superseded, or a human closed it mid-run).  Distinct from FAILED
    # because there is nothing for the operator to follow up on.
    CLOSED = "closed"


@dataclass
class MergeResult:
    """Result of a PR merge operation."""

    pr_info: PullRequestInfo
    status: MergeStatus
    error: str | None = None
    # Non-fatal note attached to a *successful* (or otherwise non-error)
    # outcome — e.g. a preview MERGED result for a PR that is behind its
    # base branch and would be rebased first. Kept separate from ``error``
    # so a MERGED status never carries a contradictory error message.
    warning: str | None = None
    attempts: int = 0
    duration: float = 0.0


def _merged_from_payload(payload: dict[str, Any]) -> bool | None:
    """Whether a PR REST payload says the PR merged.

    Prefers the explicit ``merged`` boolean.  A trimmed or proxied
    payload may omit it, so fall back to ``merged_at`` --- the full REST
    object always carries that key (an ISO timestamp when merged,
    ``null`` for closed-but-unmerged).  Returns ``None`` only when
    neither is usable, so an ambiguous payload is never mistaken for a
    definite "not merged".

    Shared so every caller derives merged-ness identically; the same
    rule is applied by ``_recheck_pr_before_retry`` and
    ``_fetch_pr_state_now``.
    """
    merged_field = payload.get("merged")
    if isinstance(merged_field, bool):
        return merged_field
    if "merged_at" not in payload:
        return None
    merged_at = payload.get("merged_at")
    if merged_at is not None and not isinstance(merged_at, str):
        return None
    return merged_at is not None


def _merge_already_in_progress(error_msg: str) -> bool:
    """Whether a 405 body says GitHub is already merging this PR.

    GitHub's wording is ``Merge already in progress``.  Matched
    case-insensitively and without punctuation assumptions so a minor
    upstream rewording does not silently reinstate the old
    fail-after-6-seconds behaviour.
    """
    lowered = error_msg.lower()
    return "merge already in progress" in lowered or (
        "already in progress" in lowered and "merge" in lowered
    )
