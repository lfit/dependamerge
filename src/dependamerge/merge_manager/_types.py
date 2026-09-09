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
    # Terminal: the run did not merge the PR, and nothing about it needs
    # a human either.  Whatever GitHub refused the merge over has since
    # stopped holding --- a required check finished, a rebase landed, a
    # base branch settled --- so re-running is expected to merge it.
    #
    # Deliberately not ``PENDING``, which is the *initial* non-terminal
    # state every result starts in, nor ``AUTO_MERGE_PENDING``, where
    # GitHub finishes the merge server-side without another run.  The
    # distinction this carries is the one the counts were missing:
    # "could not merge" and "had not finished yet" call for different
    # responses, and reporting both as FAILED sent operators looking for
    # a cause on PRs that had none.
    UNSETTLED = "unsettled"


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
    # Whether ``error`` records GitHub *refusing the merge*, as opposed
    # to the run itself failing --- an unhandled exception, a rebase that
    # did not complete, a token without the rights.
    #
    # Only a refusal's stated cause is a snapshot that can expire, so
    # only a refusal may be withdrawn on a later reading of the PR.  A
    # run-side failure says nothing about mergeability, and rewriting one
    # because the PR happens to look clean would bury the actionable
    # error and tell the operator to re-run something that will fail the
    # same way.  Defaults to False so a new failure path is excluded
    # until it opts in, rather than silently inheriting the correction.
    merge_refused: bool = False


class RecreateCause(Enum):
    """Why a Dependabot recreate was requested.

    The two causes are unrelated, and the gates that apply to one do not
    apply to the other.  ``UNSIGNED`` is the original case: branch
    protection requires signed commits and this PR carries unverified
    ones, so the signature checks decide whether a recreate would help.
    ``STUCK_CHECK`` is a *timing* problem --- a required check that never
    reported --- where signatures are irrelevant.
    """

    UNSIGNED = "unsigned"
    STUCK_CHECK = "stuck_check"


class RecreateOutcome(Enum):
    """What became of a recreate request.

    ``NONE`` covers only the paths where **no replacement was acted
    on**: the recreate was not applicable, was not triggered, or no
    replacement was ever found.

    ``MERGED`` is a success --- auto-merge is armed on the replacement
    before the wait begins, so it can complete between polls.

    ``ABANDONED`` is a *resolved* failure: the replacement is closed or
    conflicted, so there is nothing left to wait for and nothing to
    merge.

    ``PENDING`` is the replacement we found, armed and then stopped
    waiting on --- because the ceiling arrived, the poll budget ran
    out, or ``--max-wait 0`` asked us not to block at all.  It is
    deliberately distinct from ``NONE``: the replacement is real, open
    and expected to merge on its own, so reporting the original as a
    failure would under-report a success exactly as the closed/merged
    case did.  ``_confirm_failure`` cannot rescue that, since it
    rechecks only the original PR.
    """

    NONE = "none"
    READY = "ready"
    MERGED = "merged"
    ABANDONED = "abandoned"
    PENDING = "pending"


@dataclass(frozen=True)
class RecreateResult:
    """The terminal state of a recreate, and the PR it refers to.

    Carrying the outcome separately from the PR is what stops a caller
    merging a replacement that has already merged, or waiting on one
    that has closed.  ``pr_info`` is populated for ``READY``,
    ``MERGED`` and ``PENDING``; it may be present for ``ABANDONED`` to
    name the PR in a message, and is always ``None`` for ``NONE``.
    """

    outcome: RecreateOutcome
    pr_info: PullRequestInfo | None = None

    @classmethod
    def none(cls) -> RecreateResult:
        """No replacement was acted on."""
        return cls(RecreateOutcome.NONE)


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
