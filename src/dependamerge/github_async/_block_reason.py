# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Why a pull request cannot merge.

Turns the mergeable state, review decision, check results and
protection configuration of a pull request into one human-readable
reason, memoised briefly because the merge pipeline asks repeatedly.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

# ``_now`` stays an attribute of the package rather than a name bound
# here: it was a module-level attribute of ``dependamerge.github_async``
# before the split, and callers substitute it there.
import dependamerge.github_async as _pkg

from ._base import _GitHubAsyncBase
from ._block_reason_probes import (
    _check_block_reason,
    _collect_check_signals,
    _collect_required_check_gaps,
    _collect_review_signals,
    _count_unresolved_copilot_comments,
    _guarded_block_reason,
    _pending_block_reason,
    _resolve_block_reason_base_branch,
    _review_block_reason,
)


class _BlockReasonMixin(_GitHubAsyncBase):
    """Block-reason analysis and its cache for ``GitHubAsync``."""

    def clear_block_reasons(self) -> None:
        """Forget every memoised block reason.

        For run boundaries: the merge manager supports reuse, and a
        non-confirmed invocation runs the whole batch as a preview
        first.  Expiry alone is not enough --- a second run can begin
        inside the window, and checks complete while a head SHA stays
        constant, so the earlier run's answer can be both cached and
        wrong.
        """
        self._block_reason_cache.clear()

    def invalidate_block_reason(self, owner: str, repo: str, number: int) -> None:
        """Forget any memoised block reason for a PR.

        Called after operations that change *why* a PR is blocked ---
        approving it, or attempting a merge.  Without this, the memo
        outlives the state it describes: approving a PR that reported
        "requires approval", then failing the retry for a different
        reason, would replay the stale approval message and could send
        the failure down the wrong recovery path.
        """
        for key in [
            k for k in self._block_reason_cache if k[:3] == (owner, repo, number)
        ]:
            self._block_reason_cache.pop(key, None)

    # How long an ``analyze_block_reason`` result stays usable.
    #
    # Deliberately short.  The obvious design --- cache per
    # ``(repo, head_sha)`` for the run --- is unsafe: the reason a PR is
    # blocked changes as checks complete, while its head SHA does not,
    # and callers re-analyse after waiting precisely to observe that
    # change.  A run-lifetime cache would answer "still blocked" forever.
    #
    # The waste worth removing is the *burst*: a single evaluation pass
    # calls this several times in quick succession with nothing changing
    # in between, at five or more requests each.  A few seconds collapses
    # that burst and has long expired by the time any wait loop
    # re-checks.
    _BLOCK_REASON_TTL_SECONDS = 10.0

    async def analyze_block_reason(
        self,
        owner: str,
        repo: str,
        number: int,
        head_sha: str,
        base_branch: str | None = None,
    ) -> str:
        """
        Analyze why a PR is blocked and return appropriate status.

        This is the async version that should be used from async contexts.

        ``base_branch`` lets callers that already know the PR's base ref
        (e.g. the merge pipeline, which carries it on ``PullRequestInfo``)
        skip the PR-detail fetch this method otherwise performs just to
        read ``base.ref`` — one request saved per invocation, and this
        method runs several times per blocked PR.

        Results are memoised briefly; see
        ``_BLOCK_REASON_TTL_SECONDS`` for why the window is short.  The
        base branch is part of the memo key because it selects which
        protection and required-check configuration is consulted: a
        retargeted PR, or two callers supplying different bases, must
        not share an answer computed against the other's branch.
        """
        cache_key = (owner, repo, number, head_sha, base_branch)
        cached = self._block_reason_cache.get(cache_key)
        if cached is not None:
            cached_at, cached_reason = cached
            if _pkg._now() - cached_at < self._BLOCK_REASON_TTL_SECONDS:
                return cached_reason

        reason = await self._analyze_block_reason_uncached(
            owner, repo, number, head_sha, base_branch
        )
        self._block_reason_cache[cache_key] = (_pkg._now(), reason)
        return reason

    async def _analyze_block_reason_uncached(
        self,
        owner: str,
        repo: str,
        number: int,
        head_sha: str,
        base_branch: str | None = None,
    ) -> str:
        """Compute the block reason, ignoring the memo."""
        reviews = await _collect_review_signals(self, owner, repo, number)
        unresolved_copilot_comments = await _count_unresolved_copilot_comments(
            self, owner, repo, number
        )
        checks = await _collect_check_signals(self, owner, repo, head_sha)
        base_branch = await _resolve_block_reason_base_branch(
            self, owner, repo, number, base_branch
        )
        (
            missing_required_checks,
            pending_required_checks,
        ) = await _collect_required_check_gaps(
            self, owner, repo, number, base_branch, checks
        )

        # Prioritize blocking conditions by specificity
        # Most specific blockers first
        reason = _check_block_reason(
            checks.failing, missing_required_checks, pending_required_checks
        )
        if reason is not None:
            return reason

        reason = _review_block_reason(
            reviews.human_changes_requested,
            reviews.unresolved_copilot_reviews,
            unresolved_copilot_comments,
        )
        if reason is not None:
            return reason

        reason = _pending_block_reason(checks.pending)
        if reason is not None:
            return reason

        if not reviews.approved:
            return "Blocked by branch protection (requires approval)"

        return await _guarded_block_reason(self, owner, repo, base_branch)
