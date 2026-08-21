# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Interpretation of GitHub's mergeable state and block reasons.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._models import (
    MergeResult,
    MergeStatus,
)


class _MergeabilityMixin(_MergeManagerBase):
    """Interpretation of GitHub's mergeable state and block reasons."""

    async def _handle_not_mergeable_pr(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> MergeResult:
        """Classify and report a PR that failed the mergeability gate.

        Extracted from ``_merge_single_pr`` to keep that method's
        branch count manageable. Produces a detailed skip/block
        reason, sets ``result`` accordingly, and returns it.
        """
        # Get detailed status for a more informative skip message
        # Use async method to avoid event loop conflicts
        repo_owner, repo_name = pr_info.repository_full_name.split("/")

        # Check if blocked to get more detailed status
        if pr_info.mergeable_state == "blocked" and self._github_client:
            try:
                detailed_status = await self._github_client.analyze_block_reason(
                    repo_owner,
                    repo_name,
                    pr_info.number,
                    pr_info.head_sha,
                    base_branch=pr_info.base_branch,
                )
            except Exception:
                detailed_status = f"Blocked (state: {pr_info.mergeable_state})"
        else:
            # For non-blocked states, provide basic status
            if pr_info.mergeable_state == "dirty":
                detailed_status = "Merge conflicts"
            elif pr_info.mergeable_state == "behind":
                detailed_status = "Rebase required (out of date)"
            elif pr_info.mergeable_state == "draft":
                detailed_status = "Draft PR"
            else:
                detailed_status = f"Not mergeable (state: {pr_info.mergeable_state})"

        # Use the detailed status as the skip reason, with fallback
        if detailed_status and detailed_status != "Status unclear":
            skip_reason = detailed_status.lower()
        else:
            # Fallback to basic mapping if detailed status is unclear
            # aislop-ignore-next-line ai-slop/python-repetitive-dispatch -- branches set distinct reasons with sub-conditions, not a uniform table
            if pr_info.mergeable_state == "dirty":
                skip_reason = "merge conflicts"
            elif pr_info.mergeable_state == "behind":
                skip_reason = "behind"
            elif pr_info.mergeable_state == "blocked":
                if pr_info.mergeable is True:
                    skip_reason = "blocked, requires review"
                else:
                    skip_reason = "blocked by failing checks"
            elif pr_info.mergeable_state == "unstable":
                skip_reason = "unstable"
            elif pr_info.mergeable is False:
                skip_reason = "not mergeable"
            else:
                skip_reason = "unknown"

        # Determine if this is truly blocked (unmergeable) or just skipped
        if pr_info.mergeable_state == "dirty" or (
            pr_info.mergeable_state == "behind" and pr_info.mergeable is False
        ):
            result.status = MergeStatus.BLOCKED
            icon = "🛑"
            status = "Blocked"
        else:
            result.status = MergeStatus.SKIPPED
            icon = "⏭️"
            status = "Skipped"

        self._pr_status(
            f"{icon} {status}: {pr_info.html_url} [{skip_reason}]",
            level="info",
        )

        result.error = f"PR is not mergeable (state: {pr_info.mergeable_state}, mergeable: {pr_info.mergeable})"

        # For the result error (used in CLI output), use the detailed status if it's more informative
        if detailed_status and detailed_status != "Status unclear":
            result.error = detailed_status

        return result

    def _simulate_preview_merge(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> None:
        """Simulate the Step 6 merge outcome for preview mode.

        Mutates ``result`` in place.  No console output: preview
        progress is conveyed by the Rich tracker counters (with
        preview-accurate labels such as "Mergeable") and the per-PR
        outcomes/reasons are reported in the end-of-run summary, so
        per-PR lines here go to the log only.
        """
        if pr_info.mergeable_state == "behind" and not self.fix_out_of_date:
            result.status = MergeStatus.SKIPPED
            result.error = "PR is behind base branch and --no-fix option is set"
            self._pr_status(
                f"\u23ed\ufe0f Skipped: {pr_info.html_url} [behind, rebase disabled]",
                level="debug",
            )
        elif pr_info.mergeable_state == "behind" and self.fix_out_of_date:
            # Behind PRs merge directly unless branch protection
            # requires up-to-date heads, in which case the real run
            # refreshes the branch first — either way the PR counts
            # as mergeable.
            result.status = MergeStatus.MERGED
            # Use ``warning`` (not ``error``) so the MERGED result
            # does not carry a contradictory error message.
            result.warning = "behind base branch"
            self._pr_status(
                f"\u2611\ufe0f Approve/merge: {pr_info.html_url} [behind base branch]",
                level="debug",
            )
        elif pr_info.mergeable_state == "dirty":
            result.status = MergeStatus.BLOCKED
            result.error = "PR has merge conflicts"
            self._pr_status(
                f"\U0001f6d1 Blocked: {pr_info.html_url} [merge conflicts]",
                level="debug",
            )
        elif pr_info.mergeable is False and pr_info.mergeable_state == "blocked":
            result.status = MergeStatus.BLOCKED
            result.error = "PR blocked by failing checks"
            self._pr_status(
                f"\U0001f6d1 Blocked: {pr_info.html_url} [blocked by failing checks]",
                level="debug",
            )
        else:
            # Simulate successful merge in preview mode
            result.status = MergeStatus.MERGED
            self._pr_status(
                f"\u2611\ufe0f Approve/merge: {pr_info.html_url}",
                level="debug",
            )

    @staticmethod
    def _block_reason_indicates_pending_checks(
        block_reason: str | None,
    ) -> bool:
        """Return True if a block reason indicates pending required checks.

        Both Step 5.5 (whether to enter the wait loop) and Step 6
        (whether to defer to auto-merge instead of attempting a
        manual merge) need to recognise the same set of phrasings
        returned by ``GitHubAsync.analyze_block_reason()``.
        Centralising the predicate here keeps the two call sites
        consistent so a new phrasing only has to be added once.

        The predicate matches **only** wording that explicitly
        signals a check is still in progress / waiting to start.
        It deliberately excludes:

        - ``Blocked by failing check: …`` — the check has run and
          reported a non-pending failure; auto-merge will not
          rescue this.
        - ``Blocked by missing required status: …`` — the check
          has not been registered against the commit at all;
          auto-merge will not retry it on its own.
        - any reason where ``failing`` or ``missing`` appears
          before a service name (defensive: covers future GitHub
          phrasing changes that include both keywords).

        Args:
            block_reason: The string returned by
                ``analyze_block_reason()``, or ``None`` if the
                analysis failed or returned nothing.

        Returns:
            True when the reason mentions pending required checks
            in any of the recognised phrasings; False otherwise
            (including when ``block_reason`` is ``None``).
        """
        if block_reason is None:
            return False
        reason_lower = block_reason.lower()

        # Defensive negative gate: never classify a reason as
        # 'pending' if it explicitly says the check has failed or
        # is missing. This guards the bare-substring matches
        # below against future phrasings that combine both terms
        # (e.g. "failing check (pending retry): pre-commit.ci").
        if "failing check" in reason_lower:
            return False
        if "missing required status" in reason_lower:
            return False
        if "missing required check" in reason_lower:
            return False

        return (
            "pending required check" in reason_lower
            or "pending check" in reason_lower
            or ("required" in reason_lower and "pending" in reason_lower)
            or "waiting for status" in reason_lower
            or "queued" in reason_lower
        )

    @staticmethod
    def _block_reason_indicates_check_blockage(
        block_reason: str | None,
    ) -> bool:
        """Return True if a block reason concerns status checks at all.

        Broader sibling of
        :meth:`_block_reason_indicates_pending_checks`: matches any
        ``analyze_block_reason()`` phrasing about checks — failing,
        missing, or pending — while rejecting reasons a rebase cannot
        influence (missing approvals, requested changes, unresolved
        Copilot feedback, opaque ruleset blocks).

        Step 5 uses this to decide whether a ``blocked`` PR is worth
        probing for staleness: refreshing the branch re-runs checks
        against the current base, so only check-related blockage can
        possibly be cured by a rebase.

        Args:
            block_reason: The string returned by
                ``analyze_block_reason()``, or ``None`` if the
                analysis failed or returned nothing.

        Returns:
            True when the reason mentions failing, missing, or
            pending checks; False otherwise (including ``None``).
        """
        if block_reason is None:
            return False
        reason_lower = block_reason.lower()
        return (
            "failing check" in reason_lower
            or "missing required status" in reason_lower
            or "missing required check" in reason_lower
            or "pending required check" in reason_lower
            or ("required" in reason_lower and "pending" in reason_lower)
            or "waiting for status" in reason_lower
            or "queued" in reason_lower
        )
