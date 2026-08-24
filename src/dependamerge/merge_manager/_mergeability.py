# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Predicates over a pull request's mergeability.

GitHub's ``mergeable_state`` is coarse and its block reasons are
prose, so these helpers interpret both to decide whether a PR is
merely waiting, genuinely blocked, or simply behind its base.
"""

from __future__ import annotations

from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _MergeabilityMixin(_MergeManagerBase):
    """Interpreting mergeable state and block reasons."""

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

    async def _behind_pr_requires_rebase(
        self,
        pr_info: PullRequestInfo,
        repo_owner: str,
        repo_name: str,
    ) -> bool:
        """Return True when GitHub would refuse to merge this behind PR.

        A ``behind`` PR only needs a branch refresh before merging
        when the base branch's protection enforces the *strict*
        status-check policy ("require branches to be up to date
        before merging").  Everywhere else GitHub merges a
        behind-but-green PR directly, and a proactive rebase would
        just restart CI (minutes per PR), invalidate sibling merges'
        check runs, and — on signature-requiring repos — drag in the
        signing workflow.  The policy probe is cached per
        repo/branch, so same-repo batches pay for it once.

        Any probe failure counts as "not required": the merge attempt
        itself is the authoritative test, and the reactive path in
        ``_handle_merge_failure`` recovers if GitHub rejects it.
        """
        if self._github_client is None:
            return False
        try:
            strict = await self._github_client.requires_strict_status_checks(
                repo_owner,
                repo_name,
                pr_info.base_branch or "main",
            )
        except Exception as exc:
            self.log.debug(
                "requires_strict_status_checks failed for %s/%s: %s",
                repo_owner,
                repo_name,
                exc,
            )
            return False
        # Strict ``is True`` comparison so AsyncMock defaults in tests
        # (truthy Mock objects) never route PRs into the rebase path.
        if strict is not True:
            return False
        self._pr_status(
            f"\U0001f504 Stale head: {pr_info.html_url} "
            "[behind base; branch protection requires up-to-date "
            "heads — refreshing before merge]",
            level="debug",
        )
        return True

    async def _blocked_pr_needs_rebase(
        self,
        pr_info: PullRequestInfo,
        repo_owner: str,
        repo_name: str,
        block_reason: str | None,
    ) -> bool:
        """Decide whether a ``blocked`` PR is really stale-and-fixable.

        Implements the staleness probe behind Step 5's
        blocked-masks-behind handling: a ``blocked`` PR is treated
        like ``behind`` when **all** hold:

        1. Its block reason is check-related (failing or missing
           checks) — the only class of blockage a branch refresh can
           cure, because the refresh re-runs checks against the
           current base.
        2. The block reason is NOT merely *pending* checks: those
           resolve on their own once the checks finish, so a rebase
           would restart them for nothing (and, mid-batch, restart
           them repeatedly as sibling merges advance the base).
        3. The compare API confirms the head is at least one commit
           behind the base branch.  ``None`` (comparison failed)
           counts as "not behind": a rebase is a write action and a
           CI-time expense, so it must rest on positive evidence.

        The classification gates run first so the extra compare call
        is only spent on plausible candidates.

        Args:
            pr_info: The pull request under evaluation.
            repo_owner: Base repository owner.
            repo_name: Base repository name.
            block_reason: The result of ``analyze_block_reason()``
                computed once at the top of ``_merge_single_pr`` (may
                be ``None`` when the analysis returned nothing).

        Returns:
            True when the PR should take the Step 5 rebase path.
        """
        if self._github_client is None:
            return False
        if not self._block_reason_indicates_check_blockage(block_reason):
            return False
        if self._block_reason_indicates_pending_checks(block_reason):
            return False

        behind_by = await self._github_client.get_behind_by(
            repo_owner,
            repo_name,
            pr_info.base_branch or "main",
            pr_info.head_sha,
        )
        if behind_by is None or behind_by <= 0:
            return False

        pr_info.behind_by = behind_by
        self._pr_status(
            f"\U0001f504 Stale head: {pr_info.html_url} "
            f"[blocked ({block_reason}); {behind_by} commit(s) behind "
            f"base — rebasing to re-run checks]",
            level="debug",
        )
        return True

    def _is_pr_mergeable(self, pr_info: PullRequestInfo) -> bool:
        """Check whether a PR is worth attempting to merge.

        This returns ``True`` for any state where dependamerge can
        plausibly make progress — either by approving + merging,
        rebasing, or enabling auto-merge and waiting (Step 5.5).
        We deliberately err on the side of letting Step 5.5 see the
        PR: it has finer-grained logic (block-reason analysis,
        merge-timeout-bounded waits) than this gate, so a False here
        denies a PR the chance to be auto-merge-rescued.

        Returns False only for states where no amount of waiting,
        approving, or auto-merging can help:

        - ``dirty``: real merge conflict; the branch must be
          rebased by a human (or by ``--fix``).
        - ``draft``: GitHub blocks merging draft PRs by design.

        For all other states (``blocked``, ``behind``, ``unstable``,
        empty/``"unknown"``) we return True regardless of the
        ``mergeable`` boolean. ``mergeable=False`` from the API can
        mean "definitely failing", but it can also mean "GitHub is
        still computing" or "a non-required check failed" — the
        downstream Step 5.5 + Step 6 gates have the context to make
        the right call.
        """
        # Hard skips: states where merging is impossible regardless
        # of mergeable value or downstream rescue logic.
        if pr_info.mergeable_state == "dirty":
            self.log.debug(
                "🛑 Skipping PR %s/%s#%s: merge conflict (dirty)",
                pr_info.repository_full_name.split("/", 1)[0]
                if "/" in pr_info.repository_full_name
                else pr_info.repository_full_name,
                pr_info.repository_full_name.split("/", 1)[-1],
                pr_info.number,
            )
            return False
        if pr_info.mergeable_state == "draft":
            self.log.debug(
                "⏭️ Skipping draft PR %s#%s",
                pr_info.repository_full_name,
                pr_info.number,
            )
            return False

        # Everything else — ``blocked``, ``behind``, ``unstable``,
        # ``clean``, empty/None state, plus any ``mergeable`` value
        # — reaches the merge flow. Step 5.5 will route
        # not-yet-merge-ready cases to AUTO_MERGE_PENDING after
        # consulting block-reason analysis and bounded by
        # ``merge_timeout``, which is a much friendlier outcome than
        # a hard skip from here.
        self.log.debug(
            "✅ PR %s#%s eligible for merge flow (mergeable=%s, state=%s)",
            pr_info.repository_full_name,
            pr_info.number,
            pr_info.mergeable,
            pr_info.mergeable_state,
        )
        return True

    def _has_blocking_reviews(self, pr_info: PullRequestInfo) -> bool:
        """
        Check if a PR has reviews that would block automatic approval.

        Args:
            pr_info: Pull request information

        Returns:
            True if there are blocking reviews (changes requested), False otherwise
        """
        for review in pr_info.reviews:
            if review.state == "CHANGES_REQUESTED":
                self.log.info(
                    f"⚠️ PR {pr_info.number} has changes requested by {review.user} - will not override human feedback"
                )
                return True
        return False
