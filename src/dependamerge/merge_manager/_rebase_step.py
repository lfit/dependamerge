# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Step 5: the block-reason analysis, and the rebase it may require.

Decides whether GitHub genuinely requires the pull request to be
brought up to date before it will merge, and dispatches the rebase when
it does.  The analysis of *why* a pull request is blocked is produced
here too, because it is the input to that decision and is reused by the
two later phases that would otherwise pay for it again.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from .. import rebase
from ._base import _MergeManagerBase
from ._merge_state import _Attempt, _RebaseOutcome
from ._models import (
    MergeResult,
    MergeStatus,
)


class _RebaseStepMixin(_MergeManagerBase):
    """Step 5: the block-reason analysis, and the rebase it may require."""

    async def _rebase_if_required(self, attempt: _Attempt) -> _RebaseOutcome:
        """Analyse the block reason and rebase when GitHub demands it.

        The analysis and the decision are returned together so the wait
        pre-check and the auto-merge skip gate can reuse them: each
        ``analyze_block_reason`` call costs roughly four API requests,
        and all three consumers want the same snapshot of the same head.

        ``_RebaseOutcome.result`` is populated only when the rebase
        itself failed, in which case the attempt is over.
        """
        blocked_reason, blocked_analysis_ok = await self._analyze_block_reason_once(
            attempt
        )
        needs_rebase = await self._rebase_is_required(
            attempt, blocked_reason, blocked_analysis_ok
        )
        failure = await self._perform_rebase(attempt) if needs_rebase else None
        return _RebaseOutcome(
            result=failure,
            needs_rebase=needs_rebase,
            blocked_reason=blocked_reason,
            blocked_analysis_ok=blocked_analysis_ok,
        )

    async def _analyze_block_reason_once(
        self, attempt: _Attempt
    ) -> tuple[str | None, bool]:
        """Analyse once, per attempt, why a pull request is blocked.

        Step 5's staleness probe, Step 5.5's wait pre-check and Step 6's
        auto-merge skip gate all consult the same analysis, and each
        call costs ~4 API requests (reviews, comments, check runs,
        combined status).  Fetching it once here and passing the result
        through collapses the previous two to three calls per blocked
        pull request into one.

        Returns ``(reason, analysis_ok)``.  ``analysis_ok`` records
        whether the analysis itself succeeded: the Step 5.5 pre-check
        treats an analysis *failure* (as opposed to a ``None`` /
        inconclusive reason) as "do not wait".
        """
        pr_info = attempt.pr_info
        blocked_reason: str | None = None
        blocked_analysis_ok = False
        if (
            pr_info.mergeable_state == "blocked"
            and not self.preview_mode
            and self._github_client is not None
        ):
            try:
                blocked_reason = await self._github_client.analyze_block_reason(
                    attempt.owner,
                    attempt.repo,
                    pr_info.number,
                    pr_info.head_sha,
                    base_branch=pr_info.base_branch,
                )
                blocked_analysis_ok = True
            except Exception as exc:
                self.log.debug(
                    "analyze_block_reason failed for %s/%s#%s: %s",
                    attempt.owner,
                    attempt.repo,
                    pr_info.number,
                    exc,
                )
        return blocked_reason, blocked_analysis_ok

    async def _rebase_is_required(
        self,
        attempt: _Attempt,
        blocked_reason: str | None,
        blocked_analysis_ok: bool,
    ) -> bool:
        """Report whether GitHub actually requires a rebase first.

        Rebases are expensive: they restart every required CI check
        (minutes of wall-clock time per pull request), and same-repo
        batches compound the cost because every sibling merge moves the
        base again.  So Step 5 rebases **only when GitHub actually
        requires it**:

        - ``behind`` alone is NOT enough.  GitHub happily merges a
          behind-but-green pull request unless the branch's protection
          enforces the *strict* status-check policy ("require branches
          to be up to date before merging"), so we probe that policy
          (cached per repo/branch) and otherwise send the pull request
          straight to the merge attempt.  Should a merge still be
          rejected for staleness, the reactive path in
          ``_handle_merge_failure`` recovers.
        - ``blocked`` masks ``behind`` (``mergeable_state`` is a single
          value).  A required check that *failed* on a head demonstrably
          behind base was judged against pre-rebase content — e.g. an
          org-required workflow audit that the base branch has since
          fixed — and only a rebase re-runs it against the current base.
          Pending checks are excluded: they resolve on their own, no
          rebase required.
        """
        pr_info = attempt.pr_info
        needs_rebase = False
        if (
            self.fix_out_of_date
            and not self.preview_mode
            and self._github_client is not None
        ):
            if pr_info.mergeable_state == "behind":
                needs_rebase = await self._behind_pr_requires_rebase(
                    pr_info, attempt.owner, attempt.repo
                )
            elif pr_info.mergeable_state == "blocked" and blocked_analysis_ok:
                needs_rebase = await self._blocked_pr_needs_rebase(
                    pr_info, attempt.owner, attempt.repo, blocked_reason
                )
        return needs_rebase

    async def _perform_rebase(self, attempt: _Attempt) -> MergeResult | None:
        """Run the rebase, returning the failed result or ``None``.

        The rebase itself is dispatched to the dedicated ``rebase``
        module so the macro-vs-local-vs-REST decision tree, the
        local-git workflow, and the post-rebase polling loop all live in
        one place where they can be tested in isolation.  Assembling the
        context it needs is the only work done here, which is why that
        assembly is kept out of the decision above.
        """
        rebase_ctx = rebase.RebaseContext(
            github_client=self._github_client,
            token=self.token,
            rebase_local=self.rebase_local,
            preview_mode=self.preview_mode,
            merge_recheck_interval=self._merge_recheck_interval,
            merge_poll_max_attempts=self._merge_poll_max_attempts,
            log=self.log,
            console=self._console,
            rebased_prs=self._rebased_prs,
            enable_auto_merge=self._enable_auto_merge_with_approval,
            track_pr_state=self._track_pr_state,
            record_rebase=self._record_rebase,
            request_dependabot_rebase=self._request_dependabot_rebase,
        )
        outcome = await rebase.perform_step5_rebase(
            ctx=rebase_ctx,
            pr_info=attempt.pr_info,
            owner=attempt.owner,
            repo=attempt.repo,
        )
        if outcome.failed:
            attempt.result.status = MergeStatus.FAILED
            attempt.result.error = outcome.error_message
            return attempt.result
        return None
