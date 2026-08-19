# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Step 6: leaving the merge to auto-merge, or dispatching it ourselves.

Decides whether GitHub's auto-merge will complete the merge without us,
and otherwise performs the merge under the per-repository dispatch lock
and runs the two recoveries that apply to a rejection: approving a head
whose only missing requirement is our review, and waiting out
ruleset-required workflows that are still executing.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from ._base import _MergeManagerBase
from ._merge_state import (
    _Attempt,
    _DispatchOutcome,
    _RebaseOutcome,
    _WaitOutcome,
)


class _DirectMergeMixin(_MergeManagerBase):
    """Step 6: auto-merge, or a merge dispatched by us."""

    async def _dispatch_merge(
        self, attempt: _Attempt, rebased: _RebaseOutcome, waited: _WaitOutcome
    ) -> _DispatchOutcome:
        """Merge the pull request, or leave it to auto-merge.

        Returns ``merged=None`` when auto-merge is armed and will
        complete the merge server-side, so the attempt is neither a
        success nor a failure; and ``conflicted=True`` when the pull
        request turned ``dirty`` around the dispatch and belongs in
        conflict recovery rather than the failure classifier.
        """
        pr_info = attempt.pr_info
        if self.progress_tracker:
            self.progress_tracker.update_operation(
                f"Merging PR {pr_info.number} in {pr_info.repository_full_name}"
            )

        if await self._auto_merge_handles_pr(attempt, rebased, waited):
            # Sentinel: auto-merge pending
            return _DispatchOutcome(merged=None, conflicted=False)
        return await self._merge_directly(attempt, waited)

    async def _auto_merge_handles_pr(
        self, attempt: _Attempt, rebased: _RebaseOutcome, waited: _WaitOutcome
    ) -> bool:
        """Report whether auto-merge will complete this merge for us.

        If auto-merge is enabled and the pull request is in a state that
        auto-merge can rescue (blocked, behind, or unstable), the manual
        merge attempt is skipped — GitHub will merge automatically once
        branch protection is satisfied.

        Any ``mergeable`` value (including ``False``) is accepted here
        for the same reason the wait decision accepts it:
        ``mergeable=False`` from the API can mean "definitely failing",
        "still computing", or "a non-required check failed".  Letting
        auto-merge decide whether the failing thing actually blocks the
        merge is more accurate than treating ``False`` as terminal here.

        For ``blocked`` pull requests ``analyze_block_reason()`` is still
        consulted, to weed out cases auto-merge cannot resolve (missing
        approvals, code-owner reviews, etc.).  ``behind`` and
        ``unstable`` are accepted directly: ``behind`` resolves once
        GitHub re-runs checks against the rebased commit, and
        ``unstable`` means a non-required check failed, which does not
        actually block auto-merge.

        Auto-merge is NOT left to handle the pull request when
        ``force_level == "all"`` — force semantics intentionally proceed
        despite the blocked state and must not be overridden — or when
        the block reason is something other than pending required
        checks.
        """
        pr_info = attempt.pr_info
        auto_merge_pending_checks = False
        if (
            attempt.pr_key in self._auto_merge_enabled
            and pr_info.mergeable_state in ("blocked", "behind", "unstable")
            and self.force_level != "all"
        ):
            if pr_info.mergeable_state in ("behind", "unstable"):
                # ``behind``: still behind after rebase polling
                # timed out; auto-merge will pick the PR up
                # once GitHub finishes rebase + required
                # checks.
                # ``unstable``: a non-required check failed but
                # required checks may still be pending or
                # passing; auto-merge will fire when branch
                # protection allows.
                auto_merge_pending_checks = True
            else:
                block_reason: str | None = None
                analysis_fresh = False
                if (
                    rebased.blocked_analysis_ok
                    and not waited.should_wait
                    and not waited.already_rebased
                ):
                    # Nothing has changed since the analysis
                    # at the top of the flow (no Step 5
                    # rebase, no Step 5.5 wait), so reuse it
                    # instead of re-spending its ~4 API calls.
                    block_reason = rebased.blocked_reason
                    analysis_fresh = True
                if not analysis_fresh and self._github_client is not None:
                    try:
                        block_reason = await self._github_client.analyze_block_reason(
                            attempt.owner,
                            attempt.repo,
                            pr_info.number,
                            pr_info.head_sha,
                            base_branch=pr_info.base_branch,
                        )
                    except Exception as exc:
                        self.log.debug(
                            "analyze_block_reason failed for %s: %s",
                            attempt.pr_key,
                            exc,
                        )
                # Treat any pending-checks-style block reason
                # as auto-merge eligible. We previously matched
                # only the literal substring "pending required
                # check", but analyze_block_reason returns a
                # range of phrasings (e.g. "required status
                # check… pending") depending on which
                # checks are outstanding. The same predicate is
                # used by the Step 5.5 pre-check.
                auto_merge_pending_checks = self._block_reason_indicates_pending_checks(
                    block_reason
                )
        return auto_merge_pending_checks

    async def _merge_directly(
        self, attempt: _Attempt, waited: _WaitOutcome
    ) -> _DispatchOutcome:
        """Dispatch the merge and run the recoveries a rejection allows.

        Kept apart from the auto-merge decision above because this is
        the only part of the flow that takes the per-repository dispatch
        lock, and because the two conflict exits it can reach must be
        handled outside that lock by the orchestrator.
        """
        pr_info = attempt.pr_info
        repo_owner = attempt.owner
        repo_name = attempt.repo
        pr_key = attempt.pr_key

        # Proactive approval: some organizations mandate an
        # approving review via a repository ruleset before
        # *any* merge is allowed.  When this PR's base branch
        # is governed that way a merge-first attempt is
        # guaranteed to be rejected, so approve the current
        # head up-front and skip the doomed round-trip plus
        # reactive recovery.  See the helper for details; on
        # any lookup failure it no-ops and the reactive
        # approve-on-demand path still covers us.
        await self._approve_if_review_mandated(pr_info, repo_owner, repo_name, pr_key)

        # Serialise the actual merge dispatch per repo so
        # back-to-back merges don't race GitHub's branch
        # protection propagation.  Workers on the same
        # repo queue here; workers on different repos run
        # in parallel.  See ``_get_merge_dispatch_lock``.
        dispatch_lock = await self._get_merge_dispatch_lock(repo_owner, repo_name)
        dirty_before_dispatch = False
        async with dispatch_lock:
            # Re-read live merge state *before* dispatch — a
            # single GET, no recompute poll.  In a
            # repo-scoped batch an earlier sibling merge can
            # turn this PR ``dirty`` between the one-shot
            # fetch and dispatch (the classic shared
            # ``uv.lock`` conflict); routing it straight to
            # conflict recovery avoids dispatching a doomed
            # merge that 405s and then churns the retry loop
            # against the stale ``clean`` snapshot.  We keep
            # this to a single GET (not the polling
            # ``_refresh_pr_mergeability``) because the
            # dispatch lock is the one point serialised *and*
            # ordered after a sibling merge, so polling
            # GitHub's recompute window here would serialise
            # the whole repo batch.
            if self._repo_scoped:
                dirty_before_dispatch = await self._is_pr_dirty_now(
                    pr_info, repo_owner, repo_name
                )
            if dirty_before_dispatch:
                merged = False
            else:
                merged = await self._merge_pr_with_retry(pr_info, repo_owner, repo_name)

        # Conflict recovery runs *outside* the dispatch lock
        # so the rebase wait never blocks sibling merges.
        if dirty_before_dispatch:
            return _DispatchOutcome(merged=merged, conflicted=True)

        # A PR can also turn ``dirty`` *during* our own merge
        # window (a sibling merged between the pre-dispatch
        # check and the merge call).  The post-failure
        # refresh — off the lock, with its recompute poll —
        # catches that so a freshly-dirty PR is never
        # reported as a generic merge failure.
        if not merged and self._repo_scoped:
            await self._refresh_pr_mergeability(pr_info, repo_owner, repo_name)
            if pr_info.mergeable_state == "dirty":
                return _DispatchOutcome(merged=merged, conflicted=True)

        # Approve-on-demand (merge-path trigger): if the
        # direct merge was rejected solely because our review
        # is missing, approve the current head and retry once.
        # Returns True only if the retry merged the PR; any
        # other failure is left for the classifier below.
        if not merged:
            if await self._approve_and_retry_if_review_required(
                pr_info, repo_owner, repo_name
            ):
                merged = True

        # A 405 "Required workflows … are not satisfied"
        # rejection means ruleset-required workflows are
        # still *executing* on the head commit — a pending
        # condition, unlike the terminal "… failed"
        # variant.  Wait for them to finish and retry
        # rather than failing a PR whose checks are still
        # green.  Skipped when Step 5/5.5 already spent
        # this PR's wait budget (the workflows are then
        # genuinely slower than merge_timeout) and under
        # --force=all (force semantics bypass waits).
        if (
            not merged
            and not waited.should_wait
            and not waited.already_rebased
            and self.force_level != "all"
        ):
            last_merge_exc = self._last_merge_exception.get(pr_key)
            if (
                last_merge_exc is not None
                and self._merge_error_indicates_pending_workflows(str(last_merge_exc))
            ):
                merged = await self._wait_for_required_workflows_and_retry(
                    pr_info, repo_owner, repo_name
                )
        return _DispatchOutcome(merged=merged, conflicted=False)
