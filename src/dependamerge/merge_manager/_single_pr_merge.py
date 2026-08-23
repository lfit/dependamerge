# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Step 6: the merge itself, or the decision to leave it to auto-merge.

If auto-merge is enabled and the PR is in a state that auto-merge can
rescue (blocked, behind, or unstable), the manual merge attempt is
skipped --- GitHub will merge automatically once branch protection is
satisfied.

We accept any ``mergeable`` value (including ``False``) here for the
same reason Step 5.5 does: ``mergeable=False`` from the API can mean
"definitely failing", "still computing", or "a non-required check
failed".  Letting auto-merge decide whether the failing thing actually
blocks merge is more accurate than us treating ``False`` as terminal.

For ``blocked`` PRs we still consult ``analyze_block_reason()`` to
weed out cases auto-merge cannot resolve (missing approvals,
code-owner reviews, etc.).  For ``behind`` and ``unstable`` we accept
directly: ``behind`` resolves once GitHub re-runs checks against the
rebased commit, and ``unstable`` means a non-required check failed
(which doesn't actually block auto-merge).

Do NOT skip when:
  * force_level == "all" --- force semantics intentionally proceed
    despite the blocked state and must not be overridden by
    auto-merge.
  * the block reason (for ``blocked`` PRs) is something other than
    pending required checks (e.g. missing approvals).
"""

from __future__ import annotations

from ._single_pr_context import _MergeFlow
from ._single_pr_outcome import _SinglePrOutcomeMixin
from ._types import MergeResult, MergeStatus


class _SinglePrMergeMixin(_SinglePrOutcomeMixin):
    """The Step 6 dispatch and the auto-merge deferral that precedes it."""

    async def _perform_merge(self, flow: _MergeFlow) -> MergeResult | None:
        """Merge the PR (or defer to auto-merge) and record the outcome."""
        pr_info = flow.pr_info
        flow.result.status = MergeStatus.MERGING
        if self.preview_mode:
            self._simulate_preview_merge(pr_info, flow.result)
            return None

        if self.progress_tracker:
            self.progress_tracker.update_operation(
                f"Merging PR {pr_info.number} in {pr_info.repository_full_name}"
            )

        merged: bool | None
        if await self._auto_merge_will_handle(flow):
            merged = None  # Sentinel: auto-merge pending
        else:
            merged, early = await self._attempt_direct_merge(flow)
            if early is not None:
                return early
            merged = await self._retry_after_review_or_workflows(flow, merged)

        if merged is None:
            self._record_auto_merge_pending(flow)
        elif merged:
            self._record_merged(flow)
        else:
            return await self._handle_failed_merge(flow)
        return None

    async def _auto_merge_will_handle(self, flow: _MergeFlow) -> bool:
        """Whether to leave this PR for GitHub's auto-merge to complete."""
        pr_info = flow.pr_info
        if not (
            flow.pr_key in self._auto_merge_enabled
            and pr_info.mergeable_state in ("blocked", "behind", "unstable")
            and self.force_level != "all"
        ):
            return False

        if pr_info.mergeable_state in ("behind", "unstable"):
            # ``behind``: still behind after rebase polling timed out;
            # auto-merge will pick the PR up once GitHub finishes rebase
            # + required checks.
            # ``unstable``: a non-required check failed but required
            # checks may still be pending or passing; auto-merge will
            # fire when branch protection allows.
            return True
        return await self._blocked_pr_defers_to_auto_merge(flow)

    async def _blocked_pr_defers_to_auto_merge(self, flow: _MergeFlow) -> bool:
        """Whether a ``blocked`` PR is blocked only on pending checks."""
        pr_info = flow.pr_info
        block_reason: str | None = None
        analysis_fresh = False
        if (
            flow.blocked_analysis_ok
            and not flow.should_wait
            and not flow.already_rebased
        ):
            # Nothing has changed since the analysis at the top of the
            # flow (no Step 5 rebase, no Step 5.5 wait), so reuse it
            # instead of re-spending its ~4 API calls.
            block_reason = flow.blocked_reason
            analysis_fresh = True
        if not analysis_fresh and self._github_client is not None:
            try:
                block_reason = await self._github_client.analyze_block_reason(
                    flow.repo_owner,
                    flow.repo_name,
                    pr_info.number,
                    pr_info.head_sha,
                    base_branch=pr_info.base_branch,
                )
            except Exception as exc:
                self.log.debug(
                    "analyze_block_reason failed for %s: %s",
                    flow.pr_key,
                    exc,
                )
        # Treat any pending-checks-style block reason as auto-merge
        # eligible. We previously matched only the literal substring
        # "pending required check", but analyze_block_reason returns a
        # range of phrasings (e.g. "required status check… pending")
        # depending on which checks are outstanding. The same predicate
        # is used by the Step 5.5 pre-check.
        return self._block_reason_indicates_pending_checks(block_reason)

    async def _attempt_direct_merge(
        self, flow: _MergeFlow
    ) -> tuple[bool, MergeResult | None]:
        """Dispatch the merge, routing a freshly-dirty PR to recovery.

        Returns ``(merged, early_result)``; ``early_result`` is a
        terminal result from conflict recovery, in which case ``merged``
        is meaningless.
        """
        pr_info = flow.pr_info
        # Proactive approval: some organizations mandate an approving
        # review via a repository ruleset before *any* merge is allowed.
        # When this PR's base branch is governed that way a merge-first
        # attempt is guaranteed to be rejected, so approve the current
        # head up-front and skip the doomed round-trip plus reactive
        # recovery.  See the helper for details; on any lookup failure
        # it no-ops and the reactive approve-on-demand path still covers
        # us.
        await self._approve_if_review_mandated(
            pr_info, flow.repo_owner, flow.repo_name, flow.pr_key
        )
        merged, dirty_before_dispatch = await self._merge_under_dispatch_lock(flow)

        # Conflict recovery runs *outside* the dispatch lock so the
        # rebase wait never blocks sibling merges.
        if dirty_before_dispatch:
            return False, await self._handle_merge_conflict(
                pr_info, flow.repo_owner, flow.repo_name, flow.result
            )
        # A PR can also turn ``dirty`` *during* our own merge window (a
        # sibling merged between the pre-dispatch check and the merge
        # call).  The post-failure refresh — off the lock, with its
        # recompute poll — catches that so a freshly-dirty PR is never
        # reported as a generic merge failure.
        if not merged and self._repo_scoped:
            await self._refresh_pr_mergeability(
                pr_info, flow.repo_owner, flow.repo_name
            )
            if pr_info.mergeable_state == "dirty":
                return False, await self._handle_merge_conflict(
                    pr_info, flow.repo_owner, flow.repo_name, flow.result
                )
        return merged, None

    async def _merge_under_dispatch_lock(self, flow: _MergeFlow) -> tuple[bool, bool]:
        """Merge with the per-repo dispatch lock held.

        Serialises the actual merge dispatch per repo so back-to-back
        merges don't race GitHub's branch protection propagation.
        Workers on the same repo queue here; workers on different repos
        run in parallel.  See ``_get_merge_dispatch_lock``.

        Returns ``(merged, dirty_before_dispatch)``.
        """
        pr_info = flow.pr_info
        dispatch_lock = await self._get_merge_dispatch_lock(
            flow.repo_owner, flow.repo_name
        )
        dirty_before_dispatch = False
        async with dispatch_lock:
            # Re-read live merge state *before* dispatch — a single GET,
            # no recompute poll.  In a repo-scoped batch an earlier
            # sibling merge can turn this PR ``dirty`` between the
            # one-shot fetch and dispatch (the classic shared
            # ``uv.lock`` conflict); routing it straight to conflict
            # recovery avoids dispatching a doomed merge that 405s and
            # then churns the retry loop against the stale ``clean``
            # snapshot.  We keep this to a single GET (not the polling
            # ``_refresh_pr_mergeability``) because the dispatch lock is
            # the one point serialised *and* ordered after a sibling
            # merge, so polling GitHub's recompute window here would
            # serialise the whole repo batch.
            if self._repo_scoped:
                dirty_before_dispatch = await self._is_pr_dirty_now(
                    pr_info, flow.repo_owner, flow.repo_name
                )
            if dirty_before_dispatch:
                merged = False
            else:
                merged = await self._merge_pr_with_retry(
                    pr_info, flow.repo_owner, flow.repo_name
                )
        return merged, dirty_before_dispatch

    async def _retry_after_review_or_workflows(
        self, flow: _MergeFlow, merged: bool
    ) -> bool:
        """Retry a rejected merge when the rejection is recoverable."""
        pr_info = flow.pr_info
        # Approve-on-demand (merge-path trigger): if the direct merge was
        # rejected solely because our review is missing, approve the
        # current head and retry once.  Returns True only if the retry
        # merged the PR; any other failure is left for the classifier.
        if not merged:
            if await self._approve_and_retry_if_review_required(
                pr_info, flow.repo_owner, flow.repo_name
            ):
                merged = True

        # A 405 "Required workflows … are not satisfied" rejection means
        # ruleset-required workflows are still *executing* on the head
        # commit — a pending condition, unlike the terminal "… failed"
        # variant.  Wait for them to finish and retry rather than failing
        # a PR whose checks are still green.  Skipped when Step 5/5.5
        # already spent this PR's wait budget (the workflows are then
        # genuinely slower than merge_timeout) and under --force=all
        # (force semantics bypass waits).
        if (
            not merged
            and not flow.should_wait
            and not flow.already_rebased
            and self.force_level != "all"
        ):
            last_merge_exc = self._last_merge_exception.get(flow.pr_key)
            if (
                last_merge_exc is not None
                and self._merge_error_indicates_pending_workflows(str(last_merge_exc))
            ):
                merged = await self._wait_for_required_workflows_and_retry(
                    pr_info, flow.repo_owner, flow.repo_name
                )
        return merged
