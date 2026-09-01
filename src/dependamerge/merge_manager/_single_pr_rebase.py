# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Step 5: refresh a stale branch, but only when GitHub insists.

Rebases are expensive: they restart every required CI check (minutes
of wall-clock time per PR), and same-repo batches compound the cost
because every sibling merge moves the base again.  So Step 5 rebases
**only when GitHub actually requires it**:

- ``behind`` alone is NOT enough.  GitHub happily merges a
  behind-but-green PR unless the branch's protection enforces the
  *strict* status-check policy ("require branches to be up to date
  before merging"), so we probe that policy (cached per repo/branch)
  and otherwise send the PR straight to the merge attempt.  Should a
  merge still be rejected for staleness, the reactive path in
  ``_handle_merge_failure`` recovers.
- ``blocked`` masks ``behind`` (``mergeable_state`` is a single
  value).  A required check that *failed* on a head demonstrably
  behind base was judged against pre-rebase content --- e.g. an
  org-required workflow audit that the base branch has since fixed
  --- and only a rebase re-runs it against the current base.  Pending
  checks are excluded: they resolve on their own, no rebase required.

The rebase itself is dispatched to the dedicated ``rebase`` module so
the macro-vs-local-vs-REST decision tree, the local-git workflow, and
the post-rebase polling loop all live in one place where they can be
tested in isolation.
"""

from __future__ import annotations

from .. import rebase
from ._base import _MergeManagerBase
from ._single_pr_context import _MergeFlow
from ._types import MergeResult, MergeStatus


class _SinglePrRebaseMixin(_MergeManagerBase):
    """The Step 5 staleness probe and its dispatch to ``rebase``."""

    async def _pr_needs_rebase(self, flow: _MergeFlow) -> bool:
        """Whether GitHub requires this PR's branch refreshed first."""
        pr_info = flow.pr_info
        if not (
            self.fix_out_of_date
            and not self.preview_mode
            and self._github_client is not None
        ):
            return False

        if pr_info.mergeable_state == "behind":
            return await self._behind_pr_requires_rebase(
                pr_info, flow.repo_owner, flow.repo_name
            )
        if pr_info.mergeable_state == "blocked" and flow.blocked_analysis_ok:
            return await self._blocked_pr_needs_rebase(
                pr_info, flow.repo_owner, flow.repo_name, flow.blocked_reason
            )
        return False

    async def _run_step5_rebase(self, flow: _MergeFlow) -> MergeResult | None:
        """Rebase the PR, returning a terminal result on failure."""
        rebase_ctx = rebase.RebaseContext(
            github_client=self._github_client,
            token=self.token,
            host=self.host,
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
            pr_info=flow.pr_info,
            owner=flow.repo_owner,
            repo=flow.repo_name,
        )
        if outcome.failed:
            flow.result.status = MergeStatus.FAILED
            flow.result.error = outcome.error_message
            return flow.result
        return None
