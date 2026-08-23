# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Asking dependabot to recreate a PR that cannot merge as it stands.

Before giving up on a failed merge, check whether this is a dependabot
PR that failed for a reason recreation resolves.  Two triggers are
considered:

  1. Branch-protection failures (the original unsigned-commit case).
  2. A *required* verification check that has been stuck (queued /
     in_progress / pending) for longer than
     ``STUCK_CHECK_THRESHOLD_SECONDS`` on a PR that itself was created
     and last updated that long ago.  Required checks (DCO, lint,
     build, etc.) normally start reporting in seconds; when one stalls
     indefinitely, the only reliable recovery for dependabot PRs is to
     recreate the PR so the checks fire again on a fresh head SHA.
     pre-commit.ci is excluded here --- it has its own dedicated
     recovery via ``_trigger_stale_precommit_ci`` (which posts
     ``pre-commit.ci run``).
"""

from __future__ import annotations

from ..bot_identity import is_dependabot
from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._single_pr_context import _MergeFlow
from ._types import MergeStatus


class _SinglePrRecreateMixin(_MergeManagerBase):
    """The dependabot recreate trigger and the merge of the new PR."""

    async def _maybe_recreate_dependabot_pr(
        self, flow: _MergeFlow, failure_reason: str
    ) -> PullRequestInfo | None:
        """Ask dependabot to recreate the PR, when that can help."""
        pr_info = flow.pr_info
        if not (is_dependabot(pr_info.author) and not self.preview_mode):
            return None
        if not await self._should_request_recreate(flow, failure_reason):
            return None

        self._track_pr_state(pr_info, "recreating")
        try:
            return await self._trigger_dependabot_recreate(pr_info)
        finally:
            self._track_pr_state(pr_info, None)

    async def _should_request_recreate(
        self, flow: _MergeFlow, failure_reason: str
    ) -> bool:
        """Whether the failure is one a fresh head SHA would clear."""
        pr_info = flow.pr_info
        reason_lower = failure_reason.lower()
        # Branch protection *and* repository rulesets can both block a
        # dependabot PR for reasons recreation resolves (most commonly
        # an unsigned-commit / required-signature rule).  Treat them
        # alike so the recreate path is not silently skipped on repos
        # that have migrated from classic protection to rulesets.
        if "branch protection" in reason_lower or "ruleset" in reason_lower:
            return True

        try:
            (
                is_stuck,
                stuck_check,
                stuck_age,
            ) = await self._detect_stuck_required_check(pr_info)
        except Exception as exc:
            self.log.debug(
                "_detect_stuck_required_check failed for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            is_stuck = False
            stuck_check = None
            stuck_age = 0.0
        if is_stuck:
            self._pr_status(
                f"⏳ Stuck required check detected: {pr_info.html_url} "
                f"[{stuck_check} pending for "
                f"{stuck_age:.0f}s, requesting recreate]",
                level="info",
            )
            return True
        return False

    async def _merge_recreated_pr(
        self, flow: _MergeFlow, recreated_pr: PullRequestInfo
    ) -> None:
        """Approve and merge the PR dependabot created in place of ours."""
        result = flow.result
        # We have a fresh PR — approve and merge it
        new_owner, new_repo = recreated_pr.repository_full_name.split("/", 1)
        await self._approve_pr(new_owner, new_repo, recreated_pr.number)

        new_merged = await self._dispatch_recreated_merge(
            new_owner, new_repo, recreated_pr
        )

        if new_merged:
            result.status = MergeStatus.MERGED
            result.pr_info = recreated_pr
            self._pr_status(
                f"✅ Merged (recreated): {recreated_pr.html_url}",
                level="debug",
            )
        else:
            result.status = MergeStatus.FAILED
            result.error = (
                f"Dependabot recreated PR #{recreated_pr.number} but merge still failed"
            )
            self.log.error(
                "Failed to merge recreated PR %s#%s",
                recreated_pr.repository_full_name,
                recreated_pr.number,
            )
            self._pr_status(
                f"❌ Failed: {recreated_pr.html_url} [recreated PR merge failed]",
                level="error",
            )

    async def _dispatch_recreated_merge(
        self, new_owner: str, new_repo: str, recreated_pr: PullRequestInfo
    ) -> bool:
        """Merge the recreated PR under the per-repo dispatch lock."""
        new_merge_method = self._pr_merge_methods.get(
            f"{new_owner}/{new_repo}", self.default_merge_method
        )
        try:
            if self._github_client is None:
                raise RuntimeError("GitHub client not initialized")
            # Same per-repo dispatch serialisation as the main merge
            # path — see ``_get_merge_dispatch_lock``.
            new_dispatch_lock = await self._get_merge_dispatch_lock(new_owner, new_repo)
            async with new_dispatch_lock:
                new_merged = await self._github_client.merge_pull_request(
                    new_owner,
                    new_repo,
                    recreated_pr.number,
                    new_merge_method,
                )
        except Exception as merge_err:
            self.log.error(
                "Failed to merge recreated PR %s#%s: %s",
                recreated_pr.repository_full_name,
                recreated_pr.number,
                merge_err,
            )
            new_merged = False
        return new_merged
