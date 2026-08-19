# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Turning the merge dispatch's answer into the reported result.

A rejected merge is not automatically a failure: the pull request may
have merged or closed elsewhere while we worked, or auto-merge may
still be poised to finish it.  What remains after those are ruled out
is classified, and for a dependabot pull request may still be recovered
by asking for the pull request to be recreated.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from ..bot_identity import is_dependabot
from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._merge_state import _Attempt
from ._models import (
    MergeStatus,
)


class _MergeOutcomeMixin(_MergeManagerBase):
    """Turning the merge dispatch's answer into the reported result."""

    async def _report_merge_outcome(
        self, attempt: _Attempt, merged: bool | None
    ) -> None:
        """Record the attempt's outcome on its ``MergeResult``.

        ``merged`` is tri-state: ``None`` means auto-merge is active and
        the pull request will merge asynchronously, ``True`` that it
        merged, and ``False`` that the dispatch was rejected — which is
        handed to the classifier rather than reported as a failure
        outright.
        """
        pr_info = attempt.pr_info
        result = attempt.result
        if merged is None:
            # Auto-merge is active — PR will merge asynchronously.
            # Tailor the reason to the actual ``mergeable_state``
            # so the end-of-run summary shows what auto-merge is
            # waiting on, rather than always "pending checks".
            result.status = MergeStatus.AUTO_MERGE_PENDING
            if pr_info.mergeable_state == "behind":
                wait_reason = "behind base branch"
            elif pr_info.mergeable_state == "unstable":
                wait_reason = "non-required check failure"
            else:
                # ``blocked`` (the only other state that
                # reaches this branch) routed through
                # ``analyze_block_reason()`` and was
                # classified as pending required checks by
                # ``_block_reason_indicates_pending_checks``.
                wait_reason = "pending checks"
            result.error = f"auto-merge pending: {wait_reason}"
            self._pr_status(
                f"⏳ Waiting: {pr_info.html_url} [{wait_reason}]",
                level="debug",
            )
        elif merged:
            result.status = MergeStatus.MERGED
            self._pr_status(
                f"✅ Merged: {pr_info.html_url}",
                level="debug",
            )
        else:
            await self._classify_merge_failure(attempt)

    async def _classify_merge_failure(self, attempt: _Attempt) -> None:
        """Work out whether a rejected merge is really a failure.

        Kept apart from the reporting above because most of it is not
        reporting at all: it is a sequence of checks that can each
        reclassify the rejection as something needing no operator
        follow-up, and only the residue is reported as a failure.
        """
        if await self._report_externally_closed(attempt):
            return
        if self._report_behind_with_auto_merge(attempt):
            return

        # Compute failure summary once — used for both the
        # recreate decision and the final error reporting.
        failure_reason = await self._get_failure_summary(attempt.pr_info)

        recreated_pr = await self._request_recreate_if_warranted(
            attempt, failure_reason
        )
        if recreated_pr is not None:
            await self._merge_recreated_pr(attempt, recreated_pr)
        else:
            await self._report_merge_failure(
                attempt.pr_info,
                attempt.owner,
                attempt.repo,
                attempt.result,
                failure_reason,
            )

    async def _report_externally_closed(self, attempt: _Attempt) -> bool:
        """Report a pull request that closed outside this attempt.

        A failed merge attempt can mask two benign races: the pull
        request merged externally (a concurrent dependamerge run at org
        scope, or a human admin), or dependabot closed it without
        merging once sibling merges advanced the base.  Neither outcome
        needs human follow-up, so both are reported as terminal states
        rather than failures.

        Returns ``True`` when the live state explained the rejection and
        the result has been recorded.
        """
        pr_info = attempt.pr_info
        result = attempt.result
        ext_state, ext_merged = await self._fetch_pr_state_now(
            pr_info, attempt.owner, attempt.repo
        )
        if ext_state == "closed" and ext_merged:
            result.status = MergeStatus.SKIPPED
            result.error = "already merged externally"
            self._pr_status(
                f"⏭️ Skipped: {pr_info.html_url} [already merged externally]",
                level="info",
            )
            return True
        if ext_state == "closed":
            result.status = MergeStatus.CLOSED
            result.error = (
                "PR closed without merging during the run "
                "(no operator follow-up needed)"
            )
            self._pr_status(
                f"\U0001f6aa Closed without merging: {pr_info.html_url}",
                level="info",
            )
            return True
        return False

    def _report_behind_with_auto_merge(self, attempt: _Attempt) -> bool:
        """Report a rejected ``behind`` merge that auto-merge will finish.

        A ``behind`` pull request whose merge was rejected but that has
        auto-merge armed is not a failure: the reactive recovery in
        ``_handle_merge_failure`` requested a dependabot rebase and armed
        auto-merge, so GitHub completes the merge server-side once the
        rebase lands and required checks pass.

        Returns ``True`` when the result was recorded that way.
        """
        pr_info = attempt.pr_info
        result = attempt.result
        if not (
            pr_info.mergeable_state == "behind"
            and attempt.pr_key in self._auto_merge_enabled
        ):
            return False
        result.status = MergeStatus.AUTO_MERGE_PENDING
        result.error = "auto-merge pending: behind base branch (rebase requested)"
        self._pr_status(
            f"\u23f3 Waiting: {pr_info.html_url} "
            "[behind base branch; rebase requested]",
            level="debug",
        )
        return True

    async def _request_recreate_if_warranted(
        self, attempt: _Attempt, failure_reason: str
    ) -> PullRequestInfo | None:
        """Ask dependabot to recreate the pull request, where that helps.

        Two triggers are considered.  The first is a branch-protection
        failure — the original unsigned-commit case.  Branch protection
        *and* repository rulesets can both block a dependabot pull
        request for reasons recreation resolves (most commonly an
        unsigned-commit / required-signature rule), so they are treated
        alike and the recreate path is not silently skipped on
        repositories that have migrated from classic protection to
        rulesets.  The second is a stuck required check, delegated to
        :meth:`_stuck_check_warrants_recreate`.

        Returns the recreated pull request, or ``None`` when no recreate
        was warranted or dependabot did not produce one.
        """
        pr_info = attempt.pr_info
        if not is_dependabot(pr_info.author) or self.preview_mode:
            return None

        reason_lower = failure_reason.lower()
        should_recreate = (
            "branch protection" in reason_lower or "ruleset" in reason_lower
        )
        if not should_recreate:
            should_recreate = await self._stuck_check_warrants_recreate(pr_info)
        if not should_recreate:
            return None

        self._track_pr_state(pr_info, "recreating")
        try:
            return await self._trigger_dependabot_recreate(pr_info)
        finally:
            self._track_pr_state(pr_info, None)

    async def _stuck_check_warrants_recreate(self, pr_info: PullRequestInfo) -> bool:
        """Report whether a stuck required check justifies a recreate.

        A *required* verification check that has been queued /
        in_progress / pending for longer than
        ``STUCK_CHECK_THRESHOLD_SECONDS``, on a pull request that itself
        was created and last updated that long ago, will not recover on
        its own.  Required checks (DCO, lint, build, etc.) normally
        start reporting in seconds; when one stalls indefinitely the
        only reliable recovery for a dependabot pull request is to
        recreate it so the checks fire again on a fresh head SHA.
        pre-commit.ci is excluded by the detector — it has its own
        dedicated recovery via ``_trigger_stale_precommit_ci``, which
        posts ``pre-commit.ci run``.

        A detection failure is answered ``False``: without evidence the
        recreate is not worth the churn.
        """
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
                f"⏳ Stuck required check detected: "
                f"{pr_info.html_url} "
                f"[{stuck_check} pending for "
                f"{stuck_age:.0f}s, requesting recreate]",
                level="info",
            )
            return True
        return False

    async def _merge_recreated_pr(
        self, attempt: _Attempt, recreated_pr: PullRequestInfo
    ) -> None:
        """Approve and merge the pull request dependabot recreated.

        The recreated pull request carries a fresh head SHA, so its
        checks run again and the rule that rejected the original no
        longer applies.  Success and failure are both reported against
        the *new* pull request, which is the one an operator would open.
        """
        result = attempt.result
        # We have a fresh PR — approve and merge it
        new_owner, new_repo = recreated_pr.repository_full_name.split("/", 1)
        await self._approve_pr(new_owner, new_repo, recreated_pr.number)

        new_merge_method = self._pr_merge_methods.get(
            f"{new_owner}/{new_repo}", self.default_merge_method
        )
        try:
            if self._github_client is None:
                raise RuntimeError("GitHub client not initialized")
            # Same per-repo dispatch serialisation as
            # the main merge path — see
            # ``_get_merge_dispatch_lock``.
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
