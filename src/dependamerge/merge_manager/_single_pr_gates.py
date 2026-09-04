# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The gates a pull request clears before the merge is attempted.

Everything from Step 0 (is this PR really ours to merge?) through the
Copilot review gate and the one block-reason analysis the later steps
share.  Each gate returns a terminal ``MergeResult`` when it decides
the PR's fate, or ``None`` to let it through to the next one.
"""

from __future__ import annotations

from ..github2gerrit_detector import build_gerrit_skip_message
from ._base import _MergeManagerBase
from ._single_pr_context import _MergeFlow
from ._types import MergeResult, MergeStatus

#: Mergeable states worth attempting a self-inflicted-block repair on.
#: ``blocked`` is the reported cause; ``unknown`` is what GitHub returns
#: while it is still working the answer out, which is precisely when a
#: PR's checks have not settled.
_REPAIRABLE_STATES = frozenset({"blocked", "unknown"})


class _SinglePrGatesMixin(_MergeManagerBase):
    """The pre-merge screening steps of the single-PR sequence."""

    async def _gate_github2gerrit(self, flow: _MergeFlow) -> MergeResult | None:
        """Route a PR that mirrors a Gerrit change, or let it through.

        Returns a terminal result when the PR is skipped outright or its
        Gerrit change was submitted (successfully or not), and ``None``
        when the mode is ``ignore`` or no mapping was detected.
        """
        if self.github2gerrit_mode == "ignore":
            return None

        pr_info = flow.pr_info
        result = flow.result
        g2g_result = await self._detect_github2gerrit(
            flow.repo_owner, flow.repo_name, pr_info.number
        )

        if not (g2g_result.has_mapping and g2g_result.mapping):
            return None

        mapping = g2g_result.mapping
        skip_msg = build_gerrit_skip_message(mapping)

        if self.github2gerrit_mode == "skip":
            # Skip this PR entirely
            result.status = MergeStatus.SKIPPED
            result.error = f"Skipped: {skip_msg}"
            self._pr_status(
                f"⏩ Skipped: {pr_info.html_url} [{skip_msg}]",
                level="info",
            )
            return result

        # Default: "submit" mode - submit the Gerrit change
        if self.preview_mode:
            self._pr_status(
                f"🔄 Gerrit submit: {pr_info.html_url} [{skip_msg}]",
                level="info",
            )
            result.status = MergeStatus.MERGED
            return result

        # Attempt to submit the Gerrit change
        self._pr_status(
            f"🔄 Submitting Gerrit change for {pr_info.html_url} [{skip_msg}]",
            level="info",
        )
        submitted = await self._submit_gerrit_change(
            mapping, pr_info, flow.repo_owner, flow.repo_name
        )

        if submitted:
            result.status = MergeStatus.MERGED
            self._pr_status(
                f"✅ Gerrit submitted: {pr_info.html_url}",
                level="info",
            )
            return result

        # Gerrit submission failed - report as failed
        result.status = MergeStatus.FAILED
        result.error = f"Failed to submit Gerrit change ({skip_msg})"
        self._pr_status(
            f"❌ Failed: {pr_info.html_url} [Gerrit submit failed for {skip_msg}]",
            level="error",
        )
        return result

    async def _gate_pr_still_open(self, flow: _MergeFlow) -> MergeResult | None:
        """Stop on a PR that is no longer open.

        A PR closed *and merged* by another process (a concurrent
        dependamerge run, a human admin, an auto-merge that landed
        mid-flight, etc.) is a skip rather than a failure: there is no
        remaining work or human follow-up to perform.
        """
        pr_info = flow.pr_info
        result = flow.result
        if pr_info.state == "open":
            return None

        already_merged = await self._is_pr_already_merged(
            pr_info, flow.repo_owner, flow.repo_name
        )
        if already_merged:
            result.status = MergeStatus.SKIPPED
            result.error = "already merged externally"
            self._pr_status(
                f"⏭️ Skipped: {pr_info.html_url} [already merged externally]",
                level="info",
            )
            return result
        result.status = MergeStatus.CLOSED
        result.error = "PR was already closed without merging"
        self._pr_status(
            f"🚪 Closed: {pr_info.html_url} [already closed]",
            level="info",
        )
        return result

    async def _gate_mergeable_state(self, flow: _MergeFlow) -> MergeResult | None:
        """Route conflicts, unmergeable PRs and blocking reviews.

        A merge conflict (``dirty``) has no merge path of its own: it
        goes to the conflict handler (dependabot → ``@dependabot
        rebase`` + wait; other authors → report and fail fast) rather
        than the generic not-mergeable skip below.  Skipped in preview
        (no side effects) and under ``force=all`` (which intentionally
        attempts the merge regardless of state).
        """
        pr_info = flow.pr_info
        result = flow.result
        if (
            pr_info.mergeable_state == "dirty"
            and not self.preview_mode
            and self.force_level != "all"
        ):
            return await self._handle_merge_conflict(
                pr_info, flow.repo_owner, flow.repo_name, result
            )

        if not self._is_pr_mergeable(pr_info):
            return await self._handle_not_mergeable_pr(pr_info, result)

        # Check for blocking reviews (changes requested)
        if self._has_blocking_reviews(pr_info):
            # Only skip if not forcing with 'all' level
            if self.force_level != "all":
                result.status = MergeStatus.SKIPPED
                result.error = "PR has reviews requesting changes - will not override human feedback"
                self._pr_status(
                    f"⏭️ Skipped: {pr_info.html_url} [has reviews requesting changes]",
                    level="debug",
                )
                return result
            else:
                # Only log during preview evaluation to avoid duplicate messages
                if self.preview_mode:
                    self.log.warning(
                        f"⚠️ Overriding blocking reviews for {pr_info.repository_full_name}#{pr_info.number} (--force=all)"
                    )
        return None

    async def _repair_blocked_pr(self, flow: _MergeFlow) -> None:
        """Clear the self-inflicted reasons a PR is ``blocked``.

        Re-runs a stale pre-commit.ci and repairs a Dependabot
        title/commit-subject mismatch (which fails the semantic check
        permanently, so waiting out the merge timeout would only delay
        discovering it).  Avoids triggering side effects in preview.

        ``unknown`` is repaired alongside ``blocked``.  GitHub computes
        mergeability in the background and reports ``unknown`` until it
        has, which is the window a freshly pushed pull request sits in
        --- so the PRs whose checks were still settling, the ones this
        repair exists for, were the ones it skipped.  Acting there is
        safe because the repair's own preconditions do not consult the
        mergeable state: it fires only when pre-commit.ci is a required
        check and its status is missing or stalled, and the title repair
        only when the subject genuinely disagrees with the title.
        """
        pr_info = flow.pr_info
        if not (
            not self.preview_mode
            and pr_info.mergeable_state in _REPAIRABLE_STATES
            and self._github_client
        ):
            return

        precommit_fixed = await self._trigger_stale_precommit_ci(pr_info)
        if precommit_fixed:
            # Re-fetch PR state now that pre-commit.ci has passed
            try:
                updated = await self._github_client.get(
                    f"/repos/{flow.repo_owner}/{flow.repo_name}/pulls/{pr_info.number}"
                )
                if isinstance(updated, dict):
                    pr_info.mergeable = updated.get("mergeable")
                    pr_info.mergeable_state = updated.get("mergeable_state")
            except Exception as e:
                self.log.debug(
                    "Failed to refresh PR %s mergeable state after "
                    "pre-commit.ci rerun: %s",
                    f"{pr_info.repository_full_name}#{pr_info.number}",
                    e,
                )

        await self._align_semantic_title(pr_info)

    async def _gate_merge_requirements(self, flow: _MergeFlow) -> MergeResult | None:
        """Stop on a PR whose branch protection requirements are unmet."""
        pr_info = flow.pr_info
        result = flow.result
        can_merge, merge_check_reason = await self._check_merge_requirements(pr_info)

        if not can_merge:
            result.status = MergeStatus.SKIPPED
            result.error = f"Merge requirements not met: {merge_check_reason}"
            self._pr_status(
                f"⏭️ Skipped: {pr_info.html_url} [{merge_check_reason.lower()}]",
                level="debug",
            )
            return result
        return None

    async def _gate_copilot_reviews(self, flow: _MergeFlow) -> MergeResult | None:
        """Dismiss Copilot feedback, and stop if that did not complete.

        Gates on Copilot processing but does NOT approve up-front.
        Approval is performed on demand: either just before arming
        auto-merge (see ``_enable_auto_merge_with_approval``) or after a
        direct merge is rejected specifically for a missing review (see
        ``_approve_and_retry_if_review_required``).  This avoids
        approving PRs that did not actually need our review, while the
        gate below still prevents acting on a PR with unresolved
        Copilot feedback.
        """
        pr_info = flow.pr_info
        result = flow.result
        copilot_processing_successful = True
        if self.dismiss_copilot and self._copilot_handler:
            # Analyze what types of reviews we have
            self._copilot_handler.analyze_copilot_review_dismissibility(pr_info)

            try:
                (
                    processed_count,
                    total_count,
                ) = await self._copilot_handler.dismiss_copilot_comments_for_pr(pr_info)
                if total_count > 0:
                    # Silent processing in background
                    pass
            except Exception as e:
                self.log.warning(
                    f"⚠️ Failed to process Copilot items for PR {pr_info.number}: {e}"
                )
                copilot_processing_successful = False

        if not copilot_processing_successful:
            result.status = MergeStatus.FAILED
            result.error = "Copilot review processing incomplete - not approving to avoid pollution"
            self._pr_status(
                f"❌ Failed: {pr_info.html_url} [copilot processing incomplete]",
                level="error",
            )
            return result
        return None

    async def _analyze_blocked_state(self, flow: _MergeFlow) -> None:
        """Analyse why a ``blocked`` PR is blocked, once per PR snapshot.

        Step 5's staleness probe, Step 5.5's wait pre-check, and Step 6's
        auto-merge skip gate all consult the same analysis, and each call
        costs ~4 API requests (reviews, comments, check runs, combined
        status).  Fetching it once here and passing the result through
        ``flow`` collapses the previous two-to-three calls per blocked PR
        into one.
        """
        pr_info = flow.pr_info
        if not (
            pr_info.mergeable_state == "blocked"
            and not self.preview_mode
            and self._github_client is not None
        ):
            return

        try:
            flow.blocked_reason = await self._github_client.analyze_block_reason(
                flow.repo_owner,
                flow.repo_name,
                pr_info.number,
                pr_info.head_sha,
                base_branch=pr_info.base_branch,
            )
            flow.blocked_analysis_ok = True
        except Exception as exc:
            self.log.debug(
                "analyze_block_reason failed for %s/%s#%s: %s",
                flow.repo_owner,
                flow.repo_name,
                pr_info.number,
                exc,
            )
