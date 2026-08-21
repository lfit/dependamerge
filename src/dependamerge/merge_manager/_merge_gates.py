# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The gates a pull request clears before a merge is attempted.

Gerrit routing for github2gerrit-managed repositories, the state,
conflict, review and requirement checks, and the Copilot review pass.
Each gate answers a populated ``MergeResult`` when the attempt ends
there, and ``None`` when the flow should carry on.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from ..github2gerrit_detector import (
    build_gerrit_skip_message,
)
from ._base import _MergeManagerBase
from ._merge_state import _Attempt
from ._models import (
    MergeResult,
    MergeStatus,
)


class _MergeGatesMixin(_MergeManagerBase):
    """The gates a pull request clears before a merge is attempted."""

    async def _route_to_gerrit(self, attempt: _Attempt) -> MergeResult | None:
        """Skip or submit a pull request that mirrors a Gerrit change.

        ``github2gerrit`` replicates Gerrit changes as GitHub pull
        requests, so merging the pull request here would strand the
        change it mirrors.  Depending on ``github2gerrit_mode`` the
        attempt therefore either skips the pull request outright or
        submits the Gerrit change and reports that as the outcome.

        Separate from the rest of the gates because it is not a
        judgement about *this* pull request's readiness: it decides
        whether GitHub is the right place to act at all.

        Returns ``None`` for a pull request with no Gerrit mapping, and
        when detection is disabled, which is the ordinary case.
        """
        pr_info = attempt.pr_info
        result = attempt.result
        if self.github2gerrit_mode != "ignore":
            g2g_result = await self._detect_github2gerrit(
                attempt.owner, attempt.repo, pr_info.number
            )

            if g2g_result.has_mapping and g2g_result.mapping:
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
                    mapping, pr_info, attempt.owner, attempt.repo
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
                    f"❌ Failed: {pr_info.html_url} "
                    f"[Gerrit submit failed for {skip_msg}]",
                    level="error",
                )
                return result
        return None

    async def _check_merge_eligibility(self, attempt: _Attempt) -> MergeResult | None:
        """Run the gates that can end the attempt before any merge.

        In order: a pull request that closed before we reached it, a
        real merge conflict, a state GitHub reports as unmergeable,
        reviews requesting changes, and finally this repository's own
        merge requirements.  The order matters — a ``dirty`` pull
        request must reach the conflict handler rather than the generic
        not-mergeable skip below it — so the sequence is kept intact
        here rather than distributed across the callers.

        Returns ``None`` when every gate passed.
        """
        pr_info = attempt.pr_info
        result = attempt.result

        closed = await self._check_pr_still_open(attempt)
        if closed is not None:
            return closed

        # A merge conflict (``dirty``) has no merge path of its
        # own: route it to the conflict handler (dependabot →
        # ``@dependabot rebase`` + wait; other authors → report
        # and fail fast) rather than the generic not-mergeable
        # skip below.  Skipped in preview (no side effects) and
        # under ``force=all`` (which intentionally attempts the
        # merge regardless of state).
        if (
            pr_info.mergeable_state == "dirty"
            and not self.preview_mode
            and self.force_level != "all"
        ):
            return await self._handle_merge_conflict(
                pr_info, attempt.owner, attempt.repo, result
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

        await self._repair_blocked_pr(attempt)

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

    async def _check_pr_still_open(self, attempt: _Attempt) -> MergeResult | None:
        """Report a pull request that was closed before we reached it.

        If it has been closed *and merged* by another process (a
        concurrent dependamerge run, a human admin, an auto-merge that
        landed mid-flight, etc.) we treat it as a skip rather than a
        failure: there is no remaining work or human follow-up to
        perform.  A close without a merge is reported as CLOSED, which
        likewise needs no operator action.

        Returns ``None`` for a pull request that is still open.
        """
        pr_info = attempt.pr_info
        result = attempt.result
        if pr_info.state == "open":
            return None

        already_merged = await self._is_pr_already_merged(
            pr_info, attempt.owner, attempt.repo
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

    async def _repair_blocked_pr(self, attempt: _Attempt) -> None:
        """Fix the two blocks we can fix before judging requirements.

        A ``blocked`` pull request may be waiting on a pre-commit.ci run
        that has gone stale, which a comment macro restarts, and its
        state is re-read afterwards so the gate below judges the outcome
        rather than the stale snapshot.  A Dependabot title that does
        not match its commit subject fails the semantic check
        permanently, so it is repaired here rather than discovered after
        the merge timeout has elapsed.

        Both are side effects, so neither runs in preview mode.
        """
        pr_info = attempt.pr_info
        # If the PR is blocked, check for stale pre-commit.ci
        # and trigger a re-run before evaluating merge requirements.
        # Avoid triggering side effects when running in preview mode.
        if (
            not self.preview_mode
            and pr_info.mergeable_state == "blocked"
            and self._github_client
        ):
            precommit_fixed = await self._trigger_stale_precommit_ci(pr_info)
            if precommit_fixed:
                # Re-fetch PR state now that pre-commit.ci has passed
                try:
                    updated = await self._github_client.get(
                        f"/repos/{attempt.owner}/{attempt.repo}/pulls/{pr_info.number}"
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

    async def _process_copilot_feedback(self, attempt: _Attempt) -> MergeResult | None:
        """Dismiss Copilot review items, and gate the merge on doing so.

        Approval is deliberately *not* performed here.  It happens on
        demand later — either just before arming auto-merge (see
        ``_enable_auto_merge_with_approval``) or after a direct merge is
        rejected specifically for a missing review (see
        ``_approve_and_retry_if_review_required``) — which avoids
        approving pull requests that did not actually need our review.
        This gate still prevents acting on a pull request whose Copilot
        feedback we could not resolve.

        Returns the failed ``MergeResult`` when the Copilot pass did not
        complete, and ``None`` otherwise.
        """
        pr_info = attempt.pr_info
        result = attempt.result
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
