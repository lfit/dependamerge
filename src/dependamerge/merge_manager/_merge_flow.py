# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The end-to-end merge attempt for a single pull request.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import logging
import time
from typing import Any

from .. import rebase
from ..bot_identity import is_dependabot
from ..github2gerrit_detector import (
    build_gerrit_skip_message,
)
from ..github_async import PermissionError as GitHubPermissionError
from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._models import (
    MergeResult,
    MergeStatus,
)


class _MergeFlowMixin(_MergeManagerBase):
    """The end-to-end merge attempt for a single pull request."""

    async def _merge_single_pr_impl(self, pr_info: PullRequestInfo) -> MergeResult:
        """
        Merge a single pull request with retry logic.

        Args:
            pr_info: Pull request information

        Returns:
            MergeResult with operation status and details
        """
        start_time = time.time()
        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)

        # Fast-fail when a previous PR in this batch has already
        # hit a permission error against the same repository.  In
        # that case the token genuinely lacks the rights to act on
        # any PR in this repo, so attempting the GitHub API calls
        # again would only produce another 403 and another copy of
        # the token-guidance block.  Report the failure cleanly
        # (single ❌ line, no traceback) and move on.
        if pr_info.repository_full_name in self._permission_failed_repos:
            result = MergeResult(pr_info=pr_info, status=MergeStatus.FAILED)
            result.error = (
                f"token lacks required permissions on {pr_info.repository_full_name}"
            )
            self._pr_status(
                f"❌ Failed: {pr_info.html_url} "
                "[token lacks permissions on this repository]",
                level="error",
            )
            result.duration = time.time() - start_time
            return result

        # Early determination of merge method based on repository settings
        merge_method = await self._get_merge_method_for_repo(repo_owner, repo_name)

        # Store the determined merge method for this PR
        self._pr_merge_methods[f"{repo_owner}/{repo_name}"] = merge_method

        result = MergeResult(pr_info=pr_info, status=MergeStatus.PENDING)

        try:
            if self.github2gerrit_mode != "ignore":
                g2g_result = await self._detect_github2gerrit(
                    repo_owner, repo_name, pr_info.number
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
                        f"🔄 Submitting Gerrit change for {pr_info.html_url} "
                        f"[{skip_msg}]",
                        level="info",
                    )
                    submitted = await self._submit_gerrit_change(
                        mapping, pr_info, repo_owner, repo_name
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

            # Check if PR is closed before processing.  If it has
            # been closed *and merged* by another process (a
            # concurrent dependamerge run, a human admin, an
            # auto-merge that landed mid-flight, etc.) we treat it
            # as a skip rather than a failure: there is no
            # remaining work or human follow-up to perform.
            if pr_info.state != "open":
                already_merged = await self._is_pr_already_merged(
                    pr_info, repo_owner, repo_name
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
                    pr_info, repo_owner, repo_name, result
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
                            f"/repos/{repo_owner}/{repo_name}/pulls/{pr_info.number}"
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

                # A Dependabot title/commit-subject mismatch fails the
                # semantic check permanently, so repair it here rather
                # than waiting out the merge timeout to discover it.
                await self._align_semantic_title(pr_info)

            can_merge, merge_check_reason = await self._check_merge_requirements(
                pr_info
            )

            if not can_merge:
                result.status = MergeStatus.SKIPPED
                result.error = f"Merge requirements not met: {merge_check_reason}"
                self._pr_status(
                    f"⏭️ Skipped: {pr_info.html_url} [{merge_check_reason.lower()}]",
                    level="debug",
                )
                return result

            copilot_processing_successful = True
            if self.dismiss_copilot and self._copilot_handler:
                # Analyze what types of reviews we have
                self._copilot_handler.analyze_copilot_review_dismissibility(pr_info)

                try:
                    (
                        processed_count,
                        total_count,
                    ) = await self._copilot_handler.dismiss_copilot_comments_for_pr(
                        pr_info
                    )
                    if total_count > 0:
                        # Silent processing in background
                        pass
                except Exception as e:
                    self.log.warning(
                        f"⚠️ Failed to process Copilot items for PR {pr_info.number}: {e}"
                    )
                    copilot_processing_successful = False

            # Gate on Copilot processing, but DO NOT approve
            # up-front. Approval is now performed on demand (approve-on-
            # demand): either just before arming auto-merge (see
            # _enable_auto_merge_with_approval) or after a direct merge is
            # rejected specifically for a missing review (see
            # _approve_and_retry_if_review_required). This avoids approving
            # PRs that did not actually need our review, while the Copilot
            # gate below still prevents acting on a PR with unresolved
            # Copilot feedback.
            if not copilot_processing_successful:
                result.status = MergeStatus.FAILED
                result.error = "Copilot review processing incomplete - not approving to avoid pollution"
                self._pr_status(
                    f"❌ Failed: {pr_info.html_url} [copilot processing incomplete]",
                    level="error",
                )
                return result

            # Analyse the block reason once per PR snapshot.  Step 5's
            # staleness probe, Step 5.5's wait pre-check, and Step 6's
            # auto-merge skip gate all consult the same analysis, and
            # each call costs ~4 API requests (reviews, comments,
            # check runs, combined status).  Fetching it once here and
            # passing the result through collapses the previous two to
            # three calls per blocked PR into one.
            # ``blocked_analysis_ok`` records whether the analysis
            # itself succeeded: the Step 5.5 pre-check treats an
            # analysis *failure* (as opposed to a None/inconclusive
            # reason) as "do not wait".
            blocked_reason: str | None = None
            blocked_analysis_ok = False
            if (
                pr_info.mergeable_state == "blocked"
                and not self.preview_mode
                and self._github_client is not None
            ):
                try:
                    blocked_reason = await self._github_client.analyze_block_reason(
                        repo_owner,
                        repo_name,
                        pr_info.number,
                        pr_info.head_sha,
                        base_branch=pr_info.base_branch,
                    )
                    blocked_analysis_ok = True
                except Exception as exc:
                    self.log.debug(
                        "analyze_block_reason failed for %s/%s#%s: %s",
                        repo_owner,
                        repo_name,
                        pr_info.number,
                        exc,
                    )

            # Handle rebase if needed before merge.
            #
            # Rebases are expensive: they restart every required CI
            # check (minutes of wall-clock time per PR), and same-repo
            # batches compound the cost because every sibling merge
            # moves the base again.  So Step 5 rebases **only when
            # GitHub actually requires it**:
            #
            # - ``behind`` alone is NOT enough.  GitHub happily merges
            #   a behind-but-green PR unless the branch's protection
            #   enforces the *strict* status-check policy ("require
            #   branches to be up to date before merging"), so we
            #   probe that policy (cached per repo/branch) and
            #   otherwise send the PR straight to the merge attempt.
            #   Should a merge still be rejected for staleness, the
            #   reactive path in ``_handle_merge_failure`` recovers.
            # - ``blocked`` masks ``behind`` (``mergeable_state`` is a
            #   single value).  A required check that *failed* on a
            #   head demonstrably behind base was judged against
            #   pre-rebase content — e.g. an org-required workflow
            #   audit that the base branch has since fixed — and only
            #   a rebase re-runs it against the current base.  Pending
            #   checks are excluded: they resolve on their own, no
            #   rebase required.
            #
            # The rebase itself is dispatched to the dedicated
            # ``rebase`` module so the macro-vs-local-vs-REST decision
            # tree, the local-git workflow, and the post-rebase
            # polling loop all live in one place where they can be
            # tested in isolation.
            needs_rebase = False
            if (
                self.fix_out_of_date
                and not self.preview_mode
                and self._github_client is not None
            ):
                if pr_info.mergeable_state == "behind":
                    needs_rebase = await self._behind_pr_requires_rebase(
                        pr_info, repo_owner, repo_name
                    )
                elif pr_info.mergeable_state == "blocked" and blocked_analysis_ok:
                    needs_rebase = await self._blocked_pr_needs_rebase(
                        pr_info, repo_owner, repo_name, blocked_reason
                    )
            if needs_rebase:
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
                    pr_info=pr_info,
                    owner=repo_owner,
                    repo=repo_name,
                )
                if outcome.failed:
                    result.status = MergeStatus.FAILED
                    result.error = outcome.error_message
                    return result

            # If the PR is still blocked (e.g. by a pending
            # required status check such as pre-commit.ci) or
            # unstable (a non-required check failed), enable
            # auto-merge and wait for required checks to
            # complete. Skipped when:
            #   * preview_mode (no side effects)
            #   * force_level == "all" (force semantics bypass wait)
            #   * Step 5 already ran a rebase + wait for this PR
            #     (avoid doubling the configured merge_timeout)
            #   * mergeable_state == "blocked" for a reason that
            #     cannot resolve on its own (e.g. "requires approval",
            #     missing code-owner reviews) — waiting would just
            #     delay the inevitable failure/merge by up to
            #     merge_timeout.
            #
            # ``behind`` PRs deliberately do NOT wait here: unless
            # branch protection enforces the strict up-to-date policy
            # (in which case Step 5 already refreshed the branch), a
            # behind-but-green PR merges directly, so parking it in
            # the wait loop — where the state never advances on its
            # own — would just burn the full ``merge_timeout`` before
            # the merge attempt that was going to succeed anyway.
            #
            # Exception: when Step 5 just dispatched an *asynchronous*
            # rebase (local force-push or the ``@dependabot rebase``
            # macro) and auto-merge could not be armed, those paths
            # leave ``_rebased_prs`` unset precisely so this wait can
            # bridge the gap while the rebase lands and GitHub
            # recomputes mergeability — the snapshot still reads
            # ``behind`` because neither path refreshes ``pr_info``.
            # Without the wait, Step 6 would fire a manual merge
            # against the stale state and 405.  ``needs_rebase``
            # captures "Step 5 actually ran" and the
            # ``not already_rebased`` guard below excludes the
            # auto-merge-armed case.
            #
            # We accept any ``mergeable`` value (including ``False``)
            # when the state is one of these auto-merge-rescuable
            # states, because GitHub returns ``mergeable=False``
            # transiently while computing the value or when a
            # non-required check failed. The block-reason pre-check
            # below still weeds out genuinely-stuck cases (missing
            # approvals, etc.) so we don't burn ``merge_timeout`` on
            # them.
            pr_key_for_wait = f"{repo_owner}/{repo_name}#{pr_info.number}"
            already_rebased = pr_key_for_wait in self._rebased_prs
            # ``unstable`` means a non-required check is failing or
            # pending but the PR is otherwise mergeable.  When GitHub
            # also reports ``mergeable is True`` the green button is
            # live and a direct merge succeeds *now*, so entering the
            # auto-merge wait would be pure waste: the state never
            # reaches ``clean`` (the non-required check stays red, e.g.
            # an excluded Zizmor scan), so the loop burns the full
            # ``merge_timeout``; and ``enablePullRequestAutoMerge``
            # is rejected outright on an already-mergeable PR, so the
            # wait isn't even backed by auto-merge.  Route those
            # straight to the Step 6 direct merge.  We still wait on
            # ``unstable`` when ``mergeable`` is not literally True
            # (GitHub still computing the value, or a required check
            # transiently failing) so a genuinely not-yet-ready PR is
            # not merged prematurely.
            state_is_waitable = (
                pr_info.mergeable_state == "blocked"
                or (
                    pr_info.mergeable_state == "unstable"
                    and pr_info.mergeable is not True
                )
                or (pr_info.mergeable_state == "behind" and needs_rebase)
            )
            base_should_wait = (
                not self.preview_mode
                and self._github_client is not None
                and state_is_waitable
                and self.force_level != "all"
                and not already_rebased
            )

            # For ``blocked`` PRs, consult the block-reason analysis
            # (computed once above) before entering the wait loop so
            # we don't burn the full merge_timeout on PRs blocked for
            # reasons that cannot resolve on their own.
            should_wait = base_should_wait
            if base_should_wait and pr_info.mergeable_state == "blocked":
                if not blocked_analysis_ok:
                    # Treat analysis failures as 'do not wait' so we
                    # don't burn the full ``merge_timeout`` on a PR
                    # whose block reason we cannot classify. The PR
                    # will fall through to the Step 6 skip gate (which
                    # re-consults the analysis) and either defer to
                    # auto-merge or surface a manual-merge error
                    # promptly.
                    should_wait = False
                elif blocked_reason is not None:
                    if not self._block_reason_indicates_pending_checks(blocked_reason):
                        self.log.debug(
                            "Skipping Step 5.5 wait for %s: block "
                            "reason '%s' will not resolve on its own",
                            pr_key_for_wait,
                            blocked_reason,
                        )
                        should_wait = False

            if should_wait:
                if pr_key_for_wait not in self._auto_merge_enabled:
                    # Approve-on-demand: arming auto-merge implies we want
                    # the PR to merge once checks pass, so approve the
                    # current head first (idempotent) before enabling.
                    auto_ok_pre = await self._enable_auto_merge_with_approval(
                        pr_info, repo_owner, repo_name
                    )
                    if auto_ok_pre:
                        self._pr_status(
                            f"🤖 Auto-merge: {pr_info.html_url}",
                            level="debug",
                        )

                # Wait (bounded by ``merge_timeout``) for required
                # checks to complete and auto-merge to fire.  The
                # continue-states mirror the ``base_should_wait`` entry
                # condition above (blocked / behind / unstable).
                self._track_pr_state(pr_info, "waiting")
                (
                    closed_during_wait,
                    merged_during_wait,
                ) = await self._wait_for_auto_merge(
                    pr_info,
                    repo_owner,
                    repo_name,
                    continue_states=("blocked", "behind", "unstable"),
                    measures_checks=True,
                )
                self._track_pr_state(pr_info, None)

                # If the wait revealed the PR is already closed,
                # short-circuit before attempting a manual merge.
                # Distinguish auto-merge success from
                # closed-without-merge using the ``merged`` boolean
                # captured from the refresh payload.
                if closed_during_wait:
                    if merged_during_wait:
                        result.status = MergeStatus.MERGED
                        self._pr_status(
                            f"✅ Merged (auto-merge): {pr_info.html_url}",
                            level="debug",
                        )
                    else:
                        result.status = MergeStatus.CLOSED
                        result.error = (
                            "PR closed without merging during auto-merge wait "
                            "(no operator follow-up needed)"
                        )
                        self._pr_status(
                            f"🚪 Closed without merging: {pr_info.html_url}",
                            level="warning",
                        )
                    return result

                # The wait can expire with the PR still blocked on a
                # pre-commit.ci run that will never finish.  Step 0.5
                # only retriggers a run that was already stale when
                # processing *started*; a run that went pending shortly
                # before this run began crosses the stuck threshold
                # *during* the wait above, so without this re-check the
                # merge below fails on the pending check without the
                # recovery macro ever being posted.  The helper re-gates
                # on required-check status, pending age, and duplicate
                # comments, so this is a no-op unless the run is
                # genuinely stuck.
                if (
                    pr_info.mergeable_state == "blocked"
                    and self._github_client is not None
                ):
                    late_precommit_fixed = await self._trigger_stale_precommit_ci(
                        pr_info
                    )
                    if late_precommit_fixed:
                        # pre-commit.ci now reports success.  Auto-merge
                        # was armed before the wait, so GitHub may merge
                        # the PR the moment the check lands — re-fetch
                        # and short-circuit on a closed PR before
                        # attempting a manual merge below.
                        late_updated: Any = None
                        try:
                            late_updated = await self._github_client.get(
                                f"/repos/{repo_owner}/{repo_name}"
                                f"/pulls/{pr_info.number}"
                            )
                        except Exception as e:
                            self.log.debug(
                                "Failed to refresh PR %s state after "
                                "post-wait pre-commit.ci rerun: %s",
                                pr_key_for_wait,
                                e,
                            )
                        if isinstance(late_updated, dict):
                            if late_updated.get("state") == "closed":
                                pr_info.state = "closed"
                                if late_updated.get("merged", False):
                                    result.status = MergeStatus.MERGED
                                    self._pr_status(
                                        f"✅ Merged (auto-merge): {pr_info.html_url}",
                                        level="debug",
                                    )
                                else:
                                    result.status = MergeStatus.CLOSED
                                    result.error = (
                                        "PR closed without merging during "
                                        "auto-merge wait "
                                        "(no operator follow-up needed)"
                                    )
                                    self._pr_status(
                                        f"🚪 Closed without merging: "
                                        f"{pr_info.html_url}",
                                        level="warning",
                                    )
                                return result
                            # Only accept concrete values: GitHub
                            # returns null / "" / "unknown" while it
                            # recomputes mergeability right after the
                            # check lands, and clobbering the known
                            # snapshot with those would change the
                            # downstream routing (mirrors the guards in
                            # the ``_wait_for_auto_merge`` refresh).
                            if late_updated.get("mergeable") is not None:
                                pr_info.mergeable = late_updated.get("mergeable")
                            late_state = late_updated.get("mergeable_state")
                            if late_state not in (None, "", "unknown"):
                                pr_info.mergeable_state = late_state
                            updated_head = (late_updated.get("head") or {}).get("sha")
                            if updated_head:
                                pr_info.head_sha = updated_head

            result.status = MergeStatus.MERGING
            if self.preview_mode:
                self._simulate_preview_merge(pr_info, result)
            else:
                if self.progress_tracker:
                    self.progress_tracker.update_operation(
                        f"Merging PR {pr_info.number} in {pr_info.repository_full_name}"
                    )

                # If auto-merge is enabled and the PR is in a state
                # that auto-merge can rescue (blocked, behind, or
                # unstable), skip the manual merge attempt — GitHub
                # will merge automatically once branch protection is
                # satisfied.
                #
                # We accept any ``mergeable`` value (including
                # ``False``) here for the same reason Step 5.5 does:
                # ``mergeable=False`` from the API can mean
                # "definitely failing", "still computing", or "a
                # non-required check failed". Letting auto-merge
                # decide whether the failing thing actually blocks
                # merge is more accurate than us treating
                # ``False`` as terminal here.
                #
                # For ``blocked`` PRs we still consult
                # ``analyze_block_reason()`` to weed out cases
                # auto-merge cannot resolve (missing approvals,
                # code-owner reviews, etc.). For ``behind`` and
                # ``unstable`` we accept directly: ``behind``
                # resolves once GitHub re-runs checks against the
                # rebased commit, and ``unstable`` means a
                # non-required check failed (which doesn't actually
                # block auto-merge).
                #
                # Do NOT skip when:
                #   * force_level == "all" — force semantics
                #     intentionally proceed despite the blocked
                #     state and must not be overridden by
                #     auto-merge.
                #   * the block reason (for ``blocked`` PRs) is
                #     something other than pending required
                #     checks (e.g. missing approvals).
                pr_key = f"{repo_owner}/{repo_name}#{pr_info.number}"
                auto_merge_pending_checks = False
                if (
                    pr_key in self._auto_merge_enabled
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
                            blocked_analysis_ok
                            and not should_wait
                            and not already_rebased
                        ):
                            # Nothing has changed since the analysis
                            # at the top of the flow (no Step 5
                            # rebase, no Step 5.5 wait), so reuse it
                            # instead of re-spending its ~4 API calls.
                            block_reason = blocked_reason
                            analysis_fresh = True
                        if not analysis_fresh and self._github_client is not None:
                            try:
                                block_reason = (
                                    await self._github_client.analyze_block_reason(
                                        repo_owner,
                                        repo_name,
                                        pr_info.number,
                                        pr_info.head_sha,
                                        base_branch=pr_info.base_branch,
                                    )
                                )
                            except Exception as exc:
                                self.log.debug(
                                    "analyze_block_reason failed for %s: %s",
                                    pr_key,
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
                        auto_merge_pending_checks = (
                            self._block_reason_indicates_pending_checks(block_reason)
                        )

                if auto_merge_pending_checks:
                    merged = None  # Sentinel: auto-merge pending
                else:
                    # Proactive approval: some organizations mandate an
                    # approving review via a repository ruleset before
                    # *any* merge is allowed.  When this PR's base branch
                    # is governed that way a merge-first attempt is
                    # guaranteed to be rejected, so approve the current
                    # head up-front and skip the doomed round-trip plus
                    # reactive recovery.  See the helper for details; on
                    # any lookup failure it no-ops and the reactive
                    # approve-on-demand path still covers us.
                    await self._approve_if_review_mandated(
                        pr_info, repo_owner, repo_name, pr_key
                    )
                    # Serialise the actual merge dispatch per repo so
                    # back-to-back merges don't race GitHub's branch
                    # protection propagation.  Workers on the same
                    # repo queue here; workers on different repos run
                    # in parallel.  See ``_get_merge_dispatch_lock``.
                    dispatch_lock = await self._get_merge_dispatch_lock(
                        repo_owner, repo_name
                    )
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
                            merged = await self._merge_pr_with_retry(
                                pr_info, repo_owner, repo_name
                            )
                    # Conflict recovery runs *outside* the dispatch lock
                    # so the rebase wait never blocks sibling merges.
                    if dirty_before_dispatch:
                        return await self._handle_merge_conflict(
                            pr_info, repo_owner, repo_name, result
                        )
                    # A PR can also turn ``dirty`` *during* our own merge
                    # window (a sibling merged between the pre-dispatch
                    # check and the merge call).  The post-failure
                    # refresh — off the lock, with its recompute poll —
                    # catches that so a freshly-dirty PR is never
                    # reported as a generic merge failure.
                    if not merged and self._repo_scoped:
                        await self._refresh_pr_mergeability(
                            pr_info, repo_owner, repo_name
                        )
                        if pr_info.mergeable_state == "dirty":
                            return await self._handle_merge_conflict(
                                pr_info, repo_owner, repo_name, result
                            )

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
                        and not should_wait
                        and not already_rebased
                        and self.force_level != "all"
                    ):
                        last_merge_exc = self._last_merge_exception.get(pr_key)
                        if (
                            last_merge_exc is not None
                            and self._merge_error_indicates_pending_workflows(
                                str(last_merge_exc)
                            )
                        ):
                            merged = await self._wait_for_required_workflows_and_retry(
                                pr_info, repo_owner, repo_name
                            )

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
                    # A failed merge attempt can mask two benign
                    # races: the PR merged externally (a concurrent
                    # dependamerge run at org scope, or a human
                    # admin), or dependabot closed it without merging
                    # once sibling merges advanced the base. Neither
                    # outcome needs human follow-up.
                    ext_state, ext_merged = await self._fetch_pr_state_now(
                        pr_info, repo_owner, repo_name
                    )
                    if ext_state == "closed" and ext_merged:
                        result.status = MergeStatus.SKIPPED
                        result.error = "already merged externally"
                        self._pr_status(
                            f"⏭️ Skipped: {pr_info.html_url} "
                            "[already merged externally]",
                            level="info",
                        )
                        return result
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
                        return result

                    # A ``behind`` PR whose merge was rejected but that
                    # has auto-merge armed is not a failure: the
                    # reactive recovery in ``_handle_merge_failure``
                    # requested a dependabot rebase and armed
                    # auto-merge, so GitHub completes the merge
                    # server-side once the rebase lands and required
                    # checks pass.
                    if (
                        pr_info.mergeable_state == "behind"
                        and pr_key in self._auto_merge_enabled
                    ):
                        result.status = MergeStatus.AUTO_MERGE_PENDING
                        result.error = (
                            "auto-merge pending: behind base branch (rebase requested)"
                        )
                        self._pr_status(
                            f"\u23f3 Waiting: {pr_info.html_url} "
                            "[behind base branch; rebase requested]",
                            level="debug",
                        )
                        return result

                    # Compute failure summary once — used for both the
                    # recreate decision and the final error reporting.
                    failure_reason = await self._get_failure_summary(pr_info)

                    # Before giving up, check if this is a dependabot PR
                    # that failed due to unsigned commits.  If so, ask
                    # dependabot to recreate the PR and merge the new one.
                    #
                    # Two recreate triggers are considered:
                    #   1. Branch-protection failures (the original
                    #      unsigned-commit case).
                    #   2. A *required* verification check that has
                    #      been stuck (queued / in_progress / pending)
                    #      for longer than
                    #      ``STUCK_CHECK_THRESHOLD_SECONDS`` on a PR
                    #      that itself was created and last updated
                    #      that long ago. Required checks (DCO, lint,
                    #      build, etc.) normally start reporting in
                    #      seconds; when one stalls indefinitely, the
                    #      only reliable recovery for dependabot PRs
                    #      is to recreate the PR so the checks fire
                    #      again on a fresh head SHA. pre-commit.ci is
                    #      excluded here — it has its own dedicated
                    #      recovery via ``_trigger_stale_precommit_ci``
                    #      (which posts ``pre-commit.ci run``).
                    recreated_pr = None
                    if is_dependabot(pr_info.author) and not self.preview_mode:
                        reason_lower = failure_reason.lower()
                        # Branch protection *and* repository rulesets can
                        # both block a dependabot PR for reasons recreation
                        # resolves (most commonly an unsigned-commit /
                        # required-signature rule).  Treat them alike so the
                        # recreate path is not silently skipped on repos that
                        # have migrated from classic protection to rulesets.
                        should_recreate = (
                            "branch protection" in reason_lower
                            or "ruleset" in reason_lower
                        )
                        if not should_recreate:
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
                                should_recreate = True
                        if should_recreate:
                            self._track_pr_state(pr_info, "recreating")
                            try:
                                recreated_pr = await self._trigger_dependabot_recreate(
                                    pr_info
                                )
                            finally:
                                self._track_pr_state(pr_info, None)

                    if recreated_pr is not None:
                        # We have a fresh PR — approve and merge it
                        new_owner, new_repo = recreated_pr.repository_full_name.split(
                            "/", 1
                        )
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
                            new_dispatch_lock = await self._get_merge_dispatch_lock(
                                new_owner, new_repo
                            )
                            async with new_dispatch_lock:
                                new_merged = (
                                    await self._github_client.merge_pull_request(
                                        new_owner,
                                        new_repo,
                                        recreated_pr.number,
                                        new_merge_method,
                                    )
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
                                f"Dependabot recreated PR #{recreated_pr.number} "
                                "but merge still failed"
                            )
                            self.log.error(
                                "Failed to merge recreated PR %s#%s",
                                recreated_pr.repository_full_name,
                                recreated_pr.number,
                            )
                            self._pr_status(
                                f"❌ Failed: {recreated_pr.html_url} "
                                "[recreated PR merge failed]",
                                level="error",
                            )
                    else:
                        await self._report_merge_failure(
                            pr_info,
                            repo_owner,
                            repo_name,
                            result,
                            failure_reason,
                        )

        except GitHubPermissionError as e:
            # Handle permission errors with detailed guidance.
            #
            # When the token lacks rights on a repository the same
            # error fires for every PR processed.  Record the repo
            # so subsequent PRs in the batch short-circuit via the
            # fast-fail check at the top of this method, and emit
            # the verbose guidance block only the first time we
            # see the failure for a given repository.
            result.status = MergeStatus.FAILED
            result.error = str(e)

            first_failure_for_repo = (
                pr_info.repository_full_name not in self._permission_failed_repos
            )
            self._permission_failed_repos.add(pr_info.repository_full_name)

            operation_desc = e.operation.replace("_", " ")
            self._pr_status(
                f"❌ Failed: {pr_info.html_url} [permission denied: {operation_desc}]",
                level="error",
            )

            if not first_failure_for_repo:
                # Already printed the full guidance for this repo;
                # do not repeat it for every remaining PR.
                return result

            # Provide token-specific guidance (printed once per repo)
            self._console.print("\n💡 Token Permission Issue:")
            self._console.print(f"   Problem: {e}")

            if e.token_type_guidance:
                self._console.print("\n   For Classic Tokens:")
                self._console.print(
                    f"   • {e.token_type_guidance.get('classic', 'Check token scopes')}"
                )
                self._console.print("\n   For Fine-Grained Tokens:")
                self._console.print(
                    f"   • {e.token_type_guidance.get('fine_grained', 'Check token permissions')}"
                )
                if "fix" in e.token_type_guidance:
                    self._console.print("\n   Quick Fix:")
                    self._console.print(f"   • {e.token_type_guidance['fix']}")

            self._console.print()

        except Exception as e:
            result.status = MergeStatus.FAILED
            result.error = str(e)

            # Provide clean single-line error messages for other errors.
            # The stack trace is attached only when the logger is in
            # DEBUG mode (i.e. the user passed ``--verbose``).  In the
            # default WARNING setup the trace would otherwise be
            # printed to stderr for every failure, swamping a
            # repo-scoped batch run with several hundred lines of
            # noise per PR when the underlying cause is something
            # uniform (e.g. token without the required scope) that a
            # single clean line already conveys.
            self.log.error(
                "Failed to process PR %s: %s",
                pr_info.html_url,
                e,
                exc_info=self.log.isEnabledFor(logging.DEBUG),
            )
            self._pr_status(
                f"❌ Failed: {pr_info.html_url} [processing error: {e}]",
                level="error",
            )

        finally:
            result.duration = time.time() - start_time
            # Clean up recently-approved tracking to avoid unbounded growth
            pr_key = f"{repo_owner}/{repo_name}#{pr_info.number}"
            self._recently_approved.discard(pr_key)

        return result
