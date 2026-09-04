# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Explaining why a merge did not happen.

Turning GitHub's block reasons, check results and API errors into a
sentence a human can act on.
"""

from __future__ import annotations

import asyncio

from ..bot_identity import is_dependabot
from ..models import PullRequestInfo
from ._failure_summary import _FailureSummaryFromExceptionMixin, _is_state_verdict
from ._types import MergeResult, MergeStatus


class _FailureReportingMixin(_FailureSummaryFromExceptionMixin):
    """Diagnosing and reporting merge failures."""

    async def _report_merge_failure(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        result: MergeResult,
        failure_reason: str,
        refused: bool = False,
    ) -> MergeResult:
        """Report a failed merge, upgrading to a stuck-check cause if found.

        Called when ``_merge_pr_with_retry`` failed and no dependabot
        recreate produced a replacement PR.  For a non-dependabot PR
        we check whether a required check is stuck (Option A): if so,
        print ``⚠️ Stuck check`` and arm auto-merge (when the PR is
        otherwise mergeable) so it lands once the check is
        re-triggered, without a force-push that would break this org's
        self-merge rule.  Otherwise emit the generic failure line.

        Sets ``result`` to ``FAILED`` and returns it.

        *refused* marks the result as GitHub declining the merge on the
        pull request's state, which is what lets the confirmation step
        re-judge it later: such a reason is a snapshot of one instant and
        may have expired by the time the summary prints.  A stuck check
        is also a state, so that branch qualifies too.  A missing token
        scope, a 403 or a 502 does not, and passing them through here
        unmarked keeps the message that explains what to fix.
        """
        stuck_reported = False
        if not is_dependabot(pr_info.author) and not self.preview_mode:
            try:
                detection = await self._detect_stuck_required_check(pr_info)
            except Exception as exc:
                self.log.debug(
                    "_detect_stuck_required_check failed for %s#%s: %s",
                    pr_info.repository_full_name,
                    pr_info.number,
                    exc,
                )
                detection = None
            if detection is not None and detection[0]:
                stuck_check = detection[1]
                self._pr_status(
                    f"⚠️ Stuck check: {pr_info.html_url} [{stuck_check}]",
                    level="warning",
                )
                # Arm auto-merge when the PR is otherwise mergeable
                # (not dirty) so it lands automatically once the stuck
                # check is re-triggered, without a second review round.
                # Approve the current head first (approve-on-demand): the
                # PR is no longer approved up-front, so auto-merge would
                # otherwise wait forever on a missing review.
                if pr_info.mergeable_state != "dirty":
                    await self._enable_auto_merge_with_approval(pr_info, owner, repo)
                result.error = f"stuck check: {stuck_check}"
                stuck_reported = True

        result.status = MergeStatus.FAILED
        # A stuck check is a reading of the PR's state, so it expires the
        # same way a rejection does.
        result.merge_refused = refused or stuck_reported
        if not stuck_reported:
            # Use the (now informative) failure reason as the result
            # error too, so the end-of-run summary surfaces the real
            # cause rather than a generic "all retry attempts" line.
            result.error = failure_reason or "Failed to merge after all retry attempts"
        # Keep the live output terse: the full (often long) reason is
        # shown in the end-of-run summary via ``result.error``, so
        # repeating it inline only duplicates it.  The ``⚠️ Stuck
        # check`` line above already carries the cause for stuck PRs.
        if not stuck_reported:
            self._pr_status(f"❌ Failed: {pr_info.html_url}", level="error")
        return result

    async def _analyze_block_reason_async(self, pr_info: PullRequestInfo) -> str:
        """Detailed reason a PR is blocked, using the async client.

        Replaces a call into ``GitHubClient._analyze_block_reason``, the
        synchronous wrapper.  That method detects a running event loop
        and, unable to call ``asyncio.run`` inside one, returns the
        placeholder ``"Blocked by branch protection"`` **without making
        any request**.  Since every caller here runs under the merge
        manager's loop, the detailed analysis was unreachable in
        production: every blocked PR resolved through the
        ``"branch protection"`` branch below and reported
        ``branch protection rules prevent merge`` whatever the true
        cause --- a failing check, a Copilot review, a ruleset, a human
        reviewer.  The surrounding branches were dead code that happened
        to agree with the fallback.

        Returns an empty string when no client is available, letting the
        caller fall through to its own generic handling.
        """
        if self._github_client is None:
            return ""
        owner, repo = pr_info.repository_full_name.split("/", 1)
        reason = await self._github_client.analyze_block_reason(
            owner,
            repo,
            pr_info.number,
            pr_info.head_sha,
            base_branch=pr_info.base_branch,
        )
        # Guard the contract rather than trusting it: every other API
        # payload in this module is type-checked before use, and a
        # non-string here would propagate into the branch matching below
        # as a silently truthy value.
        return reason if isinstance(reason, str) else ""

    async def _get_failure_summary(self, pr_info: PullRequestInfo) -> tuple[str, bool]:
        """
        Generate a detailed failure summary based on PR state.

        Args:
            pr_info: Pull request information

        Returns:
            A ``(reason, refused)`` pair.  ``refused`` says the reason
            describes GitHub declining the merge on the pull request's
            *state*, which is the only kind of reason a later reading may
            withdraw.

            With no exception to explain the failure, every branch below
            reads the PR's own state, so the answer is a snapshot of it
            by construction.  With an exception this code could not
            classify --- a retry-exhausted network timeout carries no
            ``GitHub:`` body and no recognisable status --- the state
            summary is still the best description available, but nothing
            about the failure says the pull request was judged, so it is
            not withdrawable.  Only a status GitHub uses for a verdict
            makes it so.
        """
        # Check if we have a stored exception for this PR
        pr_key = f"{pr_info.repository_full_name}#{pr_info.number}"
        last_exception = self._last_merge_exception.get(pr_key)
        self.log.debug(
            f"_get_failure_summary called for {pr_key}, mergeable_state={pr_info.mergeable_state}, mergeable={pr_info.mergeable}, has_exception={last_exception is not None}"
        )
        if last_exception:
            from_exception = self._failure_summary_from_exception(
                pr_key, last_exception, pr_info
            )
            if from_exception is not None:
                return from_exception
            return (
                await self._failure_summary_from_state(pr_info),
                _is_state_verdict(str(last_exception)),
            )

        return await self._failure_summary_from_state(pr_info), True

    async def _failure_summary_from_state(self, pr_info: PullRequestInfo) -> str:
        """Infer why a merge failed from the PR's reported state."""
        # aislop-ignore-next-line ai-slop/python-repetitive-dispatch -- branches run distinct analysis (block-reason parsing), not a uniform table
        if pr_info.mergeable_state == "behind":
            return "behind base branch"
        elif pr_info.mergeable_state == "blocked":
            # Use detailed block analysis for blocked PRs
            return await self._blocked_failure_summary(pr_info)
        elif pr_info.mergeable_state == "dirty":
            return "merge conflicts"
        elif pr_info.mergeable_state == "draft":
            return "draft PR"
        elif pr_info.mergeable is False:
            return "cannot update protected ref - organization or branch protection rules prevent merge"
        elif pr_info.mergeable_state == "unknown":
            # For unknown state, try to get more details
            return await self._unknown_state_failure_summary(pr_info)
        else:
            return f"merge failed: {pr_info.mergeable_state}"

    async def _blocked_failure_summary(self, pr_info: PullRequestInfo) -> str:
        """Summarise a ``blocked`` PR, naming the blocker where possible.

        The guard covers **only** the analysis probe.  Wrapping the
        conversion chain as well would make a defect in that chain
        indistinguishable from a genuine analysis failure --- both would
        surface as the generic fallback, which is plausible enough that a
        silently broken conversion could persist indefinitely, with every
        blocked PR simply reporting the generic reason.
        """
        try:
            detailed_reason = await self._analyze_block_reason_async(pr_info)
        except Exception as e:
            self.log.debug(f"Failed to get detailed block reason: {e}")
            # Fallback logic when detailed analysis fails
            if pr_info.mergeable is True:
                return "branch protection rules prevent merge"
            return "blocked by failing status checks"

        # Convert the detailed reason to a more concise format for console output
        if detailed_reason.startswith("Blocked by failing check:"):
            check_name = detailed_reason.replace("Blocked by failing check: ", "")
            return f"failing check: {check_name}"
        elif (
            detailed_reason.startswith("Blocked by")
            and "failing checks" in detailed_reason
        ):
            return detailed_reason.replace("Blocked by ", "").lower()
        elif "Human reviewer requested changes" in detailed_reason:
            return "human reviewer requested changes"
        elif "Copilot" in detailed_reason:
            return detailed_reason.replace("Blocked by ", "").lower()
        elif "ruleset" in detailed_reason.lower():
            return "repository ruleset prevents merge"
        elif "undetermined reason" in detailed_reason.lower():
            return "blocked for an undetermined reason"
        elif "branch protection" in detailed_reason.lower():
            return "branch protection rules prevent merge"
        else:
            return detailed_reason.replace("Blocked by ", "").lower()

    async def _unknown_state_failure_summary(self, pr_info: PullRequestInfo) -> str:
        """Summarise a PR whose mergeable state GitHub reports as unknown.

        Scoped like :meth:`_blocked_failure_summary`, and for the same
        reason: only the probe is guarded, so a conversion defect cannot
        masquerade as a failed analysis.
        """
        try:
            detailed_reason = await self._analyze_block_reason_async(pr_info)
        except Exception as e:
            self.log.debug(f"Failed to analyze unknown state: {e}")
            return "status checks pending or failed"

        if "failing check" in detailed_reason.lower():
            if detailed_reason.startswith("Blocked by failing check:"):
                check_name = detailed_reason.replace("Blocked by failing check: ", "")
                return f"failing check: {check_name}"
            else:
                return detailed_reason.replace("Blocked by ", "").lower()
        else:
            return detailed_reason.replace("Blocked by ", "").lower()

    async def _get_merge_method_for_repo(self, owner: str, repo: str) -> str:
        """
        Get the appropriate merge method for a specific repository based on branch protection settings.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Merge method to use: "merge", "squash", or "rebase"
        """
        if not self._github_service:
            self.log.warning("GitHubService not available, using default merge method")
            return self.default_merge_method

        try:
            protection_settings = (
                await self._github_service.get_branch_protection_settings(
                    owner, repo, "main"
                )
            )

            # Determine appropriate merge method
            merge_method = self._github_service.determine_merge_method(
                protection_settings, self.default_merge_method
            )

            if merge_method != self.default_merge_method:
                self.log.debug(
                    f"Repository {owner}/{repo} requires '{merge_method}' merge method "
                    f"(protection: requiresLinearHistory={protection_settings and protection_settings.get('requiresLinearHistory', False)})"
                )

            return merge_method

        except Exception as e:
            self.log.warning(
                f"Failed to determine merge method for {owner}/{repo}, using default '{self.default_merge_method}': {e}"
            )
            return self.default_merge_method

    async def _handle_merge_failure(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """
        Handle a merge failure and determine if we should retry.

        Args:
            pr_info: Pull request information
            owner: Repository owner
            repo: Repository name

        Returns:
            True if we should retry, False otherwise
        """
        if not self._github_client:
            return False

        # Check if the branch is out of date and we can fix it
        if self.fix_out_of_date and pr_info.mergeable_state == "behind":
            if is_dependabot(pr_info.author):
                # Prefer the ``@dependabot rebase`` macro over REST
                # ``update-branch``: the REST endpoint creates an
                # unsigned merge commit that can violate
                # signature-requiring branch protection, while
                # dependabot force-pushes a freshly signed rebase.
                # The macro completes asynchronously (minutes), so an
                # immediate retry is pointless — arm auto-merge so
                # GitHub finishes the merge server-side once the
                # rebase lands and checks pass, and let the caller's
                # not-merged classification report AUTO_MERGE_PENDING.
                self.log.info(
                    f"PR {owner}/{repo}#{pr_info.number} is behind - "
                    "requesting dependabot rebase"
                )
                if self._dependabot_is_rebasing(
                    pr_info.body
                ) or await self._request_dependabot_rebase(pr_info, owner, repo):
                    await self._enable_auto_merge_with_approval(pr_info, owner, repo)
                return False
            try:
                self.log.info(
                    f"PR {owner}/{repo}#{pr_info.number} is behind - updating branch"
                )
                await self._github_client.update_branch(owner, repo, pr_info.number)
                self._record_rebase()
                # Wait a moment for GitHub to process the update
                await asyncio.sleep(min(2.0, self._merge_recheck_interval))
                return True
            except Exception as e:
                self.log.error(
                    f"Failed to update branch for PR {owner}/{repo}#{pr_info.number}: {e}"
                )

        # For other failure types, don't retry
        return False
