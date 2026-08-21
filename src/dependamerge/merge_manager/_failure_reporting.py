# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Explaining why a merge did not happen.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from ..bot_identity import is_dependabot
from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._models import (
    MergeResult,
    MergeStatus,
)


class _FailureReportingMixin(_MergeManagerBase):
    """Explaining why a merge did not happen."""

    async def _report_merge_failure(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        result: MergeResult,
        failure_reason: str,
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

    async def _get_failure_summary(self, pr_info: PullRequestInfo) -> str:
        """
        Generate a detailed failure summary based on PR state.

        Args:
            pr_info: Pull request information

        Returns:
            Detailed description of why the merge failed
        """
        # Check if we have a stored exception for this PR
        pr_key = f"{pr_info.repository_full_name}#{pr_info.number}"
        last_exception = self._last_merge_exception.get(pr_key)
        self.log.debug(
            f"_get_failure_summary called for {pr_key}, mergeable_state={pr_info.mergeable_state}, mergeable={pr_info.mergeable}, has_exception={last_exception is not None}"
        )
        if last_exception:
            error_msg = str(last_exception)
            self.log.debug(f"Last exception for {pr_key}: {error_msg[:200]}")
            from_exception = self._summarize_merge_exception(pr_info, error_msg)
            if from_exception is not None:
                return from_exception

        # aislop-ignore-next-line ai-slop/python-repetitive-dispatch -- branches run distinct analysis (block-reason parsing), not a uniform table
        if pr_info.mergeable_state == "behind":
            return "behind base branch"
        elif pr_info.mergeable_state == "blocked":
            # Use detailed block analysis for blocked PRs
            return await self._summarize_blocked_state(pr_info)
        elif pr_info.mergeable_state == "dirty":
            return "merge conflicts"
        elif pr_info.mergeable_state == "draft":
            return "draft PR"
        elif pr_info.mergeable is False:
            return "cannot update protected ref - organization or branch protection rules prevent merge"
        elif pr_info.mergeable_state == "unknown":
            # For unknown state, try to get more details
            return await self._summarize_unknown_state(pr_info)
        else:
            return f"merge failed: {pr_info.mergeable_state}"

    def _summarize_merge_exception(
        self, pr_info: PullRequestInfo, error_msg: str
    ) -> str | None:
        """Classify the exception the last merge attempt raised.

        Held apart from ``_get_failure_summary`` so the two sources of a
        reason stay distinguishable: what the failure said, handled
        here, and what the PR's state implies, handled by the caller.

        Returns ``None`` when the exception says nothing conclusive and
        the caller should fall through to state-based analysis.  A 405
        against a PR that is not itself clean is one such case.
        """
        detail = self._github_supplied_detail(error_msg)
        if detail is not None:
            return detail
        # Workflow-scope failures surface in several phrasings: the
        # PermissionError messages we raise ("Missing 'workflow' scope",
        # "Missing workflow permissions") and GitHub's own response body
        # ("refusing to allow ... without `workflow` scope").  Match all
        # of them, but require the word "workflow" so unrelated 403s do
        # not get mislabelled as a scope problem.
        error_lower = error_msg.lower()
        if "workflow" in error_lower and (
            "missing 'workflow' scope" in error_lower
            or "missing workflow permissions" in error_lower
            or "refusing to allow" in error_lower
        ):
            return "missing 'workflow' token scope"
        # The token already had the 'workflow' scope but GitHub still
        # refused the workflow-file update — a ruleset or SSO problem,
        # not a scope problem.  Report it as such rather than telling the
        # user to add a scope they already hold.
        elif "blocked by something other than token scope" in error_lower:
            return (
                "workflow update blocked by repository ruleset or SSO "
                "(token already has 'workflow' scope)"
            )
        elif "403" in error_msg and "forbidden" in error_lower:
            return "insufficient permissions"
        # Surface transient HTTP errors (502, 405 etc.) accurately instead
        # of falling through to infer a reason from mergeable_state, which
        # may be stale or misleading (e.g. "clean" → "branch protection").
        elif "405" in error_msg and "Method Not Allowed" in error_msg:
            if pr_info.mergeable_state in ("clean", "unstable"):
                return (
                    "GitHub API returned transient 405 error "
                    "(PR appears mergeable — GitHub may be experiencing issues, "
                    "see https://www.githubstatus.com)"
                )
            # For non-clean states, fall through to state-based analysis below
        elif "502" in error_msg:
            return (
                "GitHub API returned 502 Bad Gateway "
                "(GitHub may be experiencing issues, "
                "see https://www.githubstatus.com)"
            )
        return None

    @staticmethod
    def _github_supplied_detail(error_msg: str) -> str | None:
        """Extract GitHub's own explanation from a merge exception.

        The merge layer (github_async.merge_pull_request) embeds
        GitHub's own explanation after a "GitHub: " marker — the
        ruleset violation, "Required workflows ... are not
        satisfied", required-check names, etc.  This is the
        actionable cause, so surface it ahead of any generic
        state-based inference.  We trim the PR-state context we
        appended after it so the reason stays concise.

        Separate from the phrase matching that follows it because this
        is the one branch that parses a payload rather than recognising
        a phrase, and because a marker carrying no text must leave that
        matching to run.

        Returns ``None`` when the marker is absent or carries no text.
        """
        marker = "GitHub: "
        if marker not in error_msg:
            return None
        detail = error_msg.split(marker, 1)[1]
        detail = detail.split(" (PR state:", 1)[0].strip()
        if not detail:
            return None
        return detail[:300]

    async def _summarize_blocked_state(self, pr_info: PullRequestInfo) -> str:
        """Explain a ``blocked`` PR, preferring detailed block analysis.

        Separate because this is the only state that issues a further
        API call, and so the only one needing its own failure handling:
        when the analysis raises, the reason falls back to what
        ``mergeable`` alone implies.
        """
        try:
            detailed_reason = await self._analyze_block_reason_async(pr_info)
            # Convert the detailed reason to a more concise format for console output
            return self._concise_block_reason(detailed_reason)
        except Exception as e:
            self.log.debug(f"Failed to get detailed block reason: {e}")
            # Fallback to generic message

        # Fallback logic when detailed analysis fails
        if pr_info.mergeable is True:
            return "branch protection rules prevent merge"
        else:
            return "blocked by failing status checks"

    @staticmethod
    def _concise_block_reason(detailed_reason: str) -> str:
        """Condense a detailed block reason into a console-sized phrase.

        Kept apart from the API call that produces the reason so the
        phrase matching — which the end-of-run summary and several tests
        depend on verbatim — reads without the surrounding error
        handling.  It still runs inside the caller's ``try``, so an
        unexpected reason falls back exactly as it did before.
        """
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

    async def _summarize_unknown_state(self, pr_info: PullRequestInfo) -> str:
        """Explain an ``unknown`` PR state through block analysis.

        Separate from the ``blocked`` case despite the shared call: an
        ``unknown`` state carries no information of its own, so a failed
        analysis leaves only "pending or failed" to report, where a
        blocked PR can still fall back on ``mergeable``.
        """
        try:
            detailed_reason = await self._analyze_block_reason_async(pr_info)
            return self._concise_unknown_reason(detailed_reason)
        except Exception as e:
            self.log.debug(f"Failed to analyze unknown state: {e}")
            return "status checks pending or failed"

    @staticmethod
    def _concise_unknown_reason(detailed_reason: str) -> str:
        """Condense the block reason found for an ``unknown`` PR state.

        A named failing check is surfaced with its name so the operator
        knows where to look; every other reason keeps the analysis
        wording, minus its ``Blocked by`` prefix.  Extracted for the
        same reason as ``_concise_block_reason``, and likewise runs
        inside the caller's ``try``.
        """
        if "failing check" in detailed_reason.lower():
            if detailed_reason.startswith("Blocked by failing check:"):
                check_name = detailed_reason.replace("Blocked by failing check: ", "")
                return f"failing check: {check_name}"
            else:
                return detailed_reason.replace("Blocked by ", "").lower()
        else:
            return detailed_reason.replace("Blocked by ", "").lower()
