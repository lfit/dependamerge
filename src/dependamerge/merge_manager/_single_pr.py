# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The core single-PR merge sequence.

``_merge_single_pr_impl`` is the spine of the whole tool: it takes a
single pull request from 'not yet looked at' to a terminal
``MergeResult``, delegating each specialised step to the other
mixins.  What lives here is the shape of the sequence --- the gates in
order, the two waits, the merge, and the two error classifiers --- with
each step's substance in a ``_single_pr_*`` sibling:

``_single_pr_context``
    The ``_MergeFlow`` scratchpad every step reads and writes.
``_single_pr_gates``
    Steps 0 to 4: Gerrit mirroring, PR state, mergeability, blocking
    reviews, pre-commit.ci repair, merge requirements, Copilot.
``_single_pr_rebase``
    Step 5: whether the branch must be refreshed, and doing it.
``_single_pr_wait``
    Step 5.5: arming auto-merge and waiting for required checks.
``_single_pr_merge``
    Step 6: the merge dispatch, or deferral to auto-merge.
``_single_pr_outcome`` / ``_single_pr_recreate``
    Classifying the result, including the dependabot recreate path.
"""

from __future__ import annotations

import logging
import time

from ..github_async import PermissionError as GitHubPermissionError
from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._single_pr_context import _MergeFlow
from ._single_pr_gates import _SinglePrGatesMixin
from ._single_pr_merge import _SinglePrMergeMixin
from ._single_pr_rebase import _SinglePrRebaseMixin
from ._single_pr_wait import _SinglePrWaitMixin
from ._types import MergeResult, MergeStatus


class _SinglePullRequestMixin(
    _SinglePrGatesMixin,
    _SinglePrRebaseMixin,
    _SinglePrWaitMixin,
    _SinglePrMergeMixin,
    _MergeManagerBase,
):
    """The end-to-end merge of one pull request."""

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
            return self._permission_fast_fail(pr_info, start_time)

        # Early determination of merge method based on repository settings
        merge_method = await self._get_merge_method_for_repo(repo_owner, repo_name)

        # Store the determined merge method for this PR
        self._pr_merge_methods[f"{repo_owner}/{repo_name}"] = merge_method

        result = MergeResult(pr_info=pr_info, status=MergeStatus.PENDING)
        flow = _MergeFlow(
            pr_info=pr_info,
            repo_owner=repo_owner,
            repo_name=repo_name,
            result=result,
        )
        outcome = result

        try:
            outcome = await self._run_merge_sequence(flow)
        except GitHubPermissionError as e:
            self._report_permission_error(flow, e)
        except Exception as e:
            self._report_processing_error(flow, e)
        finally:
            result.duration = time.time() - start_time
            # Clean up recently-approved tracking to avoid unbounded growth
            pr_key = f"{repo_owner}/{repo_name}#{pr_info.number}"
            self._recently_approved.discard(pr_key)

        return outcome

    async def _run_merge_sequence(self, flow: _MergeFlow) -> MergeResult:
        """Run every step in order, stopping at the first terminal one."""
        early = await self._gate_github2gerrit(flow)
        if early is not None:
            return early
        early = await self._gate_pr_still_open(flow)
        if early is not None:
            return early
        early = await self._gate_mergeable_state(flow)
        if early is not None:
            return early

        await self._repair_blocked_pr(flow)

        early = await self._gate_merge_requirements(flow)
        if early is not None:
            return early
        early = await self._gate_copilot_reviews(flow)
        if early is not None:
            return early

        await self._analyze_blocked_state(flow)

        flow.needs_rebase = await self._pr_needs_rebase(flow)
        if flow.needs_rebase:
            early = await self._run_step5_rebase(flow)
            if early is not None:
                return early

        flow.already_rebased = flow.pr_key in self._rebased_prs
        flow.should_wait = self._should_wait_for_checks(flow)
        if flow.should_wait:
            early = await self._wait_for_checks(flow)
            if early is not None:
                return early

        early = await self._perform_merge(flow)
        if early is not None:
            return early
        return flow.result

    def _permission_fast_fail(
        self, pr_info: PullRequestInfo, start_time: float
    ) -> MergeResult:
        """Report a repository already known to reject this token."""
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

    def _report_permission_error(
        self, flow: _MergeFlow, e: GitHubPermissionError
    ) -> None:
        """Handle permission errors with detailed guidance.

        When the token lacks rights on a repository the same error fires
        for every PR processed.  Record the repo so subsequent PRs in the
        batch short-circuit via the fast-fail check at the top of
        ``_merge_single_pr_impl``, and emit the verbose guidance block
        only the first time we see the failure for a given repository.
        """
        pr_info = flow.pr_info
        result = flow.result
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
            return

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

    def _report_processing_error(self, flow: _MergeFlow, e: Exception) -> None:
        """Provide clean single-line error messages for other errors.

        The stack trace is attached only when the logger is in DEBUG mode
        (i.e. the user passed ``--verbose``).  In the default WARNING
        setup the trace would otherwise be printed to stderr for every
        failure, swamping a repo-scoped batch run with several hundred
        lines of noise per PR when the underlying cause is something
        uniform (e.g. token without the required scope) that a single
        clean line already conveys.
        """
        pr_info = flow.pr_info
        flow.result.status = MergeStatus.FAILED
        flow.result.error = str(e)

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
