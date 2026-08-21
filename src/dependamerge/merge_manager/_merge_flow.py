# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The end-to-end merge attempt for a single pull request.

Runs one pull request through the phases that live in the sibling
mixins — Gerrit routing, the eligibility gates, the Copilot pass, the
rebase step, the wait on required checks, the merge dispatch and the
outcome report — and owns the two exception handlers that cover all of
them.  Each phase runs inside this module's ``try``, so a phase moved
out of it would escape those handlers; each returns the state the later
phases read, rather than leaving it on the manager, which every
concurrent pull-request worker shares.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import logging
import time

from ..github_async import PermissionError as GitHubPermissionError
from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._merge_state import _Attempt
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

        denied = self._fail_fast_on_denied_repo(pr_info, start_time)
        if denied is not None:
            return denied

        # Early determination of merge method based on repository settings
        merge_method = await self._get_merge_method_for_repo(repo_owner, repo_name)

        # Store the determined merge method for this PR
        self._pr_merge_methods[f"{repo_owner}/{repo_name}"] = merge_method

        result = MergeResult(pr_info=pr_info, status=MergeStatus.PENDING)
        attempt = _Attempt(
            pr_info=pr_info,
            owner=repo_owner,
            repo=repo_name,
            pr_key=f"{repo_owner}/{repo_name}#{pr_info.number}",
            result=result,
        )

        try:
            gerrit = await self._route_to_gerrit(attempt)
            if gerrit is not None:
                return gerrit

            ineligible = await self._check_merge_eligibility(attempt)
            if ineligible is not None:
                return ineligible

            copilot_failure = await self._process_copilot_feedback(attempt)
            if copilot_failure is not None:
                return copilot_failure

            rebased = await self._rebase_if_required(attempt)
            if rebased.result is not None:
                return rebased.result

            waited = await self._wait_for_required_checks(attempt, rebased)
            if waited.result is not None:
                return waited.result

            result.status = MergeStatus.MERGING
            if self.preview_mode:
                self._simulate_preview_merge(pr_info, result)
            else:
                dispatch = await self._dispatch_merge(attempt, rebased, waited)
                if dispatch.conflicted:
                    return await self._handle_merge_conflict(
                        pr_info, repo_owner, repo_name, result
                    )
                await self._report_merge_outcome(attempt, dispatch.merged)

        except GitHubPermissionError as e:
            self._report_permission_denied(attempt, e)

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
            self._recently_approved.discard(attempt.pr_key)

        return result

    def _fail_fast_on_denied_repo(
        self, pr_info: PullRequestInfo, start_time: float
    ) -> MergeResult | None:
        """Report immediately when this repository has already denied us.

        When a previous pull request in this batch has already hit a
        permission error against the same repository, the token
        genuinely lacks the rights to act on any pull request in it, so
        attempting the GitHub API calls again would only produce another
        403 and another copy of the token-guidance block.  Report the
        failure cleanly (single ❌ line, no traceback) and let the caller
        move on.

        Runs before the attempt's ``try``, and returns a fully-timed
        result of its own, because there is nothing left to guard: no
        API call is made and no later phase runs.

        Returns ``None`` when the repository has no recorded denial.
        """
        if pr_info.repository_full_name not in self._permission_failed_repos:
            return None
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

    def _report_permission_denied(
        self, attempt: _Attempt, error: GitHubPermissionError
    ) -> None:
        """Record the repository's denial and print the guidance once.

        When the token lacks rights on a repository the same error fires
        for every pull request processed.  The repository is recorded so
        subsequent pull requests in the batch short-circuit via
        :meth:`_fail_fast_on_denied_repo`, and the verbose guidance
        block is emitted only the first time the failure is seen for a
        given repository.
        """
        pr_info = attempt.pr_info
        result = attempt.result
        result.status = MergeStatus.FAILED
        result.error = str(error)

        first_failure_for_repo = (
            pr_info.repository_full_name not in self._permission_failed_repos
        )
        self._permission_failed_repos.add(pr_info.repository_full_name)

        operation_desc = error.operation.replace("_", " ")
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
        self._console.print(f"   Problem: {error}")

        if error.token_type_guidance:
            self._console.print("\n   For Classic Tokens:")
            self._console.print(
                f"   • {error.token_type_guidance.get('classic', 'Check token scopes')}"
            )
            self._console.print("\n   For Fine-Grained Tokens:")
            self._console.print(
                f"   • {error.token_type_guidance.get('fine_grained', 'Check token permissions')}"
            )
            if "fix" in error.token_type_guidance:
                self._console.print("\n   Quick Fix:")
                self._console.print(f"   • {error.token_type_guidance['fix']}")

        self._console.print()
