# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Pre-merge inspection of a repository's merge requirements.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import PullRequestInfo
from ._base import _MergeManagerBase

if TYPE_CHECKING:
    from ..github_async import GitHubAsync


class _RequirementsMixin(_MergeManagerBase):
    """Pre-merge inspection of a repository's merge requirements."""

    async def _check_merge_requirements(
        self, pr_info: PullRequestInfo
    ) -> tuple[bool, str]:
        """
        Check if a PR meets all requirements for merging, including branch protection rules.

        Args:
            pr_info: Pull request information

        Returns:
            Tuple of (can_merge: bool, reason: str)
        """
        if not self._github_client:
            return False, "GitHub client not initialized"

        repo_owner, repo_name = pr_info.repository_full_name.split("/")

        verdict = await self._check_code_owner_reviews(
            pr_info, repo_owner, repo_name, self._github_client
        )
        if verdict is not None:
            return verdict

        # Predictive merge probe. This is a *best-effort* dry-run verdict
        # only: GitHub's mergeable_state can lag, and repository rulesets
        # are invisible to it, so it must never gate the real merge. Run it
        # only in preview mode to render the evaluation; the execution path
        # is attempt-first and lets the actual merge response be
        # authoritative (Step 6 + _merge_pr_with_retry).
        if self.preview_mode:
            verdict = await self._probe_predicted_merge(pr_info, repo_owner, repo_name)
            if verdict is not None:
                return verdict

        # Additional checks based on PR state
        return self._evaluate_mergeable_state(pr_info, repo_owner, repo_name)

    async def _check_code_owner_reviews(
        self,
        pr_info: PullRequestInfo,
        repo_owner: str,
        repo_name: str,
        client: GitHubAsync,
    ) -> tuple[bool, str] | None:
        """
        Inspect the base branch's protection rules for a code owner review requirement.

        Kept apart so the protection lookup retains its own exception
        boundary: a failed or unavailable lookup is logged and expresses
        no opinion, leaving the later checks to decide.

        Returns a verdict when code owner reviews are required, or
        ``None`` when this requirement does not settle the outcome.
        """
        try:
            base_branch = pr_info.base_branch or "main"
            protection_rules = await client.get_branch_protection(
                repo_owner, repo_name, base_branch
            )

            if protection_rules:
                required_reviews = protection_rules.get(
                    "required_pull_request_reviews", {}
                )
                if required_reviews:
                    require_code_owner = required_reviews.get(
                        "require_code_owner_reviews", False
                    )

                    # If code owner reviews are required, our automated approval might not be sufficient
                    if require_code_owner:
                        return self._code_owner_bypass_verdict(
                            pr_info, repo_owner, repo_name
                        )

        except Exception as exc:
            # Don't fail the merge attempt if we can't check protection rules.
            self.log.debug(
                "Branch protection check failed for %s/%s#%s: %s",
                repo_owner,
                repo_name,
                pr_info.number,
                exc,
            )

        return None

    def _code_owner_bypass_verdict(
        self, pr_info: PullRequestInfo, repo_owner: str, repo_name: str
    ) -> tuple[bool, str]:
        """
        Decide whether the configured force level waives required code owner reviews.

        Separate from the lookup above so the force-level decision does
        not add a nesting level inside the protection-rule traversal.
        """
        # Check if user wants to bypass code owner checks
        if self.force_level in [
            "code-owners",
            "protection-rules",
            "all",
        ]:
            # Only log during preview evaluation to avoid duplicate messages
            if self.preview_mode:
                self.log.warning(
                    f"⚠️ Bypassing code owner review requirement for {repo_owner}/{repo_name}#{pr_info.number} (--force={self.force_level})"
                )
            return (
                True,
                "code owner review requirement bypassed by force level",
            )
        else:
            return (
                False,
                "code owner reviews are required - cannot auto-approve",
            )

    async def _probe_predicted_merge(
        self, pr_info: PullRequestInfo, repo_owner: str, repo_name: str
    ) -> tuple[bool, str] | None:
        """
        Predict the merge outcome and turn a negative prediction into a verdict.

        Kept apart so the prediction keeps the exception boundary that
        makes it advisory: an outcome that cannot be predicted is logged
        and the remaining checks continue.

        Returns a verdict when the merge is predicted to fail, or
        ``None`` when it is predicted to succeed or could not be
        determined.
        """
        try:
            # Use pre-determined merge method for this repository
            cache_key = f"{repo_owner}/{repo_name}"
            merge_method = self._pr_merge_methods.get(
                cache_key, self.default_merge_method
            )

            # Predict the outcome to detect hidden branch protection rules
            test_result = await self._predict_merge_outcome(
                repo_owner, repo_name, pr_info.number, merge_method
            )
            if not test_result[0]:
                # Check if we should bypass protection rules
                if self.force_level in [
                    "code-owners",
                    "protection-rules",
                    "all",
                ]:
                    return await self._protection_bypass_verdict(
                        pr_info, repo_owner, repo_name, test_result[1]
                    )
                else:
                    return False, test_result[1]

        except Exception as e:
            # If we can't predict the outcome, continue with other checks
            self.log.debug(
                f"Could not predict merge outcome for {repo_owner}/{repo_name}#{pr_info.number}: {e}"
            )

        return None

    async def _protection_bypass_verdict(
        self,
        pr_info: PullRequestInfo,
        repo_owner: str,
        repo_name: str,
        blocked_reason: str,
    ) -> tuple[bool, str]:
        """
        Confirm the authenticated user may actually bypass branch protection.

        Separate from the prediction above so the permission round-trip
        and its two outcomes read as one unit, rather than as three more
        nesting levels inside the probe.
        """
        if self._github_client:
            self.log.debug(
                f"Checking bypass permissions for {repo_owner}/{repo_name} with force_level={self.force_level}"
            )
            (
                can_bypass,
                bypass_reason,
            ) = await self._github_client.check_user_can_bypass_protection(
                repo_owner, repo_name, self.force_level
            )
            self.log.debug(
                f"Bypass check result: can_bypass={can_bypass}, reason={bypass_reason}"
            )
            if not can_bypass:
                self.log.warning(
                    f"Cannot bypass branch protection for {repo_owner}/{repo_name}#{pr_info.number}: {bypass_reason}"
                )
                return (
                    False,
                    f"cannot bypass branch protection: {bypass_reason}",
                )

        self.log.warning(
            f"⚠️ Bypassing branch protection check for {repo_owner}/{repo_name}#{pr_info.number}: {blocked_reason} (--force={self.force_level})"
        )
        # When bypassing, return early to allow merge
        return (
            True,
            f"branch protection check bypassed (--force={self.force_level})",
        )

    def _evaluate_mergeable_state(
        self, pr_info: PullRequestInfo, repo_owner: str, repo_name: str
    ) -> tuple[bool, str]:
        """
        Translate GitHub's reported ``mergeable_state`` into a merge verdict.

        Separate from the checks above because it consults only PR state
        already fetched, so it makes no API call and cannot fail.
        """
        # aislop-ignore-next-line ai-slop/python-repetitive-dispatch -- branch drives multi-step control flow, not a value lookup
        if pr_info.mergeable_state == "blocked":
            return self._evaluate_blocked_state(pr_info, repo_owner, repo_name)
        elif pr_info.mergeable_state == "behind":
            return self._evaluate_behind_state(pr_info, repo_owner, repo_name)
        elif pr_info.mergeable_state == "unstable":
            # ``unstable`` means a non-required check failed but
            # the PR is otherwise mergeable. Auto-merge can still
            # fire because non-required checks don't block branch
            # protection. Let Step 5.5 handle it.
            return (
                True,
                "PR unstable — Step 5.5 will enable auto-merge",
            )
        elif pr_info.mergeable_state == "dirty":
            if self.force_level == "all":
                self.log.warning(
                    f"⚠️ Attempting merge despite conflicts for {repo_owner}/{repo_name}#{pr_info.number} (--force=all)"
                )
                return True, "PR has conflicts but forcing merge attempt (--force=all)"
            else:
                return (False, "merge conflicts")

        return True, "All merge requirements appear to be met"

    def _evaluate_blocked_state(
        self, pr_info: PullRequestInfo, repo_owner: str, repo_name: str
    ) -> tuple[bool, str]:
        """
        Decide the verdict for a PR GitHub reports as ``blocked``.

        Held apart from the state dispatch because a block has three
        distinct causes to weigh — dismissable Copilot comments, a
        missing approval, and failing checks — each with its own verdict.
        """
        # Check if Copilot comments might be the blocker
        if self.dismiss_copilot and self._copilot_handler:
            has_copilot_comments = self._copilot_handler.has_blocking_copilot_comments(
                pr_info
            )
            if has_copilot_comments:
                return (
                    True,
                    "PR blocked but has Copilot comments that will be dismissed",
                )

        # For blocked PRs, if mergeable is True, it just needs approval - we can handle that
        if pr_info.mergeable is True:
            return True, "PR ready for approval and merge"
        else:
            # If mergeable is False and state is blocked, it's blocked by failing checks
            if self.force_level == "all":
                # Only log during preview evaluation to avoid duplicate messages
                if self.preview_mode:
                    self.log.warning(
                        f"⚠️ Bypassing failing status checks for {repo_owner}/{repo_name}#{pr_info.number} (--force=all)"
                    )
                return True, "PR blocked but forcing merge attempt (--force=all)"
            else:
                # Don't hard-fail here: let Step 5.5 enable
                # auto-merge and route to AUTO_MERGE_PENDING.
                # The block reason might be "failing required
                # check" right now but the check could still
                # complete successfully — GitHub returns
                # ``mergeable=False`` transiently for several
                # reasons (still computing, non-required check
                # failed). Step 5.5's analyze_block_reason
                # pre-check still weeds out genuinely-stuck
                # cases (missing approvals, etc.).
                return (
                    True,
                    "PR blocked — Step 5.5 will enable auto-merge",
                )

    def _evaluate_behind_state(
        self, pr_info: PullRequestInfo, repo_owner: str, repo_name: str
    ) -> tuple[bool, str]:
        """
        Decide the verdict for a PR GitHub reports as ``behind`` its base branch.

        Held apart from the state dispatch because the outcome turns on
        whether we may rebase the branch ourselves, which is unrelated to
        the other states' reasoning.
        """
        if not self.fix_out_of_date:
            if self.force_level == "all":
                self.log.warning(
                    f"⚠️ Attempting merge despite being behind for {repo_owner}/{repo_name}#{pr_info.number} (--force=all)"
                )
                return True, "PR behind but forcing merge attempt (--force=all)"
            else:
                # Don't hard-fail when behind + --no-fix: the
                # user opted out of *us* rebasing the branch,
                # but enabling auto-merge in Step 5.5 is a
                # separate, non-rewriting operation. If a third
                # party (Dependabot, pre-commit-ci) rebases the
                # PR while we wait, auto-merge will fire.
                return (
                    True,
                    "PR behind — Step 5.5 will enable auto-merge",
                )
        else:
            return True, "PR is behind - will rebase before merge"
