# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Pre-merge inspection of a repository's merge requirements.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from ..models import PullRequestInfo
from ._base import _MergeManagerBase


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

        try:
            base_branch = pr_info.base_branch or "main"
            protection_rules = await self._github_client.get_branch_protection(
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

        except Exception as exc:
            # Don't fail the merge attempt if we can't check protection rules.
            self.log.debug(
                "Branch protection check failed for %s/%s#%s: %s",
                repo_owner,
                repo_name,
                pr_info.number,
                exc,
            )

        # Predictive merge probe. This is a *best-effort* dry-run verdict
        # only: GitHub's mergeable_state can lag, and repository rulesets
        # are invisible to it, so it must never gate the real merge. Run it
        # only in preview mode to render the evaluation; the execution path
        # is attempt-first and lets the actual merge response be
        # authoritative (Step 6 + _merge_pr_with_retry).
        if self.preview_mode:
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
                            f"⚠️ Bypassing branch protection check for {repo_owner}/{repo_name}#{pr_info.number}: {test_result[1]} (--force={self.force_level})"
                        )
                        # When bypassing, return early to allow merge
                        return (
                            True,
                            f"branch protection check bypassed (--force={self.force_level})",
                        )
                    else:
                        return False, test_result[1]

            except Exception as e:
                # If we can't predict the outcome, continue with other checks
                self.log.debug(
                    f"Could not predict merge outcome for {repo_owner}/{repo_name}#{pr_info.number}: {e}"
                )

        # Additional checks based on PR state
        # aislop-ignore-next-line ai-slop/python-repetitive-dispatch -- branch drives multi-step control flow, not a value lookup
        if pr_info.mergeable_state == "blocked":
            # Check if Copilot comments might be the blocker
            if self.dismiss_copilot and self._copilot_handler:
                has_copilot_comments = (
                    self._copilot_handler.has_blocking_copilot_comments(pr_info)
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
        elif pr_info.mergeable_state == "behind":
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
