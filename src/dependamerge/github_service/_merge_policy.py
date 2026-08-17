# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""Branch-protection lookup and merge-method selection."""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import Any

from ..github_graphql import GET_BRANCH_PROTECTION
from ._base import _GitHubServiceBase


class _MergePolicyMixin(_GitHubServiceBase):
    """Branch-protection and merge-method policy for ``GitHubService``."""

    async def get_branch_protection_settings(
        self, owner: str, repo: str, branch: str = "main"
    ) -> dict[str, Any] | None:
        """
        Get branch protection settings for a repository branch.

        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name (defaults to "main")

        Returns:
            Branch protection settings dict, or None if no protection or error
        """
        cache_key = f"{owner}/{repo}:{branch}"

        if cache_key in self._branch_protection_cache:
            return self._branch_protection_cache[cache_key]

        if not self._api:
            return None

        try:
            variables = {"owner": owner, "name": repo, "branch": f"refs/heads/{branch}"}

            response = await self._api.graphql(GET_BRANCH_PROTECTION, variables)

            # Debug: Log the actual response structure
            self.log.debug(f"GraphQL response for {owner}/{repo}: {response}")

            repo_data = response.get("repository")
            if not repo_data:
                self.log.debug(f"No repository data for {owner}/{repo}")
                self._branch_protection_cache[cache_key] = None
                return None

            protection = {
                "allowsMergeCommits": repo_data.get("mergeCommitAllowed", True),
                "allowsSquashMerges": repo_data.get("squashMergeAllowed", True),
                "allowsRebaseMerges": repo_data.get("rebaseMergeAllowed", True),
            }

            # Add branch protection rule settings if they exist
            ref_data = repo_data.get("ref")
            if ref_data:
                branch_protection = ref_data.get("branchProtectionRule")
                if branch_protection:
                    protection.update(branch_protection)

            self._branch_protection_cache[cache_key] = protection

            self.log.debug(
                f"Branch protection for {owner}/{repo}:{branch}: "
                f"requiresLinearHistory={protection.get('requiresLinearHistory', False)}, "
                f"allowsMergeCommits={protection.get('allowsMergeCommits')}, "
                f"allowsSquashMerges={protection.get('allowsSquashMerges')}, "
                f"allowsRebaseMerges={protection.get('allowsRebaseMerges')}"
            )

            return protection

        except Exception as e:
            error_str = str(e)
            if (
                "FORBIDDEN" in error_str
                and "Resource not accessible by personal access token" in error_str
            ):
                self.log.debug(
                    f"Cannot access branch protection for {owner}/{repo}:{branch}: Missing 'Administration: Read-only' permission. "
                    f"For fine-grained tokens, enable 'Administration: Read-only'. For classic tokens, ensure 'repo' scope is enabled."
                )
            else:
                self.log.warning(
                    f"Failed to get branch protection for {owner}/{repo}:{branch}: {e}"
                )
            # Cache the None result to avoid repeated failures
            self._branch_protection_cache[cache_key] = None
            return None

    def determine_merge_method(
        self, branch_protection: dict[str, Any] | None, default_method: str = "merge"
    ) -> str:
        """
        Determine the appropriate merge method based on branch protection settings.

        Args:
            branch_protection: Branch protection settings from GraphQL
            default_method: Default merge method to use if no restrictions

        Returns:
            Recommended merge method: "merge", "squash", or "rebase"
        """
        if not branch_protection:
            return default_method

        # If linear history is required, only rebase merge is allowed
        if branch_protection.get("requiresLinearHistory", False):
            if branch_protection.get("allowsRebaseMerges", True):
                return "rebase"
            else:
                self.log.warning(
                    "Repository requires linear history but doesn't allow rebase merges"
                )
                return default_method

        # Otherwise, prefer the default method if it's allowed
        if default_method == "merge" and branch_protection.get(
            "allowsMergeCommits", True
        ):
            return "merge"
        elif default_method == "squash" and branch_protection.get(
            "allowsSquashMerges", True
        ):
            return "squash"
        elif default_method == "rebase" and branch_protection.get(
            "allowsRebaseMerges", True
        ):
            return "rebase"

        # Fall back to first available method
        if branch_protection.get("allowsMergeCommits", True):
            return "merge"
        elif branch_protection.get("allowsSquashMerges", True):
            return "squash"
        elif branch_protection.get("allowsRebaseMerges", True):
            return "rebase"

        self.log.warning(
            f"No merge methods allowed by branch protection: {branch_protection}"
        )
        return default_method

    def _split_owner_repo(self, full_name: str) -> tuple[str, str]:
        try:
            owner, name = full_name.split("/", 1)
            return owner, name
        except Exception:
            return "unknown", "unknown"
