# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""PR statistics behind the organization status report."""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..bot_identity import is_automation_author
from ._base import _GitHubServiceBase


class _StatusStatsMixin(_GitHubServiceBase):
    """PR statistics for ``GitHubService``."""

    async def _gather_pr_statistics(
        self, owner: str, name: str, since_date: str | None
    ) -> dict[str, int]:
        """
        Gather PR statistics for a repository.

        Returns dict with counts for:
        - open_prs_human, open_prs_automation
        - merged_prs_human, merged_prs_automation
        - action_prs_human, action_prs_automation
        - workflow_prs_human, workflow_prs_automation
        """
        stats = {
            "open_prs_human": 0,
            "open_prs_automation": 0,
            "merged_prs_human": 0,
            "merged_prs_automation": 0,
            "action_prs_human": 0,
            "action_prs_automation": 0,
            "workflow_prs_human": 0,
            "workflow_prs_automation": 0,
        }

        try:
            first_nodes, page_info = await self._fetch_repo_prs_first_page(owner, name)
            open_prs = list(first_nodes)

            # Get additional pages if needed
            if page_info.get("hasNextPage"):
                async for pr_node in self._iter_repo_open_prs_pages(
                    owner, name, page_info.get("endCursor")
                ):
                    open_prs.append(pr_node)

            # Count open PRs
            for pr in open_prs:
                author = (pr.get("author") or {}).get("login", "").lower()
                is_automation = self._is_automation_author(author)

                if is_automation:
                    stats["open_prs_automation"] += 1
                else:
                    stats["open_prs_human"] += 1

                # Check if PR affects actions or workflows
                files = (pr.get("files") or {}).get("nodes", []) or []
                affects_action = self._affects_action_files(files)
                affects_workflow = self._affects_workflow_files(files)

                if affects_action:
                    if is_automation:
                        stats["action_prs_automation"] += 1
                    else:
                        stats["action_prs_human"] += 1

                if affects_workflow:
                    if is_automation:
                        stats["workflow_prs_automation"] += 1
                    else:
                        stats["workflow_prs_human"] += 1

            # Get merged PRs since the last tag/release
            if since_date:
                merged_prs = await self._get_merged_prs_since(owner, name, since_date)
                for pr in merged_prs:
                    author = (pr.get("user") or {}).get("login", "").lower()
                    is_automation = self._is_automation_author(author)

                    if is_automation:
                        stats["merged_prs_automation"] += 1
                    else:
                        stats["merged_prs_human"] += 1

        except Exception as e:
            self.log.debug(f"Error gathering PR statistics for {owner}/{name}: {e}")

        return stats

    def _is_automation_author(self, author: str) -> bool:
        """Check if author is an automation tool.

        Delegates to the shared :func:`bot_identity.is_automation_author`
        so REST and GraphQL login forms are classified identically.
        """
        return is_automation_author(author)

    def _affects_action_files(self, files: list[dict[str, Any]]) -> bool:
        """Check if files include action definition or implementation files."""
        action_patterns = [
            "action.yaml",
            "action.yml",
            "Dockerfile",  # Action Dockerfiles
        ]

        for file_node in files:
            path = file_node.get("path", "")
            filename = path.split("/")[-1] if "/" in path else path

            if filename.lower() in [p.lower() for p in action_patterns]:
                return True

            # Check for JavaScript action files (in src/ or lib/ directories)
            if path.startswith(("src/", "lib/")) and path.endswith(".js"):
                return True

        return False

    def _affects_workflow_files(self, files: list[dict[str, Any]]) -> bool:
        """Check if files include GitHub workflow or configuration files."""
        for file_node in files:
            path = file_node.get("path", "")

            # Check if file is in .github directory
            if path.startswith(".github/"):
                # Exclude non-workflow files
                if path.endswith((".md", ".txt", ".png", ".jpg", ".gif")):
                    continue

                # Include workflow files and other YAML configs
                if path.startswith(".github/workflows/") or path.endswith(
                    (".yml", ".yaml")
                ):
                    return True

        return False

    async def _get_merged_prs_since(
        self, owner: str, name: str, since_date: str
    ) -> list[dict[str, Any]]:
        """Get merged PRs since a specific date."""
        try:
            # Convert date format from YYYY/MM/DD to ISO format
            date_obj = datetime.strptime(since_date, "%Y/%m/%d")
            iso_date = date_obj.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Use REST API to get merged PRs
            merged_prs = []
            page = 1
            per_page = 100

            while True:
                params = {
                    "state": "closed",
                    "sort": "updated",
                    "direction": "desc",
                    "per_page": per_page,
                    "page": page,
                }

                prs_data = await self._api.get(
                    f"/repos/{owner}/{name}/pulls", params=params
                )

                if not isinstance(prs_data, list) or len(prs_data) == 0:
                    break

                for pr in prs_data:
                    # Check if PR was merged
                    merged_at = pr.get("merged_at")
                    if merged_at:
                        # Check if merged after the since_date
                        if merged_at >= iso_date:
                            merged_prs.append(pr)

                # Check if we've reached the last page
                if len(prs_data) < per_page:
                    break

                page += 1

                # Limit to avoid excessive API calls
                if page > 10:
                    break

            return merged_prs

        except Exception as e:
            self.log.debug(f"Error getting merged PRs for {owner}/{name}: {e}")
            return []
