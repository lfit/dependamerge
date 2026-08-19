# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""Organization status reporting: tags, releases and status icons."""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from ..models import (
    OrganizationStatus,
    RepositoryStatus,
)
from ._base import _GitHubServiceBase


class _StatusMixin(_GitHubServiceBase):
    """Organization status gathering for ``GitHubService``."""

    async def gather_organization_status(self, org: str) -> OrganizationStatus:
        """
        Gather repository status information for an organization.

        This collects:
        - Latest tags and releases
        - Open and merged pull requests
        - PRs affecting action files or workflows

        Returns:
            OrganizationStatus with aggregated data and errors.
        """
        errors: list[str] = []
        repository_statuses: list[RepositoryStatus] = []
        total_repositories = 0
        scanned_repositories = 0

        # Process repositories with bounded parallelism
        # (repo total is set automatically by _iter_org_repositories
        # on the first GraphQL page via totalCount)
        async def process_repo_status(
            repo_node: dict[str, Any],
        ) -> tuple[RepositoryStatus | None, int, list[str]]:
            async with self._repo_semaphore:
                repo_errors: list[str] = []
                repo_full_name = repo_node.get("nameWithOwner", "unknown/unknown")
                if self._progress:
                    self._progress.start_repository(repo_full_name)
                try:
                    owner, name = self._split_owner_repo(repo_full_name)

                    latest_tag, tag_date = await self._get_latest_tag(owner, name)
                    latest_release, release_date = await self._get_latest_release(
                        owner, name
                    )

                    # Determine status icon
                    status_icon = self._determine_status_icon(
                        latest_tag, latest_release, tag_date, release_date
                    )

                    pr_stats = await self._gather_pr_statistics(
                        owner, name, tag_date or release_date
                    )

                    repo_status = RepositoryStatus(
                        repository_name=name,
                        latest_tag=latest_tag,
                        latest_release=latest_release,
                        tag_date=tag_date,
                        release_date=release_date,
                        status_icon=status_icon,
                        **pr_stats,
                    )

                    if self._progress:
                        self._progress.complete_repository(0)

                    return repo_status, 1, repo_errors
                except Exception as e:
                    if self._progress:
                        self._progress.add_error()
                    return None, 0, [f"Error scanning repository {repo_full_name}: {e}"]

        tasks: list[asyncio.Task[Any]] = []
        async for repo in self._iter_org_repositories(org):
            tasks.append(asyncio.create_task(process_repo_status(repo)))

        total_repositories = len(tasks)

        if tasks:
            results = await asyncio.gather(*tasks)
            for repo_status, scanned_inc, repo_errors in results:
                if repo_status:
                    repository_statuses.append(repo_status)
                scanned_repositories += scanned_inc
                if repo_errors:
                    errors.extend(repo_errors)

        return OrganizationStatus(
            organization=org,
            total_repositories=total_repositories,
            scanned_repositories=scanned_repositories,
            repository_statuses=repository_statuses,
            scan_timestamp=datetime.now().isoformat(),
            errors=errors,
        )

    async def _get_latest_tag(
        self, owner: str, name: str
    ) -> tuple[str | None, str | None]:
        """Get the latest tag and its date."""
        try:
            # Use REST API to get tags
            tags_data = await self._api.get(
                f"/repos/{owner}/{name}/tags", params={"per_page": 1}
            )
            if isinstance(tags_data, list) and len(tags_data) > 0:
                tag_name = tags_data[0].get("name")
                commit_sha = (tags_data[0].get("commit") or {}).get("sha")
                if commit_sha:
                    commit_data = await self._api.get(
                        f"/repos/{owner}/{name}/commits/{commit_sha}"
                    )
                    if isinstance(commit_data, dict):
                        commit_date = (
                            commit_data.get("commit", {})
                            .get("committer", {})
                            .get("date")
                        )
                        if commit_date:
                            # Convert ISO date to YYYY/MM/DD
                            date_obj = datetime.fromisoformat(
                                commit_date.replace("Z", "+00:00")
                            )
                            formatted_date = date_obj.strftime("%Y/%m/%d")
                            return tag_name, formatted_date
                return tag_name, None
            return None, None
        except Exception as e:
            self.log.debug(f"Error getting latest tag for {owner}/{name}: {e}")
            return None, None

    async def _get_latest_release(
        self, owner: str, name: str
    ) -> tuple[str | None, str | None]:
        """Get the latest production release (not draft/pre-release) and its date."""
        try:
            # Use REST API to get releases
            releases_data = await self._api.get(f"/repos/{owner}/{name}/releases")
            if isinstance(releases_data, list):
                # Find first non-draft, non-prerelease
                for release in releases_data:
                    if not release.get("draft") and not release.get("prerelease"):
                        release_name = release.get("tag_name") or release.get("name")
                        published_at = release.get("published_at")
                        if published_at:
                            # Convert ISO date to YYYY/MM/DD
                            date_obj = datetime.fromisoformat(
                                published_at.replace("Z", "+00:00")
                            )
                            formatted_date = date_obj.strftime("%Y/%m/%d")
                            return release_name, formatted_date
                        return release_name, None
            return None, None
        except Exception as e:
            self.log.debug(f"Error getting latest release for {owner}/{name}: {e}")
            return None, None

    def _determine_status_icon(
        self,
        latest_tag: str | None,
        latest_release: str | None,
        tag_date: str | None,
        release_date: str | None,
    ) -> str:
        """
        Determine status icon based on tag and release status.

        ✅ = Tag has matching release
        ⚠️ = Tag exists but no matching release
        ❌ = Release is more recent than tag (or no tag but has release)
        """
        if latest_tag and latest_release:
            # Check if tag and release match
            if latest_tag == latest_release:
                return "✅"
            # Check if release is more recent than tag
            if tag_date and release_date:
                try:
                    tag_dt = datetime.strptime(tag_date, "%Y/%m/%d")
                    release_dt = datetime.strptime(release_date, "%Y/%m/%d")
                    if release_dt > tag_dt:
                        return "❌"
                except Exception as exc:
                    # Date parsing failed, fall through to warning icon
                    self.log.debug(f"Tag/release date parse failed: {exc}")
            return "⚠️"
        elif latest_tag and not latest_release:
            return "⚠️"
        elif latest_release and not latest_tag:
            return "❌"
        else:
            return "❌"
