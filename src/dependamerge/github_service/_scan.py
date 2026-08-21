# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""The organization-wide unmergeable-PR scan."""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from ..models import (
    OrganizationScanResult,
    UnmergeablePR,
)
from ._base import _GitHubServiceBase


class _ScanMixin(_GitHubServiceBase):
    """Organization scanning for ``GitHubService``."""

    async def scan_organization(
        self, org: str, include_drafts: bool = False
    ) -> OrganizationScanResult:
        """
        Scan an organization for unmergeable PRs using GraphQL in a batched,
        parallel fashion with bounded concurrency.

        Args:
            org: The organization name to scan.
            include_drafts: If True, include draft PRs in the results. If False (default),
                          filter out PRs that are only blocked due to draft status.

        Returns:
            OrganizationScanResult with aggregated data and errors.
        """
        errors: list[str] = []
        unmergeable_prs: list[UnmergeablePR] = []
        total_repositories = 0
        scanned_repositories = 0
        total_prs = 0

        # Process repositories with bounded parallelism
        # (repo total is set automatically by _iter_org_repositories
        # on the first GraphQL page via totalCount)
        async def process_repo(
            repo_node: dict[str, Any],
        ) -> tuple[list[UnmergeablePR], int, int, list[str]]:
            async with self._repo_semaphore:
                repo_errors: list[str] = []
                repo_full_name = repo_node.get("nameWithOwner", "unknown/unknown")
                if self._progress:
                    self._progress.start_repository(repo_full_name)
                try:
                    owner, name = self._split_owner_repo(repo_full_name)
                    first_nodes, page_info = await self._fetch_repo_prs_first_page(
                        owner, name
                    )
                    prs_nodes: list[dict[str, Any]] = list(first_nodes)
                    has_next = bool(page_info.get("hasNextPage"))
                    end_cursor = page_info.get("endCursor")

                    # Include additional pages of PRs if present
                    if has_next:
                        async for pr_node in self._iter_repo_open_prs_pages(
                            owner, name, end_cursor
                        ):
                            prs_nodes.append(pr_node)

                    repo_total_prs = len(prs_nodes)

                    # Analyze PRs concurrently within this repository
                    tasks = [
                        self._analyze_pr_node(repo_full_name, pr_node, include_drafts)
                        for pr_node in prs_nodes
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    repo_unmergeables: list[UnmergeablePR] = []
                    for r in results:
                        if isinstance(r, Exception):
                            repo_errors.append(
                                f"Error analyzing PR in {repo_full_name}: {r}"
                            )
                            if self._progress:
                                self._progress.add_error()
                            continue
                        if r is not None and isinstance(r, UnmergeablePR):
                            repo_unmergeables.append(r)

                    if self._progress:
                        self._progress.complete_repository(len(repo_unmergeables))

                    return repo_unmergeables, repo_total_prs, 1, repo_errors
                except Exception as e:
                    if self._progress:
                        self._progress.add_error()
                    # Return no unmergeables, no prs counted, no scanned increment, but record error
                    return (
                        [],
                        0,
                        0,
                        [f"Error scanning repository {repo_full_name}: {e}"],
                    )

        tasks: list[asyncio.Task[Any]] = []
        async for repo in self._iter_org_repositories_with_open_prs(org):
            tasks.append(asyncio.create_task(process_repo(repo)))

        total_repositories = len(tasks)

        if tasks:
            results = await asyncio.gather(*tasks)
            for repo_unmergeables, repo_prs_count, scanned_inc, repo_errors in results:
                unmergeable_prs.extend(repo_unmergeables)
                total_prs += repo_prs_count
                scanned_repositories += scanned_inc
                if repo_errors:
                    errors.extend(repo_errors)

        return OrganizationScanResult(
            organization=org,
            total_repositories=total_repositories,
            scanned_repositories=scanned_repositories,
            total_prs=total_prs,
            unmergeable_prs=unmergeable_prs,
            scan_timestamp=datetime.now().isoformat(),
            errors=errors,
        )
