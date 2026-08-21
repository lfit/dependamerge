# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""Bulk open-PR collection for a repository or a whole owner."""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from typing import Any

from ..github_async import (
    RateLimitError,
    SecondaryRateLimitError,
)
from ..models import PullRequestInfo
from ._base import _GitHubServiceBase


class _FetchMixin(_GitHubServiceBase):
    """Open-PR collection for ``GitHubService``."""

    async def _collect_repo_open_prs(
        self,
        owner: str,
        repo: str,
        *,
        only_automation: bool,
    ) -> list[PullRequestInfo]:
        """Fetch + convert + filter the open PRs of a single repository.

        This is the shared body used by both :meth:`fetch_repo_open_prs`
        (single-repository bulk merge) and :meth:`fetch_owner_open_prs`
        (owner-wide bulk merge), so the GraphQL-node-to-PullRequestInfo
        conversion and automation filtering live in exactly one place.

        Unlike its callers it does not emit ``start_repository`` /
        ``complete_repository`` progress events; the caller owns the
        per-repository progress lifecycle.
        """
        repo_full_name = f"{owner}/{repo}"

        first_nodes, page_info = await self._fetch_repo_prs_first_page(owner, repo)
        pr_nodes = list(first_nodes)
        has_next = bool(page_info.get("hasNextPage"))
        end_cursor = page_info.get("endCursor") or None

        # Fetch additional pages if present
        if has_next:
            async for pr_node in self._iter_repo_open_prs_pages(
                owner, repo, end_cursor
            ):
                pr_nodes.append(pr_node)

        results: list[PullRequestInfo] = []
        for pr_node in pr_nodes:
            pr_info = self.to_pull_request_info(repo_full_name, pr_node)

            if self._progress:
                self._progress.analyze_pr(pr_info.number, repo_full_name)

            # Filter by automation author if requested
            if only_automation and not self._is_automation_author(pr_info.author):
                continue

            results.append(pr_info)

        return results

    async def fetch_repo_open_prs(
        self,
        owner: str,
        repo: str,
        *,
        only_automation: bool = True,
    ) -> list[PullRequestInfo]:
        """
        Fetch all open PRs for a specific repository.

        This is used for repository-scoped bulk operations where we don't
        need to scan across an organization. It reuses the same GraphQL
        pagination infrastructure used by find_similar_prs.

        Args:
            owner: Repository owner (user or organization).
            repo: Repository name.
            only_automation: If True, only return PRs from automation tools.
                           If False, return all open PRs.

        Returns:
            List of PullRequestInfo for matching open PRs.
        """
        repo_full_name = f"{owner}/{repo}"

        if self._progress:
            self._progress.start_repository(repo_full_name)
            self._progress.update_operation(f"Fetching open PRs from {repo_full_name}")

        results = await self._collect_repo_open_prs(
            owner, repo, only_automation=only_automation
        )

        if self._progress:
            self._progress.complete_repository(len(results))

        return results

    async def fetch_owner_open_prs(
        self,
        owner: str,
        *,
        only_automation: bool = True,
    ) -> tuple[list[PullRequestInfo], list[str]]:
        """Fetch open PRs across every in-scope repository of an owner.

        Enumerates the owner's non-archived, non-fork repositories
        (organization or user account, resolved at runtime) and fetches
        their open PRs with bounded per-repository concurrency, reusing
        the same per-repo body as :meth:`fetch_repo_open_prs`.

        Per-repository failures are isolated: a transient error scanning
        one repository is recorded and enumeration continues, so a single
        bad repository never aborts an owner-wide run.  Global
        rate-limit / secondary-rate-limit errors are *not* swallowed —
        they propagate so the API layer's backoff governs the whole run.

        Args:
            owner: The organization or user login.
            only_automation: If True, only return PRs from automation tools.

        Returns:
            A ``(prs, errors)`` tuple: the collected PRs across all
            repositories, and a list of human-readable per-repository
            error strings (empty when everything succeeded).
        """

        async def process_repo(
            repo_node: dict[str, Any],
        ) -> tuple[list[PullRequestInfo], list[str]]:
            async with self._repo_semaphore:
                repo_full_name = repo_node.get("nameWithOwner", "unknown/unknown")
                if self._progress:
                    self._progress.start_repository(repo_full_name)
                try:
                    repo_owner, repo_name = self._split_owner_repo(repo_full_name)
                    prs = await self._collect_repo_open_prs(
                        repo_owner, repo_name, only_automation=only_automation
                    )
                    if self._progress:
                        self._progress.complete_repository(len(prs))
                    return prs, []
                except (RateLimitError, SecondaryRateLimitError):
                    # Global rate limiting must abort the whole run rather
                    # than be recorded as a per-repo error and skipped.
                    raise
                except Exception as e:
                    if self._progress:
                        # Count the error *and* mark the repository as
                        # processed.  Without the ``complete_repository``
                        # call the per-repo counter would never advance
                        # for a failed repo, leaving the progress fraction
                        # stuck below 100% and the "Scanning <repo>"
                        # label stale once the run finishes.  Passing 0
                        # adds nothing to the unmergeable tally.
                        self._progress.add_error()
                        self._progress.complete_repository(0)
                    return [], [f"Error scanning repository {repo_full_name}: {e}"]

        all_prs: list[PullRequestInfo] = []
        errors: list[str] = []

        # Bounded producer/consumer pipeline.  A fixed pool of workers
        # (sized to the repo-concurrency limit) drains repository nodes
        # from a queue fed by the paginated iterator.  This caps in-flight
        # work — both pending tasks and buffered nodes — instead of
        # materialising one task per repository up front, which matters
        # for owners with thousands of repositories.  A propagated
        # rate-limit error from any worker tears the whole pipeline down
        # (see the teardown below), preserving the global-throttle
        # semantics of aborting the run rather than skipping repos.
        worker_count = max(1, self._max_repo_tasks)
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=worker_count * 2
        )

        async def producer() -> None:
            async for repo in self._iter_owner_repositories(owner):
                await queue.put(repo)
            # One sentinel per worker so each terminates once the backlog
            # drains.
            for _ in range(worker_count):
                await queue.put(None)

        async def worker() -> None:
            while True:
                repo = await queue.get()
                if repo is None:
                    return
                repo_prs, repo_errors = await process_repo(repo)
                # Safe to mutate the shared lists without a lock: asyncio
                # is single-threaded and ``extend`` does not await.
                all_prs.extend(repo_prs)
                errors.extend(repo_errors)

        producer_task = asyncio.create_task(producer())
        worker_tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]
        pipeline = [producer_task, *worker_tasks]
        try:
            # No return_exceptions: a propagated rate-limit error aborts
            # the pipeline (the desired behaviour for global throttling).
            await asyncio.gather(*pipeline)
        except BaseException:
            # Tear the pipeline down so no worker is left blocked on the
            # queue, then re-raise the original (e.g. rate-limit) error.
            for task in pipeline:
                task.cancel()
            await asyncio.gather(*pipeline, return_exceptions=True)
            raise

        return all_prs, errors
