# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum

from rich.console import Console

from .github_async import GitHubAsync
from .models import ComparisonResult, PullRequestInfo
from .progress_tracker import MergeProgressTracker


class CloseStatus(Enum):
    """Status of a PR close operation."""

    PENDING = "pending"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CloseResult:
    """Result of a PR close operation."""

    pr_info: PullRequestInfo
    status: CloseStatus
    error: str | None = None
    attempts: int = 0
    duration: float = 0.0


class AsyncCloseManager:
    """
    Manages parallel closing of pull requests.

    This class handles:
    - Concurrent closing of PRs
    - Progress tracking and error handling
    - Rate limit-aware processing
    """

    def __init__(
        self,
        token: str,
        max_retries: int = 2,
        concurrency: int = 5,
        progress_tracker: MergeProgressTracker | None = None,
        dry_run: bool = False,
    ):
        self.token = token
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.progress_tracker = progress_tracker
        self.dry_run = dry_run
        self.log = logging.getLogger(__name__)

        # Track close operations
        self._close_semaphore = asyncio.Semaphore(concurrency)
        self._results: list[CloseResult] = []
        self._github_client: GitHubAsync | None = None
        self._console = Console()

    async def __aenter__(self):
        """Async context manager entry."""
        self._github_client = GitHubAsync(token=self.token)
        await self._github_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._github_client:
            await self._github_client.__aexit__(exc_type, exc_val, exc_tb)

    async def close_prs_parallel(
        self,
        pr_list: list[tuple[PullRequestInfo, ComparisonResult | None]],
    ) -> list[CloseResult]:
        """
        Close multiple PRs in parallel.

        Args:
            pr_list: List of (PullRequestInfo, ComparisonResult) tuples

        Returns:
            List of CloseResult objects with operation results
        """
        if not pr_list:
            return []

        # Reset results for this batch
        self._results = []

        # Create tasks for all PRs
        tasks = [self._close_single_pr(pr_info) for pr_info, _ in pr_list]

        # Execute in parallel
        await asyncio.gather(*tasks, return_exceptions=True)

        return self._results

    async def _close_single_pr(self, pr_info: PullRequestInfo) -> CloseResult:
        """
        Close a single pull request.

        Args:
            pr_info: Pull request information

        Returns:
            CloseResult with operation status
        """
        start_time = time.time()
        result = CloseResult(
            pr_info=pr_info,
            status=CloseStatus.PENDING,
        )

        async with self._close_semaphore:
            try:
                # Update progress if tracker is available
                if self.progress_tracker:
                    self.progress_tracker.update_operation(
                        f"Closing {pr_info.repository_full_name}#{pr_info.number}"
                    )

                # Check if PR is already closed
                if pr_info.state != "open":
                    result.status = CloseStatus.SKIPPED
                    result.error = f"PR is already {pr_info.state}"
                    self._console.print(
                        f"⏭️ Skipped: {pr_info.html_url} [already {pr_info.state}]"
                    )
                    self.log.info(
                        f"⏭️  Skipping {pr_info.repository_full_name}#{pr_info.number}: {result.error}"
                    )
                    self._results.append(result)
                    return result

                # Check if PR is a draft
                if pr_info.mergeable_state == "draft":
                    result.status = CloseStatus.SKIPPED
                    result.error = "PR is a draft"
                    self._console.print(f"⏭️ Skipped: {pr_info.html_url} [draft PR]")
                    self.log.info(
                        f"⏭️  Skipping {pr_info.repository_full_name}#{pr_info.number}: {result.error}"
                    )
                    self._results.append(result)
                    return result

                # Parse repository info
                repo_parts = pr_info.repository_full_name.split("/")
                if len(repo_parts) != 2:
                    result.status = CloseStatus.FAILED
                    result.error = (
                        f"Invalid repository name: {pr_info.repository_full_name}"
                    )
                    self._console.print(
                        f"❌ Failed: {pr_info.html_url} [{result.error}]"
                    )
                    self._results.append(result)
                    return result

                repo_owner, repo_name = repo_parts

                # Perform close operation
                if self.dry_run:
                    # Dry-run mode: just mark as would-close
                    result.status = CloseStatus.CLOSED
                    self._console.print(f"☑️ Would close: {pr_info.html_url}")
                    self.log.info(
                        f"☑️  Would close {pr_info.repository_full_name}#{pr_info.number} (dry-run)"
                    )
                else:
                    # Actually close the PR
                    result.status = CloseStatus.CLOSING
                    if self.progress_tracker:
                        self.progress_tracker.increment_closed()

                    attempt = 0
                    success = False

                    while attempt < self.max_retries and not success:
                        attempt += 1
                        result.attempts = attempt

                        try:
                            if self._github_client is None:
                                raise RuntimeError("GitHub client not initialized")

                            await self._github_client.close_pull_request(
                                repo_owner, repo_name, pr_info.number
                            )

                            result.status = CloseStatus.CLOSED
                            success = True
                            self._console.print(f"✅ Success: {pr_info.html_url}")
                            self.log.info(
                                f"✅  Closed {pr_info.repository_full_name}#{pr_info.number}"
                            )

                        except Exception as e:
                            error_msg = str(e)
                            self.log.warning(
                                f"Attempt {attempt}/{self.max_retries} failed for "
                                f"{pr_info.repository_full_name}#{pr_info.number}: {error_msg}"
                            )

                            if attempt >= self.max_retries:
                                result.status = CloseStatus.FAILED
                                result.error = error_msg
                                self._console.print(
                                    f"❌ Failed: {pr_info.html_url} [{error_msg}]"
                                )
                                self.log.error(
                                    f"❌  Failed to close {pr_info.repository_full_name}#{pr_info.number}: {error_msg}"
                                )
                            else:
                                # Wait before retrying
                                await asyncio.sleep(2**attempt)

            except Exception as e:
                result.status = CloseStatus.FAILED
                result.error = f"Unexpected error: {e}"
                self._console.print(f"❌ Failed: {pr_info.html_url} [{result.error}]")
                self.log.error(
                    f"❌  Unexpected error closing {pr_info.repository_full_name}#{pr_info.number}: {e}"
                )

            finally:
                result.duration = time.time() - start_time
                self._results.append(result)

        return result

    def get_results(self) -> list[CloseResult]:
        """Get all close results."""
        return self._results

    def get_summary(self) -> dict[str, int]:
        """
        Get summary statistics of close operations.

        Returns:
            Dictionary with counts of closed, failed, and skipped PRs
        """
        closed = sum(1 for r in self._results if r.status == CloseStatus.CLOSED)
        failed = sum(1 for r in self._results if r.status == CloseStatus.FAILED)
        skipped = sum(1 for r in self._results if r.status == CloseStatus.SKIPPED)

        return {
            "closed": closed,
            "failed": failed,
            "skipped": skipped,
            "total": len(self._results),
        }
