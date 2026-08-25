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
from .github_async import PermissionError as GitHubPermissionError
from .models import ComparisonResult, PullRequestInfo
from .output_utils import log_and_print
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
        preview_mode: bool = False,
    ):
        self.token = token
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.progress_tracker = progress_tracker
        self.preview_mode = preview_mode
        self.log = logging.getLogger(__name__)

        # Track close operations
        self._close_semaphore = asyncio.Semaphore(concurrency)
        self._results: list[CloseResult] = []
        self._github_client: GitHubAsync | None = None
        self._console = Console()

    def __repr__(self) -> str:
        """Safe repr that never exposes the token value."""
        return "AsyncCloseManager(token=***)"

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

        tasks = [self._close_single_pr(pr_info) for pr_info, _ in pr_list]

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

                target = self._resolve_close_target(pr_info, result)
                if target is None:
                    # No append here: the ``finally`` below is the single
                    # append point.  Appending on this path too would
                    # record every skipped PR twice, because ``return``
                    # inside ``try`` runs ``finally`` first.
                    return result

                repo_owner, repo_name = target

                # Perform close operation
                if self.preview_mode:
                    # Preview mode: just mark as would-close
                    result.status = CloseStatus.CLOSED
                    log_and_print(
                        self.log,
                        self._console,
                        f"☑️ Would close: {pr_info.html_url}",
                        level="info",
                    )
                else:
                    # Actually close the PR
                    await self._close_with_retries(
                        pr_info, result, repo_owner, repo_name
                    )

            except GitHubPermissionError as e:
                self._record_permission_error(pr_info, result, e)

            except Exception as e:
                self._record_unexpected_error(pr_info, result, e)

            finally:
                result.duration = time.time() - start_time
                # The single point at which a result enters ``_results``.
                # ``get_summary`` and ``get_results`` both read this list,
                # so a second append anywhere would inflate the counts the
                # close command reports.
                self._results.append(result)

        return result

    def _resolve_close_target(
        self, pr_info: PullRequestInfo, result: CloseResult
    ) -> tuple[str, str] | None:
        """
        Decide whether a PR may be closed, and against which repository.

        Returns:
            The ``(owner, name)`` pair to close against, or None when the
            PR must not be closed. In the None case ``result`` already
            carries its terminal status and the outcome has been logged.
        """
        # Check if PR is already closed
        if pr_info.state != "open":
            result.status = CloseStatus.SKIPPED
            result.error = f"PR is already {pr_info.state}"
            log_and_print(
                self.log,
                self._console,
                f"⏭️ Skipped: {pr_info.html_url} [already {pr_info.state}]",
                level="info",
            )
            return None

        # Check if PR is a draft
        if pr_info.mergeable_state == "draft":
            result.status = CloseStatus.SKIPPED
            result.error = "PR is a draft"
            log_and_print(
                self.log,
                self._console,
                f"⏭️ Skipped: {pr_info.html_url} [draft PR]",
                level="info",
            )
            return None

        repo_parts = pr_info.repository_full_name.split("/")
        if len(repo_parts) != 2:
            result.status = CloseStatus.FAILED
            result.error = f"Invalid repository name: {pr_info.repository_full_name}"
            log_and_print(
                self.log,
                self._console,
                f"❌ Failed: {pr_info.html_url} [{result.error}]",
                level="error",
            )
            return None

        repo_owner, repo_name = repo_parts
        return repo_owner, repo_name

    async def _close_with_retries(
        self,
        pr_info: PullRequestInfo,
        result: CloseResult,
        repo_owner: str,
        repo_name: str,
    ) -> None:
        """Issue the close call, recording the outcome on ``result``.

        Retries up to ``max_retries`` times with exponential backoff.
        """
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
                log_and_print(
                    self.log,
                    self._console,
                    f"✅ Closed: {pr_info.html_url}",
                    level="info",
                )

            except GitHubPermissionError:
                # Broader than this clause, and deliberately ahead of it:
                # a permission denial must reach the dedicated handler in
                # ``_close_single_pr``, which explains *which* scope the
                # token lacks.  Letting the generic clause below catch it
                # would retry pointlessly --- the token cannot gain scopes
                # between attempts --- and discard that guidance, turning
                # an actionable failure into a vague one.
                raise

            except Exception as e:
                error_msg = str(e)
                self.log.warning(
                    f"Attempt {attempt}/{self.max_retries} failed for "
                    f"{pr_info.repository_full_name}#{pr_info.number}: {error_msg}"
                )

                if attempt >= self.max_retries:
                    result.status = CloseStatus.FAILED
                    result.error = error_msg
                    self._console.print(f"❌ Failed: {pr_info.html_url} [{error_msg}]")
                    self.log.error(
                        f"Failed to close {pr_info.repository_full_name}#{pr_info.number}: {error_msg}"
                    )
                else:
                    # Wait before retrying
                    await asyncio.sleep(2**attempt)

    def _record_permission_error(
        self,
        pr_info: PullRequestInfo,
        result: CloseResult,
        error: GitHubPermissionError,
    ) -> None:
        """Fail the close on permission denial, and explain the fix."""
        result.status = CloseStatus.FAILED
        result.error = str(error)

        operation_desc = error.operation.replace("_", " ")
        log_and_print(
            self.log,
            self._console,
            f"❌ Failed: {pr_info.html_url} [permission denied: {operation_desc}]",
            level="error",
        )

        # Provide token-specific guidance
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
        self.log.error(
            f"Permission error closing {pr_info.repository_full_name}#{pr_info.number}: {error}"
        )

    def _record_unexpected_error(
        self,
        pr_info: PullRequestInfo,
        result: CloseResult,
        error: Exception,
    ) -> None:
        """Fail the close after an error we do not handle specifically."""
        result.status = CloseStatus.FAILED
        result.error = f"Unexpected error: {error}"
        self._console.print(f"❌ Failed: {pr_info.html_url} [{result.error}]")
        self.log.error(
            f"Unexpected error closing {pr_info.repository_full_name}#{pr_info.number}: {error}"
        )

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
