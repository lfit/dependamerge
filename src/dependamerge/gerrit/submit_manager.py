# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Gerrit submit manager for parallel review and submit operations.

This module provides the GerritSubmitManager class for handling bulk
approval (+2 Code-Review) and submit operations on Gerrit changes.

It supports:
- Parallel submission of multiple changes
- Review (vote) operations with configurable labels
- Submit with pre-flight checks (submittable status)
- Error handling and result tracking
- Dry-run mode for previewing operations
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING, Any

from dependamerge.gerrit.client import (
    GerritAuthError,
    GerritRestError,
    build_client,
)
from dependamerge.gerrit.models import (
    GerritChangeInfo,
    GerritComparisonResult,
    GerritSubmitResult,
)

if TYPE_CHECKING:
    from dependamerge.progress_tracker import ProgressTracker


log = logging.getLogger("dependamerge.gerrit.submit_manager")


class SubmitStatus(str, Enum):
    """Status values for submit operations."""

    PENDING = "pending"
    REVIEWING = "reviewing"
    REVIEWED = "reviewed"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class GerritSubmitManager:
    """
    Manages parallel approval and submission of Gerrit changes.

    This class handles the workflow of reviewing changes (applying
    Code-Review +2 votes) and submitting them.
    """

    def __init__(
        self,
        host: str,
        base_path: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 30.0,
        max_workers: int = 5,
        progress_tracker: ProgressTracker | None = None,
    ) -> None:
        """
        Initialize the submit manager.

        Args:
            host: Gerrit server hostname.
            base_path: Optional base path (e.g., "infra").
            username: HTTP username for authentication.
            password: HTTP password for authentication.
            timeout: Request timeout in seconds.
            max_workers: Maximum parallel workers for submissions.
            progress_tracker: Optional progress tracker for UI feedback.
        """
        self.host = host
        self.base_path = base_path
        self._max_workers = max_workers
        self._progress_tracker = progress_tracker

        # Build REST client
        self._client = build_client(
            host,
            base_path=base_path,
            timeout=timeout,
            username=username,
            password=password,
        )

        if not self._client.is_authenticated:
            log.warning(
                "GerritSubmitManager initialized without authentication. "
                "Review and submit operations will fail."
            )

        log.debug(
            "GerritSubmitManager initialized: host=%s, base_path=%s, "
            "max_workers=%d, auth=%s",
            host,
            base_path,
            max_workers,
            "yes" if self._client.is_authenticated else "no",
        )

    @property
    def is_authenticated(self) -> bool:
        """Check if the manager has authentication credentials."""
        return self._client.is_authenticated

    def submit_changes(
        self,
        changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
        review_labels: dict[str, int] | None = None,
        dry_run: bool = False,
    ) -> list[GerritSubmitResult]:
        """
        Submit multiple changes sequentially.

        Args:
            changes: List of (change, comparison_result) tuples.
            review_labels: Labels to apply (default: {"Code-Review": 2}).
            dry_run: If True, simulate operations without making changes.

        Returns:
            List of GerritSubmitResult for each change.
        """
        if review_labels is None:
            review_labels = {"Code-Review": 2}

        results: list[GerritSubmitResult] = []

        for change, _comparison in changes:
            result = self._submit_single_change(
                change, review_labels, dry_run
            )
            results.append(result)

        return results

    def submit_changes_parallel(
        self,
        changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
        review_labels: dict[str, int] | None = None,
        dry_run: bool = False,
    ) -> list[GerritSubmitResult]:
        """
        Submit multiple changes in parallel.

        Args:
            changes: List of (change, comparison_result) tuples.
            review_labels: Labels to apply (default: {"Code-Review": 2}).
            dry_run: If True, simulate operations without making changes.

        Returns:
            List of GerritSubmitResult for each change.
        """
        if review_labels is None:
            review_labels = {"Code-Review": 2}

        if not changes:
            return []

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [
                executor.submit(
                    self._submit_single_change, change, review_labels, dry_run
                )
                for change, _comparison in changes
            ]

            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    log.error("Unexpected error in parallel submit: %s", exc)
                    # Create a generic failure result
                    results.append(
                        GerritSubmitResult.failure_result(
                            change_number=0,
                            project="unknown",
                            error=str(exc),
                        )
                    )

        return results

    def _submit_single_change(
        self,
        change: GerritChangeInfo,
        review_labels: dict[str, int],
        dry_run: bool,
    ) -> GerritSubmitResult:
        """
        Submit a single change (review + submit).

        Args:
            change: The change to submit.
            review_labels: Labels to apply.
            dry_run: If True, simulate without making changes.

        Returns:
            GerritSubmitResult indicating success or failure.
        """
        start_time = time.time()
        reviewed = False
        submitted = False

        try:
            # Check if change can be submitted
            if not change.is_open:
                return GerritSubmitResult.failure_result(
                    change_number=change.number,
                    project=change.project,
                    error=f"Change is not open (status: {change.status})",
                    duration=time.time() - start_time,
                )

            if change.work_in_progress:
                return GerritSubmitResult.failure_result(
                    change_number=change.number,
                    project=change.project,
                    error="Change is marked as Work In Progress",
                    duration=time.time() - start_time,
                )

            if dry_run:
                log.info(
                    "[DRY RUN] Would review and submit %s #%d",
                    change.project,
                    change.number,
                )
                return GerritSubmitResult.success_result(
                    change_number=change.number,
                    project=change.project,
                    reviewed=True,
                    submitted=True,
                    duration=time.time() - start_time,
                )

            # Step 1: Apply review (vote)
            review_success = self._review_change(
                change.number, review_labels
            )
            if review_success:
                reviewed = True
                log.info(
                    "Applied review to %s #%d: %s",
                    change.project,
                    change.number,
                    review_labels,
                )
            else:
                return GerritSubmitResult.failure_result(
                    change_number=change.number,
                    project=change.project,
                    error="Failed to apply review",
                    reviewed=False,
                    duration=time.time() - start_time,
                )

            # Step 2: Submit the change
            submit_success = self._submit_change(change.number)
            if submit_success:
                submitted = True
                log.info(
                    "Submitted %s #%d",
                    change.project,
                    change.number,
                )
            else:
                return GerritSubmitResult.failure_result(
                    change_number=change.number,
                    project=change.project,
                    error="Failed to submit (change may not be submittable)",
                    reviewed=reviewed,
                    duration=time.time() - start_time,
                )

            return GerritSubmitResult.success_result(
                change_number=change.number,
                project=change.project,
                reviewed=reviewed,
                submitted=submitted,
                duration=time.time() - start_time,
            )

        except GerritAuthError as exc:
            log.error(
                "Authentication error for %s #%d: %s",
                change.project,
                change.number,
                exc,
            )
            return GerritSubmitResult.failure_result(
                change_number=change.number,
                project=change.project,
                error=f"Authentication error: {exc}",
                reviewed=reviewed,
                duration=time.time() - start_time,
            )

        except GerritRestError as exc:
            log.error(
                "REST error for %s #%d: %s",
                change.project,
                change.number,
                exc,
            )
            return GerritSubmitResult.failure_result(
                change_number=change.number,
                project=change.project,
                error=f"REST error: {exc}",
                reviewed=reviewed,
                duration=time.time() - start_time,
            )

        except Exception as exc:
            log.exception(
                "Unexpected error for %s #%d: %s",
                change.project,
                change.number,
                exc,
            )
            return GerritSubmitResult.failure_result(
                change_number=change.number,
                project=change.project,
                error=f"Unexpected error: {exc}",
                reviewed=reviewed,
                duration=time.time() - start_time,
            )

    def _review_change(
        self,
        change_number: int,
        labels: dict[str, int],
    ) -> bool:
        """
        Apply a review (vote) to a change.

        Args:
            change_number: The change number.
            labels: Labels to apply (e.g., {"Code-Review": 2}).

        Returns:
            True if successful, False otherwise.
        """
        endpoint = f"/changes/{change_number}/revisions/current/review"
        payload = {"labels": labels}

        try:
            self._client.post(endpoint, data=payload)
            return True
        except GerritRestError as exc:
            log.warning(
                "Failed to review change %d: %s", change_number, exc
            )
            return False

    def _submit_change(self, change_number: int) -> bool:
        """
        Submit a change.

        Args:
            change_number: The change number.

        Returns:
            True if successful, False otherwise.
        """
        endpoint = f"/changes/{change_number}/submit"

        try:
            self._client.post(endpoint)
            return True
        except GerritRestError as exc:
            log.warning(
                "Failed to submit change %d: %s", change_number, exc
            )
            return False

    def review_only(
        self,
        changes: list[GerritChangeInfo],
        review_labels: dict[str, int] | None = None,
        dry_run: bool = False,
    ) -> list[GerritSubmitResult]:
        """
        Apply reviews without submitting.

        Useful for approving changes that need additional verification.

        Args:
            changes: List of changes to review.
            review_labels: Labels to apply.
            dry_run: If True, simulate without making changes.

        Returns:
            List of results indicating review success/failure.
        """
        if review_labels is None:
            review_labels = {"Code-Review": 2}

        results: list[GerritSubmitResult] = []

        for change in changes:
            start_time = time.time()

            if dry_run:
                log.info(
                    "[DRY RUN] Would review %s #%d with %s",
                    change.project,
                    change.number,
                    review_labels,
                )
                results.append(
                    GerritSubmitResult.success_result(
                        change_number=change.number,
                        project=change.project,
                        reviewed=True,
                        submitted=False,
                        duration=time.time() - start_time,
                    )
                )
                continue

            success = self._review_change(change.number, review_labels)
            if success:
                results.append(
                    GerritSubmitResult.success_result(
                        change_number=change.number,
                        project=change.project,
                        reviewed=True,
                        submitted=False,
                        duration=time.time() - start_time,
                    )
                )
            else:
                results.append(
                    GerritSubmitResult.failure_result(
                        change_number=change.number,
                        project=change.project,
                        error="Failed to apply review",
                        duration=time.time() - start_time,
                    )
                )

        return results

    def get_submit_summary(
        self, results: list[GerritSubmitResult]
    ) -> dict[str, Any]:
        """
        Generate a summary of submit results.

        Args:
            results: List of submit results.

        Returns:
            Dictionary with summary statistics.
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        reviewed = sum(1 for r in results if r.reviewed)
        submitted = sum(1 for r in results if r.submitted)
        total_duration = sum(r.duration_seconds for r in results)

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "reviewed": reviewed,
            "submitted": submitted,
            "total_duration_seconds": round(total_duration, 2),
            "average_duration_seconds": (
                round(total_duration / total, 2) if total > 0 else 0.0
            ),
        }


def create_submit_manager(
    host: str,
    base_path: str | None = None,
    username: str | None = None,
    password: str | None = None,
    max_workers: int = 5,
    progress_tracker: ProgressTracker | None = None,
) -> GerritSubmitManager:
    """
    Factory function to create a GerritSubmitManager.

    Args:
        host: Gerrit server hostname.
        base_path: Optional base path.
        username: HTTP username for authentication.
        password: HTTP password for authentication.
        max_workers: Maximum parallel workers.
        progress_tracker: Optional progress tracker.

    Returns:
        Configured GerritSubmitManager instance.
    """
    return GerritSubmitManager(
        host=host,
        base_path=base_path,
        username=username,
        password=password,
        max_workers=max_workers,
        progress_tracker=progress_tracker,
    )


__all__ = [
    "GerritSubmitManager",
    "SubmitStatus",
    "create_submit_manager",
]
