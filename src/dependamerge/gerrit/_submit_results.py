# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Submit status vocabulary and result construction for the submit manager.

This module holds the ``SubmitStatus`` values and
:class:`_GerritSubmitResultMixin`, which carries the parts of
``GerritSubmitManager`` that only build results: the pre-flight check
that short-circuits unsubmittable changes and dry runs, the mapping from
an exception to a failure result, and the run summary.  None of it
touches the network.

It lives here rather than in ``dependamerge.gerrit.submit_manager``
purely to keep that module reviewable.  Nothing in here references
``build_client``: that name is only resolved in ``submit_manager``'s own
namespace, so that patching it there stays effective.  The logger name
is unchanged so records keep reporting as
``dependamerge.gerrit.submit_manager``.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any

from dependamerge.gerrit.client import GerritAuthError, GerritRestError
from dependamerge.gerrit.models import GerritChangeInfo, GerritSubmitResult

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


class _GerritSubmitResultMixin:
    """Result construction shared into ``GerritSubmitManager``."""

    def _precheck_submit(
        self,
        change: GerritChangeInfo,
        dry_run: bool,
        start_time: float,
    ) -> GerritSubmitResult | None:
        """Return a short-circuit result, or None when submission may proceed.

        Rejects changes that cannot be submitted (not open, work in
        progress) and satisfies dry runs without any network calls.
        """
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
            # A dry run performs no review or submit, so report the
            # simulated success with reviewed/submitted left False.
            # Callers gate real side effects on ``submitted`` (e.g.
            # closing the corresponding GitHub PR after a Gerrit
            # submit), so a dry run must never claim it submitted.
            return GerritSubmitResult.success_result(
                change_number=change.number,
                project=change.project,
                reviewed=False,
                submitted=False,
                duration=time.time() - start_time,
            )

        return None

    def _submit_error_result(
        self,
        change: GerritChangeInfo,
        exc: Exception,
        reviewed: bool,
        start_time: float,
    ) -> GerritSubmitResult:
        """Map an exception raised during submission to a failure result.

        Classifies the exception, logs it at the matching level (auth
        and REST errors are expected; anything else is logged with a
        traceback), and returns a failure result.
        """
        if isinstance(exc, GerritAuthError):
            log.error(
                "Authentication error for %s #%d: %s",
                change.project,
                change.number,
                exc,
            )
            message = f"Authentication error: {exc}"
        elif isinstance(exc, GerritRestError):
            log.error(
                "REST error for %s #%d: %s",
                change.project,
                change.number,
                exc,
            )
            message = f"REST error: {exc}"
        else:
            log.exception(
                "Unexpected error for %s #%d: %s",
                change.project,
                change.number,
                exc,
            )
            message = f"Unexpected error: {exc}"

        return GerritSubmitResult.failure_result(
            change_number=change.number,
            project=change.project,
            error=message,
            reviewed=reviewed,
            duration=time.time() - start_time,
        )

    def get_submit_summary(self, results: list[GerritSubmitResult]) -> dict[str, Any]:
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
