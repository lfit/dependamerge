# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The per-change review and submit calls behind the submit manager.

:class:`_GerritSubmitOperationMixin` carries the REST round-trips that
act on a single change: applying a review (vote), submitting, the
review-then-submit pipeline that combines them, and the review-only bulk
variant.

It lives here rather than in ``dependamerge.gerrit.submit_manager``
purely to keep that module reviewable.  Nothing in here references
``build_client``: that name is only resolved in ``submit_manager``'s own
namespace, so that patching it there stays effective.  Every attribute
this mixin reads is established by ``GerritSubmitManager.__init__``.
The logger name is unchanged so records keep reporting as
``dependamerge.gerrit.submit_manager``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import logging
import time

from dependamerge.gerrit.client import GerritRestClient, GerritRestError
from dependamerge.gerrit.models import GerritChangeInfo, GerritSubmitResult

from ._submit_results import _GerritSubmitResultMixin

log = logging.getLogger("dependamerge.gerrit.submit_manager")


class _GerritSubmitOperationMixin(_GerritSubmitResultMixin):
    """Review and submit calls shared into ``GerritSubmitManager``."""

    # Established by GerritSubmitManager.__init__.
    _client: GerritRestClient

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

        precheck = self._precheck_submit(change, dry_run, start_time)
        if precheck is not None:
            return precheck

        try:
            review_success = self._review_change(change.number, review_labels)
            if not review_success:
                return GerritSubmitResult.failure_result(
                    change_number=change.number,
                    project=change.project,
                    error="Failed to apply review",
                    reviewed=False,
                    duration=time.time() - start_time,
                )
            reviewed = True
            log.info(
                "Applied review to %s #%d: %s",
                change.project,
                change.number,
                review_labels,
            )

            submit_success = self._submit_change(change.number)
            if not submit_success:
                return GerritSubmitResult.failure_result(
                    change_number=change.number,
                    project=change.project,
                    error="Failed to submit (change may not be submittable)",
                    reviewed=reviewed,
                    duration=time.time() - start_time,
                )
            log.info(
                "Submitted %s #%d",
                change.project,
                change.number,
            )

            return GerritSubmitResult.success_result(
                change_number=change.number,
                project=change.project,
                reviewed=reviewed,
                submitted=True,
                duration=time.time() - start_time,
            )

        except Exception as exc:
            return self._submit_error_result(change, exc, reviewed, start_time)

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
            log.warning("Failed to review change %d: %s", change_number, exc)
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
            log.warning("Failed to submit change %d: %s", change_number, exc)
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
