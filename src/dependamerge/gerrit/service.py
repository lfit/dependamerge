# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Gerrit service layer for dependamerge.

This module provides a high-level service class for querying and operating
on Gerrit changes. It abstracts the REST API interactions and provides
methods for:

- Fetching change details
- Enumerating open changes across a server
- Finding similar changes for bulk operations
- Pagination handling for large result sets

The change queries, the rebase call and the fallback comparison scoring
live in the sibling modules ``_service_queries``, ``_service_rebase`` and
``_service_compare``, and are mixed back into ``GerritService`` here, so
this module's surface is unchanged.  ``create_url_builder`` and
``build_client`` are deliberately resolved in *this* module's namespace
only, so that substituting them here is observed by the code that uses
them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

# The redundant ``as`` aliases below mark deliberate re-exports: every name
# here has always been reachable as ``dependamerge.gerrit.service.<name>``,
# even where only the sibling modules still reference it.
from dependamerge.gerrit.client import GerritNotFoundError as GerritNotFoundError
from dependamerge.gerrit.client import GerritRestError as GerritRestError
from dependamerge.gerrit.client import (
    build_client,
)
from dependamerge.gerrit.models import (
    GerritChangeInfo,
    GerritComparisonResult,
)
from dependamerge.gerrit.urls import GerritUrlBuilder, create_url_builder

from ._service_change_details import DEFAULT_CHANGE_OPTIONS as DEFAULT_CHANGE_OPTIONS
from ._service_change_details import _GerritChangeDetailsMixin

# The sibling modules below carry the parts of this module that never touch
# create_url_builder / build_client.
from ._service_compare import _GerritCompareMixin
from ._service_errors import GerritServiceError as GerritServiceError
from ._service_queries import CHANGE_ID_MATCH_LIMIT as CHANGE_ID_MATCH_LIMIT
from ._service_queries import DEFAULT_CHANGE_ID_OPTIONS as DEFAULT_CHANGE_ID_OPTIONS
from ._service_queries import DEFAULT_LIST_OPTIONS as DEFAULT_LIST_OPTIONS
from ._service_queries import _GerritQueryMixin
from ._service_rebase import _GerritRebaseMixin

if TYPE_CHECKING:
    from dependamerge.progress_tracker import ProgressTracker


log = logging.getLogger("dependamerge.gerrit.service")


class GerritService(
    _GerritQueryMixin,
    _GerritChangeDetailsMixin,
    _GerritRebaseMixin,
    _GerritCompareMixin,
):
    """
    High-level service for Gerrit operations.

    This class provides methods for querying changes, finding similar
    changes, and managing change operations across a Gerrit server.
    """

    # Default similarity threshold for fallback comparison
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.8

    def __init__(
        self,
        host: str,
        base_path: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 15.0,
        max_attempts: int = 5,
        progress_tracker: ProgressTracker | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        """
        Initialize the Gerrit service.

        Args:
            host: Gerrit server hostname.
            base_path: Optional base path (e.g., "infra").
            username: Optional HTTP username for authentication.
            password: Optional HTTP password for authentication.
            timeout: Request timeout in seconds.
            max_attempts: Maximum retry attempts for transient failures.
            progress_tracker: Optional progress tracker for UI feedback.
            similarity_threshold: Minimum confidence score (0.0 to 1.0) for
                changes to be considered similar in the fallback comparison.
                This is used by _basic_compare when no external comparator
                is provided. Should match the threshold used by
                GerritChangeComparator for consistent behavior.
        """
        self.host = host
        self.base_path = base_path
        self._progress_tracker = progress_tracker
        self._similarity_threshold = similarity_threshold

        self._url_builder = create_url_builder(
            host, base_path=base_path, auto_discover=False
        )

        self._client = build_client(
            host,
            base_path=base_path,
            timeout=timeout,
            max_attempts=max_attempts,
            username=username,
            password=password,
        )

        log.debug(
            "GerritService initialized: host=%s, base_path=%s, auth=%s",
            host,
            base_path,
            "yes" if self._client.is_authenticated else "no",
        )

    @property
    def is_authenticated(self) -> bool:
        """Check if the service has authentication credentials."""
        return self._client.is_authenticated

    @property
    def url_builder(self) -> GerritUrlBuilder:
        """Get the URL builder for constructing URLs."""
        return self._url_builder

    def find_similar_changes(
        self,
        source_change: GerritChangeInfo,
        comparator: Any,
        only_automation: bool = True,
        limit: int = 500,
        candidates: list[GerritChangeInfo] | None = None,
    ) -> list[tuple[GerritChangeInfo, GerritComparisonResult]]:
        """
        Find changes similar to the source change.

        This method uses the provided comparator to identify similar
        changes among the candidates. When no candidate list is given,
        it falls back to scanning all open changes on the server.

        Args:
            source_change: The change to find similar changes for.
            comparator: A comparator object with a compare_gerrit_changes()
                       method (or compare_pull_requests for compatibility).
            only_automation: Whether to only match automation changes.
            limit: Maximum number of changes to scan when no candidate
                   list is given.
            candidates: Optional pre-scoped candidate changes (e.g. the
                        open changes sharing the source change's topic).
                        Scoping via a server-side query is far cheaper
                        and more reliable than the whole-server scan.

        Returns:
            List of (change_info, comparison_result) tuples for similar
            changes, sorted by confidence score descending.
        """
        log.info(
            "Finding similar changes for %s #%d",
            source_change.project,
            source_change.number,
        )

        if candidates is None:
            candidates = self.get_all_open_changes(limit=limit)

        log.debug("Scanning %d open changes for similarity", len(candidates))

        similar_changes: list[tuple[GerritChangeInfo, GerritComparisonResult]] = []

        for change in candidates:
            # Skip the source change itself
            if change.number == source_change.number:
                continue

            if not only_automation and self._owners_differ(source_change, change):
                log.debug(
                    "Skipping change %d because owner %r does not match source owner %r",
                    change.number,
                    change.owner,
                    source_change.owner,
                )
                continue

            # Compare using the provided comparator
            try:
                if hasattr(comparator, "compare_gerrit_changes"):
                    result = comparator.compare_gerrit_changes(
                        source_change, change, only_automation=only_automation
                    )
                else:
                    # Fall back to generic comparison if available
                    result = self._basic_compare(source_change, change, only_automation)
            except Exception as exc:
                log.debug("Error comparing change %d: %s", change.number, exc)
                continue

            if result.is_similar:
                similar_changes.append((change, result))
                log.debug(
                    "Found similar change: %s #%d (score=%.2f)",
                    change.project,
                    change.number,
                    result.confidence_score,
                )

        # Sort by confidence score descending
        similar_changes.sort(key=lambda x: x[1].confidence_score, reverse=True)

        log.info("Found %d similar changes", len(similar_changes))
        return similar_changes


def create_gerrit_service(
    host: str,
    base_path: str | None = None,
    username: str | None = None,
    password: str | None = None,
    progress_tracker: ProgressTracker | None = None,
) -> GerritService:
    """
    Factory function to create a GerritService instance.

    Args:
        host: Gerrit server hostname.
        base_path: Optional base path.
        username: Optional HTTP username.
        password: Optional HTTP password.
        progress_tracker: Optional progress tracker.

    Returns:
        Configured GerritService instance.
    """
    return GerritService(
        host=host,
        base_path=base_path,
        username=username,
        password=password,
        progress_tracker=progress_tracker,
    )


__all__ = [
    "CHANGE_ID_MATCH_LIMIT",
    "DEFAULT_CHANGE_ID_OPTIONS",
    "DEFAULT_CHANGE_OPTIONS",
    "DEFAULT_LIST_OPTIONS",
    "GerritService",
    "GerritServiceError",
    "create_gerrit_service",
]
