# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Single-change detail lookups behind ``GerritService``.

:class:`_GerritChangeDetailsMixin` carries the reads that address one
change by its number: its full change info, and the mergeable status
Gerrit computes only on request.  The default option set those reads
send lives here too.

Separated from ``_service_queries`` --- which searches for changes and
returns lists --- so neither module outgrows a reviewable size.  The
split is by question asked: "tell me about this change" here, "find me
changes matching this" there.

Every attribute this mixin reads is established by
``GerritService.__init__``.  The logger name is unchanged so records
keep reporting as ``dependamerge.gerrit.service``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import logging
from typing import Any

from dependamerge.gerrit.client import (
    GerritNotFoundError,
    GerritRestClient,
    GerritRestError,
)
from dependamerge.gerrit.models import GerritChangeInfo

from ._service_errors import GerritServiceError

log = logging.getLogger("dependamerge.gerrit.service")


# Default query options for fetching change details
DEFAULT_CHANGE_OPTIONS: list[str] = [
    "CURRENT_REVISION",
    "CURRENT_FILES",
    "CURRENT_COMMIT",
    "DETAILED_LABELS",
    "DETAILED_ACCOUNTS",
    "SUBMITTABLE",
    # Both action options are needed for permission checks: CHANGE_ACTIONS
    # returns change-level actions, while CURRENT_ACTIONS returns
    # revision-level actions (including 'submit') under
    # revisions[<rev>].actions.
    "CURRENT_ACTIONS",
    "CHANGE_ACTIONS",
]


class _GerritChangeDetailsMixin:
    """Single-change lookups shared into ``GerritService``."""

    # Established by GerritService.__init__.
    _client: GerritRestClient
    host: str
    base_path: str | None

    def get_mergeable_status(
        self,
        change_number: int,
    ) -> dict[str, Any]:
        """
        Fetch the mergeable status for a change.

        This makes an explicit API call to compute merge status,
        which is not included in the standard change info query.

        Args:
            change_number: The Gerrit change number.

        Returns:
            A dict with mergeable info including:
            - mergeable: bool - whether the change can be merged
            - submit_type: str - the submit type (e.g., MERGE_IF_NECESSARY)
            - commit_merged: bool - whether commit is already merged
            - content_merged: bool - whether content is already merged

        Raises:
            GerritServiceError: If the status cannot be fetched.
        """
        endpoint = f"/changes/{change_number}/revisions/current/mergeable"
        log.debug("Fetching mergeable status: %s", endpoint)

        try:
            result: dict[str, Any] = self._client.get(endpoint)
            return result
        except GerritNotFoundError:
            # Change doesn't exist or has no current revision
            return {"mergeable": None}
        except GerritRestError as exc:
            log.warning(
                "Failed to fetch mergeable status for %d: %s", change_number, exc
            )
            return {"mergeable": None}

    def get_change_info(
        self,
        change_number: int,
        options: list[str] | None = None,
        check_mergeable: bool = True,
    ) -> GerritChangeInfo:
        """
        Fetch detailed information about a specific change.

        Args:
            change_number: The Gerrit change number.
            options: Optional list of query options. Defaults to
                    DEFAULT_CHANGE_OPTIONS.
            check_mergeable: If True, make an additional API call to
                           fetch the actual mergeable status.

        Returns:
            A GerritChangeInfo instance with full change details.

        Raises:
            GerritServiceError: If the change cannot be fetched.
            GerritNotFoundError: If the change does not exist.
        """
        if options is None:
            options = DEFAULT_CHANGE_OPTIONS

        endpoint = f"/changes/{change_number}"
        if options:
            params = "&".join(f"o={opt}" for opt in options)
            endpoint += "?" + params

        log.debug("Fetching change info: %s", endpoint)

        try:
            data = self._client.get(endpoint)
            change_info = GerritChangeInfo.from_api_response(
                data, host=self.host, base_path=self.base_path
            )

            # Fetch actual mergeable status if requested and change is open
            if check_mergeable and change_info.status == "NEW":
                mergeable_data = self.get_mergeable_status(change_number)
                if mergeable_data.get("mergeable") is not None:
                    change_info = change_info.model_copy(
                        update={"mergeable": mergeable_data.get("mergeable")}
                    )

            return change_info
        except GerritNotFoundError:
            raise
        except GerritRestError as exc:
            msg = f"Failed to fetch change {change_number}: {exc}"
            log.error(msg)
            raise GerritServiceError(msg) from exc
