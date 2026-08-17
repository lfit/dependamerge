# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The Gerrit change queries behind ``GerritService``.

:class:`_GerritQueryMixin` carries the read-only REST calls: fetching a
single change, listing open changes (optionally scoped by project,
branch, owner or topic), listing projects, and the paginated query loop
that backs them.  The default option sets those queries send to Gerrit
live here too.

It lives here rather than in ``dependamerge.gerrit.service`` purely to
keep that module reviewable.  Nothing in here references
``create_url_builder`` or ``build_client``: those names are only
resolved in ``service``'s own namespace, so that patching them there
stays effective.  Every attribute this mixin reads is established by
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

# Default query options for listing changes
DEFAULT_LIST_OPTIONS: list[str] = [
    "CURRENT_REVISION",
    "CURRENT_FILES",
    "CURRENT_COMMIT",
    "LABELS",
    "DETAILED_ACCOUNTS",
]


class _GerritQueryMixin:
    """Change and project queries shared into ``GerritService``."""

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

    def get_open_changes(
        self,
        project: str | None = None,
        branch: str | None = None,
        owner: str | None = None,
        limit: int = 500,
        offset: int = 0,
        options: list[str] | None = None,
    ) -> list[GerritChangeInfo]:
        """
        Get open changes, optionally filtered by project/branch/owner.

        Args:
            project: Optional project name to filter by.
            branch: Optional branch name to filter by.
            owner: Optional owner username to filter by.
            limit: Maximum number of changes to return.
            offset: Starting offset for pagination.
            options: Optional list of query options.

        Returns:
            List of GerritChangeInfo for matching open changes.
        """
        if options is None:
            options = DEFAULT_LIST_OPTIONS

        query_parts = ["status:open"]
        if project:
            query_parts.append(f"project:{project}")
        if branch:
            query_parts.append(f"branch:{branch}")
        if owner:
            query_parts.append(f"owner:{owner}")

        query = " ".join(query_parts)
        return self._query_changes(query, limit, offset, options)

    def get_all_open_changes(
        self,
        limit: int = 1000,
        options: list[str] | None = None,
    ) -> list[GerritChangeInfo]:
        """
        Get all open changes across the entire Gerrit server.

        This method handles pagination automatically to fetch up to
        the specified limit of changes.

        Args:
            limit: Maximum number of changes to return.
            options: Optional list of query options.

        Returns:
            List of GerritChangeInfo for all open changes.
        """
        return self.get_open_changes(limit=limit, options=options)

    def get_changes_by_topic(
        self,
        topic: str,
        include_merged: bool = False,
        limit: int = 100,
        options: list[str] | None = None,
    ) -> list[GerritChangeInfo]:
        """
        Get changes with a specific topic.

        Args:
            topic: The topic name to search for.
            include_merged: Whether to include merged changes.
            limit: Maximum number of changes to return.
            options: Optional list of query options.

        Returns:
            List of GerritChangeInfo for matching changes.
        """
        if options is None:
            options = DEFAULT_LIST_OPTIONS

        # Topics containing whitespace (or quotes) must be quoted in
        # Gerrit query syntax, otherwise the bare value splits into
        # separate query terms and the search silently matches nothing
        # useful.  Plain topics stay unquoted.
        topic_term = f"topic:{topic}"
        if '"' in topic or any(ch.isspace() for ch in topic):
            escaped = topic.replace("\\", "\\\\").replace('"', '\\"')
            topic_term = f'topic:"{escaped}"'

        if include_merged:
            query = f"{topic_term} (status:open OR status:merged)"
        else:
            query = f"{topic_term} status:open"

        return self._query_changes(query, limit, 0, options)

    def get_projects(self, limit: int = 500) -> list[str]:
        """
        Get a list of project names from the Gerrit server.

        Args:
            limit: Maximum number of projects to return.

        Returns:
            List of project names.
        """
        log.debug("Fetching project list (limit=%d)", limit)

        try:
            endpoint = f"/projects/?n={limit}"
            data = self._client.get(endpoint)

            # Gerrit returns a dict with project names as keys
            if isinstance(data, dict):
                return sorted(data.keys())
            return []

        except GerritRestError as exc:
            log.warning("Failed to fetch projects: %s", exc)
            return []

    def _query_changes(
        self,
        query: str,
        limit: int,
        offset: int,
        options: list[str],
    ) -> list[GerritChangeInfo]:
        """Execute a change query with pagination."""
        all_changes: list[GerritChangeInfo] = []
        page_size = min(limit, 100)
        current_offset = offset

        while len(all_changes) < limit:
            remaining = limit - len(all_changes)
            current_limit = min(page_size, remaining)

            params = [
                f"q={query}",
                f"n={current_limit}",
                f"S={current_offset}",
            ]
            for opt in options:
                params.append(f"o={opt}")

            endpoint = "/changes/?" + "&".join(params)
            log.debug("Querying changes: %s", endpoint)

            try:
                data = self._client.get(endpoint)
            except GerritRestError as exc:
                log.warning(
                    "Failed to query changes (offset=%d): %s",
                    current_offset,
                    exc,
                )
                break

            if not data or not isinstance(data, list):
                break

            page_changes = []
            for item in data:
                try:
                    change = GerritChangeInfo.from_api_response(
                        item, host=self.host, base_path=self.base_path
                    )
                    page_changes.append(change)
                except Exception as exc:
                    log.debug("Skipping malformed change: %s", exc)
                    continue

            all_changes.extend(page_changes)

            # Check if we've reached the end
            if len(page_changes) < current_limit:
                break

            current_offset += len(page_changes)

        return all_changes[:limit]
