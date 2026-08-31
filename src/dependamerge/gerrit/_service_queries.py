# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The Gerrit change queries behind ``GerritService``.

:class:`_GerritQueryMixin` carries the read-only REST calls that
*search* for changes: listing open changes (optionally scoped by
project, branch, owner or topic), resolving a Change-Id to the change
it names, listing projects, and the paginated query loop that backs
them.  The default option sets those queries send to Gerrit live here
too.

Reads that address a single change by number live in
``_service_change_details`` instead, so neither module outgrows a
reviewable size.  The split is by question asked: "find me changes
matching this" here, "tell me about this change" there.

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
import re

from dependamerge.gerrit.client import (
    GerritRestClient,
)
from dependamerge.gerrit.models import GerritChangeInfo

from ._service_errors import GerritServiceError

log = logging.getLogger("dependamerge.gerrit.service")


# A Gerrit Change-Id: ``I`` followed by 40 hexadecimal characters.
# Validated before interpolation because the value lands in Gerrit's
# query language, where an unchecked string such as
# ``I... OR status:open`` would parse as extra query terms and quietly
# broaden a lookup that advertises an exact-key match.
_CHANGE_ID_RE = re.compile(r"\AI[0-9a-fA-F]{40}\Z")

# How many matches a Change-Id lookup fetches before settling on the
# first.  Above 1 so an ambiguous Change-Id --- the same one appears on
# every cherry-pick of a change --- stays visible to the diagnostic
# rather than being silently truncated by the server.
CHANGE_ID_MATCH_LIMIT = 5

# Default query options for a Change-Id lookup.  Narrower than
# DEFAULT_CHANGE_OPTIONS: a caller resolving a Change-Id is deciding
# whether the change can be submitted, so the label and
# submit-requirement detail is worth the round trip while the file and
# account detail is not.
DEFAULT_CHANGE_ID_OPTIONS: list[str] = [
    "CURRENT_REVISION",
    "LABELS",
    "DETAILED_LABELS",
    "SUBMIT_REQUIREMENTS",
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
            List of GerritChangeInfo for matching open changes.  An
            empty list means nothing matched; a query that failed
            raises instead.

        Raises:
            GerritRestError: If the query cannot be run --- including
                ``GerritAuthError`` when the credentials are rejected.
            GerritServiceError: If Gerrit's response cannot be parsed.
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
            List of GerritChangeInfo for all open changes.  An empty
            list means nothing matched; a query that failed raises
            instead.

        Raises:
            GerritRestError: If the query cannot be run --- including
                ``GerritAuthError`` when the credentials are rejected.
            GerritServiceError: If Gerrit's response cannot be parsed.
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
            List of GerritChangeInfo for matching changes.  An empty
            list means nothing matched; a query that failed raises
            instead.

        Raises:
            GerritRestError: If the query cannot be run --- including
                ``GerritAuthError`` when the credentials are rejected.
            GerritServiceError: If Gerrit's response cannot be parsed.
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

    def find_open_change_by_change_id(
        self,
        change_id: str,
        options: list[str] | None = None,
    ) -> GerritChangeInfo | None:
        """
        Find the open change carrying ``change_id``.

        Distinct from :meth:`get_change_info`, which takes a change
        *number* and raises when it does not exist.  A Change-Id is not
        unique on its own --- the same one appears on every cherry-pick
        of a change across branches and projects --- so this is a search
        that may legitimately match nothing, and returns None rather
        than raising.

        Where several open changes share the Change-Id, the first Gerrit
        returns is used.  Callers needing to disambiguate should query by
        project or branch instead.

        Args:
            change_id: The Gerrit Change-Id (the ``I``-prefixed key from
                the commit message), not a change number.
            options: Optional list of query options. Defaults to
                    DEFAULT_CHANGE_ID_OPTIONS.

        Returns:
            The first matching open change, or None when none match.
            None means exactly that: a query that failed raises rather
            than reporting an absence it never established.

        Raises:
            ValueError: If ``change_id`` is not a well-formed Change-Id.
            GerritRestError: If the query cannot be run --- including
                ``GerritAuthError`` when the credentials are rejected.
            GerritServiceError: If Gerrit's response cannot be parsed.
        """
        if not _CHANGE_ID_RE.match(change_id):
            raise ValueError(
                f"Not a well-formed Gerrit Change-Id: {change_id!r} "
                "(expected 'I' followed by 40 hexadecimal characters)"
            )

        if options is None:
            options = DEFAULT_CHANGE_ID_OPTIONS

        # A small limit rather than 1: fetching a few matches keeps the
        # ambiguous case visible to the debug log below, at no extra cost
        # in the overwhelmingly common single-match case.
        changes = self._query_changes(
            query=f"change:{change_id} status:open",
            limit=CHANGE_ID_MATCH_LIMIT,
            offset=0,
            options=options,
        )

        if not changes:
            return None

        if len(changes) > 1:
            # The count is what was fetched and parsed, not what Gerrit
            # matched: _query_changes caps the page and skips malformed
            # items.  So it may understate, never overstate --- fine for
            # a diagnostic, and the reason this hedges rather than
            # reporting a total.
            qualifier = "at least " if len(changes) >= CHANGE_ID_MATCH_LIMIT else ""
            log.debug(
                "Change-Id %s matches %s%d open changes; using %s #%d",
                change_id,
                qualifier,
                len(changes),
                changes[0].project,
                changes[0].number,
            )

        return changes[0]

    def get_projects(self, limit: int = 500) -> list[str]:
        """
        Get a list of project names from the Gerrit server.

        Args:
            limit: Maximum number of projects to return.

        Returns:
            List of project names.  An empty list means the server has
            none; a request that failed raises instead.

        Raises:
            GerritRestError: If the request cannot be run --- including
                ``GerritAuthError`` when the credentials are rejected.
            GerritServiceError: If Gerrit's response cannot be parsed.
        """
        log.debug("Fetching project list (limit=%d)", limit)

        endpoint = f"/projects/?n={limit}"
        data = self._client.get(endpoint)

        # Gerrit answers with a map of project name to detail.
        if not isinstance(data, dict):
            raise GerritServiceError(
                f"Gerrit returned {type(data).__name__} rather than a map "
                "of projects; the response schema may have changed"
            )

        return sorted(data.keys())

    def _query_changes(
        self,
        query: str,
        limit: int,
        offset: int,
        options: list[str],
    ) -> list[GerritChangeInfo]:
        """Execute a change query with pagination.

        Raises rather than degrading to a short or empty list.  A merge
        tool that treats "the query failed" as "nothing matched" will
        cheerfully report success having done nothing, or merge a subset
        of a topic believing it merged the whole; both are worse than
        stopping.

        Raises:
            GerritRestError: If a page cannot be fetched.  Propagated
                rather than wrapped, because the callers already
                distinguish it --- and its ``GerritAuthError`` subclass
                --- from other failures.
            GerritServiceError: If a page contains changes but none of
                them parse, which indicates a schema mismatch rather
                than an absence of results.
        """
        all_changes: list[GerritChangeInfo] = []
        skipped = 0
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

            data = self._client.get(endpoint)

            if not isinstance(data, list):
                # A changes query answers with a JSON array.  Anything
                # else --- a dict, a bare null --- is the request
                # succeeding and the response being unusable, which is
                # the same schema mismatch handled below and must not
                # masquerade as the end of the results.
                raise GerritServiceError(
                    f"Gerrit returned {type(data).__name__} rather than a "
                    f"list of changes for query {query!r}; the response "
                    "schema may have changed"
                )

            if not data:
                # A genuinely empty page: no more results.
                break

            page_changes = []
            page_skipped = 0
            for item in data:
                try:
                    change = GerritChangeInfo.from_api_response(
                        item, host=self.host, base_path=self.base_path
                    )
                    page_changes.append(change)
                except Exception as exc:
                    page_skipped += 1
                    log.debug("Skipping malformed change: %s", exc)
                    continue

            if page_skipped and not page_changes:
                # Gerrit sent changes and not one of them parsed.  Left
                # to fall through, this page would break the loop and
                # surface as an empty result --- a schema mismatch
                # wearing the costume of "no matches".
                raise GerritServiceError(
                    f"Gerrit returned {page_skipped} change(s) for query "
                    f"{query!r} but none could be parsed; the response "
                    "schema may have changed"
                )

            skipped += page_skipped
            all_changes.extend(page_changes)

            # Paginate on what the *server* returned, not on what parsed.
            # Skipped items still occupy offsets, so advancing by the
            # parsed count would re-request them, and treating a
            # short-after-skipping page as the end would stop early.
            if len(data) < current_limit:
                break

            current_offset += len(data)

        if skipped:
            log.warning(
                "Skipped %d unparsable change(s) of %d returned for query "
                "%r; results are incomplete",
                skipped,
                skipped + len(all_changes),
                query,
            )

        return all_changes[:limit]
