# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""Owner and repository enumeration, plus the open-PR page fetches."""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..github_async import GraphQLError
from ..github_graphql import (
    ORG_REPOS_ONLY,
    REPO_OPEN_PRS_PAGE,
    USER_REPOS_ONLY,
)
from ._base import _GitHubServiceBase
from ._constants import (
    DEFAULT_COMMENTS_PAGE_SIZE,
    DEFAULT_CONTEXTS_PAGE_SIZE,
    DEFAULT_FILES_PAGE_SIZE,
    DEFAULT_PRS_PAGE_SIZE,
)


class _RepositoriesMixin(_GitHubServiceBase):
    """Repository enumeration and open-PR paging for ``GitHubService``."""

    async def _iter_org_repositories(self, org: str) -> AsyncIterator[dict[str, Any]]:
        """Iterate an owner's non-archived repositories (forks included).

        Despite the historical name, this works for both organizations
        and personal user accounts: the correct GraphQL root
        (``organization`` vs ``user``) is resolved once at runtime via
        :meth:`_resolve_owner_root`, so org-wide *read* operations
        (``status``, ``blocked``, and ``close``'s similar-PR scan) no
        longer fail with a ``NOT_FOUND`` error when handed a user
        account.  This mirrors the owner-aware enumeration the merge
        path already uses via :meth:`_iter_owner_repositories`.

        Unlike :meth:`_iter_owner_repositories`, fork repositories are
        *not* skipped here: the read-only reporting commands want a
        complete picture of every repository the owner has, whereas the
        bulk-merge path deliberately excludes forks.  The progress total
        is published from the first page's ``totalCount``, which counts
        *all* of the owner's repositories — including the archived repos
        this iterator filters out — so the denominator is approximate and
        the percentage can finish below 100%.  It is close enough for a
        progress bar.
        """
        async for repo in self._iter_owner_repositories(org, skip_forks=False):
            yield repo

    async def _iter_org_repositories_with_open_prs(
        self, org: str
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Iterate organization repositories only; PRs are fetched per repository.

        This reduces per-query node pressure. Consumers should fetch PR pages
        using _fetch_repo_prs_first_page and _iter_repo_open_prs_pages.
        """
        async for repo in self._iter_org_repositories(org):
            yield repo

    async def _resolve_owner_root(self, owner: str) -> tuple[str, str]:
        """Resolve whether ``owner`` is an organization or a user account.

        Probes the ``organization(login:)`` repositories query first; if
        that root resolves to null (the login is not an org), falls back
        to the ``user(login:)`` query.  The verdict is cached so repeated
        pagination pages do not re-probe.

        Returns:
            A ``(root_key, query)`` tuple where ``root_key`` is the
            top-level GraphQL field (``"organization"`` or ``"user"``)
            and ``query`` is the matching query document.
        """
        cached = self._owner_root_cache.get(owner)
        if cached is not None:
            return cached

        variables = {"org": owner, "reposCursor": None}
        # GitHub answers ``organization(login:)`` for a *user* login with
        # ``data.organization = null`` AND a NOT_FOUND error in the
        # ``errors`` array ("Could not resolve to an Organization ...").
        # ``GitHubAsync.graphql`` raises ``GraphQLError`` on any non-transient
        # ``errors`` payload, so the null-organization case never reaches the
        # ``data`` check below — it surfaces as an exception instead.  Treat a
        # NOT_FOUND-on-organization error as "not an org" and fall back to the
        # user root; re-raise anything else (e.g. genuine transport or schema
        # errors).
        try:
            data = await self._api.graphql(ORG_REPOS_ONLY, variables)
            is_org = (data or {}).get("organization") is not None
        except GraphQLError as exc:
            if not self._is_not_an_organization_error(exc):
                raise
            is_org = False

        if is_org:
            resolved = ("organization", ORG_REPOS_ONLY)
        else:
            resolved = ("user", USER_REPOS_ONLY)
        self._owner_root_cache[owner] = resolved
        return resolved

    @staticmethod
    def _is_not_an_organization_error(exc: GraphQLError) -> bool:
        """Return True when a GraphQL error means "login is not an org".

        ``GitHubAsync.graphql`` raises ``GraphQLError(json.dumps(errors))``,
        so the exception text is the structured GraphQL ``errors`` array.
        Parse it and match only a ``NOT_FOUND`` error reported against the
        top-level ``organization`` field (``path == ["organization"]``),
        so an unrelated ``NOT_FOUND`` on a nested field that merely mentions
        "organization" cannot trigger the user-account fallback.

        If the payload is not the expected JSON shape (e.g. the
        retries-exhausted sentinel), fall back to the conservative
        substring heuristic so a genuinely-missing org still falls back
        rather than aborting.
        """
        try:
            errors = json.loads(str(exc))
        except (ValueError, TypeError):
            errors = None

        if isinstance(errors, list):
            for error in errors:
                if not isinstance(error, dict):
                    continue
                if str(error.get("type", "")).upper() != "NOT_FOUND":
                    continue
                path = error.get("path")
                if path == ["organization"]:
                    return True
            return False

        # Payload was not the structured errors array; degrade gracefully.
        msg = str(exc).lower()
        return "not_found" in msg and "organization" in msg

    async def _iter_owner_repositories(
        self, owner: str, *, skip_forks: bool = True
    ) -> AsyncIterator[dict[str, Any]]:
        """Iterate an owner's non-archived repositories.

        Works for both organizations and personal user accounts: the
        correct GraphQL root is resolved once via
        :meth:`_resolve_owner_root` and reused for every page.

        Archived repositories are always skipped.  Fork repositories are
        skipped by default (``skip_forks=True``): owner-wide bulk merges
        target the owner's own automation PRs, not PRs on mirrored
        forks.  Read-only reporting paths pass ``skip_forks=False`` to
        include forks for a complete picture.  The progress total is
        published from the first page's ``totalCount``, which counts
        *all* of the owner's repositories — including the archived and
        (when filtered) fork repos this iterator skips — so the
        denominator is approximate and the percentage can finish below
        100%.  It is close enough for a progress bar.
        """
        root_key, query = await self._resolve_owner_root(owner)
        cursor: str | None = None
        total_set = False
        while True:
            variables = {"org": owner, "reposCursor": cursor}
            data = await self._api.graphql(query, variables)
            root = (data or {}).get(root_key) or {}
            repos = root.get("repositories") or {}

            if not total_set:
                total_count = repos.get("totalCount")
                if total_count is not None and self._progress:
                    self._progress.update_total_repositories(total_count)
                total_set = True

            nodes: list[dict[str, Any]] = repos.get("nodes", []) or []
            for repo in nodes:
                if repo.get("isArchived"):
                    continue
                if skip_forks and repo.get("isFork"):
                    continue
                yield repo

            page_info = repos.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")

    async def _iter_repo_open_prs_pages(
        self, owner: str, name: str, cursor: str | None
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Iterate additional pages of open PRs for a specific repository.
        """
        prs_cursor = cursor
        while prs_cursor:
            prs_size = DEFAULT_PRS_PAGE_SIZE
            files_size = DEFAULT_FILES_PAGE_SIZE
            comments_size = DEFAULT_COMMENTS_PAGE_SIZE
            contexts_size = DEFAULT_CONTEXTS_PAGE_SIZE
            if getattr(self, "_rate_limited", False):
                prs_size = max(10, prs_size // 2)
                files_size = max(20, files_size // 2)
                comments_size = max(5, comments_size // 2)
                contexts_size = max(10, contexts_size // 2)
            variables = {
                "owner": owner,
                "name": name,
                "prsCursor": prs_cursor,
                "prsPageSize": prs_size,
                "filesPageSize": files_size,
                "commentsPageSize": comments_size,
                "contextsPageSize": contexts_size,
            }
            async with self._page_semaphore:
                data = await self._api.graphql(REPO_OPEN_PRS_PAGE, variables)
            repo = (data or {}).get("repository") or {}
            prs = repo.get("pullRequests") or {}
            nodes: list[dict[str, Any]] = prs.get("nodes", []) or []
            for pr in nodes:
                yield pr

            page_info = prs.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                break
            prs_cursor = page_info.get("endCursor")

    async def _fetch_repo_prs_first_page(
        self, owner: str, name: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Fetch the first page of open PRs for a repository using GraphQL.
        Returns a tuple of (nodes, pageInfo).
        """
        prs_size = DEFAULT_PRS_PAGE_SIZE
        files_size = DEFAULT_FILES_PAGE_SIZE
        comments_size = DEFAULT_COMMENTS_PAGE_SIZE
        contexts_size = DEFAULT_CONTEXTS_PAGE_SIZE
        if getattr(self, "_rate_limited", False):
            prs_size = max(10, prs_size // 2)
            files_size = max(20, files_size // 2)
            comments_size = max(5, comments_size // 2)
            contexts_size = max(10, contexts_size // 2)
        variables = {
            "owner": owner,
            "name": name,
            "prsCursor": None,
            "prsPageSize": prs_size,
            "filesPageSize": files_size,
            "commentsPageSize": comments_size,
            "contextsPageSize": contexts_size,
        }
        async with self._page_semaphore:
            data = await self._api.graphql(REPO_OPEN_PRS_PAGE, variables)
        repo = (data or {}).get("repository") or {}
        prs = repo.get("pullRequests") or {}
        nodes: list[dict[str, Any]] = prs.get("nodes", []) or []
        page_info: dict[str, Any] = prs.get("pageInfo") or {}
        return nodes, page_info
