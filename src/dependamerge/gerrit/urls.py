# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Gerrit URL construction utilities.

This module provides a centralized way to construct Gerrit URLs with
consistent handling of base paths (e.g., "/infra/") that some Gerrit
servers require.

Usage:
    from dependamerge.gerrit.urls import GerritUrlBuilder

    builder = GerritUrlBuilder("gerrit.linuxfoundation.org", base_path="infra")
    api_url = builder.api_url("/changes/")
    change_url = builder.change_url("releng/project", 12345)
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import quote, urljoin, urlparse

log = logging.getLogger("dependamerge.gerrit.urls")


# Module-level cache for discovered base paths
_BASE_PATH_CACHE: dict[str, str] = {}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """HTTP handler that captures redirects instead of following them."""

    def http_error_301(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any
    ) -> Any:
        return fp

    def http_error_302(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any
    ) -> Any:
        return fp

    def http_error_303(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any
    ) -> Any:
        return fp

    def http_error_307(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any
    ) -> Any:
        return fp

    def http_error_308(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any
    ) -> Any:
        return fp


def discover_base_path(host: str, timeout: float = 5.0) -> str:
    """
    Discover the HTTP base path for a Gerrit host.

    This function probes the Gerrit server to detect if it uses a base path
    (like "/infra/") by checking for redirects from common endpoints.

    The discovery result is cached for the process lifetime.

    Args:
        host: The Gerrit hostname (without scheme).
        timeout: Connection timeout in seconds.

    Returns:
        The base path (e.g., "infra") or empty string if none.
    """
    if not host:
        return ""

    # Check cache first
    cached = _BASE_PATH_CACHE.get(host)
    if cached is not None:
        return cached

    # Known Gerrit endpoints that should exist
    known_endpoints = {
        "changes",
        "accounts",
        "dashboard",
        "c",
        "q",
        "admin",
        "login",
        "settings",
        "plugins",
        "Documentation",
    }

    opener = urllib.request.build_opener(_NoRedirect)
    opener.addheaders = [("User-Agent", "dependamerge/gerrit-urls")]

    # Probe endpoints that typically redirect to the base path
    probes = ["/dashboard/self", "/"]

    for scheme in ("https", "http"):
        for probe in probes:
            url = f"{scheme}://{host}{probe}"

            try:
                resp = opener.open(url, timeout=timeout)
                code = getattr(resp, "getcode", lambda: 0)() or getattr(
                    resp, "status", 0
                )

                # 200 OK means no base path needed
                if code == 200:
                    log.debug("Discovered base path for %s: (none)", host)
                    _BASE_PATH_CACHE[host] = ""
                    return ""

                # Handle redirects
                if code in (301, 302, 303, 307, 308):
                    headers = getattr(resp, "headers", {}) or {}
                    location = (
                        headers.get("Location")
                        or headers.get("location")
                        or ""
                    )
                    if location:
                        base_path = _extract_base_path(
                            host, location, known_endpoints
                        )
                        _BASE_PATH_CACHE[host] = base_path
                        log.debug(
                            "Discovered base path for %s: %r", host, base_path
                        )
                        return base_path

            except urllib.error.HTTPError as http_err:
                # HTTPError also contains response info
                code = http_err.code
                if code in (301, 302, 303, 307, 308):
                    location = (
                        http_err.headers.get("Location")
                        or http_err.headers.get("location")
                        or ""
                    )
                    if location:
                        base_path = _extract_base_path(
                            host, location, known_endpoints
                        )
                        _BASE_PATH_CACHE[host] = base_path
                        log.debug(
                            "Discovered base path for %s: %r", host, base_path
                        )
                        return base_path

            except Exception as exc:
                log.debug(
                    "Base path probe failed for %s%s: %s", host, probe, exc
                )
                continue

    # Default to no base path
    _BASE_PATH_CACHE[host] = ""
    log.debug("No base path discovered for %s", host)
    return ""


def _extract_base_path(
    host: str, location: str, known_endpoints: set[str]
) -> str:
    """Extract the base path from a redirect Location header."""
    parsed = urlparse(location)

    # Get the path component
    path = parsed.path if parsed.scheme or parsed.netloc else location

    # Split into segments
    segments = [s for s in path.split("/") if s]

    if not segments:
        return ""

    # The first segment is the base path if it's not a known endpoint
    first = segments[0]
    if first not in known_endpoints:
        return first

    return ""


class GerritUrlBuilder:
    """
    Builder for Gerrit URLs with consistent base path handling.

    This class encapsulates all Gerrit URL construction logic, ensuring
    that the base path is properly included in all URL types.
    """

    def __init__(
        self,
        host: str,
        base_path: str | None = None,
        auto_discover: bool = True,
    ) -> None:
        """
        Initialize the URL builder for a Gerrit host.

        Args:
            host: Gerrit hostname (without protocol).
            base_path: Optional base path override. If None, reads from
                      GERRIT_HTTP_BASE_PATH environment variable or
                      auto-discovers from the server.
            auto_discover: Whether to auto-discover base path if not
                          provided. Set to False for offline/testing use.
        """
        self.host = host.strip()
        self._base_path: str = ""

        # Determine base path
        if base_path is not None:
            self._base_path = base_path.strip().strip("/")
        else:
            # Check environment variable first
            env_bp = os.getenv("GERRIT_HTTP_BASE_PATH", "").strip().strip("/")
            if env_bp:
                self._base_path = env_bp
            elif auto_discover:
                # Auto-discover from server
                self._base_path = discover_base_path(self.host)

        log.debug(
            "GerritUrlBuilder: host=%s, base_path=%r",
            self.host,
            self._base_path,
        )

    @property
    def base_path(self) -> str:
        """Get the normalized base path (without leading/trailing slashes)."""
        return self._base_path

    @property
    def has_base_path(self) -> bool:
        """Check if a base path is configured."""
        return bool(self._base_path)

    def _build_base_url(self) -> str:
        """Build the base URL with protocol and base path."""
        if self._base_path:
            return f"https://{self.host}/{self._base_path}/"
        return f"https://{self.host}/"

    def api_url(self, endpoint: str = "") -> str:
        """
        Build a Gerrit REST API URL.

        Args:
            endpoint: API endpoint path (e.g., "/changes/", "/accounts/self").

        Returns:
            Complete API URL.
        """
        base_url = self._build_base_url()
        if endpoint:
            # Ensure proper joining
            endpoint = endpoint.lstrip("/")
            return urljoin(base_url, endpoint)
        return base_url

    def web_url(self, path: str = "") -> str:
        """
        Build a Gerrit web UI URL.

        Args:
            path: Web path (e.g., "c/project/+/123", "dashboard").

        Returns:
            Complete web URL.
        """
        base_url = self._build_base_url()
        if path:
            path = path.lstrip("/")
            return urljoin(base_url, path)
        return base_url.rstrip("/")

    def change_url(self, project: str, change_number: int) -> str:
        """
        Build a URL for a specific Gerrit change.

        Args:
            project: Gerrit project name (e.g., "releng/tool").
            change_number: Gerrit change number.

        Returns:
            Complete change URL.
        """
        # Gerrit change URL format: /c/project/+/number
        path = f"c/{project}/+/{change_number}"
        return self.web_url(path)

    def changes_api_url(
        self,
        query: str | None = None,
        options: list[str] | None = None,
        limit: int | None = None,
        start: int | None = None,
    ) -> str:
        """
        Build a URL for querying changes.

        Args:
            query: Gerrit query string (e.g., "status:open project:foo").
            options: List of query options (e.g., ["CURRENT_REVISION"]).
            limit: Maximum number of results.
            start: Starting offset for pagination.

        Returns:
            Complete changes query URL.
        """
        params: list[str] = []

        if query:
            params.append(f"q={quote(query, safe='')}")
        if options:
            for opt in options:
                params.append(f"o={opt}")
        if limit is not None:
            params.append(f"n={limit}")
        if start is not None:
            params.append(f"S={start}")

        endpoint = "/changes/"
        if params:
            endpoint += "?" + "&".join(params)

        return self.api_url(endpoint)

    def change_api_url(
        self,
        change_id: str | int,
        options: list[str] | None = None,
    ) -> str:
        """
        Build a URL for fetching a specific change.

        Args:
            change_id: Change number or Change-Id.
            options: List of query options (e.g., ["CURRENT_REVISION"]).

        Returns:
            Complete change detail URL.
        """
        endpoint = f"/changes/{change_id}"

        if options:
            params = "&".join(f"o={opt}" for opt in options)
            endpoint += "?" + params

        return self.api_url(endpoint)

    def review_url(self, change_id: str | int, revision: str = "current") -> str:
        """
        Build a URL for posting a review.

        Args:
            change_id: Change number or Change-Id.
            revision: Revision ID or "current".

        Returns:
            Review endpoint URL.
        """
        return self.api_url(f"/changes/{change_id}/revisions/{revision}/review")

    def submit_url(self, change_id: str | int) -> str:
        """
        Build a URL for submitting a change.

        Args:
            change_id: Change number or Change-Id.

        Returns:
            Submit endpoint URL.
        """
        return self.api_url(f"/changes/{change_id}/submit")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"GerritUrlBuilder(host={self.host!r}, "
            f"base_path={self._base_path!r})"
        )


def create_url_builder(
    host: str,
    base_path: str | None = None,
    auto_discover: bool = True,
) -> GerritUrlBuilder:
    """
    Factory function to create a GerritUrlBuilder.

    This is the preferred way to create URL builders.

    Args:
        host: Gerrit hostname.
        base_path: Optional base path override.
        auto_discover: Whether to auto-discover base path from server.

    Returns:
        Configured GerritUrlBuilder instance.
    """
    return GerritUrlBuilder(host, base_path, auto_discover)


__all__ = [
    "GerritUrlBuilder",
    "create_url_builder",
    "discover_base_path",
]
