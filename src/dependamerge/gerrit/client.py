# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Gerrit REST client with retry, timeout, and transient error handling.

This module provides a typed wrapper for Gerrit REST API calls with:
- Bounded retries using exponential backoff with jitter
- Request timeouts
- Transient error classification (HTTP 5xx/429 and network errors)

The client uses pygerrit2 for all Gerrit REST API interactions.

The error types, the pure retry helpers and the request machinery live in
the sibling modules ``_client_errors`` and ``_client_requests`` and are
re-exported here, so this module's surface is unchanged.  ``GerritRestAPI``,
``HTTPBasicAuth`` and ``get_credentials_for_host`` are deliberately resolved
in *this* module's namespace only, so that substituting them here is
observed by the code that uses them.

Usage:
    from dependamerge.gerrit.client import GerritRestClient, build_client

    client = build_client("gerrit.example.org", timeout=10.0)
    changes = client.get("/changes/?q=status:open&n=10")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pygerrit2 import GerritRestAPI, HTTPBasicAuth

from dependamerge.netrc import NetrcParseError, get_credentials_for_host

# The sibling modules below carry the parts of this module that never touch
# GerritRestAPI / HTTPBasicAuth / get_credentials_for_host.  Redundant ``as``
# aliases mark deliberate re-exports: every name here has always been
# reachable as ``dependamerge.gerrit.client.<name>``.
from ._client_errors import _RETRYABLE_HTTP_CODES as _RETRYABLE_HTTP_CODES
from ._client_errors import _TRANSIENT_ERR_SUBSTRINGS as _TRANSIENT_ERR_SUBSTRINGS
from ._client_errors import (
    GerritAuthError,
    GerritNotFoundError,
    GerritRestError,
    _Auth,
    _mask_secret,
)
from ._client_errors import _calculate_backoff as _calculate_backoff
from ._client_errors import _extract_status_code as _extract_status_code
from ._client_errors import _is_transient_error as _is_transient_error
from ._client_requests import _GerritRequestMixin

log = logging.getLogger("dependamerge.gerrit.client")


class GerritRestClient(_GerritRequestMixin):
    """
    REST client for Gerrit with retry and timeout handling.

    This client provides methods for making authenticated requests to
    Gerrit's REST API with automatic retry on transient failures.

    Uses pygerrit2 for all Gerrit REST API interactions.
    """

    def __init__(
        self,
        *,
        base_url: str,
        auth: tuple[str, str] | None = None,
        timeout: float = 10.0,
        max_attempts: int = 5,
    ) -> None:
        """
        Initialize the Gerrit REST client.

        Args:
            base_url: The base URL of the Gerrit server (e.g.,
                     "https://gerrit.example.org/").
            auth: Optional tuple of (username, password) for HTTP Basic auth.
            timeout: Request timeout in seconds.
            max_attempts: Maximum number of retry attempts for transient errors.
        """
        # Normalize base URL to end with '/'
        self._base_url: str = base_url.rstrip("/") + "/"
        self._timeout: float = float(timeout)
        self._max_attempts: int = int(max_attempts)
        self._auth: _Auth | None = None

        if auth and auth[0] and auth[1]:
            self._auth = _Auth(auth[0], auth[1])

        if self._auth is not None:
            self._client = GerritRestAPI(
                url=self._base_url,
                auth=HTTPBasicAuth(self._auth.user, self._auth.password),
            )
        else:
            self._client = GerritRestAPI(url=self._base_url)

        log.debug(
            "GerritRestClient initialized: base_url=%s, timeout=%.1fs, "
            "max_attempts=%d, auth_user=%s",
            self._base_url,
            self._timeout,
            self._max_attempts,
            self._auth.user if self._auth else "(none)",
        )

    @property
    def base_url(self) -> str:
        """Get the base URL of the Gerrit server."""
        return self._base_url

    @property
    def is_authenticated(self) -> bool:
        """Check if the client has authentication credentials."""
        return self._auth is not None

    def get(self, path: str) -> Any:
        """
        Perform an HTTP GET request.

        Args:
            path: The API path (e.g., "/changes/12345").

        Returns:
            The parsed JSON response.

        Raises:
            GerritRestError: On non-retryable errors or exhausted retries.
            GerritAuthError: On authentication failures.
            GerritNotFoundError: When the resource is not found.
        """
        return self._request_with_retry("GET", path)

    def post(self, path: str, data: Any | None = None) -> Any:
        """
        Perform an HTTP POST request.

        Args:
            path: The API path.
            data: Optional JSON-serializable data to send.

        Returns:
            The parsed JSON response.

        Raises:
            GerritRestError: On non-retryable errors or exhausted retries.
            GerritAuthError: On authentication failures.
        """
        return self._request_with_retry("POST", path, data=data)

    def put(self, path: str, data: Any | None = None) -> Any:
        """
        Perform an HTTP PUT request.

        Args:
            path: The API path.
            data: Optional JSON-serializable data to send.

        Returns:
            The parsed JSON response.

        Raises:
            GerritRestError: On non-retryable errors or exhausted retries.
            GerritAuthError: On authentication failures.
        """
        return self._request_with_retry("PUT", path, data=data)

    def delete(self, path: str) -> Any:
        """
        Perform an HTTP DELETE request.

        Args:
            path: The API path.

        Returns:
            The parsed JSON response (may be empty).

        Raises:
            GerritRestError: On non-retryable errors or exhausted retries.
            GerritAuthError: On authentication failures.
        """
        return self._request_with_retry("DELETE", path)

    def __repr__(self) -> str:
        """String representation for debugging."""
        masked = ""
        if self._auth is not None:
            masked = f"{self._auth.user}:{_mask_secret(self._auth.password)}@"
        return f"GerritRestClient(base_url='{masked}{self._base_url}')"


def build_client(
    host: str,
    *,
    base_path: str | None = None,
    timeout: float = 10.0,
    max_attempts: int = 5,
    username: str | None = None,
    password: str | None = None,
    use_netrc: bool = True,
    netrc_file: Path | None = None,
) -> GerritRestClient:
    """
    Build a GerritRestClient for a given host.

    This factory function constructs the appropriate base URL and reads
    authentication credentials from multiple sources in priority order.

    Credential resolution order:
    1. Explicit username/password arguments
    2. .netrc file (if use_netrc=True)
    3. Environment variables: GERRIT_USERNAME/GERRIT_PASSWORD or
       GERRIT_HTTP_USER/GERRIT_HTTP_PASSWORD

    Args:
        host: Gerrit hostname (without scheme).
        base_path: Optional base path (e.g., "infra"). If None, no base path.
        timeout: Request timeout in seconds.
        max_attempts: Maximum retry attempts for transient failures.
        username: HTTP username. Takes priority over netrc and env vars.
        password: HTTP password. Takes priority over netrc and env vars.
        use_netrc: Whether to try .netrc for credentials (default: True).
        netrc_file: Explicit path to a .netrc file (optional).

    Returns:
        A configured GerritRestClient instance.
    """
    if base_path:
        base_url = f"https://{host}/{base_path.strip('/')}/"
    else:
        # aislop-ignore-next-line ai-slop/hardcoded-url -- host is caller-supplied
        base_url = f"https://{host}/"

    # Start with explicit credentials if provided
    user = (username or "").strip()
    passwd = (password or "").strip()

    # Try .netrc if explicit credentials not provided
    if (not user or not passwd) and use_netrc:
        try:
            netrc_creds = get_credentials_for_host(
                host=host,
                netrc_file=netrc_file,
                use_netrc=True,
                netrc_optional=True,
            )
            if netrc_creds:
                if not user:
                    user = netrc_creds.login
                if not passwd:
                    passwd = netrc_creds.password
                log.debug("Using credentials from .netrc for %s", host)
        except NetrcParseError as e:
            log.warning("Error parsing .netrc file: %s", e)

    # Fall back to environment variables
    if not user:
        user = (
            os.getenv("GERRIT_USERNAME", "").strip()
            or os.getenv("GERRIT_HTTP_USER", "").strip()
        )
    if not passwd:
        passwd = (
            os.getenv("GERRIT_PASSWORD", "").strip()
            or os.getenv("GERRIT_HTTP_PASSWORD", "").strip()
        )

    auth: tuple[str, str] | None = None
    if user and passwd:
        auth = (user, passwd)

    return GerritRestClient(
        base_url=base_url,
        auth=auth,
        timeout=timeout,
        max_attempts=max_attempts,
    )


__all__ = [
    "GerritAuthError",
    "GerritNotFoundError",
    "GerritRestClient",
    "GerritRestError",
    "build_client",
]
