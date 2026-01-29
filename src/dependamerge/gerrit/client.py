# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Gerrit REST client with retry, timeout, and transient error handling.

This module provides a typed wrapper for Gerrit REST API calls with:
- Bounded retries using exponential backoff with jitter
- Request timeouts
- Transient error classification (HTTP 5xx/429 and network errors)

The client uses pygerrit2 for all Gerrit REST API interactions.

Usage:
    from dependamerge.gerrit.client import GerritRestClient, build_client

    client = build_client("gerrit.example.org", timeout=10.0)
    changes = client.get("/changes/?q=status:open&n=10")
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Final

from pygerrit2 import GerritRestAPI
from pygerrit2 import HTTPBasicAuth
from requests.exceptions import RequestException

log = logging.getLogger("dependamerge.gerrit.client")


_TRANSIENT_ERR_SUBSTRINGS: Final[tuple[str, ...]] = (
    "timed out",
    "temporarily unavailable",
    "temporary failure",
    "connection reset",
    "connection aborted",
    "broken pipe",
    "connection refused",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
)

_RETRYABLE_HTTP_CODES: Final[frozenset[int]] = frozenset(
    {429, 500, 502, 503, 504}
)


class GerritRestError(RuntimeError):
    """Raised for non-retryable REST errors or exhausted retries."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class GerritAuthError(GerritRestError):
    """Raised for authentication failures (401/403)."""


class GerritNotFoundError(GerritRestError):
    """Raised when a resource is not found (404)."""


@dataclass(frozen=True)
class _Auth:
    """Authentication credentials."""

    user: str
    password: str


def _mask_secret(s: str) -> str:
    """Mask a secret for logging, preserving first/last 2 chars."""
    if not s:
        return s
    if len(s) <= 4:
        return "****"
    return s[:2] + "*" * (len(s) - 4) + s[-2:]


def _is_transient_error(exc: Exception) -> bool:
    """Check if an exception represents a transient/retryable error."""
    exc_str = str(exc).lower()
    return any(sub in exc_str for sub in _TRANSIENT_ERR_SUBSTRINGS)


def _calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.5,
) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = min(base_delay * (2**attempt), max_delay)
    jitter_amount = delay * jitter * float(random.random())
    return float(delay + jitter_amount)


def _extract_status_code(exc: Exception) -> int | None:
    """Extract HTTP status code from a requests exception if available."""
    # Check for response attribute (requests.HTTPError)
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            return int(status_code)
    # Check string representation for status codes
    exc_str = str(exc)
    for code in (401, 403, 404, 429, 500, 502, 503, 504):
        if str(code) in exc_str:
            return code
    return None


class GerritRestClient:
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

        # Build pygerrit2 client
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

    def _request_with_retry(
        self,
        method: str,
        path: str,
        data: Any | None = None,
    ) -> Any:
        """Perform a request with automatic retry on transient failures."""
        last_exception: Exception | None = None

        for attempt in range(self._max_attempts):
            try:
                return self._request(method, path, data)
            except GerritAuthError:
                # Don't retry authentication failures
                raise
            except GerritNotFoundError:
                # Don't retry not found errors
                raise
            except GerritRestError as exc:
                last_exception = exc
                # Check if this is a retryable HTTP error or transient network error
                is_retryable_http = (
                    exc.status_code and exc.status_code in _RETRYABLE_HTTP_CODES
                )
                is_transient = _is_transient_error(exc)

                if is_retryable_http or is_transient:
                    if attempt < self._max_attempts - 1:
                        delay = _calculate_backoff(attempt)
                        if exc.status_code:
                            log.warning(
                                "Gerrit REST %s %s failed (HTTP %d), "
                                "retrying in %.1fs (attempt %d/%d)",
                                method,
                                path,
                                exc.status_code,
                                delay,
                                attempt + 1,
                                self._max_attempts,
                            )
                        else:
                            log.warning(
                                "Gerrit REST %s %s failed (%s), "
                                "retrying in %.1fs (attempt %d/%d)",
                                method,
                                path,
                                exc,
                                delay,
                                attempt + 1,
                                self._max_attempts,
                            )
                        time.sleep(delay)
                        continue
                raise
            except Exception as exc:
                last_exception = exc
                # Check for transient network errors
                if _is_transient_error(exc):
                    if attempt < self._max_attempts - 1:
                        delay = _calculate_backoff(attempt)
                        log.warning(
                            "Gerrit REST %s %s failed (%s), "
                            "retrying in %.1fs (attempt %d/%d)",
                            method,
                            path,
                            exc,
                            delay,
                            attempt + 1,
                            self._max_attempts,
                        )
                        time.sleep(delay)
                        continue
                raise GerritRestError(
                    f"Gerrit REST {method} {path} failed: {exc}"
                ) from exc

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise GerritRestError(f"Gerrit REST {method} {path} failed unexpectedly")

    def _request(
        self,
        method: str,
        path: str,
        data: Any | None = None,
    ) -> Any:
        """Perform a single HTTP request (no retry) using pygerrit2."""
        if not path:
            raise GerritRestError("path is required")

        # Normalize path to start with /
        api_path = path if path.startswith("/") else f"/{path}"

        log.debug(
            "Gerrit REST %s %s (auth=%s)",
            method,
            api_path,
            "yes" if self._auth else "no",
        )

        try:
            if method == "GET":
                return self._client.get(api_path, timeout=self._timeout)
            elif method == "POST":
                if data is not None:
                    return self._client.post(api_path, data=data, timeout=self._timeout)
                return self._client.post(api_path, timeout=self._timeout)
            elif method == "PUT":
                if data is not None:
                    return self._client.put(api_path, data=data, timeout=self._timeout)
                return self._client.put(api_path, timeout=self._timeout)
            elif method == "DELETE":
                return self._client.delete(api_path, timeout=self._timeout)
            else:
                raise GerritRestError(f"Unsupported HTTP method: {method}")

        except RequestException as exc:
            # Handle requests exceptions from pygerrit2
            status_code = _extract_status_code(exc)
            exc_str = str(exc).lower()

            if status_code == 401 or "401" in exc_str or "unauthorized" in exc_str:
                raise GerritAuthError(
                    f"Authentication failed for {path}",
                    status_code=401,
                ) from exc
            if status_code == 403 or "403" in exc_str or "forbidden" in exc_str:
                raise GerritAuthError(
                    f"Access forbidden for {path}",
                    status_code=403,
                ) from exc
            if status_code == 404 or "404" in exc_str or "not found" in exc_str:
                raise GerritNotFoundError(
                    f"Resource not found: {path}",
                    status_code=404,
                ) from exc

            raise GerritRestError(
                f"Gerrit REST {method} {path} failed: {exc}",
                status_code=status_code,
            ) from exc

        except Exception as exc:
            # Handle any other exceptions
            exc_str = str(exc).lower()
            if "401" in exc_str or "unauthorized" in exc_str:
                raise GerritAuthError(
                    f"Authentication failed: {exc}", status_code=401
                ) from exc
            if "403" in exc_str or "forbidden" in exc_str:
                raise GerritAuthError(
                    f"Access forbidden: {exc}", status_code=403
                ) from exc
            if "404" in exc_str or "not found" in exc_str:
                raise GerritNotFoundError(
                    f"Resource not found: {exc}", status_code=404
                ) from exc
            raise GerritRestError(f"Gerrit REST {method} failed: {exc}") from exc

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
) -> GerritRestClient:
    """
    Build a GerritRestClient for a given host.

    This factory function constructs the appropriate base URL and reads
    authentication credentials from arguments or environment variables.

    Args:
        host: Gerrit hostname (without scheme).
        base_path: Optional base path (e.g., "infra"). If None, no base path.
        timeout: Request timeout in seconds.
        max_attempts: Maximum retry attempts for transient failures.
        username: HTTP username. Falls back to GERRIT_USERNAME or
                  GERRIT_HTTP_USER environment variables.
        password: HTTP password. Falls back to GERRIT_PASSWORD or
                  GERRIT_HTTP_PASSWORD environment variables.

    Returns:
        A configured GerritRestClient instance.
    """
    # Build base URL
    if base_path:
        base_url = f"https://{host}/{base_path.strip('/')}/"
    else:
        base_url = f"https://{host}/"

    # Resolve authentication from arguments or environment
    user = (
        (username or "").strip()
        or os.getenv("GERRIT_USERNAME", "").strip()
        or os.getenv("GERRIT_HTTP_USER", "").strip()
    )
    passwd = (
        (password or "").strip()
        or os.getenv("GERRIT_PASSWORD", "").strip()
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
