# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The request and retry machinery behind ``GerritRestClient``.

:class:`_GerritRequestMixin` carries the two methods that issue Gerrit
REST calls: the bounded retry loop and the single-shot request that
dispatches to pygerrit2 and translates failures into the
:mod:`dependamerge.gerrit._client_errors` hierarchy.

It lives here rather than in ``dependamerge.gerrit.client`` purely to
keep that module reviewable.  Nothing in here references ``GerritRestAPI``,
``HTTPBasicAuth`` or ``get_credentials_for_host``: those names are only
resolved in ``client``'s own namespace, so that patching them there stays
effective.  Every attribute this mixin reads is established by
``GerritRestClient.__init__``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from requests.exceptions import RequestException

from ._client_errors import (
    _RETRYABLE_HTTP_CODES,
    GerritAuthError,
    GerritNotFoundError,
    GerritRestError,
    _Auth,
    _calculate_backoff,
    _extract_status_code,
    _is_transient_error,
)

log = logging.getLogger("dependamerge.gerrit.client")


class _GerritRequestMixin:
    """Request/retry behaviour shared into ``GerritRestClient``."""

    # Established by GerritRestClient.__init__.
    _client: Any
    _timeout: float
    _max_attempts: int
    _auth: _Auth | None

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
            # aislop-ignore-next-line ai-slop/python-repetitive-dispatch -- each verb has distinct argument handling (data payload for POST/PUT)
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
