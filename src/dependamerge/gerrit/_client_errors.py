# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Error types and pure helpers for the Gerrit REST client.

This module holds the exception hierarchy raised by
:mod:`dependamerge.gerrit.client`, the tables used to classify an error
as retryable, and the small side-effect-free helpers that support retry
and logging.  None of it touches the network, so it is safe to import
and exercise on its own.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

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

_RETRYABLE_HTTP_CODES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})


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
    exc_str = str(exc)
    for code in (401, 403, 404, 429, 500, 502, 503, 504):
        if str(code) in exc_str:
            return code
    return None
