# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Exception types and transient-failure predicates for the async client.

The exception classes, the retry/rate-limit predicates and the retry
budget constants used across the ``GitHubAsync`` mixins.  Split out of
``dependamerge.github_async`` unchanged so the client's transport,
permission and merge code can share them without a cycle.
"""

from __future__ import annotations

import json
import time
from typing import (
    Any,
)


class RateLimitError(Exception):
    """Raised when the primary GitHub API rate limit is reached."""


class SecondaryRateLimitError(Exception):
    """Raised when GitHub's secondary rate limit (abuse detection) triggers."""


class GraphQLError(Exception):
    """Raised for GraphQL errors returned by GitHub."""


class PermissionError(Exception):
    """Raised when GitHub API returns a permission/authorization error.

    Attributes:
        operation: The operation that failed (e.g., 'approve', 'merge', 'close')
        message: Human-readable error message
        token_type_guidance: Guidance for both classic and fine-grained tokens
    """

    def __init__(
        self,
        operation: str,
        message: str,
        token_type_guidance: dict[str, str] | None = None,
    ):
        self.operation = operation
        self.token_type_guidance = token_type_guidance or {}
        super().__init__(message)


class RetryableError(Exception):
    """Internal exception to signal tenacity that a retry should occur."""


# Ceiling of tenacity's ``wait_random_exponential`` on ``_request``.  Named
# so the secondary-rate-limit path can subtract it from a ``Retry-After``
# sleep rather than stacking a full sleep on top of the retry backoff.
_TENACITY_MAX_BACKOFF = 10.0


def _now() -> float:
    return time.time()


def _is_secondary_rate_limited(body_text: str) -> bool:
    text = body_text.lower()
    # GitHub may return messages like:
    # "You have exceeded a secondary rate limit. Please wait a few minutes..."
    # Or "abuse detection mechanism"
    return "secondary rate limit" in text or "abuse detection" in text


def _is_primary_rate_limited(body_text: str) -> bool:
    text = body_text.lower()
    return "api rate limit exceeded" in text


def _is_transient_graphql_error(errors: Any) -> bool:
    try:
        # The structure is usually a list of dicts with "message".
        message_blob = json.dumps(errors).lower()
    except Exception:
        message_blob = str(errors).lower()
    # Heuristics for retryable GraphQL responses
    return any(
        needle in message_blob
        for needle in [
            "rate limit",  # may appear in graphql errors as well
            "something went wrong",  # generic GH error
            "timeout",
            "internal server error",
            "network timeout",
        ]
    )


# Approve-specific retry policy.  ``POST .../reviews`` returns transient
# 500 often enough to matter in bulk runs, but a blanket retry of every
# POST is unsafe, so this is applied only where duplicate-suppression is
# possible (see ``approve_pull_request``).
_APPROVE_MAX_ATTEMPTS = 3
_APPROVE_RETRY_BASE_DELAY = 2.0

# Server-side statuses worth retrying for an operation that can verify
# its own effect afterwards.  Note 500 is intentionally *not* in
# ``_is_retryable_status``: generic retries must not replay arbitrary
# non-idempotent writes.
# Statuses the outer approve retry handles.  Deliberately **only** 500:
# ``_request`` already retries 429/502/503/504 (``_is_retryable_status``)
# plus transport and rate-limit errors via tenacity, six attempts each.
# Including those here too would nest the loops --- up to 18 requests and
# two sets of backoff sleeps for one approval --- which is precisely the
# API-budget waste this work is trying to remove.  500 is the one status
# ``_request`` does not retry, because a blanket replay of failed POSTs
# is unsafe; it is safe *here* only because this call can verify its own
# effect first (see ``approve_pull_request``).
_TRANSIENT_SERVER_STATUSES = frozenset({500})


def _is_transient_server_error(exc: Exception) -> bool:
    """Whether ``exc`` is a server-side failure the outer retry should own.

    Anything already covered by ``_request``'s tenacity policy returns
    ``False`` here: by the time such an exception surfaces it has been
    retried six times, and trying again adds cost without adding hope.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in _TRANSIENT_SERVER_STATUSES


def _is_retryable_status(status: int) -> bool:
    # Treat common transient statuses as retryable.
    return status in (429, 502, 503, 504)
