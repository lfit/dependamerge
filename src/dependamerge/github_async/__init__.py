# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Asynchronous GitHub API client for dependamerge.

``GitHubAsync`` provides bounded-concurrency, rate-limited access to the
GitHub REST and GraphQL APIs, with adaptive throttling, permission
diagnostics, merge/review operations, branch-protection and ruleset
lookups, and block-reason analysis.

The implementation is split across private sibling modules purely to
keep each one reviewable; every name this package exposed as a single
module is still reachable as ``dependamerge.github_async.<name>``.
"""

from __future__ import annotations

from ._client import (
    GITHUB_API,
    GITHUB_GQL,
    GitHubAsync,
)
from ._errors import (
    _APPROVE_MAX_ATTEMPTS,
    _APPROVE_RETRY_BASE_DELAY,
    _TENACITY_MAX_BACKOFF,
    _TRANSIENT_SERVER_STATUSES,
    GraphQLError,
    PermissionError,
    RateLimitError,
    RetryableError,
    SecondaryRateLimitError,
    _is_primary_rate_limited,
    _is_retryable_status,
    _is_secondary_rate_limited,
    _is_transient_graphql_error,
    _is_transient_server_error,
    _now,
)
from ._permissions import (
    OPERATION_PERMISSIONS,
)
from ._throttling import (
    _Budget,
    _maybe_await,
    _ResizableSemaphore,
)

__all__ = [
    "GitHubAsync",
    "RateLimitError",
    "SecondaryRateLimitError",
    "GraphQLError",
    "PermissionError",
]
