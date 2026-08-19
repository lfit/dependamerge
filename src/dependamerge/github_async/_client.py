# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The concrete asynchronous GitHub API client.

``GitHubAsync`` itself: construction of the HTTP client, the shared
throttling state and the session caches, plus the context-manager
protocol.  Its methods live in the sibling ``_Xxx`` mixin modules and
are assembled here, so the class surface is unchanged.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from typing import (
    Any,
)

import httpx
from aiolimiter import AsyncLimiter

from ._block_reason import _BlockReasonMixin
from ._checks import _ChecksMixin
from ._merge import _MergeMixin
from ._permission_checks import _PermissionChecksMixin
from ._permissions import _PermissionsMixin
from ._required_checks import _RequiredChecksMixin
from ._reviews import _ReviewsMixin
from ._signatures import _SignaturesMixin
from ._strict_checks import _StrictChecksMixin
from ._throttling import _Budget, _ResizableSemaphore, _ThrottleMixin
from ._transport import _TransportMixin

GITHUB_API = "https://api.github.com"
GITHUB_GQL = "https://api.github.com/graphql"


class GitHubAsync(
    _TransportMixin,
    _ThrottleMixin,
    _PermissionsMixin,
    _PermissionChecksMixin,
    _ReviewsMixin,
    _MergeMixin,
    _SignaturesMixin,
    _StrictChecksMixin,
    _RequiredChecksMixin,
    _BlockReasonMixin,
    _ChecksMixin,
):
    """
    Asynchronous GitHub API client with:
    - httpx AsyncClient for HTTP/2 support and connection pooling
    - Bounded concurrency via asyncio.Semaphore
    - Request rate limiting via aiolimiter.AsyncLimiter (RPS cap)
    - Robust retry with tenacity on transient errors and rate limits
    - Helpers for GraphQL and REST endpoints used by dependamerge
    """

    # Default ceiling for concurrent in-flight requests. Used both as the
    # constructor default and as the upper bound when adaptive tuning ramps
    # concurrency back up after a period of throttling.
    _DEFAULT_MAX_CONCURRENCY = 20

    def __init__(
        self,
        token: str | None = None,
        *,
        api_url: str = GITHUB_API,
        graphql_url: str = GITHUB_GQL,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        requests_per_second: float = 8.0,
        timeout: float = 20.0,
        user_agent: str = "dependamerge/async-client",
        verify: bool | str = True,
        proxies: dict[str, str] | None = None,
        logger: logging.Logger | None = None,
        on_rate_limited: Callable[[float], None | Awaitable[None]] | None = None,
        on_rate_limit_cleared: Callable[[], None | Awaitable[None]] | None = None,
        on_metrics: Callable[[int, float], None | Awaitable[None]] | None = None,
    ):
        """
        Initialize the async client.

        Args:
            token: GitHub token. If None, reads from GITHUB_TOKEN env var.
            api_url: Base REST API URL (set to your GHE base if needed).
            graphql_url: GraphQL endpoint URL.
            max_concurrency: Max concurrent in-flight requests.
            requests_per_second: Max requests per second (token bucket).
            timeout: Per-request timeout (seconds).
            user_agent: User-Agent header.
            verify: TLS verify flag or path to CA bundle.
            proxies: Optional httpx proxies mapping.
            logger: Optional logger for client messages.
            on_rate_limited: Callback invoked with reset_epoch when primary limit hit.
            on_rate_limit_cleared: Callback invoked when resuming after rate limit.
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN.")

        self.api_url = api_url.rstrip("/")
        self.graphql_url = graphql_url
        self._max_concurrency = max_concurrency
        # Remember the caller-configured ceiling so adaptive tuning ramps
        # concurrency back up to *this* value (not the class default) after
        # a period of throttling, mirroring how ``_base_rps`` bounds the RPS
        # ramp-up.
        self._base_max_concurrency = max_concurrency
        # Sized to the base ceiling and never replaced; throttling reduces
        # the *effective* capacity by holding ballast permits.  See
        # ``_ResizableSemaphore`` for why swapping the object is unsafe.
        self.semaphore = _ResizableSemaphore(max_concurrency, max_concurrency)
        self._base_rps = requests_per_second
        self._current_rps = requests_per_second
        self.limiter = AsyncLimiter(max_rate=self._current_rps, time_period=1.0)
        self.log = logger or logging.getLogger("dependamerge.github_async")
        self._timeout = timeout

        self.on_rate_limited = on_rate_limited
        self.on_rate_limit_cleared = on_rate_limit_cleared
        self.on_metrics = on_metrics

        # Adaptive throttling state.  ``_request_history`` records the
        # timestamp of every completed request and ``_error_history`` the
        # subset that failed, so the error *rate* is a real ratio rather
        # than a constant derived from a guessed request count.
        self._error_history: list[
            tuple[float, str]
        ] = []  # List of (timestamp, error_type) tuples
        self._request_history: list[float] = []
        self._error_window = 300  # 5 minutes
        self._last_retry_after: float | None = None
        self._adaptive_delay = 0.0
        self._last_adaptive_update: float | None = None
        # Per-resource rate-limit budgets, keyed by ``X-RateLimit-Resource``
        # (``core``, ``graphql``, ``search``).  Kept apart because they are
        # independent allowances; see ``_Budget``.
        self._budgets: dict[str, _Budget] = {}
        # Consecutive healthy responses observed since the last throttle
        # event.  Ramp-up requires a sustained run rather than a single
        # lucky response, so a client under pressure does not oscillate.
        self._healthy_streak = 0

        # Cache for the authenticated user's login (never changes during a session)
        self._authenticated_user_login: str | None = None

        # Session caches for repo/branch-scoped configuration.  Branch
        # protection, required status checks, and a repo's default
        # branch are effectively immutable for the lifetime of a merge
        # run, yet the merge pipeline consults them repeatedly — once
        # per PR (or several times per *blocked* PR via
        # ``analyze_block_reason``).  Caching them here collapses those
        # repeats into one fetch per repo/branch.  No locking: a
        # concurrent first miss may fetch twice, which is harmless and
        # no worse than the uncached behaviour.
        self._default_branch_cache: dict[str, str | None] = {}
        self._required_checks_cache: dict[str, list[dict[str, Any]]] = {}
        self._branch_protection_cache: dict[str, dict[str, Any]] = {}
        self._requires_signatures_cache: dict[str, bool] = {}
        self._requires_strict_checks_cache: dict[str, bool] = {}
        # Short-lived memo for ``analyze_block_reason``: keyed by
        # ``(owner, repo, number, head_sha, base_branch)`` →
        # ``(cached_at, reason)``.  The base branch belongs in the key
        # because it selects which protection configuration is read.
        # See ``_BLOCK_REASON_TTL_SECONDS`` for why it expires quickly.
        self._block_reason_cache: dict[
            tuple[str, str, int, str, str | None], tuple[float, str]
        ] = {}

        # Cache for the token's OAuth scopes.  ``_token_scopes_fetched``
        # distinguishes "not looked up yet" from "looked up, but this token
        # type does not expose scopes" (fine-grained PAT / app token, which
        # leaves ``_token_scopes`` as ``None``).
        self._token_scopes: set[str] | None = None
        self._token_scopes_fetched: bool = False

        mounts = None
        if proxies:
            mounts = {}
            if "http" in proxies and proxies["http"]:
                mounts["http://"] = httpx.AsyncHTTPTransport(proxy=proxies["http"])
            if "https" in proxies and proxies["https"]:
                mounts["https://"] = httpx.AsyncHTTPTransport(proxy=proxies["https"])
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": user_agent,
            },
            http2=True,
            timeout=timeout,
            verify=verify,
            mounts=mounts,
        )

    def __repr__(self) -> str:
        """Safe repr that never exposes the token value."""
        return f"GitHubAsync(api_url={self.api_url!r}, token=***)"

    async def __aenter__(self) -> GitHubAsync:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close underlying httpx client and stop any pending resize."""
        await self.semaphore.aclose()
        await self._client.aclose()
