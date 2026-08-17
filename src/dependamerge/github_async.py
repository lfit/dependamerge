# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import (
    Any,
    cast,
)
from urllib.parse import quote

import httpx
from aiolimiter import AsyncLimiter
from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .bot_identity import is_copilot
from .check_runs import failing_check_names

__all__ = [
    "GitHubAsync",
    "RateLimitError",
    "SecondaryRateLimitError",
    "GraphQLError",
    "PermissionError",
]

GITHUB_API = "https://api.github.com"
GITHUB_GQL = "https://api.github.com/graphql"


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


class _ResizableSemaphore:
    """A semaphore whose effective capacity can change at runtime.

    Backed by a single fixed-capacity :class:`asyncio.Semaphore` sized to
    ``maximum``.  Capacity is *reduced* by acquiring "ballast" permits and
    holding them, and restored by releasing them.

    This exists because the obvious implementation --- replacing
    ``self.semaphore`` with a smaller ``asyncio.Semaphore`` --- is unsafe.
    Tasks that acquired the old object release back into it, while new
    arrivals see a fresh object with its full count unclaimed, so the cap
    is transiently violated by up to the old capacity at exactly the
    moment the client is trying to back off.  Holding ballast keeps one
    object for the process lifetime, so every acquire/release pairs up.

    ``resize`` never blocks the caller: acquiring ballast may have to wait
    for in-flight requests to finish, so it runs in a background task.
    Shrinking is therefore best-effort and eventually consistent, which is
    the correct semantic for a throttle --- it takes effect as capacity
    frees up rather than cancelling work already in flight.
    """

    def __init__(self, capacity: int, maximum: int) -> None:
        if maximum < 1:
            raise ValueError("maximum must be >= 1")
        self._maximum = maximum
        self._sem = asyncio.Semaphore(maximum)
        self._ballast = 0
        self._desired_ballast = max(0, maximum - max(1, min(capacity, maximum)))
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    @property
    def capacity(self) -> int:
        """Effective capacity once any pending resize has settled."""
        return self._maximum - self._desired_ballast

    def resize(self, capacity: int) -> None:
        """Request a new effective capacity.  Returns immediately.

        Safe to call without a running event loop: the desired capacity is
        recorded and applied by the next call made from inside one.  The
        production caller (``_request``) always runs in a loop; tolerating
        its absence keeps the tuning logic testable in isolation and stops
        a stray call from raising inside a best-effort code path.
        """
        capacity = max(1, min(capacity, self._maximum))
        desired = self._maximum - capacity
        if desired == self._desired_ballast:
            return
        self._desired_ballast = desired
        if self._task is None or self._task.done():
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No running loop; ``_desired_ballast`` is recorded and
                # will be honoured by the next resize or explicit settle.
                self._task = None
                return
            self._task = asyncio.create_task(
                self._settle(), name="github-semaphore-resize"
            )

    async def _settle(self) -> None:
        async with self._lock:
            while self._ballast != self._desired_ballast:
                if self._ballast < self._desired_ballast:
                    await self._sem.acquire()
                    self._ballast += 1
                else:
                    self._sem.release()
                    self._ballast -= 1

    async def aclose(self) -> None:
        """Cancel any in-flight resize so the loop can shut down cleanly."""
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def __aenter__(self) -> None:
        await self._sem.acquire()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._sem.release()


class _Budget:
    """Latest known rate-limit state for one GitHub rate-limit resource.

    REST (``core``), GraphQL (``graphql``) and ``search`` have independent
    budgets.  They must be tracked separately: GraphQL responses report
    *points* remaining against a 5000-point budget, so folding them into
    the same counter as REST request budget makes a healthy GraphQL
    allowance mask an exhausted REST one, and vice versa.
    """

    __slots__ = ("remaining", "limit", "reset_epoch", "updated_at")

    def __init__(
        self, remaining: int, limit: int, reset_epoch: float | None, updated_at: float
    ) -> None:
        self.remaining = remaining
        self.limit = limit
        self.reset_epoch = reset_epoch
        self.updated_at = updated_at

    def headroom(self, now: float) -> float:
        """Fraction of the budget still available, in ``[0.0, 1.0]``.

        Past its reset the budget has replenished, so report full
        headroom rather than a stale near-zero value that would keep the
        client throttled long after the pressure ended.
        """
        if self.reset_epoch is not None and now >= self.reset_epoch:
            return 1.0
        if self.limit <= 0:
            return 1.0
        return max(0.0, min(1.0, self.remaining / self.limit))


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


# Permission requirements mapping for operations
OPERATION_PERMISSIONS = {
    "list_repos": {
        "classic": "read:org scope",
        "fine_grained": "Organization members: Read access",
        "description": "List organization repositories",
    },
    "approve": {
        "classic": "repo scope",
        "fine_grained": "Pull requests: Read and write",
        "description": "Approve pull requests",
    },
    "merge": {
        "classic": "repo scope",
        "fine_grained": "Contents: Read and write",
        "description": "Merge pull requests",
    },
    "merge_workflow": {
        "classic": "workflow scope (in addition to repo)",
        "fine_grained": "Workflows: Read and write",
        "description": "Merge pull requests that modify GitHub Actions workflows",
    },
    "update_branch": {
        "classic": "repo scope",
        "fine_grained": "Contents: Read and write, Pull requests: Read and write",
        "description": "Update/rebase pull request branches",
    },
    "close": {
        "classic": "repo scope",
        "fine_grained": "Pull requests: Read and write",
        "description": "Close pull requests",
    },
    "branch_protection": {
        "classic": "repo scope",
        "fine_grained": "Administration: Read access",
        "description": "Read branch protection rules",
    },
    "checks": {
        "classic": "repo scope (or workflow for actions)",
        "fine_grained": "Actions: Read access, Workflows: Read access",
        "description": "Read status checks and workflow runs",
    },
}


async def _maybe_await(
    cb: Callable[..., None | Awaitable[None]] | None, *args, **kwargs
) -> None:
    if cb is None:
        return None
    result = cb(*args, **kwargs)
    if not asyncio.iscoroutine(result):
        return None
    return await cast("Awaitable[None]", result)


class GitHubAsync:
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
        self.log = logger or logging.getLogger(__name__)
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

    def _parse_permission_error(
        self, error: Exception, operation: str, owner: str = "", repo: str = ""
    ) -> PermissionError | None:
        """Parse HTTP error to determine if it's a permission issue.

        Args:
            error: The exception that was raised
            operation: The operation being performed (e.g., 'approve', 'merge')
            owner: Repository owner (for context in error messages)
            repo: Repository name (for context in error messages)

        Returns:
            PermissionError if this is a permission issue, None otherwise
        """
        error_str = str(error)

        # Check for 401 (unauthorized/expired token)
        if "401" in error_str or "Unauthorized" in error_str:
            return PermissionError(
                operation=operation,
                message="Token authentication failed - token may be expired or invalid",
                token_type_guidance={
                    "classic": "Regenerate your token at: https://github.com/settings/tokens",
                    "fine_grained": "Check token expiration at: https://github.com/settings/personal-access-tokens",
                    "fix": "Run: gh auth refresh -h github.com",
                },
            )

        # Check for 403 (forbidden/permission denied)
        if "403" in error_str or "Forbidden" in error_str:
            # Try to get more detailed error info from response
            response_text = ""
            response = getattr(error, "response", None)
            if response is not None:
                try:
                    response_text = str(getattr(response, "text", "")).lower()
                except AttributeError:
                    # Response object exposes no readable body; fall
                    # back to the empty default and keep classifying.
                    pass

            error_lower = error_str.lower()

            # Check for specific permission scenarios

            # 1. Workflow scope (already handled but included for completeness)
            if (
                "refusing to allow" in response_text
                and "workflow" in response_text
                and operation == "merge"
            ):
                perms = OPERATION_PERMISSIONS.get("merge_workflow", {})
                return PermissionError(
                    operation="merge_workflow",
                    message=f"Missing workflow permissions to merge PR in {owner}/{repo} that modifies GitHub Actions workflows",
                    token_type_guidance={
                        "classic": f"Add scope: {perms.get('classic', 'workflow')}",
                        "fine_grained": f"Enable: {perms.get('fine_grained', 'Workflows: Read and write')}",
                        "fix": "Run: gh auth refresh -h github.com -s workflow",
                    },
                )

            # 2. Fine-grained token repository scope
            if (
                "resource not accessible" in response_text
                or "not in scope" in error_lower
            ):
                return PermissionError(
                    operation=operation,
                    message=f"Repository {owner}/{repo} is not accessible with this token",
                    token_type_guidance={
                        "classic": "Token should have 'repo' scope for private repositories, or 'public_repo' for public repositories",
                        "fine_grained": f"Add {owner}/{repo} to the token's repository access list at: https://github.com/settings/tokens",
                        "fix": f"Edit your fine-grained token and add '{owner}/{repo}' to repository access",
                    },
                )

            # 3. Operation-specific permission errors
            perms = OPERATION_PERMISSIONS.get(operation, {})
            if perms:
                location = f" in {owner}/{repo}" if owner and repo else ""
                return PermissionError(
                    operation=operation,
                    message=f"Insufficient permissions to {perms.get('description', operation)}{location}",
                    token_type_guidance={
                        "classic": f"Required scope: {perms.get('classic', 'repo')}",
                        "fine_grained": f"Required permission: {perms.get('fine_grained', 'unknown')}",
                        "fix": "Update your token permissions at: https://github.com/settings/tokens",
                    },
                )

            # 4. Generic 403
            return PermissionError(
                operation=operation,
                message=f"Permission denied for {operation} operation{' in ' + owner + '/' + repo if owner and repo else ''}",
                token_type_guidance={
                    "classic": "Ensure token has 'repo' scope for full repository access",
                    "fine_grained": "Check that token has appropriate permissions and repository access",
                    "fix": "Review and update token permissions at: https://github.com/settings/tokens",
                },
            )

        # Check for 422 (unprocessable entity - often approval restrictions)
        if "422" in error_str and operation == "approve":
            if (
                "review cannot be requested from pull request author"
                in error_str.lower()
            ):
                return PermissionError(
                    operation=operation,
                    message="Cannot approve your own pull request",
                    token_type_guidance={
                        "classic": "GitHub does not allow self-approval of pull requests",
                        "fine_grained": "GitHub does not allow self-approval of pull requests",
                        "fix": "Request review from another team member",
                    },
                )
            elif "unprocessable entity" in error_str.lower():
                return PermissionError(
                    operation=operation,
                    message="Pull request approval failed - repository may have approval restrictions",
                    token_type_guidance={
                        "classic": "Check repository settings for review requirements",
                        "fine_grained": "Check repository settings for review requirements",
                        "fix": "Contact repository administrator to review branch protection rules",
                    },
                )

        # Not a permission error we recognize
        return None

    def _parse_rate_limit_headers(
        self, r: httpx.Response
    ) -> tuple[int, int, float | None]:
        """
        Parse GitHub rate limit headers.

        Returns:
            (remaining, limit, reset_epoch)
        """
        remaining = int(r.headers.get("X-RateLimit-Remaining", "1"))
        limit = int(r.headers.get("X-RateLimit-Limit", "60"))
        reset = r.headers.get("X-RateLimit-Reset")
        reset_epoch = float(reset) if reset else None
        return remaining, limit, reset_epoch

    async def _sleep_until(self, reset_epoch: float) -> None:
        now = _now()
        delay = max(0.0, reset_epoch - now)
        if delay > 0:
            await _maybe_await(self.on_rate_limited, reset_epoch)
            try:
                await asyncio.sleep(delay)
            finally:
                await _maybe_await(self.on_rate_limit_cleared)

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_random_exponential(multiplier=0.5, max=_TENACITY_MAX_BACKOFF),
        retry=retry_if_exception_type(
            (
                httpx.TransportError,
                httpx.ReadTimeout,
                RetryableError,
                SecondaryRateLimitError,
            )
        ),
    )
    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """
        Low-level request with concurrency limit, RPS limit, and retry handling.
        Handles primary/secondary rate limits and transient statuses.
        """
        async with self.semaphore:
            async with self.limiter:
                r = await self._client.request(method, url, **kwargs)

        # 401 should not be retried (bad credentials)
        if r.status_code == 401:
            r.raise_for_status()

        # Primary rate limit: examine headers and body
        if r.status_code == 403:
            body_text: str
            try:
                body_text = r.text or ""
            except Exception:
                body_text = ""

            remaining, _, reset_epoch = self._parse_rate_limit_headers(r)

            # Secondary rate limit (abuse detection)
            if _is_secondary_rate_limited(body_text):
                retry_after = r.headers.get("Retry-After")
                delay: float | None = None
                if retry_after:
                    try:
                        delay = float(retry_after)
                        self._last_retry_after = delay
                        self._apply_retry_after_throttling(delay)
                    except (TypeError, ValueError):
                        delay = None
                # Track error for adaptive throttling
                self._track_error("secondary_rate_limit")
                if delay is not None:
                    # GitHub told us exactly how long to wait.  Sleeping
                    # here *and* letting tenacity back off on top would
                    # stack the two; hand tenacity a pre-slept signal
                    # instead by waiting the advised time and then
                    # raising, which tenacity adds its own (smaller,
                    # jittered) delay to.  Keep that combined wait honest
                    # by subtracting tenacity's cap from our sleep.
                    effective = max(0.0, delay - _TENACITY_MAX_BACKOFF)
                    self.log.warning(
                        "Secondary rate limit hit. Retry-After=%ss, sleeping %ss",
                        delay,
                        effective,
                    )
                    if effective:
                        await asyncio.sleep(effective)
                else:
                    self.log.warning(
                        "Secondary rate limit hit without Retry-After; "
                        "deferring to retry backoff"
                    )
                raise SecondaryRateLimitError("Secondary rate limit encountered")

            # Primary rate limit exhausted
            if remaining == 0 or _is_primary_rate_limited(body_text):
                # Honor a Retry-After header if present (primary rate
                # limits may be reported as 403 or 429).  Parse it up
                # front so that an unparsable value (e.g. an HTTP-date)
                # falls back to the reset/backoff handling below rather
                # than triggering an immediate retry.
                retry_after = r.headers.get("Retry-After")
                retry_after_delay: float | None = None
                if retry_after:
                    try:
                        retry_after_delay = float(retry_after)
                    except (TypeError, ValueError):
                        retry_after_delay = None
                if retry_after_delay is not None:
                    self._last_retry_after = retry_after_delay
                    self.log.warning(
                        "Primary rate limit with Retry-After: %ss",
                        retry_after_delay,
                    )
                    await asyncio.sleep(max(0.0, retry_after_delay))
                    self._apply_retry_after_throttling(retry_after_delay)
                elif reset_epoch:
                    self.log.warning(
                        "Primary rate limit exhausted. Waiting until reset: %s",
                        reset_epoch,
                    )
                    await self._sleep_until(reset_epoch)
                else:
                    # If no reset header, backoff and retry
                    self.log.warning(
                        "Primary rate limit suspected without reset header; backing off"
                    )
                    await asyncio.sleep(5.0)

                # Track error for adaptive throttling
                self._track_error("primary_rate_limit")
                raise RetryableError("Primary rate limit reset waited; retrying")

        # Retryable transient statuses
        if _is_retryable_status(r.status_code):
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                retry_after_delay = None
                try:
                    retry_after_delay = float(retry_after)
                except (TypeError, ValueError):
                    # Retry-After was not a numeric delay; fall through
                    # to the standard retry handling.
                    retry_after_delay = None
                if retry_after_delay is not None:
                    self._last_retry_after = retry_after_delay
                    self.log.debug(
                        "HTTP %s with Retry-After: %ss",
                        r.status_code,
                        retry_after_delay,
                    )
                    await asyncio.sleep(max(0.0, retry_after_delay))
                    self._apply_retry_after_throttling(retry_after_delay)

            self._track_error("transient_error")
            self.log.debug("Retryable HTTP status %s received", r.status_code)
            raise RetryableError(f"Transient HTTP status: {r.status_code}")

        # All other errors -> raise
        r.raise_for_status()

        self._track_request()

        # Pace the next request when recent Retry-After headers indicated
        # sustained pressure.  Decays with time (see ``_current_adaptive_delay``).
        delay = self._current_adaptive_delay()
        if delay > 0:
            await asyncio.sleep(delay)

        # Dynamic concurrency and RPS tuning from the latest headers.
        try:
            self._record_budget(r)
            self._tune(self._headroom())
        except Exception as e:
            # Tuning is best-effort; never fail the request on tuning errors.
            self.log.debug("Adaptive concurrency tuning skipped: %s", e)
        # Push current metrics to progress tracker (if provided)
        try:
            await _maybe_await(
                getattr(self, "on_metrics", None),
                self._max_concurrency,
                float(self._current_rps),
            )
        except Exception as e:
            # Metrics reporting is best-effort.
            self.log.debug("Progress metrics reporting failed: %s", e)
        return r

    async def get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        r = await self._request("GET", f"{self.api_url}{path}", params=params)
        return r.json()  # type: ignore[no-any-return]

    async def post(
        self, path: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        r = await self._request("POST", f"{self.api_url}{path}", json=json)
        if r.status_code == 204:
            return {}
        return r.json()  # type: ignore[no-any-return]

    async def put(
        self, path: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        r = await self._request("PUT", f"{self.api_url}{path}", json=json)
        if r.status_code == 204:
            return {}
        return r.json()  # type: ignore[no-any-return]

    async def patch(
        self, path: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        r = await self._request("PATCH", f"{self.api_url}{path}", json=json)
        if r.status_code == 204:
            return {}
        return r.json()  # type: ignore[no-any-return]

    async def get_check_runs_for_ref(
        self, owner: str, repo: str, ref: str
    ) -> list[dict[str, Any]]:
        """Check runs reported against *ref*.

        Returns the raw runs, including superseded duplicates: deciding
        which run is authoritative for a given name belongs to
        :mod:`dependamerge.check_runs`, not here.
        """
        data = await self.get(
            f"/repos/{owner}/{repo}/commits/{ref}/check-runs",
            params={"per_page": 100},
        )
        if not isinstance(data, dict):
            return []
        return [run for run in (data.get("check_runs") or []) if isinstance(run, dict)]

    async def get_workflow_run_names_for_sha(
        self, owner: str, repo: str, head_sha: str
    ) -> set[str]:
        """Names of Actions workflow runs that exist for *head_sha*.

        Distinct from check runs.  A required workflow that GitHub never
        dispatched has **no workflow run at all** --- not a queued one, not
        a failed one, nothing --- so its absence here is the signal that
        waiting cannot help.  Check runs cannot express that: the
        workflow simply never appears.

        Paginated: a busy commit can carry more than one page of runs,
        and a required workflow sitting on page two would otherwise read
        as absent.  Callers use absence to *stop waiting*, so a false
        absence turns a live workflow into a reported merge failure.

        Request failures **propagate**: they are not converted into an
        empty set, and callers that use absence to stop waiting must
        handle them (``_absent_workflow_runs`` does).  An empty set
        means the lookup succeeded and found nothing, which is itself
        ambiguous --- a commit whose runs are not yet visible looks
        identical --- so that too must read as "unknown" rather than
        "nothing ran".  Both readings land on the same safe action:
        keep waiting.
        """
        names: set[str] = set()
        async for page in self.get_paginated(
            f"/repos/{owner}/{repo}/actions/runs",
            params={"head_sha": head_sha},
            per_page=100,
        ):
            if not isinstance(page, dict):
                continue
            runs = page.get("workflow_runs")
            if not runs:
                break
            for run in runs:
                if isinstance(run, dict):
                    name = run.get("name")
                    if isinstance(name, str) and name:
                        names.add(name)
        return names

    async def get_failing_status_contexts(
        self, owner: str, repo: str, ref: str
    ) -> list[str]:
        """Contexts whose latest commit *status* is failing.

        Distinct from check runs: pre-commit.ci, DCO and other legacy
        integrations report through the commit status API, so a caller
        reasoning about "what is failing" from check runs alone sees only
        half the picture.

        GitHub's combined-status endpoint already collapses each context
        to its latest state, so no deduplication is needed here.
        """
        data = await self.get(f"/repos/{owner}/{repo}/commits/{ref}/status")
        if not isinstance(data, dict):
            return []
        failing: list[str] = []
        for entry in data.get("statuses") or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("state") in ("failure", "error"):
                context = entry.get("context")
                if isinstance(context, str) and context and context not in failing:
                    failing.append(context)
        return failing

    async def update_pull_request_title(
        self, owner: str, repo: str, number: int, title: str
    ) -> None:
        """Set a pull request's title.

        REST: PATCH /repos/{owner}/{repo}/pulls/{pull_number}

        GitHub emits a ``pull_request.edited`` event, which re-runs any
        workflow listening for it --- the mechanism this is used for (see
        ``semantic_title``).  Note that ruleset-*injected* required
        workflows do **not** appear to honour ``edited``, so this is only
        useful for checks the repository or org wires up conventionally.
        """
        await self.patch(
            f"/repos/{owner}/{repo}/pulls/{number}",
            json={"title": title},
        )

    async def get_pull_request_commits(
        self, owner: str, repo: str, number: int
    ) -> list[dict[str, Any]]:
        """All commits on a pull request, across pages."""
        out: list[dict[str, Any]] = []
        async for page in self.get_paginated(
            f"/repos/{owner}/{repo}/pulls/{number}/commits",
            per_page=100,
        ):
            if isinstance(page, list):
                out.extend(c for c in page if isinstance(c, dict))
        return out

    async def graphql(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute a GraphQL query with retry for transient GraphQL errors.

        Note: HTTP-level issues are handled by _request's retry. Here we add
        retry for 200 OK responses that include GraphQL-level transient errors.
        """
        payload = {"query": query, "variables": variables or {}}

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(5),
            wait=wait_random_exponential(multiplier=0.5, max=10.0),
            retry=retry_if_exception_type(
                (RetryableError, httpx.TransportError, httpx.ReadTimeout)
            ),
        ):
            with attempt:
                r = await self._request("POST", self.graphql_url, json=payload)
                data = r.json()
                if "errors" in data and data["errors"]:
                    # Retry on transient errors, otherwise raise
                    if _is_transient_graphql_error(data["errors"]):
                        self.log.debug(
                            "Transient GraphQL error encountered; retrying: %s",
                            data["errors"],
                        )
                        raise RetryableError("Transient GraphQL error")
                    # Non-transient; raise detailed error
                    raise GraphQLError(json.dumps(data["errors"]))
                if "data" not in data:
                    # Unexpected shape; treat as transient
                    self.log.debug("GraphQL response missing 'data'; retrying")
                    raise RetryableError("Malformed GraphQL response")
                return data["data"]  # type: ignore[no-any-return]

        # Should not be reached due to reraise=True; keep mypy happy
        raise GraphQLError("GraphQL request failed after retries")

    def clear_block_reasons(self) -> None:
        """Forget every memoised block reason.

        For run boundaries: the merge manager supports reuse, and a
        non-confirmed invocation runs the whole batch as a preview
        first.  Expiry alone is not enough --- a second run can begin
        inside the window, and checks complete while a head SHA stays
        constant, so the earlier run's answer can be both cached and
        wrong.
        """
        self._block_reason_cache.clear()

    def invalidate_block_reason(self, owner: str, repo: str, number: int) -> None:
        """Forget any memoised block reason for a PR.

        Called after operations that change *why* a PR is blocked ---
        approving it, or attempting a merge.  Without this, the memo
        outlives the state it describes: approving a PR that reported
        "requires approval", then failing the retry for a different
        reason, would replay the stale approval message and could send
        the failure down the wrong recovery path.
        """
        for key in [
            k for k in self._block_reason_cache if k[:3] == (owner, repo, number)
        ]:
            self._block_reason_cache.pop(key, None)

    async def approve_pull_request(
        self, owner: str, repo: str, number: int, body: str
    ) -> None:
        """
        Approve a pull request.

        REST: POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews

        This endpoint returns a transient ``500`` with some regularity.
        ``500`` is deliberately absent from ``_is_retryable_status`` --- a
        blanket retry of every failed POST is unsafe --- so the retry is
        handled here, where the operation's semantics are known.

        Crucially, a ``500`` does **not** imply the review was not
        created: in the run analysed in
        ``docs/BULK_RUN_PERFORMANCE_AUDIT.md``, 4 of the 6 PRs whose
        approval "failed" this way went on to merge, which requires the
        approval to have landed.  So before each *retry* --- and once
        more after the final attempt --- the review list is re-read, and
        an approval already present counts as success rather than
        stacking a duplicate review.  The first attempt skips that check,
        so the common path costs exactly one request.

        Raises:
            PermissionError: If token lacks required permissions
        """
        last_exc: Exception | None = None
        for attempt in range(_APPROVE_MAX_ATTEMPTS):
            if attempt and await self._has_own_approval(owner, repo, number):
                self.log.debug(
                    "Approval for %s/%s#%s already present after a %s; "
                    "treating as success",
                    owner,
                    repo,
                    number,
                    "transient error",
                )
                # Reporting success means the approval landed, so the
                # memo must go here too --- not only on the path where
                # the POST itself returned cleanly.
                self.invalidate_block_reason(owner, repo, number)
                return
            try:
                await self.post(
                    f"/repos/{owner}/{repo}/pulls/{number}/reviews",
                    json={"event": "APPROVE", "body": body},
                )
                # The PR's block reason has just changed by construction.
                self.invalidate_block_reason(owner, repo, number)
                return
            except Exception as e:
                perm_error = self._parse_permission_error(e, "approve", owner, repo)
                if perm_error:
                    raise perm_error from e
                if not _is_transient_server_error(e):
                    raise
                last_exc = e
                if attempt == _APPROVE_MAX_ATTEMPTS - 1:
                    break
                delay = _APPROVE_RETRY_BASE_DELAY * (2**attempt)
                self.log.warning(
                    "Transient error approving %s/%s#%s (attempt %d/%d); "
                    "retrying in %.1fs",
                    owner,
                    repo,
                    number,
                    attempt + 1,
                    _APPROVE_MAX_ATTEMPTS,
                    delay,
                )
                await asyncio.sleep(delay)

        # Attempts exhausted.  One last look: the final POST may have
        # created the review despite reporting failure.
        if await self._has_own_approval(owner, repo, number):
            self.log.info(
                "Approval for %s/%s#%s landed despite a reported error",
                owner,
                repo,
                number,
            )
            self.invalidate_block_reason(owner, repo, number)
            return
        assert last_exc is not None
        raise last_exc

    async def _has_own_approval(self, owner: str, repo: str, number: int) -> bool:
        """Whether the authenticated user already has an APPROVED review.

        Paginates: a single default page caps at 30 reviews, and missing
        an existing approval on a busy PR would defeat the
        duplicate-suppression this exists for and post another review.
        """
        try:
            login = await self.get_authenticated_user_login()
            if not login:
                # The lookup returns ``None`` on failure rather than
                # raising.  Without this guard a review carrying
                # ``user: null`` yields ``None == None`` and reports an
                # approval that does not exist --- which would stop
                # ``approve_pull_request`` retrying and let it report
                # success having approved nothing.
                self.log.debug(
                    "Cannot confirm existing approval on %s/%s#%s: "
                    "authenticated user unknown",
                    owner,
                    repo,
                    number,
                )
                return False
            async for page in self.get_paginated(
                f"/repos/{owner}/{repo}/pulls/{number}/reviews",
                per_page=100,
            ):
                if not isinstance(page, list):
                    continue
                for review in page:
                    if not isinstance(review, dict):
                        continue
                    if review.get("state") != "APPROVED":
                        continue
                    user = review.get("user") or {}
                    reviewer = user.get("login")
                    if reviewer and reviewer == login:
                        return True
        except Exception as exc:
            self.log.debug(
                "Could not read reviews for %s/%s#%s: %s", owner, repo, number, exc
            )
            return False
        return False

    async def merge_pull_request(
        self, owner: str, repo: str, number: int, merge_method: str = "merge"
    ) -> bool:
        """
        Merge a pull request.

        REST: PUT /repos/{owner}/{repo}/pulls/{pull_number}/merge

        Raises:
            PermissionError: If token lacks required permissions
        """
        # A merge attempt changes the PR's state whether it succeeds or
        # is refused, so any memoised block reason describes the past.
        self.invalidate_block_reason(owner, repo, number)
        try:
            self.log.debug(
                f"Attempting to merge PR #{number} in {owner}/{repo} with method={merge_method}"
            )
            data = await self.put(
                f"/repos/{owner}/{repo}/pulls/{number}/merge",
                json={"merge_method": merge_method},
            )
            # The API returns {"merged": true/false, ...}
            merged = bool(data.get("merged", False))
            if merged:
                self.log.debug(f"Successfully merged PR #{number} in {owner}/{repo}")
            else:
                self.log.warning(
                    f"GitHub API returned merged=false for PR #{number} in {owner}/{repo}: {data}"
                )
            return merged
        except Exception as e:
            # Check for permission errors first (includes workflow scope check)
            perm_error = self._parse_permission_error(e, "merge", owner, repo)
            if perm_error:
                # GitHub returns the "refusing to allow ... workflow" 403
                # only when the *classic* token lacks the ``workflow``
                # scope.  Before repeating that guidance, confirm the scope
                # really is absent: if the token already carries it (or is a
                # fine-grained/app token we cannot introspect and which
                # therefore would not produce this classic-PAT message), the
                # true cause is something else — typically a repository
                # ruleset that restricts workflow-file updates, or an
                # un-authorized SSO session.  Telling the user to add a scope
                # they already hold would be an inaccurate diagnosis.
                if perm_error.operation == "merge_workflow":
                    has_workflow = await self.check_workflow_scope()
                    if has_workflow is True:
                        perm_error = PermissionError(
                            operation="merge_workflow_restricted",
                            message=(
                                f"GitHub refused to merge PR in {owner}/{repo} "
                                "even though the token already has the "
                                "'workflow' scope. The workflow-file update is "
                                "being blocked by something other than token "
                                "scope"
                            ),
                            token_type_guidance={
                                "classic": (
                                    "Check for a repository ruleset that "
                                    "restricts updates to .github/workflows/** "
                                    "and confirm the token is SSO-authorized "
                                    "for this organization"
                                ),
                                "fine_grained": (
                                    "Check for a repository ruleset that "
                                    "restricts updates to .github/workflows/**"
                                ),
                                "fix": (
                                    "Review the repository's rulesets and "
                                    "organization SSO authorization for this "
                                    "token"
                                ),
                            },
                        )
                self.log.debug(
                    f"Permission error merging PR #{number} in {owner}/{repo}: {perm_error}"
                )
                raise perm_error from e

            error_type = type(e).__name__
            error_msg = str(e)
            self.log.debug(
                f"Merge API error for PR #{number} in {owner}/{repo}: {error_type}: {error_msg}"
            )

            github_detail = self._extract_github_error_detail(e)
            if github_detail:
                self.log.debug(
                    f"GitHub merge API response body for #{number}: {github_detail}"
                )

            # Re-check PR state: the merge may have actually succeeded
            # despite the exception (a race where the API call lands
            # but we still see an error from rate-limiting, network, or
            # JSON parsing), and the state adds context to the error we
            # raise.
            return await self._validate_merge_result(
                owner, repo, number, e, github_detail
            )

    @staticmethod
    def _extract_github_error_detail(error: Exception) -> str:
        """Extract GitHub's response-body message from a failed request.

        GitHub puts the *actual* reason here — ruleset violations,
        "Required workflows ... are not satisfied", required-check names,
        etc.  The ``HTTPStatusError`` text only carries the status line
        (e.g. "405 Method Not Allowed"), so without this the real cause is
        silently lost.  Whitespace/newlines are collapsed so the reason
        fits on a single status line.

        Returns an empty string when no detail could be extracted.
        """
        response = getattr(error, "response", None)
        if response is None:
            return ""
        try:
            body = response.json()
            if isinstance(body, dict) and isinstance(body.get("message"), str):
                return " ".join(body["message"].split())
        except Exception:
            # Response body was not JSON (or .json() failed); fall through
            # to the raw-text extraction below rather than failing here.
            pass
        try:
            raw = getattr(response, "text", "") or ""
            return " ".join(raw.split())[:500]
        except Exception:
            return ""

    async def _validate_merge_result(
        self,
        owner: str,
        repo: str,
        number: int,
        error: Exception,
        github_detail: str,
    ) -> bool:
        """Re-check PR state after a merge attempt raised an exception.

        The merge may have actually succeeded despite the exception (a race
        where the API call lands but we still see an error from
        rate-limiting, network, or JSON parsing).  When the PR is confirmed
        merged, return ``True``.  Otherwise raise an enhanced exception that
        preserves the original error text (its HTTP status line is
        string-matched by ``_merge_pr_with_retry`` to classify retryable vs
        terminal failures) and adds GitHub's actionable response body plus
        PR-state context.
        """
        try:
            pr_data_response = await self.get(f"/repos/{owner}/{repo}/pulls/{number}")
            # PR data should always be a dict, not a list
            pr_data = pr_data_response if isinstance(pr_data_response, dict) else {}

            mergeable = pr_data.get("mergeable")
            mergeable_state = pr_data.get("mergeable_state")
            state = pr_data.get("state")
            merged = pr_data.get("merged", False)
            draft = pr_data.get("draft", False)

            # Check if the merge actually succeeded despite the exception.
            # This handles race conditions where the API succeeds but we get
            # an exception due to rate limiting, network issues, JSON
            # parsing, etc.
            if state == "closed" and merged:
                self.log.info(
                    f"PR #{number} in {owner}/{repo} was successfully merged despite exception: {error}"
                )
                return True

            # Enhanced error message.  Always keep the original error text —
            # it carries the HTTP status line (e.g. "405 Method Not
            # Allowed") that ``_merge_pr_with_retry`` string-matches to
            # classify retryable vs terminal failures; dropping it made
            # every blocked/ruleset 405 fall through to the generic retry
            # path (3 attempts + sleeps).  Then *add* GitHub's response body
            # (the actionable reason) when we captured it.
            error_msg = (
                f"Failed to merge PR #{number} in {owner}/{repo}. Error: {str(error)}."
            )
            if github_detail:
                error_msg += f" GitHub: {github_detail}"
            error_msg += (
                f" (PR state: {state}, mergeable: {mergeable}, "
                f"mergeable_state: {mergeable_state})"
            )

            # Note common state-based causes for 405-style errors.
            if mergeable_state == "blocked":
                error_msg += " [blocked by branch protection / required checks]"
            elif mergeable_state == "behind":
                error_msg += " [PR branch is behind base branch]"
            elif mergeable_state == "dirty":
                error_msg += " [PR has merge conflicts]"
            elif draft:
                error_msg += " [cannot merge draft PR]"
            elif state == "closed" and not merged:
                error_msg += " [PR was closed without merging]"
            elif state != "open":
                error_msg += f" [PR is not open, state: {state}]"

            raise Exception(error_msg) from error
        except Exception as inner_e:
            # The enhanced-error path raised successfully (the message
            # starts with "Failed to merge PR") — propagate it unchanged.
            # A bare ``raise`` preserves ``inner_e`` together with its
            # existing ``__cause__`` (set to ``error`` above) and original
            # traceback, whereas ``raise inner_e from error`` would rewrite
            # the chaining.
            if "Failed to merge PR" in str(inner_e):
                raise
            # Otherwise the PR-state re-fetch itself failed.  Still surface
            # GitHub's response body (the actionable reason) when we
            # captured it, rather than dropping back to the bare
            # status-line ``HTTPStatusError``.
            if github_detail:
                raise Exception(
                    f"Failed to merge PR #{number} in {owner}/{repo}. "
                    f"Error: {str(error)}. GitHub: {github_detail}"
                ) from error
            raise error from inner_e

    async def enable_auto_merge(
        self, pull_request_node_id: str, merge_method: str = "MERGE"
    ) -> bool:
        """
        Enable auto-merge on a pull request via GraphQL.

        Auto-merge will automatically merge the PR once all required
        branch protection rules are satisfied.

        Args:
            pull_request_node_id: The GraphQL node ID of the pull request.
            merge_method: Merge method - "MERGE", "SQUASH", or "REBASE".
                Lowercase values ("merge", "squash", "rebase") are
                automatically uppercased.

        Returns:
            True if auto-merge was successfully enabled, False otherwise.
        """
        from .github_graphql import ENABLE_AUTO_MERGE

        # Normalise to the GraphQL enum (uppercase)
        graphql_method = merge_method.upper()
        if graphql_method not in ("MERGE", "SQUASH", "REBASE"):
            self.log.warning(
                "Invalid merge method for auto-merge: %s; defaulting to MERGE",
                merge_method,
            )
            graphql_method = "MERGE"

        try:
            result = await self.graphql(
                ENABLE_AUTO_MERGE,
                {
                    "pullRequestId": pull_request_node_id,
                    "mergeMethod": graphql_method,
                },
            )
            auto_merge_data = (
                result.get("enablePullRequestAutoMerge", {})
                .get("pullRequest", {})
                .get("autoMergeRequest")
            )
            if auto_merge_data:
                self.log.debug(
                    "Auto-merge enabled for PR %s (method=%s, enabledAt=%s)",
                    pull_request_node_id,
                    auto_merge_data.get("mergeMethod"),
                    auto_merge_data.get("enabledAt"),
                )
                return True
            self.log.debug(
                "Auto-merge response missing autoMergeRequest for PR %s",
                pull_request_node_id,
            )
            return False
        except Exception as e:
            error_msg = str(e)
            # Common reasons auto-merge can't be enabled:
            # - Repository doesn't have auto-merge enabled in settings
            # - PR has conflicts
            # - Required status checks not configured
            self.log.debug(
                "Could not enable auto-merge for PR %s: %s",
                pull_request_node_id,
                error_msg,
            )
            return False

    async def get_pull_request_review_comments(
        self, owner: str, repo: str, number: int
    ) -> list[dict[str, Any]]:
        """
        Get review comments for a pull request.

        REST: GET /repos/{owner}/{repo}/pulls/{pull_number}/comments
        """
        try:
            data = await self.get(f"/repos/{owner}/{repo}/pulls/{number}/comments")
            return data if isinstance(data, list) else []
        except Exception as e:
            # If we can't get review comments, return empty list
            self.log.debug(f"Could not fetch review comments for PR {number}: {e}")
            return []

    async def post_issue_comment(
        self, owner: str, repo: str, number: int, body: str
    ) -> dict[str, Any]:
        """
        Post a comment on an issue or pull request.

        REST: POST /repos/{owner}/{repo}/issues/{issue_number}/comments

        Raises:
            PermissionError: If token lacks required permissions
        """
        try:
            data = await self.post(
                f"/repos/{owner}/{repo}/issues/{number}/comments",
                json={"body": body},
            )
        except Exception as e:
            perm_error = self._parse_permission_error(
                e, f"post a comment on issue or pull request #{number}", owner, repo
            )
            if perm_error:
                raise perm_error from e
            raise
        return data if isinstance(data, dict) else {}

    async def check_pr_commit_signatures(
        self, owner: str, repo: str, number: int
    ) -> tuple[bool, list[str]]:
        """Check whether all commits on a pull request have verified signatures.

        REST: GET /repos/{owner}/{repo}/pulls/{pull_number}/commits

        Returns:
            Tuple of ``(all_verified, unverified_shas)``.
            ``all_verified`` is True when every commit carries a
            valid signature according to GitHub.
            ``unverified_shas`` contains the abbreviated SHAs of
            any commits whose verification failed.

        Raises:
            Exception: surfaces the underlying API/network error
            on failure rather than silently returning a default.
            Callers that want fail-open or fail-closed semantics
            should wrap the call in ``try``/``except`` and decide
            for themselves — the previous fail-open default
            (returning ``(True, [])``) collided with the
            signature-preservation gate in ``rebase.py``, which
            documents "verified" as a positive confirmation.
        """
        unverified: list[str] = []
        # Iterate over all pages of commits to ensure we don't miss
        # unverified commits on pull requests with >100 commits.
        async for commits in self.get_paginated(
            f"/repos/{owner}/{repo}/pulls/{number}/commits",
            per_page=100,
        ):
            if not isinstance(commits, list):
                # Unexpected response shape: the API returned 200 OK but
                # not the documented list of commits. We cannot determine
                # signature status from this, so we must not pretend every
                # commit is verified (the old fail-open ``(True, [])``
                # default collided with the signature-preservation gate in
                # ``rebase.py``). Surface the uncertainty to the caller.
                raise RuntimeError(
                    "Unexpected response shape from "
                    f"/repos/{owner}/{repo}/pulls/{number}/commits: "
                    f"expected a list, got {type(commits).__name__}"
                )

            for commit_data in commits:
                if not isinstance(commit_data, dict):
                    continue
                raw_sha = commit_data.get("sha")
                sha = str(raw_sha)[:8] if isinstance(raw_sha, str) else "unknown"
                commit_obj = commit_data.get("commit")
                if not isinstance(commit_obj, dict):
                    unverified.append(sha)
                    continue
                verification = commit_obj.get("verification")
                if not isinstance(verification, dict):
                    unverified.append(sha)
                    continue
                if not verification.get("verified", False):
                    unverified.append(sha)

        all_verified = len(unverified) == 0
        return all_verified, unverified

    async def requires_commit_signatures(
        self, owner: str, repo: str, branch: str = "main"
    ) -> bool:
        """
        Check whether a branch requires signed (verified) commits.

        Uses two complementary sources:

        1. **Classic branch protection** – the ``required_signatures``
           sub-resource of the branch protection REST endpoint.
        2. **Repository rulesets** (newer API) – any active ruleset that
           targets the given branch and contains a ``required_signatures``
           rule.

        Returns:
            True if signed commits are required by *either* mechanism.

        Results are cached per ``owner/repo@branch`` for the session:
        the requirement is branch-protection/ruleset configuration that
        does not change while dependamerge runs, and the uncached path
        costs up to 3 + N requests (classic-protection probe, repo
        metadata, ruleset list, one detail GET per ruleset).  Verdicts
        derived from transient API errors are *not* cached, so a
        momentary outage cannot pin a wrong answer for the whole run.
        """
        cache_key = f"{owner}/{repo}@{branch}"
        cached = self._requires_signatures_cache.get(cache_key)
        if cached is not None:
            return cached
        result, reliable = await self._requires_commit_signatures_uncached(
            owner, repo, branch
        )
        if reliable:
            self._requires_signatures_cache[cache_key] = result
        return result

    async def _requires_commit_signatures_uncached(
        self, owner: str, repo: str, branch: str
    ) -> tuple[bool, bool]:
        """Uncached implementation of :meth:`requires_commit_signatures`.

        Returns:
            Tuple of ``(requires_signatures, reliable)``.  ``reliable``
            is False when a transient (non-404) API error prevented a
            definitive verdict — a ``True`` verdict is always reliable
            (positive evidence), but an error-derived ``False`` must
            not be cached because the requirement may simply have been
            unreadable at that moment.
        """
        reliable = True
        try:
            # The signatures endpoint returns 200 with {"enabled": true/false}
            # or 404 when branch protection / signature requirement is absent.
            encoded_branch = quote(branch, safe="")
            sig_data = await self.get(
                f"/repos/{owner}/{repo}/branches/{encoded_branch}/protection/required_signatures"
            )
            if isinstance(sig_data, dict) and sig_data.get("enabled"):
                self.log.debug(
                    "Branch %s/%s:%s requires commit signatures (classic protection)",
                    owner,
                    repo,
                    branch,
                )
                return True, True
        except Exception as e:
            # 404 → not enabled; other errors → continue checking rulesets
            if "404" not in str(e):
                reliable = False
                self.log.debug(
                    "Error checking classic signature requirement for %s/%s:%s: %s",
                    owner,
                    repo,
                    branch,
                    e,
                )

        try:
            # Resolve the repo's actual default branch so that
            # ~DEFAULT_BRANCH ruleset conditions are evaluated correctly.
            default_branch: str | None = None
            try:
                repo_data = await self.get(f"/repos/{owner}/{repo}")
                if isinstance(repo_data, dict):
                    default_branch = repo_data.get("default_branch")
            except Exception as e:
                # Best-effort: without the default branch we fall through
                # to conservative ~DEFAULT_BRANCH matching. Log the cause
                # at debug level rather than discarding it silently.
                self.log.debug(
                    "Could not resolve default branch for %s/%s: %s",
                    owner,
                    repo,
                    e,
                )

            # Paginate through all rulesets to collect their IDs.
            # The list endpoint may not include full rules/conditions,
            # so we fetch each ruleset's detail individually (matching
            # the pattern in get_required_status_checks).
            ruleset_ids: list[int] = []
            page = 1
            per_page = 100
            while True:
                page_rulesets = await self.get(
                    f"/repos/{owner}/{repo}/rulesets?per_page={per_page}&page={page}"
                )
                if not isinstance(page_rulesets, list) or not page_rulesets:
                    break
                for rs in page_rulesets:
                    if isinstance(rs, dict):
                        rs_id = rs.get("id")
                        if rs_id is not None:
                            ruleset_ids.append(int(rs_id))
                if len(page_rulesets) < per_page:
                    break
                page += 1

            for ruleset_id in ruleset_ids:
                try:
                    detail = await self.get(
                        f"/repos/{owner}/{repo}/rulesets/{ruleset_id}"
                    )
                    if not isinstance(detail, dict):
                        continue
                except Exception as detail_err:
                    # An unreadable ruleset could hide a
                    # required_signatures rule — the eventual False
                    # verdict is no longer definitive.
                    reliable = False
                    self.log.debug(
                        "Could not fetch ruleset %s for %s/%s: %s",
                        ruleset_id,
                        owner,
                        repo,
                        detail_err,
                    )
                    continue

                # Only consider active rulesets
                if detail.get("enforcement") != "active":
                    continue
                # Check if this ruleset applies to our branch
                conditions = detail.get("conditions", {})
                if isinstance(conditions, dict) and not self._ruleset_applies_to_branch(
                    conditions, branch, default_branch
                ):
                    continue
                rules = detail.get("rules", [])
                if isinstance(rules, list):
                    for rule in rules:
                        if (
                            isinstance(rule, dict)
                            and rule.get("type") == "required_signatures"
                        ):
                            self.log.debug(
                                "Branch %s/%s:%s requires commit "
                                "signatures (ruleset: %s)",
                                owner,
                                repo,
                                branch,
                                detail.get("name", "unknown"),
                            )
                            return True, True
        except Exception as e:
            reliable = False
            self.log.debug(
                "Error checking rulesets for signature requirement on %s/%s:%s: %s",
                owner,
                repo,
                branch,
                e,
            )

        return False, reliable

    async def requires_strict_status_checks(
        self, owner: str, repo: str, branch: str = "main"
    ) -> bool:
        """Check whether a branch requires PR heads to be up to date.

        GitHub only rejects the merge of a ``behind`` PR when the
        branch's protection enforces the *strict* status-check policy
        ("Require branches to be up to date before merging").  Without
        it, a behind-but-green PR merges fine and any proactive rebase
        is wasted work (plus a full CI re-run).  The merge pipeline
        uses this to rebase **only when GitHub would actually demand
        it**.

        Uses two complementary sources:

        1. **Classic branch protection** –
           ``required_status_checks.strict`` on the branch protection
           REST payload (already cached by :meth:`get_branch_protection`).
        2. **Repository rulesets** – any active ruleset targeting the
           branch whose ``required_status_checks`` rule sets
           ``strict_required_status_checks_policy``.

        Returns:
            True if either mechanism requires the branch to be up to
            date before merging.

        Results are cached per ``owner/repo@branch`` for the session;
        verdicts derived from transient API errors are not cached so a
        momentary outage cannot pin a wrong answer for the whole run.
        """
        cache_key = f"{owner}/{repo}@{branch}"
        cached = self._requires_strict_checks_cache.get(cache_key)
        if cached is not None:
            return cached
        result, reliable = await self._requires_strict_status_checks_uncached(
            owner, repo, branch
        )
        if reliable:
            self._requires_strict_checks_cache[cache_key] = result
        return result

    async def _requires_strict_status_checks_uncached(
        self, owner: str, repo: str, branch: str
    ) -> tuple[bool, bool]:
        """Uncached implementation of :meth:`requires_strict_status_checks`.

        Returns:
            Tuple of ``(requires_strict, reliable)``.  ``reliable`` is
            False when a transient API error prevented a definitive
            verdict — a ``True`` verdict is always reliable (positive
            evidence), but an error-derived ``False`` must not be
            cached because the requirement may simply have been
            unreadable at that moment.
        """
        reliable = True
        try:
            protection = await self.get_branch_protection(owner, repo, branch)
            checks = protection.get("required_status_checks")
            if isinstance(checks, dict) and checks.get("strict") is True:
                self.log.debug(
                    "Branch %s/%s:%s requires up-to-date heads "
                    "(classic protection strict checks)",
                    owner,
                    repo,
                    branch,
                )
                return True, True
        except Exception as e:
            # get_branch_protection already maps 404 to {}; anything
            # surfacing here is a transient failure.
            reliable = False
            self.log.debug(
                "Error checking classic strict-checks policy for %s/%s:%s: %s",
                owner,
                repo,
                branch,
                e,
            )

        try:
            default_branch: str | None = None
            try:
                repo_data = await self.get(f"/repos/{owner}/{repo}")
                if isinstance(repo_data, dict):
                    default_branch = repo_data.get("default_branch")
            except Exception as e:
                self.log.debug(
                    "Could not resolve default branch for %s/%s: %s",
                    owner,
                    repo,
                    e,
                )

            ruleset_ids: list[int] = []
            page = 1
            per_page = 100
            while True:
                page_rulesets = await self.get(
                    f"/repos/{owner}/{repo}/rulesets?per_page={per_page}&page={page}"
                )
                if not isinstance(page_rulesets, list) or not page_rulesets:
                    break
                for rs in page_rulesets:
                    if isinstance(rs, dict):
                        rs_id = rs.get("id")
                        if rs_id is not None:
                            ruleset_ids.append(int(rs_id))
                if len(page_rulesets) < per_page:
                    break
                page += 1

            for ruleset_id in ruleset_ids:
                try:
                    detail = await self.get(
                        f"/repos/{owner}/{repo}/rulesets/{ruleset_id}"
                    )
                    if not isinstance(detail, dict):
                        continue
                except Exception as detail_err:
                    # An unreadable ruleset could hide a strict
                    # required_status_checks rule — the eventual False
                    # verdict is no longer definitive.
                    reliable = False
                    self.log.debug(
                        "Could not fetch ruleset %s for %s/%s: %s",
                        ruleset_id,
                        owner,
                        repo,
                        detail_err,
                    )
                    continue

                if detail.get("enforcement") != "active":
                    continue
                conditions = detail.get("conditions", {})
                if isinstance(conditions, dict) and not self._ruleset_applies_to_branch(
                    conditions, branch, default_branch
                ):
                    continue
                rules = detail.get("rules", [])
                if not isinstance(rules, list):
                    continue
                for rule in rules:
                    if (
                        isinstance(rule, dict)
                        and rule.get("type") == "required_status_checks"
                    ):
                        params = rule.get("parameters")
                        if (
                            isinstance(params, dict)
                            and params.get("strict_required_status_checks_policy")
                            is True
                        ):
                            self.log.debug(
                                "Branch %s/%s:%s requires up-to-date "
                                "heads (ruleset: %s)",
                                owner,
                                repo,
                                branch,
                                detail.get("name", "unknown"),
                            )
                            return True, True
        except Exception as e:
            reliable = False
            self.log.debug(
                "Error checking rulesets for strict-checks policy on %s/%s:%s: %s",
                owner,
                repo,
                branch,
                e,
            )

        return False, reliable

    @staticmethod
    def _ruleset_applies_to_branch(
        conditions: dict[str, Any],
        branch: str,
        default_branch: str | None = None,
    ) -> bool:
        """Check whether a ruleset's ref_name conditions match *branch*.

        Ruleset conditions use ``conditions.ref_name.include`` /
        ``conditions.ref_name.exclude`` arrays.  Recognised patterns:

        * ``~DEFAULT_BRANCH`` — matches when *branch* equals *default_branch*.
          If *default_branch* is not supplied, the match is treated as
          ``True`` (conservative) to avoid silently filtering out rulesets
          for repos whose default branch is something other than
          ``main``/``master``.
        * ``~ALL``            — matches every branch.
        * ``refs/heads/<name>`` — exact ref match.
        * Bare branch name   — treated as ``refs/heads/<name>``.

        If the conditions dict is empty or missing ``ref_name`` the
        ruleset is assumed to apply (conservative).
        """
        ref_name = conditions.get("ref_name", {})
        if not isinstance(ref_name, dict):
            return True  # No conditions — assume applies

        include = ref_name.get("include", [])
        exclude = ref_name.get("exclude", [])

        full_ref = f"refs/heads/{branch}"

        # Must match at least one include pattern (if any are specified)
        if include and not any(
            GitHubAsync._ref_pattern_matches(p, branch, full_ref, default_branch)
            for p in include
            if isinstance(p, str)
        ):
            return False

        # Must not match any exclude pattern
        if any(
            GitHubAsync._ref_pattern_matches(p, branch, full_ref, default_branch)
            for p in exclude
            if isinstance(p, str)
        ):
            return False

        return True

    @staticmethod
    def _ref_pattern_matches(
        pattern: str,
        branch: str,
        full_ref: str,
        default_branch: str | None,
    ) -> bool:
        """Check whether a single ruleset ref pattern matches *branch*.

        Defined as a static helper method (rather than a closure inside
        ``_ruleset_applies_to_branch``) so it is not re-created on every
        call and can be reused across the include/exclude comprehensions.
        """
        import fnmatch

        if pattern == "~ALL":
            return True
        if pattern == "~DEFAULT_BRANCH":
            if default_branch is None:
                # Unknown default branch — conservatively assume match
                return True
            return branch == default_branch
        # Normalise bare branch names to full refs
        pat = pattern if pattern.startswith("refs/") else f"refs/heads/{pattern}"
        return fnmatch.fnmatchcase(full_ref, pat)

    @staticmethod
    def _required_status_checks_from_detail(
        detail: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return required-status-check entries declared by a ruleset detail."""
        checks: list[dict[str, Any]] = []
        rules = detail.get("rules")
        if not isinstance(rules, list):
            return checks
        for rule in rules:
            if (
                not isinstance(rule, dict)
                or rule.get("type") != "required_status_checks"
            ):
                continue
            params = rule.get("parameters")
            if not isinstance(params, dict):
                continue
            required = params.get("required_status_checks")
            if not isinstance(required, list):
                continue
            for check in required:
                if (
                    isinstance(check, dict)
                    and isinstance(check.get("context"), str)
                    and check["context"]
                ):
                    checks.append(check)
        return checks

    async def _fetch_ruleset_required_checks(
        self, owner: str, repo: str, branch: str, default_branch: str | None
    ) -> tuple[list[dict[str, Any]], bool]:
        """Collect required checks from repo/org rulesets targeting *branch*.

        Returns ``(checks, reliable)`` where ``reliable`` is False when any
        ruleset request failed (so the caller must not cache the verdict).
        """
        checks: list[dict[str, Any]] = []
        try:
            rulesets = await self.get(f"/repos/{owner}/{repo}/rulesets?per_page=100")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.log.debug(
                f"Could not fetch rulesets for {owner}/{repo}: {e}",
                exc_info=True,
            )
            return checks, False
        if not isinstance(rulesets, list):
            return checks, True

        reliable = True
        for ruleset in rulesets:
            if not isinstance(ruleset, dict):
                continue
            ruleset_id = ruleset.get("id")
            if not ruleset_id:
                continue
            try:
                detail = await self.get(f"/repos/{owner}/{repo}/rulesets/{ruleset_id}")
                if not isinstance(detail, dict):
                    continue
                # Filter: skip rulesets that do not target this branch
                conditions = detail.get("conditions", {})
                if isinstance(conditions, dict) and not self._ruleset_applies_to_branch(
                    conditions, branch, default_branch
                ):
                    self.log.debug(
                        f"Ruleset {ruleset_id} does not apply to branch '{branch}'; skipping"
                    )
                    continue
                checks.extend(self._required_status_checks_from_detail(detail))
            except asyncio.CancelledError:
                raise
            except Exception as detail_err:
                reliable = False
                self.log.debug(
                    f"Could not fetch ruleset {ruleset_id} details: {detail_err}",
                    exc_info=True,
                )
        return checks, reliable

    async def _fetch_branch_protection_required_checks(
        self, owner: str, repo: str, branch: str
    ) -> tuple[list[dict[str, Any]], bool]:
        """Collect required checks from classic branch protection.

        Returns ``(checks, reliable)``.  Branch protection may be absent or
        inaccessible with the current token; a plain 404 is the definitive
        "no protection" answer, while anything else leaves the verdict
        unreliable.
        """
        checks: list[dict[str, Any]] = []
        try:
            data = await self.get(
                f"/repos/{owner}/{repo}/branches/{branch}/protection/required_status_checks"
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return checks, "404" in str(e)
        if isinstance(data, dict):
            for ctx in data.get("contexts", []):
                if isinstance(ctx, str) and ctx:
                    checks.append({"context": ctx})
            for check in data.get("checks", []):
                if (
                    isinstance(check, dict)
                    and isinstance(check.get("context"), str)
                    and check["context"]
                ):
                    checks.append(check)
        return checks, True

    async def get_required_status_checks(
        self, owner: str, repo: str, branch: str
    ) -> list[dict[str, Any]]:
        """
        Get required status checks for a branch by inspecting rulesets.

        Only rulesets whose ``conditions.ref_name`` patterns match *branch*
        are considered.  Falls back to branch protection rules if rulesets
        are not available.
        Returns a list of dicts with 'context' and optionally 'integration_id'.
        Results are deduplicated by ``context``.

        Results are cached per ``owner/repo@branch`` for the session:
        required-check configuration is repo/branch-level state that does
        not change while dependamerge runs, and the block-reason analysis
        consults it repeatedly (several times per blocked PR).  The
        uncached path costs 2 + N requests (repo + ruleset list + one
        detail GET per ruleset), so the cache saves a burst of API
        traffic on every repeat.  Results assembled while any of those
        requests failed are *not* cached: the fetch treats errors as
        "no required checks", and pinning that error-derived verdict
        for the whole session could misclassify blocked PRs long after
        a transient outage has passed.
        """
        cache_key = f"{owner}/{repo}@{branch}"
        cached = self._required_checks_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        required_checks: list[dict[str, Any]] = []
        seen_contexts: set[str] = set()

        def _add(candidates: list[dict[str, Any]]) -> None:
            for check in candidates:
                ctx = check.get("context")
                if not isinstance(ctx, str) or not ctx:
                    continue
                if ctx not in seen_contexts:
                    seen_contexts.add(ctx)
                    required_checks.append(check)

        # Resolve the repo's actual default branch so that ~DEFAULT_BRANCH
        # ruleset conditions are evaluated correctly (not hardcoded to
        # main/master).
        default_branch = await self._resolve_default_branch(owner, repo)

        # Try rulesets first (org-level and repo-level)
        ruleset_checks, reliable = await self._fetch_ruleset_required_checks(
            owner, repo, branch, default_branch
        )
        _add(ruleset_checks)

        # Fall back to branch protection if no ruleset checks found
        if not required_checks:
            (
                bp_checks,
                bp_reliable,
            ) = await self._fetch_branch_protection_required_checks(owner, repo, branch)
            if not bp_reliable:
                reliable = False
            _add(bp_checks)

        if reliable:
            self._required_checks_cache[cache_key] = list(required_checks)
        return required_checks

    async def get_branch_protection(
        self, owner: str, repo: str, branch: str
    ) -> dict[str, Any]:
        """
        Get branch protection rules for a branch.

        REST: GET /repos/{owner}/{repo}/branches/{branch}/protection

        Results (including the empty "no protection" result) are cached
        per ``owner/repo@branch`` for the session: the merge pipeline
        calls this once per PR via ``_check_merge_requirements``, but
        protection config is branch-level state that does not change
        mid-run.  Errors other than 404 are not cached so a transient
        failure can succeed on retry.
        """
        cache_key = f"{owner}/{repo}@{branch}"
        cached = self._branch_protection_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            protection_data = await self.get(
                f"/repos/{owner}/{repo}/branches/{branch}/protection"
            )
            # Branch protection data should always be a dict, not a list
            result = protection_data if isinstance(protection_data, dict) else {}
            self._branch_protection_cache[cache_key] = result
            return result
        except Exception as e:
            # Branch protection might not be enabled, return empty dict
            if "404" in str(e):
                self._branch_protection_cache[cache_key] = {}
                return {}
            raise

    async def get_authenticated_user_login(self) -> str | None:
        """Return the authenticated user's login, cached for the session.

        The login never changes for a given token, so the ``/user``
        round-trip is paid at most once per client instance.  Returns
        ``None`` when the lookup fails (callers should degrade
        gracefully); failures are not cached so a transient error can
        recover on the next call.
        """
        if self._authenticated_user_login is None:
            try:
                user_data = await self.get("/user")
            except Exception as e:
                self.log.debug("Could not resolve authenticated user: %s", e)
                return None
            if isinstance(user_data, dict):
                login = user_data.get("login")
                if isinstance(login, str) and login:
                    self._authenticated_user_login = login
        return self._authenticated_user_login

    async def check_user_can_bypass_protection(
        self, owner: str, repo: str, force_level: str = "code-owners"
    ) -> tuple[bool, str]:
        """
        Check if the authenticated user has permissions to bypass branch protection.

        Args:
            owner: Repository owner
            repo: Repository name
            force_level: The force level being used ("code-owners", "protection-rules", "all")

        Returns:
            Tuple of (can_bypass: bool, reason: str)
        """
        try:
            repo_data = await self.get(f"/repos/{owner}/{repo}")
            if not isinstance(repo_data, dict):
                return False, "Could not fetch repository information"

            permissions = repo_data.get("permissions", {})
            self.log.debug(
                f"Repository permissions for {owner}/{repo}: admin={permissions.get('admin')}, push={permissions.get('push')}, pull={permissions.get('pull')}"
            )

            # Check if user has admin permissions (which includes bypass)
            if permissions.get("admin"):
                self.log.debug(f"User has admin permissions for {owner}/{repo}")
                return True, "User has admin permissions"

            # Try to get more detailed permission info from user's repository membership
            try:
                # For organization repos, check if user has bypass permissions
                # This requires checking the user's role/permissions
                # Use cached login to avoid repeated /user calls
                if self._authenticated_user_login is None:
                    user_data = await self.get("/user")
                    if isinstance(user_data, dict):
                        self._authenticated_user_login = user_data.get("login")

                username = self._authenticated_user_login
                if username:
                    collab_data = await self.get(
                        f"/repos/{owner}/{repo}/collaborators/{username}/permission"
                    )
                    if isinstance(collab_data, dict):
                        permission_level = collab_data.get("permission")
                        # admin permission can bypass
                        if permission_level == "admin":
                            return True, "User has admin collaborator permissions"
            except Exception as e:
                # If we can't check detailed permissions, continue with basic check
                self.log.debug(
                    f"Could not check detailed collaborator permissions: {e}"
                )

            # If we have push permissions but not admin
            if permissions.get("push"):
                # All force levels require admin permissions to actually bypass branch protection
                # at the GitHub API level. Push permissions alone are not sufficient.
                self.log.debug(
                    f"User has push permissions for {owner}/{repo} but not admin (required to bypass branch protection at GitHub API level)"
                )
                return (
                    False,
                    "User has push permissions but not admin/bypass permissions (admin required to bypass branch protection)",
                )

            self.log.debug(
                f"User does not have sufficient permissions for {owner}/{repo}"
            )
            return False, "User does not have bypass permissions"

        except Exception as e:
            # If we can't determine permissions, return conservative result
            self.log.debug(f"Could not check bypass permissions: {e}")
            return False, f"Could not verify permissions: {str(e)}"

    async def update_branch(self, owner: str, repo: str, number: int) -> None:
        """
        Update a pull request branch (rebase).

        REST: PUT /repos/{owner}/{repo}/pulls/{pull_number}/update-branch

        Raises:
            PermissionError: If token lacks required permissions
        """
        try:
            await self.put(f"/repos/{owner}/{repo}/pulls/{number}/update-branch")
        except Exception as e:
            perm_error = self._parse_permission_error(e, "update_branch", owner, repo)
            if perm_error:
                raise perm_error from e
            raise

    async def get_token_scopes(self) -> set[str] | None:
        """Return the OAuth scopes granted to a classic personal access token.

        Classic PATs advertise their granted scopes in the
        ``X-OAuth-Scopes`` response header on every authenticated request.
        Fine-grained PATs and GitHub App installation tokens do **not** send
        this header — their permission model is per-resource and cannot be
        introspected this way.

        Returns:
            A ``set`` of scope strings for a classic PAT (possibly empty if
            the token was created with no scopes selected), or ``None`` when
            the token type does not expose scopes (fine-grained PAT / app
            token) or the lookup could not be performed.  Callers MUST treat
            ``None`` as "undeterminable", never as "no scopes granted".
        """
        if self._token_scopes_fetched:
            return self._token_scopes

        try:
            # Any authenticated REST endpoint echoes the header.
            # ``/rate_limit`` is the cheapest and is itself exempt from the
            # primary rate limit, so it never consumes quota.
            r = await self._request("GET", f"{self.api_url}/rate_limit")
        except Exception as e:
            # A transient probe failure must NOT be cached as
            # "undeterminable": doing so would let a one-off network error
            # suppress accurate scope diagnosis for the rest of the run
            # (a classic PAT that has ``workflow`` could still be reported
            # as missing it).  Leave the cache unset so a later call can
            # retry and produce an accurate result.
            self.log.debug("Could not determine token scopes: %s", e)
            return None

        raw = r.headers.get("X-OAuth-Scopes")
        if raw is None:
            # Header absent on a successful probe → fine-grained / app
            # token.  The scope set is genuinely undeterminable; cache it.
            self._token_scopes = None
        else:
            # Header present (possibly empty for a scope-less classic PAT).
            self._token_scopes = {s.strip() for s in raw.split(",") if s.strip()}
        self._token_scopes_fetched = True
        return self._token_scopes

    async def check_workflow_scope(self) -> bool | None:
        """Determine whether the token may update GitHub Actions workflows.

        Merging a PR that touches ``.github/workflows/**`` requires the
        classic ``workflow`` scope (or, for fine-grained PATs, the
        ``Workflows: Read and write`` permission).

        Returns:
            ``True``  — classic PAT that carries the ``workflow`` scope.
            ``False`` — classic PAT that is missing the ``workflow`` scope.
            ``None``  — the token type cannot be introspected (fine-grained
            PAT / app token).  The requirement cannot be verified up-front;
            callers should defer to merge-time error handling.
        """
        scopes = await self.get_token_scopes()
        if scopes is None:
            return None
        return "workflow" in scopes

    async def check_token_permissions(
        self, operations: list[str], owner: str = "", repo: str = ""
    ) -> dict[str, dict[str, Any]]:
        """Pre-flight check for token permissions.

        Tests whether the token has the necessary permissions for the specified
        operations without actually performing them. This allows failing fast
        with clear error messages before attempting bulk operations.

        Args:
            operations: List of operations to check (e.g., ['approve', 'merge', 'close'])
            owner: Repository owner (required for repository-specific checks)
            repo: Repository name (required for repository-specific checks)

        Returns:
            Dictionary mapping operation names to check results:
            {
                'operation_name': {
                    'has_permission': bool,
                    'error': str | None,
                    'guidance': dict | None
                }
            }

        Example:
            >>> results = await client.check_token_permissions(['approve', 'merge'], 'owner', 'repo')
            >>> if not results['approve']['has_permission']:
            ...     print(results['approve']['error'])
        """
        results: dict[str, dict[str, Any]] = {}

        for operation in operations:
            result: dict[str, Any] = {
                "has_permission": False,
                "error": None,
                "guidance": None,
            }

            try:
                # Perform a lightweight check for each operation
                if (
                    operation in ("approve", "merge", "close", "update_branch")
                    and owner
                    and repo
                ):
                    # Use the collaborator permission endpoint to verify
                    # the token has write access to this specific repo.
                    #
                    # The previous approach (GET /repos/{owner}/{repo} and
                    # inspecting permissions.push) is unreliable for
                    # fine-grained PATs: GitHub returns the *user's*
                    # org-level permissions regardless of token scope,
                    # producing false positives when the token is scoped
                    # to a different org.
                    #
                    # The collaborator endpoint correctly returns 403
                    # ("Resource not accessible by personal access token")
                    # when the token doesn't cover the target repo.

                    # Resolve authenticated username (cached after first call)
                    if self._authenticated_user_login is None:
                        user_data = await self.get("/user")
                        if isinstance(user_data, dict):
                            self._authenticated_user_login = user_data.get("login")

                    username = self._authenticated_user_login
                    if not username:
                        result["error"] = "Could not determine authenticated user"
                    else:
                        collab_data = await self.get(
                            f"/repos/{owner}/{repo}/collaborators/{username}/permission"
                        )
                        if isinstance(collab_data, dict):
                            perm_level = collab_data.get("permission", "none")
                            # write, maintain, or admin is required for approve/merge/close/update
                            if perm_level in ("write", "maintain", "admin"):
                                result["has_permission"] = True
                            else:
                                result["error"] = (
                                    f"Token has '{perm_level}' access to "
                                    f"{owner}/{repo} — write, maintain, or admin is required"
                                )
                                perms = OPERATION_PERMISSIONS.get(operation, {})
                                result["guidance"] = {
                                    "classic": perms.get("classic"),
                                    "fine_grained": perms.get("fine_grained"),
                                }
                        else:
                            result["error"] = (
                                "Could not determine collaborator permissions"
                            )

                elif operation == "branch_protection" and owner and repo:
                    # Verify Administration: Read permission by probing
                    # the branch protection endpoint.  A token with this
                    # permission receives either 200 (rules exist) or
                    # 404 "Branch not protected"; without it GitHub
                    # returns 403 "Resource not accessible".
                    #
                    # The repo metadata fetch is separated from the
                    # branch-protection probe so that a 404 from
                    # GET /repos/{owner}/{repo} (repo doesn't exist or
                    # token can't see it) is NOT silently treated as
                    # success.
                    default_branch = "main"
                    try:
                        repo_data = await self.get(f"/repos/{owner}/{repo}")
                        if isinstance(repo_data, dict):
                            default_branch = repo_data.get("default_branch", "main")
                    except Exception:
                        # Repo metadata fetch failed — token may lack
                        # access.  Let the error propagate to the outer
                        # handler which will surface it as a permission
                        # error.  Do NOT fall through to treat this as
                        # success.
                        raise

                    try:
                        await self.get(
                            f"/repos/{owner}/{repo}/branches/"
                            f"{default_branch}/protection"
                        )
                        result["has_permission"] = True
                    except Exception as e:
                        if "404" in str(e):
                            # 404 = branch exists but has no protection
                            # rules — the token still has the permission.
                            result["has_permission"] = True
                        else:
                            raise

                elif operation == "list_repos":
                    if owner:
                        await self.get(f"/orgs/{owner}/repos?per_page=1")
                    result["has_permission"] = True

                elif operation == "merge_workflow":
                    # Verify the token may merge PRs that modify GitHub
                    # Actions workflow files.  This is only checkable for
                    # classic PATs, which advertise their scopes via the
                    # ``X-OAuth-Scopes`` header.  Fine-grained PATs and app
                    # tokens do not expose scopes, so the check returns
                    # ``None`` and we pass it through here — the requirement
                    # cannot be verified up-front for those token types and
                    # is instead surfaced (with accurate guidance) by the
                    # merge-time handler if it actually bites.
                    has_workflow = await self.check_workflow_scope()
                    if has_workflow is False:
                        perms = OPERATION_PERMISSIONS.get("merge_workflow", {})
                        result["error"] = (
                            "Token is missing the 'workflow' scope, which is "
                            "required to merge pull requests that modify "
                            "GitHub Actions workflow files "
                            "(.github/workflows/**)"
                        )
                        result["guidance"] = {
                            "classic": perms.get("classic"),
                            "fine_grained": perms.get("fine_grained"),
                            "fix": "Run: gh auth refresh -h github.com -s workflow",
                        }
                    else:
                        # ``True`` (scope present) or ``None``
                        # (undeterminable token type) — do not block.
                        result["has_permission"] = True

                else:
                    result["error"] = f"Unknown operation: {operation}"

            except Exception as e:
                perm_error = self._parse_permission_error(e, operation, owner, repo)
                if perm_error:
                    result["has_permission"] = False
                    result["error"] = str(perm_error)
                    result["guidance"] = perm_error.token_type_guidance
                else:
                    # Unexpected error - be conservative
                    result["has_permission"] = False
                    result["error"] = f"Could not verify permissions: {str(e)}"

            results[operation] = result

        return results

    async def close_pull_request(
        self, owner: str, repo: str, number: int
    ) -> dict[str, Any]:
        """
        Close a pull request.

        Args:
            owner: Repository owner
            repo: Repository name
            number: Pull request number

        Returns:
            Updated pull request data

        Raises:
            PermissionError: If token lacks required permissions
        """
        try:
            return await self.patch(
                f"/repos/{owner}/{repo}/pulls/{number}", json={"state": "closed"}
            )
        except Exception as e:
            perm_error = self._parse_permission_error(e, "close", owner, repo)
            if perm_error:
                raise perm_error from e
            raise

    async def get_behind_by(
        self, owner: str, repo: str, base_ref: str, head_sha: str
    ) -> int | None:
        """Return how many commits ``head_sha`` is behind ``base_ref``.

        GitHub's ``mergeable_state`` is a single value, so ``blocked``
        (a failing required check) masks ``behind`` (a stale head).
        This helper answers the staleness question independently via
        the compare API, which works regardless of the reported
        mergeable state and regardless of whether the head lives on a
        fork (the SHA is resolvable in the base repository's network).

        Args:
            owner: Base repository owner
            repo: Base repository name
            base_ref: Base branch name (e.g. ``main``)
            head_sha: Head commit SHA of the pull request

        Returns:
            The ``behind_by`` commit count, or ``None`` when the
            comparison could not be performed (API error, unexpected
            payload).  ``None`` means "unknown": callers must not
            interpret it as ``behind_by == 0`` ("up to date"), and
            staleness-driven write actions (e.g. requesting a rebase)
            should require positive evidence (``behind_by > 0``)
            rather than acting on an unknown — the pattern used by
            ``AsyncMergeManager._blocked_pr_needs_rebase``.
        """
        encoded_base = quote(base_ref, safe="")
        try:
            comparison = await self.get(
                f"/repos/{owner}/{repo}/compare/{encoded_base}...{head_sha}"
            )
        except Exception as exc:
            self.log.debug(
                "Compare %s...%s failed for %s/%s: %s",
                base_ref,
                head_sha,
                owner,
                repo,
                exc,
            )
            return None
        if isinstance(comparison, dict):
            behind = comparison.get("behind_by")
            if isinstance(behind, int):
                return behind
        return None

    # How long an ``analyze_block_reason`` result stays usable.
    #
    # Deliberately short.  The obvious design --- cache per
    # ``(repo, head_sha)`` for the run --- is unsafe: the reason a PR is
    # blocked changes as checks complete, while its head SHA does not,
    # and callers re-analyse after waiting precisely to observe that
    # change.  A run-lifetime cache would answer "still blocked" forever.
    #
    # The waste worth removing is the *burst*: a single evaluation pass
    # calls this several times in quick succession with nothing changing
    # in between, at five or more requests each.  A few seconds collapses
    # that burst and has long expired by the time any wait loop
    # re-checks.
    _BLOCK_REASON_TTL_SECONDS = 10.0

    async def analyze_block_reason(
        self,
        owner: str,
        repo: str,
        number: int,
        head_sha: str,
        base_branch: str | None = None,
    ) -> str:
        """
        Analyze why a PR is blocked and return appropriate status.

        This is the async version that should be used from async contexts.

        ``base_branch`` lets callers that already know the PR's base ref
        (e.g. the merge pipeline, which carries it on ``PullRequestInfo``)
        skip the PR-detail fetch this method otherwise performs just to
        read ``base.ref`` — one request saved per invocation, and this
        method runs several times per blocked PR.

        Results are memoised briefly; see
        ``_BLOCK_REASON_TTL_SECONDS`` for why the window is short.  The
        base branch is part of the memo key because it selects which
        protection and required-check configuration is consulted: a
        retargeted PR, or two callers supplying different bases, must
        not share an answer computed against the other's branch.
        """
        cache_key = (owner, repo, number, head_sha, base_branch)
        cached = self._block_reason_cache.get(cache_key)
        if cached is not None:
            cached_at, cached_reason = cached
            if _now() - cached_at < self._BLOCK_REASON_TTL_SECONDS:
                return cached_reason

        reason = await self._analyze_block_reason_uncached(
            owner, repo, number, head_sha, base_branch
        )
        self._block_reason_cache[cache_key] = (_now(), reason)
        return reason

    async def _analyze_block_reason_uncached(
        self,
        owner: str,
        repo: str,
        number: int,
        head_sha: str,
        base_branch: str | None = None,
    ) -> str:
        """Compute the block reason, ignoring the memo."""
        # Reviews
        approved = False
        human_changes_requested = False
        unresolved_copilot_reviews = 0
        unresolved_copilot_comments = 0

        try:
            reviews = await self.get(f"/repos/{owner}/{repo}/pulls/{number}/reviews")
            if isinstance(reviews, list):
                for review in reviews:
                    if not isinstance(review, dict):
                        continue
                    state = review.get("state")
                    author = (review.get("user") or {}).get("login", "")

                    if state == "APPROVED":
                        approved = True
                    elif state == "CHANGES_REQUESTED":
                        if is_copilot(author):
                            unresolved_copilot_reviews += 1
                        else:
                            human_changes_requested = True
        except Exception:
            # Review data is best-effort; on API error leave the
            # approval/changes flags at their safe defaults.
            pass

        try:
            comments = await self.get(f"/repos/{owner}/{repo}/pulls/{number}/comments")
            if isinstance(comments, list):
                for comment in comments:
                    if not isinstance(comment, dict):
                        continue
                    author = (comment.get("user") or {}).get("login", "")
                    # Count unresolved Copilot comments (those without replies dismissing them)
                    if is_copilot(author):
                        # Simple heuristic: if comment doesn't have "DISMISSED" or similar resolution text
                        body = comment.get("body", "").lower()
                        if "dismissed" not in body and "resolved" not in body:
                            unresolved_copilot_comments += 1
        except Exception:
            # Review comments are best-effort; ignore fetch errors and
            # leave the Copilot comment count unchanged.
            pass

        # Check runs and status contexts - look for failing (check this first as it's most specific)
        failing_checks = []
        completed_check_names: set[str] = set()
        # Track all reported check names regardless of status so that
        # queued/in_progress checks are not misclassified as "missing".
        reported_check_names: set[str] = set()
        pending_check_names: set[str] = set()
        try:
            # Check runs (newer GitHub Apps API)
            runs = await self.get(
                f"/repos/{owner}/{repo}/commits/{head_sha}/check-runs"
            )
            if isinstance(runs, dict):
                raw_runs = [
                    run
                    for run in (runs.get("check_runs") or [])
                    if isinstance(run, dict)
                ]
                # Status classification deliberately considers *every*
                # reported run, not just the latest: a name carrying both
                # a completed run and a fresh in_progress re-run is still
                # pending, and must not be collapsed away here.
                for run in raw_runs:
                    name = (run.get("name") or "").strip()
                    if not name:
                        # An unnamed run cannot be matched against a
                        # required-check rule.  Recording it produces
                        # only misleading output such as "Blocked by
                        # failing check: unknown", so drop it here just
                        # as the deduplication helper does.
                        continue
                    status = run.get("status")
                    reported_check_names.add(name)
                    if status == "completed":
                        completed_check_names.add(name)
                    elif status in ("queued", "in_progress"):
                        pending_check_names.add(name)
                # Failure, by contrast, is decided by the latest run per
                # name.  A commit can carry several runs under one name
                # when a duplicate workflow event causes ``concurrency``
                # to cancel a superseded run; that cancelled run must not
                # mask the successful one that replaced it.
                failing_checks.extend(failing_check_names(raw_runs))
        except Exception:
            # Check-runs API may be unavailable; proceed with whatever
            # checks were collected so far.
            pass

        try:
            statuses = await self.get(
                f"/repos/{owner}/{repo}/commits/{head_sha}/status"
            )
            if isinstance(statuses, dict):
                for s in statuses.get("statuses") or []:
                    if not isinstance(s, dict):
                        continue
                    context = s.get("context", "unknown")
                    state = s.get("state")
                    reported_check_names.add(context)
                    if state in ["success", "neutral"]:
                        completed_check_names.add(context)
                    elif state == "pending":
                        pending_check_names.add(context)
                    if state in ["failure", "error"]:
                        # Avoid duplicates if both check-run and status exist for same service
                        if context not in failing_checks:
                            failing_checks.append(context)
        except Exception:
            # Status API may be unavailable; proceed with whatever
            # status contexts were collected so far.
            pass

        # Detect missing/pending required status checks (e.g. stale pre-commit.ci)
        missing_required_checks: list[str] = []
        pending_required_checks: list[str] = []
        # Resolve the PR's actual base branch.  It drives both the
        # required status-check lookup and the final guard-kind
        # classification, so a wrong value (e.g. assuming "main" on a repo
        # that defaults to "master") produces a misleading block reason.
        # Prefer the caller-supplied value, then the PR's own base ref; if
        # neither is available, fall back to the repository's real default
        # branch rather than a hardcoded name, and only give up (leaving
        # it ``None``) when nothing can be determined.
        if base_branch is None:
            try:
                pr_data = await self.get(f"/repos/{owner}/{repo}/pulls/{number}")
                if isinstance(pr_data, dict):
                    ref = (pr_data.get("base") or {}).get("ref")
                    if isinstance(ref, str) and ref:
                        base_branch = ref
            except Exception as pr_err:
                self.log.debug(
                    f"Could not read base branch for {owner}/{repo}#{number}: {pr_err}"
                )

        if base_branch is None:
            base_branch = await self._resolve_default_branch(owner, repo)

        # Only inspect required status checks when we know which branch to
        # query; an assumed branch would yield checks for the wrong ref.
        if base_branch is not None:
            try:
                required_checks = await self.get_required_status_checks(
                    owner, repo, base_branch
                )
                for check in required_checks:
                    ctx = check.get("context", "")
                    if not ctx:
                        continue
                    if ctx in reported_check_names:
                        if (
                            ctx not in completed_check_names
                            and ctx in pending_check_names
                        ):
                            pending_required_checks.append(ctx)
                    else:
                        # Never reported via either API — truly missing
                        missing_required_checks.append(ctx)
            except Exception as req_err:
                self.log.debug(
                    f"Could not check required status checks for "
                    f"{owner}/{repo}#{number}: {req_err}"
                )

        # Prioritize blocking conditions by specificity
        # Most specific blockers first
        if failing_checks:
            if len(failing_checks) == 1:
                return f"Blocked by failing check: {failing_checks[0]}"
            else:
                return f"Blocked by {len(failing_checks)} failing checks"

        if missing_required_checks:
            if len(missing_required_checks) == 1:
                return (
                    f"Blocked by missing required status: {missing_required_checks[0]}"
                )
            else:
                names = ", ".join(missing_required_checks)
                return f"Blocked by {len(missing_required_checks)} missing required statuses: {names}"

        if pending_required_checks:
            if len(pending_required_checks) == 1:
                return (
                    f"Blocked by pending required check: {pending_required_checks[0]}"
                )
            else:
                names = ", ".join(pending_required_checks)
                return f"Blocked by {len(pending_required_checks)} pending required checks: {names}"

        if human_changes_requested:
            return "Human reviewer requested changes"

        if unresolved_copilot_reviews > 0:
            if unresolved_copilot_comments > 0:
                return f"Blocked by {unresolved_copilot_reviews} Copilot reviews, {unresolved_copilot_comments} comments"
            else:
                return f"Blocked by {unresolved_copilot_reviews} unresolved Copilot reviews"

        if unresolved_copilot_comments > 0:
            return (
                f"Blocked by {unresolved_copilot_comments} unresolved Copilot comments"
            )

        # No *required* check is failing, missing, or pending, and no
        # human/Copilot review is blocking — but if any check on the head
        # commit is still queued or in progress, the PR is only *temporarily*
        # blocked. This matters for checks enforced through a repository
        # ruleset's "required workflows": those never appear in the classic
        # required-status-checks list, so the pending_required_checks branch
        # above cannot see them. Surface them as pending here, *before* the
        # "requires approval" fallback, so the merge pipeline waits for them
        # (and arms auto-merge) instead of failing the PR outright while its
        # workflows are still running.
        # Any name in ``pending_check_names`` has a queued/in-progress run
        # and is therefore still running. We must NOT subtract
        # ``completed_check_names``: GitHub can report two runs with the
        # same name (a re-run leaves one ``completed`` entry and a fresh
        # ``in_progress`` one), and the set difference would cancel the
        # name out and hide a check that is genuinely still running.
        #
        # Defensively filter to non-empty strings: a malformed API
        # payload can report ``name``/``context`` as ``null``, and mixing
        # ``None`` with strings would make ``sorted``/``join`` raise. This
        # branch is best-effort, so drop anything that is not a usable name.
        pending_only = sorted(
            name for name in pending_check_names if isinstance(name, str) and name
        )
        if pending_only:
            if len(pending_only) == 1:
                return f"Blocked by pending check: {pending_only[0]}"
            names = ", ".join(pending_only)
            return f"Blocked by {len(pending_only)} pending checks: {names}"

        if not approved:
            return "Blocked by branch protection (requires approval)"

        # No self-describing blocker was found: checks pass, the PR is
        # approved, and no changes are requested — yet GitHub still reports
        # the PR as blocked.  Rather than *asserting* "branch protection"
        # (which is invisible to this code path when the repository uses
        # rulesets), determine what kind of rule actually guards the branch
        # and keep the wording non-committal: we know the branch is guarded,
        # not that a specific condition is failing.
        if base_branch is None:
            # The base branch could not be resolved, so no branch-specific
            # inspection ran.  Say exactly that rather than implying we
            # looked for protection rules and found none.
            return (
                "Blocked for an undetermined reason "
                "(GitHub reports the PR as blocked, but the PR's base "
                "branch could not be determined, so its protection rules "
                "and required checks could not be inspected)"
            )
        kind = await self._detect_branch_protection_kind(owner, repo, base_branch)
        if kind == "ruleset":
            return (
                "Blocked by repository ruleset (no specific failing condition detected)"
            )
        if kind == "protection":
            return (
                "Blocked by branch protection (no specific failing condition detected)"
            )
        return (
            "Blocked for an undetermined reason "
            "(GitHub reports the PR as blocked but no failing checks, "
            "required reviews, or visible protection rules were found; "
            "the repository may use rulesets this token cannot read)"
        )

    async def _resolve_default_branch(self, owner: str, repo: str) -> str | None:
        """Return the repository's actual default branch, or ``None``.

        Many repositories default to ``master`` rather than ``main``, so
        callers must never assume a name.  This reads the authoritative
        ``default_branch`` field from the repository metadata and returns
        ``None`` when it cannot be determined (the repo is unreadable or
        the field is absent), letting callers degrade gracefully instead
        of operating on a wrong branch.

        Successful lookups are cached per ``owner/repo`` for the
        session (a repo's default branch does not change mid-run);
        failures are not cached so a transient error can recover.
        """
        cache_key = f"{owner}/{repo}"
        if cache_key in self._default_branch_cache:
            return self._default_branch_cache[cache_key]
        try:
            repo_data = await self.get(f"/repos/{owner}/{repo}")
        except Exception as e:
            self.log.debug(
                "Could not resolve default branch for %s/%s: %s", owner, repo, e
            )
            return None
        if isinstance(repo_data, dict):
            default_branch = repo_data.get("default_branch")
            if isinstance(default_branch, str) and default_branch:
                self._default_branch_cache[cache_key] = default_branch
                return default_branch
        return None

    async def _detect_branch_protection_kind(
        self, owner: str, repo: str, branch: str
    ) -> str:
        """Best-effort classification of what guards a branch.

        Used by :meth:`analyze_block_reason` to describe an otherwise
        unexplained ``BLOCKED`` state accurately instead of asserting
        "branch protection".

        Returns:
            ``"ruleset"``    — one or more repository rulesets apply to the
            branch (reported in preference to classic protection because
            rulesets are invisible to the GraphQL ``branchProtectionRule``
            field and are what most current repositories use).
            ``"protection"`` — a classic branch protection rule applies.
            ``"none"``       — neither could be found (the branch appears
            unguarded, or the token cannot read the configuration).
        """
        # Repository rulesets (newer API): the effective-rules endpoint
        # returns every rule that applies to the branch from any active
        # ruleset.  A non-empty list means a ruleset guards the branch.
        # Branch names can contain '/' (e.g. ``release/v1``), so they must
        # be URL-encoded before interpolation into the REST path.
        encoded_branch = quote(branch, safe="")
        try:
            rules = await self.get(
                f"/repos/{owner}/{repo}/rules/branches/{encoded_branch}"
            )
            if isinstance(rules, list) and rules:
                return "ruleset"
        except Exception as e:
            self.log.debug(
                "Could not read branch rules for %s/%s:%s: %s",
                owner,
                repo,
                branch,
                e,
            )

        # Classic branch protection: 200 = protected, 404 = no rule.
        try:
            await self.get(
                f"/repos/{owner}/{repo}/branches/{encoded_branch}/protection"
            )
            return "protection"
        except Exception as e:
            if "404" not in str(e):
                self.log.debug(
                    "Could not read branch protection for %s/%s:%s: %s",
                    owner,
                    repo,
                    branch,
                    e,
                )

        return "none"

    async def get_paginated(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        per_page: int = 100,
        max_pages: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Iterate through a paginated REST collection.

        Yields JSON arrays/items for each page. Caller can flatten as needed.
        """
        page = 1
        while True:
            q = dict(params or {})
            q.update({"per_page": per_page, "page": page})
            r = await self._request("GET", f"{self.api_url}{path}", params=q)
            data = r.json()
            if not data:
                return
            yield data
            page += 1
            if max_pages and page > max_pages:
                return
            # Stop when Link header doesn't include 'rel="next"'
            link = r.headers.get("Link", "")
            if 'rel="next"' not in link:
                return

    def _track_error(self, error_type: str) -> None:
        """Track an error for adaptive throttling calculations."""
        current_time = _now()
        self._error_history.append((current_time, error_type))

        cutoff = current_time - self._error_window
        self._error_history = [(t, e) for t, e in self._error_history if t > cutoff]

    def _track_request(self) -> None:
        """Record a completed request so the error *rate* has a denominator."""
        current_time = _now()
        self._request_history.append(current_time)
        cutoff = current_time - self._error_window
        if self._request_history[0] <= cutoff:
            self._request_history = [t for t in self._request_history if t > cutoff]

    def _get_recent_error_rate(self) -> float:
        """Errors as a fraction of all requests in the recent window.

        Previously this divided the error count by an *estimate* derived
        from the same error count (``errors / (errors * 10)``), which is
        the constant ``0.1`` whenever any error exists and ``0.0``
        otherwise.  Both call sites compared it against ``0.1`` and
        ``0.2``, so the error signal could never fire.  Counting requests
        as well gives a real ratio.
        """
        current_time = _now()
        cutoff = current_time - self._error_window
        errors = sum(1 for t, _ in self._error_history if t > cutoff)
        if not errors:
            return 0.0
        requests = sum(1 for t in self._request_history if t > cutoff)
        # Errors are not recorded in ``_request_history`` (they raise
        # before ``_track_request``), so the denominator is the total of
        # both.  Guard against a window holding only errors.
        total = requests + errors
        return errors / total if total else 0.0

    def _record_budget(self, r: httpx.Response) -> None:
        """Store the rate-limit state carried by a response, per resource."""
        remaining_hdr = r.headers.get("X-RateLimit-Remaining")
        limit_hdr = r.headers.get("X-RateLimit-Limit")
        if remaining_hdr is None or limit_hdr is None:
            # No rate-limit headers: nothing reliable to learn.  Notably we
            # do *not* fall back to defaults here --- the previous code
            # defaulted to remaining=1/limit=60, a headroom of 0.017, which
            # tripped the throttle on any response lacking headers.
            return
        try:
            remaining = int(remaining_hdr)
            limit = int(limit_hdr)
        except (TypeError, ValueError):
            return
        reset = r.headers.get("X-RateLimit-Reset")
        try:
            reset_epoch = float(reset) if reset else None
        except (TypeError, ValueError):
            reset_epoch = None
        resource = r.headers.get("X-RateLimit-Resource") or "core"
        self._budgets[resource] = _Budget(remaining, limit, reset_epoch, _now())

    def _headroom(self) -> float | None:
        """Smallest remaining fraction across all known budgets.

        The limiter and semaphore are shared across REST and GraphQL, so
        the binding constraint is whichever resource is most depleted.
        Returns ``None`` when nothing is known, meaning "do not tune".
        """
        if not self._budgets:
            return None
        now = _now()
        return min(b.headroom(now) for b in self._budgets.values())

    # Ramp-up requires this many consecutive healthy responses.  Recovery
    # is deliberately slower than back-off: one lucky response should not
    # undo a throttle, but a sustained healthy run must be able to.
    _RAMP_UP_STREAK = 20

    def _tune(self, headroom: float | None) -> None:
        """Adjust concurrency and RPS from budget headroom and error rate.

        Throttling down and ramping back up are the two branches of a
        single condition.  In the previous implementation the ramp-up
        branch sat in the ``else`` of ``if limit > 0:`` --- unreachable,
        because GitHub always reports a positive limit.  Back-off was
        therefore permanent for the process lifetime: a long run would
        decay to the floor of 2 concurrent / 1.0 rps and stay there.
        """
        if headroom is None:
            return
        error_rate = self._get_recent_error_rate()
        should_throttle = headroom < 0.1 or error_rate > 0.1

        if should_throttle:
            self._healthy_streak = 0
            factor = 0.3 if error_rate > 0.2 else 0.5
            new_concurrency = max(2, int(self._max_concurrency * factor))
            new_rps = max(1.0, self._current_rps * factor)
            changed = False
            if new_concurrency != self._max_concurrency:
                self._max_concurrency = new_concurrency
                self.semaphore.resize(new_concurrency)
                changed = True
            if abs(new_rps - self._current_rps) >= 0.5:
                self._current_rps = new_rps
                self.limiter = AsyncLimiter(max_rate=new_rps, time_period=1.0)
                changed = True
            if changed:
                self.log.warning(
                    "Throttling down: headroom=%.3f error_rate=%.3f "
                    "-> concurrency=%d rps=%.1f",
                    headroom,
                    error_rate,
                    self._max_concurrency,
                    self._current_rps,
                )
            return

        # Healthy.  Ramp back toward the configured base values.
        at_base = (
            self._max_concurrency >= self._base_max_concurrency
            and self._current_rps >= self._base_rps
        )
        if at_base:
            self._healthy_streak = 0
            return
        self._healthy_streak += 1
        if self._healthy_streak < self._RAMP_UP_STREAK:
            return
        self._healthy_streak = 0
        if self._max_concurrency < self._base_max_concurrency:
            self._max_concurrency = min(
                self._base_max_concurrency, self._max_concurrency + 1
            )
            self.semaphore.resize(self._max_concurrency)
        if self._current_rps < self._base_rps:
            self._current_rps = min(self._base_rps, self._current_rps + 1.0)
            self.limiter = AsyncLimiter(max_rate=self._current_rps, time_period=1.0)
        self.log.info(
            "Recovering: headroom=%.3f -> concurrency=%d rps=%.1f",
            headroom,
            self._max_concurrency,
            self._current_rps,
        )

    # Adaptive delay decays to zero over this many seconds after the last
    # Retry-After observation.
    _ADAPTIVE_DELAY_DECAY_SECONDS = 120.0

    def _current_adaptive_delay(self) -> float:
        """The pacing delay to apply right now, decayed by elapsed time.

        The decay used to live inside ``_apply_retry_after_throttling``,
        so it only ran when *another* ``Retry-After`` arrived.  A single
        long ``Retry-After`` therefore pinned a delay --- up to 5 s --- on
        every subsequent successful request for the rest of the run.  At
        roughly 10 calls per PR that is around 50 s of pure sleeping per
        PR.  Decaying on read makes the delay self-clearing.
        """
        if self._adaptive_delay <= 0 or self._last_adaptive_update is None:
            return 0.0
        elapsed = _now() - self._last_adaptive_update
        if elapsed >= self._ADAPTIVE_DELAY_DECAY_SECONDS:
            self._adaptive_delay = 0.0
            return 0.0
        remaining = 1.0 - (elapsed / self._ADAPTIVE_DELAY_DECAY_SECONDS)
        return self._adaptive_delay * remaining

    def _apply_retry_after_throttling(self, retry_after_seconds: float) -> None:
        """Set the pacing delay implied by a ``Retry-After`` header."""
        if retry_after_seconds > 30:
            # Long retry-after suggests we're hitting limits hard
            delay = min(5.0, retry_after_seconds * 0.1)
        elif retry_after_seconds > 10:
            # Medium retry-after suggests moderate pressure
            delay = min(2.0, retry_after_seconds * 0.05)
        else:
            # Short retry-after is normal, minimal delay
            delay = min(1.0, retry_after_seconds * 0.02)

        # Keep the strongest signal currently in force rather than letting
        # a mild one reset a severe one that has not yet decayed.
        self._adaptive_delay = max(delay, self._current_adaptive_delay())
        self._last_adaptive_update = _now()
