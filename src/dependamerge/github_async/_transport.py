# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
HTTP transport for the async GitHub client.

The single retrying request path plus the REST verb helpers, the
GraphQL call and the paginated iterator built on top of it.  Rate-limit
header parsing and the sleep-until-reset wait live here too, because
they are only meaningful to a request in flight.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import (
    Any,
)

import httpx
from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# ``_now`` stays an attribute of the package rather than a name bound
# here: it was a module-level attribute of ``dependamerge.github_async``
# before the split, and callers substitute it there.
import dependamerge.github_async as _pkg

from ._base import _GitHubAsyncBase
from ._errors import (
    _TENACITY_MAX_BACKOFF,
    GraphQLError,
    RetryableError,
    SecondaryRateLimitError,
    _is_primary_rate_limited,
    _is_retryable_status,
    _is_secondary_rate_limited,
    _is_transient_graphql_error,
)
from ._throttling import _maybe_await


async def _handle_secondary_rate_limit(api: _TransportMixin, r: httpx.Response) -> None:
    """Wait out an abuse-detection 403, then signal a retry.

    Always raises :class:`SecondaryRateLimitError`.
    """
    retry_after = r.headers.get("Retry-After")
    delay: float | None = None
    if retry_after:
        try:
            delay = float(retry_after)
            api._last_retry_after = delay
            api._apply_retry_after_throttling(delay)
        except (TypeError, ValueError):
            delay = None
    # Track error for adaptive throttling
    api._track_error("secondary_rate_limit")
    if delay is not None:
        # GitHub told us exactly how long to wait.  Sleeping
        # here *and* letting tenacity back off on top would
        # stack the two; hand tenacity a pre-slept signal
        # instead by waiting the advised time and then
        # raising, which tenacity adds its own (smaller,
        # jittered) delay to.  Keep that combined wait honest
        # by subtracting tenacity's cap from our sleep.
        effective = max(0.0, delay - _TENACITY_MAX_BACKOFF)
        api.log.warning(
            "Secondary rate limit hit. Retry-After=%ss, sleeping %ss",
            delay,
            effective,
        )
        if effective:
            await asyncio.sleep(effective)
    else:
        api.log.warning(
            "Secondary rate limit hit without Retry-After; deferring to retry backoff"
        )
    raise SecondaryRateLimitError("Secondary rate limit encountered")


async def _handle_primary_rate_limit(
    api: _TransportMixin, r: httpx.Response, reset_epoch: float | None
) -> None:
    """Wait out an exhausted primary rate limit, then signal a retry.

    Always raises :class:`RetryableError`.
    """
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
        api._last_retry_after = retry_after_delay
        api.log.warning(
            "Primary rate limit with Retry-After: %ss",
            retry_after_delay,
        )
        await asyncio.sleep(max(0.0, retry_after_delay))
        api._apply_retry_after_throttling(retry_after_delay)
    elif reset_epoch:
        api.log.warning(
            "Primary rate limit exhausted. Waiting until reset: %s",
            reset_epoch,
        )
        await api._sleep_until(reset_epoch)
    else:
        # If no reset header, backoff and retry
        api.log.warning(
            "Primary rate limit suspected without reset header; backing off"
        )
        await asyncio.sleep(5.0)

    # Track error for adaptive throttling
    api._track_error("primary_rate_limit")
    raise RetryableError("Primary rate limit reset waited; retrying")


async def _handle_forbidden(api: _TransportMixin, r: httpx.Response) -> None:
    """Classify a 403 as a secondary or primary rate limit, if either.

    Returns normally when the 403 is neither, leaving the caller to
    apply its ordinary error handling.
    """
    body_text: str
    try:
        body_text = r.text or ""
    except Exception:
        body_text = ""

    remaining, _, reset_epoch = api._parse_rate_limit_headers(r)

    # Secondary rate limit (abuse detection)
    if _is_secondary_rate_limited(body_text):
        await _handle_secondary_rate_limit(api, r)

    # Primary rate limit exhausted
    if remaining == 0 or _is_primary_rate_limited(body_text):
        await _handle_primary_rate_limit(api, r, reset_epoch)


async def _handle_retryable_status(api: _TransportMixin, r: httpx.Response) -> None:
    """Honour any Retry-After, then signal a retry.

    Always raises :class:`RetryableError`.
    """
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
            api._last_retry_after = retry_after_delay
            api.log.debug(
                "HTTP %s with Retry-After: %ss",
                r.status_code,
                retry_after_delay,
            )
            await asyncio.sleep(max(0.0, retry_after_delay))
            api._apply_retry_after_throttling(retry_after_delay)

    api._track_error("transient_error")
    api.log.debug("Retryable HTTP status %s received", r.status_code)
    raise RetryableError(f"Transient HTTP status: {r.status_code}")


async def _finish_successful_request(api: _TransportMixin, r: httpx.Response) -> None:
    """Post-success pacing, adaptive tuning and metrics reporting."""
    api._track_request()

    # Pace the next request when recent Retry-After headers indicated
    # sustained pressure.  Decays with time (see ``_current_adaptive_delay``).
    delay = api._current_adaptive_delay()
    if delay > 0:
        await asyncio.sleep(delay)

    # Dynamic concurrency and RPS tuning from the latest headers.
    try:
        api._record_budget(r)
        api._tune(api._headroom())
    except Exception as e:
        # Tuning is best-effort; never fail the request on tuning errors.
        api.log.debug("Adaptive concurrency tuning skipped: %s", e)
    # Push current metrics to progress tracker (if provided)
    try:
        await _maybe_await(
            getattr(api, "on_metrics", None),
            api._max_concurrency,
            float(api._current_rps),
        )
    except Exception as e:
        # Metrics reporting is best-effort.
        api.log.debug("Progress metrics reporting failed: %s", e)


class _TransportMixin(_GitHubAsyncBase):
    """REST/GraphQL request plumbing mixed into ``GitHubAsync``."""

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
        now = _pkg._now()
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
            await _handle_forbidden(self, r)

        # Retryable transient statuses
        if _is_retryable_status(r.status_code):
            await _handle_retryable_status(self, r)

        # All other errors -> raise
        r.raise_for_status()

        await _finish_successful_request(self, r)
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
