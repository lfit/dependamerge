# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
What to do with a response before its body is used.

Rate limits, transient statuses and the pacing that follows a success.
Kept apart from :mod:`_transport`, which is the request surface: this
module decides whether a response should be waited on, retried, or
allowed through, and feeds the adaptive throttler either way.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx

# ``_now`` stays an attribute of the package rather than a name bound
# here, so callers can substitute it there.
from ._errors import (
    _TENACITY_MAX_BACKOFF,
    RetryableError,
    SecondaryRateLimitError,
    _is_primary_rate_limited,
    _is_secondary_rate_limited,
)
from ._throttling import _maybe_await

if TYPE_CHECKING:
    from ._transport import _TransportMixin


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
