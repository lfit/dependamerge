# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Concurrency, rate and budget control for the async GitHub client.

Holds the resizable concurrency gate, the per-resource rate-limit
budget record, the callback dispatch helper, and the adaptive tuning
that reacts to observed error rates and ``Retry-After`` hints.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import (
    cast,
)

import httpx
from aiolimiter import AsyncLimiter

# ``_now`` stays an attribute of the package rather than a name bound
# here: it was a module-level attribute of ``dependamerge.github_async``
# before the split, and callers substitute it there.
import dependamerge.github_async as _pkg

from ._base import _GitHubAsyncBase


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


async def _maybe_await(
    cb: Callable[..., None | Awaitable[None]] | None, *args, **kwargs
) -> None:
    if cb is None:
        return None
    result = cb(*args, **kwargs)
    if not asyncio.iscoroutine(result):
        return None
    return await cast("Awaitable[None]", result)


class _ThrottleMixin(_GitHubAsyncBase):
    """Adaptive concurrency and rate tuning mixed into ``GitHubAsync``."""

    def _track_error(self, error_type: str) -> None:
        """Track an error for adaptive throttling calculations."""
        current_time = _pkg._now()
        self._error_history.append((current_time, error_type))

        cutoff = current_time - self._error_window
        self._error_history = [(t, e) for t, e in self._error_history if t > cutoff]

    def _track_request(self) -> None:
        """Record a completed request so the error *rate* has a denominator."""
        current_time = _pkg._now()
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
        current_time = _pkg._now()
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
        self._budgets[resource] = _Budget(remaining, limit, reset_epoch, _pkg._now())

    def _headroom(self) -> float | None:
        """Smallest remaining fraction across all known budgets.

        The limiter and semaphore are shared across REST and GraphQL, so
        the binding constraint is whichever resource is most depleted.
        Returns ``None`` when nothing is known, meaning "do not tune".
        """
        if not self._budgets:
            return None
        now = _pkg._now()
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
        elapsed = _pkg._now() - self._last_adaptive_update
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
        self._last_adaptive_update = _pkg._now()
