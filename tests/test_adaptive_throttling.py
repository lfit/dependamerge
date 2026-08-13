# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for adaptive rate-limit throttling in :mod:`github_async`.

These cover the defects found in the 503-PR bulk-run audit (see
``docs/BULK_RUN_PERFORMANCE_AUDIT.md``):

- the ramp-up branch was unreachable, making back-off permanent;
- the error rate was a constant by construction, so it never fired;
- the concurrency semaphore was replaced while tasks held permits;
- the adaptive delay only decayed when a new ``Retry-After`` arrived;
- REST and GraphQL budgets shared one counter;
- a missing rate-limit header defaulted to a headroom that throttled.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from dependamerge.github_async import GitHubAsync, _Budget, _ResizableSemaphore


def _response(headers: dict[str, str]) -> httpx.Response:
    return httpx.Response(
        200, headers=headers, request=httpx.Request("GET", "http://x")
    )


def _client(**kwargs) -> GitHubAsync:
    return GitHubAsync(token="t", **kwargs)


# --------------------------------------------------------------------------
# Budget accounting
# --------------------------------------------------------------------------


class TestBudget:
    def test_headroom_is_remaining_over_limit(self) -> None:
        b = _Budget(remaining=250, limit=5000, reset_epoch=None, updated_at=0.0)
        assert b.headroom(now=0.0) == pytest.approx(0.05)

    def test_headroom_is_full_after_reset(self) -> None:
        """A stale near-zero budget must not pin the client at the floor."""
        b = _Budget(remaining=0, limit=5000, reset_epoch=100.0, updated_at=0.0)
        assert b.headroom(now=99.0) == 0.0
        assert b.headroom(now=101.0) == 1.0

    def test_zero_limit_reports_full_headroom(self) -> None:
        b = _Budget(remaining=0, limit=0, reset_epoch=None, updated_at=0.0)
        assert b.headroom(now=0.0) == 1.0


class TestRecordBudget:
    def test_rest_and_graphql_tracked_separately(self) -> None:
        c = _client()
        c._record_budget(
            _response(
                {
                    "X-RateLimit-Remaining": "4900",
                    "X-RateLimit-Limit": "5000",
                    "X-RateLimit-Resource": "graphql",
                }
            )
        )
        c._record_budget(
            _response(
                {
                    "X-RateLimit-Remaining": "100",
                    "X-RateLimit-Limit": "5000",
                    "X-RateLimit-Resource": "core",
                }
            )
        )
        assert set(c._budgets) == {"graphql", "core"}
        # The binding constraint is the most depleted resource, so a
        # healthy GraphQL allowance must not mask an exhausted REST one.
        assert c._headroom() == pytest.approx(0.02)

    def test_missing_headers_are_ignored(self) -> None:
        """Absent headers previously defaulted to 1/60, tripping the throttle."""
        c = _client()
        c._record_budget(_response({}))
        assert c._budgets == {}
        assert c._headroom() is None

    def test_unparsable_headers_are_ignored(self) -> None:
        c = _client()
        c._record_budget(
            _response({"X-RateLimit-Remaining": "abc", "X-RateLimit-Limit": "5000"})
        )
        assert c._budgets == {}

    def test_resource_defaults_to_core(self) -> None:
        c = _client()
        c._record_budget(
            _response({"X-RateLimit-Remaining": "10", "X-RateLimit-Limit": "5000"})
        )
        assert "core" in c._budgets


# --------------------------------------------------------------------------
# Error rate
# --------------------------------------------------------------------------


class TestErrorRate:
    def test_no_errors_is_zero(self) -> None:
        c = _client()
        for _ in range(10):
            c._track_request()
        assert c._get_recent_error_rate() == 0.0

    def test_rate_is_a_real_ratio(self) -> None:
        """Regression: this used to be exactly 0.1 whenever any error existed."""
        c = _client()
        for _ in range(99):
            c._track_request()
        c._track_error("transient_error")
        assert c._get_recent_error_rate() == pytest.approx(0.01)

    def test_rate_rises_with_errors(self) -> None:
        c = _client()
        for _ in range(10):
            c._track_request()
        for _ in range(10):
            c._track_error("transient_error")
        assert c._get_recent_error_rate() == pytest.approx(0.5)

    def test_high_error_rate_is_reachable(self) -> None:
        """The old implementation could never exceed the 0.1 / 0.2 thresholds."""
        c = _client()
        c._track_request()
        for _ in range(9):
            c._track_error("transient_error")
        rate = c._get_recent_error_rate()
        assert rate > 0.2


# --------------------------------------------------------------------------
# Tuning: throttle down and, crucially, back up again
# --------------------------------------------------------------------------


class TestTune:
    def test_low_headroom_throttles_down(self) -> None:
        c = _client(max_concurrency=20, requests_per_second=8.0)
        c._tune(headroom=0.05)
        assert c._max_concurrency == 10
        assert c._current_rps == pytest.approx(4.0)

    def test_throttling_is_bounded_by_floors(self) -> None:
        c = _client(max_concurrency=20, requests_per_second=8.0)
        for _ in range(12):
            c._tune(headroom=0.01)
        assert c._max_concurrency == 2
        assert c._current_rps == pytest.approx(1.0)

    def test_healthy_headroom_ramps_back_up(self) -> None:
        """Regression: the ramp-up branch was unreachable, so this stayed at 2."""
        c = _client(max_concurrency=20, requests_per_second=8.0)
        for _ in range(12):
            c._tune(headroom=0.01)
        assert c._max_concurrency == 2

        for _ in range(GitHubAsync._RAMP_UP_STREAK * 3):
            c._tune(headroom=0.9)
        assert c._max_concurrency > 2
        assert c._current_rps > 1.0

    def test_ramp_up_needs_a_sustained_streak(self) -> None:
        c = _client(max_concurrency=20, requests_per_second=8.0)
        c._tune(headroom=0.01)
        throttled = c._max_concurrency
        for _ in range(GitHubAsync._RAMP_UP_STREAK - 1):
            c._tune(headroom=0.9)
        assert c._max_concurrency == throttled
        c._tune(headroom=0.9)
        assert c._max_concurrency == throttled + 1

    def test_throttling_resets_the_healthy_streak(self) -> None:
        c = _client(max_concurrency=20, requests_per_second=8.0)
        c._tune(headroom=0.01)
        for _ in range(GitHubAsync._RAMP_UP_STREAK - 1):
            c._tune(headroom=0.9)
        c._tune(headroom=0.01)
        assert c._healthy_streak == 0

    def test_ramp_up_stops_at_configured_base(self) -> None:
        c = _client(max_concurrency=6, requests_per_second=3.0)
        for _ in range(GitHubAsync._RAMP_UP_STREAK * 20):
            c._tune(headroom=0.9)
        assert c._max_concurrency == 6
        assert c._current_rps == pytest.approx(3.0)

    def test_unknown_headroom_does_not_tune(self) -> None:
        c = _client(max_concurrency=20, requests_per_second=8.0)
        c._tune(headroom=None)
        assert c._max_concurrency == 20
        assert c._current_rps == pytest.approx(8.0)

    def test_high_error_rate_throttles_even_with_budget(self) -> None:
        c = _client(max_concurrency=20, requests_per_second=8.0)
        c._track_request()
        for _ in range(9):
            c._track_error("transient_error")
        c._tune(headroom=1.0)
        assert c._max_concurrency < 20


# --------------------------------------------------------------------------
# Adaptive delay decay
# --------------------------------------------------------------------------


class TestAdaptiveDelay:
    def test_delay_is_zero_when_unset(self) -> None:
        assert _client()._current_adaptive_delay() == 0.0

    def test_delay_decays_to_zero_over_time(self, monkeypatch) -> None:
        """Regression: decay only ran when another Retry-After arrived."""
        import dependamerge.github_async as mod

        now = 1000.0
        monkeypatch.setattr(mod, "_now", lambda: now)
        c = _client()
        c._apply_retry_after_throttling(60.0)
        assert c._current_adaptive_delay() == pytest.approx(5.0)

        now = 1000.0 + GitHubAsync._ADAPTIVE_DELAY_DECAY_SECONDS / 2
        assert c._current_adaptive_delay() == pytest.approx(2.5)

        now = 1000.0 + GitHubAsync._ADAPTIVE_DELAY_DECAY_SECONDS + 1
        assert c._current_adaptive_delay() == 0.0

    def test_mild_signal_does_not_reset_a_severe_one(self, monkeypatch) -> None:
        import dependamerge.github_async as mod

        now = 1000.0
        monkeypatch.setattr(mod, "_now", lambda: now)
        c = _client()
        c._apply_retry_after_throttling(60.0)
        c._apply_retry_after_throttling(1.0)
        assert c._current_adaptive_delay() == pytest.approx(5.0)


# --------------------------------------------------------------------------
# Resizable semaphore
# --------------------------------------------------------------------------


class TestResizableSemaphore:
    @pytest.mark.asyncio
    async def test_acts_as_a_semaphore(self) -> None:
        sem = _ResizableSemaphore(2, 2)
        async with sem:
            async with sem:
                assert sem._sem.locked()

    @pytest.mark.asyncio
    async def test_shrink_reduces_effective_capacity(self) -> None:
        sem = _ResizableSemaphore(4, 4)
        sem.resize(1)
        await sem._settle()
        assert sem.capacity == 1
        async with sem:
            assert sem._sem.locked()

    @pytest.mark.asyncio
    async def test_grow_restores_capacity(self) -> None:
        sem = _ResizableSemaphore(4, 4)
        sem.resize(1)
        await sem._settle()
        sem.resize(4)
        await sem._settle()
        assert sem.capacity == 4
        held = [sem.__aenter__() for _ in range(4)]
        await asyncio.gather(*held)
        assert sem._sem.locked()
        for _ in range(4):
            await sem.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_permit_held_across_resize_releases_cleanly(self) -> None:
        """Regression: swapping the object leaked permits into a dead semaphore.

        A task holding a permit from before the resize must still be
        releasing into the same object afterwards, so total capacity stays
        consistent rather than being transiently exceeded.
        """
        sem = _ResizableSemaphore(4, 4)
        await sem.__aenter__()
        sem.resize(2)
        await sem._settle()
        await sem.__aexit__(None, None, None)
        # 4 max, 2 ballast held -> exactly 2 acquirable, no more.
        await sem.__aenter__()
        await sem.__aenter__()
        assert sem._sem.locked()
        await sem.__aexit__(None, None, None)
        await sem.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_resize_never_exceeds_maximum(self) -> None:
        sem = _ResizableSemaphore(4, 4)
        sem.resize(99)
        await sem._settle()
        assert sem.capacity == 4

    @pytest.mark.asyncio
    async def test_resize_floor_is_one(self) -> None:
        sem = _ResizableSemaphore(4, 4)
        sem.resize(0)
        await sem._settle()
        assert sem.capacity == 1

    @pytest.mark.asyncio
    async def test_resize_does_not_block_caller(self) -> None:
        """Shrinking while fully subscribed must not deadlock the response path."""
        sem = _ResizableSemaphore(2, 2)
        await sem.__aenter__()
        await sem.__aenter__()
        sem.resize(1)  # cannot take ballast yet; must return immediately
        assert sem.capacity == 1
        await sem.__aexit__(None, None, None)
        await sem.__aexit__(None, None, None)
        await sem.aclose()

    @pytest.mark.asyncio
    async def test_aclose_cancels_pending_resize(self) -> None:
        sem = _ResizableSemaphore(2, 2)
        await sem.__aenter__()
        await sem.__aenter__()
        sem.resize(1)
        await sem.aclose()
        await sem.__aexit__(None, None, None)
        await sem.__aexit__(None, None, None)


# --------------------------------------------------------------------------
# End-to-end: the ratchet no longer latches
# --------------------------------------------------------------------------


class TestNoPermanentRatchet:
    def test_recovers_after_budget_resets(self, monkeypatch) -> None:
        """The 503-PR run's failure mode, in miniature.

        Budget drains below 10%, the client throttles to the floor, the
        hourly window resets, and throughput must return. Previously it
        stayed at 2 concurrent / 1.0 rps for the life of the process.
        """
        import dependamerge.github_async as mod

        now = 1000.0
        monkeypatch.setattr(mod, "_now", lambda: now)
        c = _client(max_concurrency=20, requests_per_second=8.0)

        for _ in range(12):
            c._record_budget(
                _response(
                    {
                        "X-RateLimit-Remaining": "100",
                        "X-RateLimit-Limit": "5000",
                        "X-RateLimit-Reset": "2000",
                        "X-RateLimit-Resource": "core",
                    }
                )
            )
            c._tune(c._headroom())
        assert c._max_concurrency == 2
        assert c._current_rps == pytest.approx(1.0)

        now = 2001.0  # window reset
        for _ in range(GitHubAsync._RAMP_UP_STREAK * 25):
            c._tune(c._headroom())
        assert c._max_concurrency == 20
        assert c._current_rps == pytest.approx(8.0)
