# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for sharing one GitHub client across a merge run.

Rate limiting, concurrency and adaptive throttling all live on the
client instance, so a second one doubles the effective ceiling against a
budget GitHub shares between them and leaves each half blind to the
pressure the other causes.

Sharing has two consequences worth pinning: the service must not close a
client it does not own, and it must still receive the rate-limit
callbacks it relies on --- without them ``_rate_limited`` never sets and
the GraphQL paging silently stops shrinking under pressure.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dependamerge.github_async import GitHubAsync
from dependamerge.github_service import GitHubService, _chain_callbacks


async def _fire(callback: Any, *args: Any) -> None:
    """Invoke a callback that may be sync or async.

    The client types these as ``None | Awaitable[None]``, so awaiting the
    return value unconditionally is not type-safe.
    """
    assert callback is not None
    result = callback(*args)
    if inspect.isawaitable(result):
        await result


class TestClientOwnership:
    def test_service_creates_its_own_client_by_default(self) -> None:
        svc = GitHubService(token="t")
        assert svc._owns_api is True

    def test_supplied_client_is_used_and_not_owned(self) -> None:
        client = GitHubAsync(token="t")
        svc = GitHubService(token="t", client=client)

        assert svc._api is client
        assert svc._owns_api is False

    @pytest.mark.asyncio
    async def test_close_leaves_a_shared_client_open(self) -> None:
        """Closing a borrowed client would break the owner mid-run."""
        client = GitHubAsync(token="t")
        client.aclose = AsyncMock()  # type: ignore[method-assign]
        svc = GitHubService(token="t", client=client)

        await svc.close()

        client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_closes_an_owned_client(self) -> None:
        svc = GitHubService(token="t")
        svc._api.aclose = AsyncMock()  # type: ignore[method-assign]

        await svc.close()

        svc._api.aclose.assert_awaited_once()


class TestCallbackChaining:
    """The service's rate-limit awareness must survive sharing."""

    def test_callbacks_are_attached_to_a_shared_client(self) -> None:
        client = GitHubAsync(token="t")
        assert client.on_rate_limited is None

        GitHubService(token="t", client=client)

        assert client.on_rate_limited is not None
        assert client.on_rate_limit_cleared is not None

    @pytest.mark.asyncio
    async def test_shared_client_still_sets_the_rate_limited_flag(self) -> None:
        """Without this the paging code never reduces its page sizes."""
        client = GitHubAsync(token="t")
        svc = GitHubService(token="t", client=client)
        assert svc._rate_limited is False

        await _fire(client.on_rate_limited, 0.0)

        assert svc._rate_limited is True

        await _fire(client.on_rate_limit_cleared)
        assert svc._rate_limited is False

    @pytest.mark.asyncio
    async def test_an_existing_callback_is_preserved(self) -> None:
        seen: list[str] = []

        async def owner_callback(_reset: float) -> None:
            seen.append("owner")

        client = GitHubAsync(token="t", on_rate_limited=owner_callback)
        svc = GitHubService(token="t", client=client)

        await _fire(client.on_rate_limited, 0.0)

        assert seen == ["owner"]
        assert svc._rate_limited is True

    @pytest.mark.asyncio
    async def test_one_failing_callback_does_not_suppress_the_other(self) -> None:
        """These are observability hooks; a raise must not lose the flag."""

        async def broken(_reset: float) -> None:
            raise RuntimeError("tracker exploded")

        client = GitHubAsync(token="t", on_rate_limited=broken)
        svc = GitHubService(token="t", client=client)

        await _fire(client.on_rate_limited, 0.0)

        assert svc._rate_limited is True


class TestChainHelper:
    @pytest.mark.asyncio
    async def test_absent_sides_return_the_other(self) -> None:
        async def cb(_x: float) -> None:
            return None

        assert _chain_callbacks(None, cb) is cb
        assert _chain_callbacks(cb, None) is cb

    @pytest.mark.asyncio
    async def test_synchronous_callbacks_are_supported(self) -> None:
        seen: list[str] = []
        combined = _chain_callbacks(
            lambda _x: seen.append("a"), lambda _x: seen.append("b")
        )

        await combined(0.0)

        assert seen == ["a", "b"]


class TestCallbackDetach:
    """A closed service must not keep receiving events.

    Leaving callbacks attached retains the closed service and, worse,
    attaching a replacement to the same client stacks a second copy so
    every rate-limit update fires twice.
    """

    @pytest.mark.asyncio
    async def test_close_restores_the_clients_callbacks(self) -> None:
        client = GitHubAsync(token="t")
        svc = GitHubService(token="t", client=client)
        assert client.on_rate_limited is not None

        await svc.close()

        assert client.on_rate_limited is None
        assert client.on_rate_limit_cleared is None
        assert client.on_metrics is None
        await client.aclose()

    @pytest.mark.asyncio
    async def test_close_preserves_a_pre_existing_callback(self) -> None:
        async def owner_callback(_reset: float) -> None:
            return None

        client = GitHubAsync(token="t", on_rate_limited=owner_callback)
        svc = GitHubService(token="t", client=client)

        await svc.close()

        assert client.on_rate_limited is owner_callback
        await client.aclose()

    @pytest.mark.asyncio
    async def test_a_closed_service_no_longer_receives_events(self) -> None:
        client = GitHubAsync(token="t")
        first = GitHubService(token="t", client=client)
        await first.close()

        second = GitHubService(token="t", client=client)
        await _fire(client.on_rate_limited, 0.0)

        assert second._rate_limited is True
        assert first._rate_limited is False
        await client.aclose()
