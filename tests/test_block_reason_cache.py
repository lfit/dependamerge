# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the short-lived block-reason memo.

``analyze_block_reason`` costs five or more requests and runs several
times per blocked PR, from seven call sites in the merge manager.

The memo is deliberately short-lived. Caching per ``(repo, head_sha)``
for the lifetime of a run --- the literal recommendation in
``docs/BULK_RUN_PERFORMANCE_AUDIT.md`` --- would be unsafe: the reason a
PR is blocked changes as its checks complete, while its head SHA does
not, and callers re-analyse after waiting precisely to observe that
change. These tests pin both halves: the burst is collapsed, and a
result never outlives the window.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dependamerge.github_async import GitHubAsync


def _client() -> GitHubAsync:
    return GitHubAsync(token="t")


class TestBlockReasonMemo:
    @pytest.mark.asyncio
    async def test_repeated_calls_within_the_window_hit_the_memo(self) -> None:
        c = _client()
        c._analyze_block_reason_uncached = AsyncMock(return_value="Blocked by X")  # type: ignore[method-assign]

        results = [await c.analyze_block_reason("o", "r", 1, "sha") for _ in range(5)]

        assert results == ["Blocked by X"] * 5
        assert c._analyze_block_reason_uncached.await_count == 1
        await c.aclose()

    @pytest.mark.asyncio
    async def test_result_expires(self, monkeypatch) -> None:
        """A stale reason must not outlive a wait.

        Callers re-analyse after waiting for checks; serving the old
        answer would report a PR as still blocked forever.
        """
        import dependamerge.github_async as mod

        now = 1000.0
        monkeypatch.setattr(mod, "_now", lambda: now)
        c = _client()
        c._analyze_block_reason_uncached = AsyncMock(  # type: ignore[method-assign]
            side_effect=["Blocked by X", "Blocked by Y"]
        )

        assert await c.analyze_block_reason("o", "r", 1, "sha") == "Blocked by X"

        now = 1000.0 + GitHubAsync._BLOCK_REASON_TTL_SECONDS + 1
        assert await c.analyze_block_reason("o", "r", 1, "sha") == "Blocked by Y"
        assert c._analyze_block_reason_uncached.await_count == 2
        await c.aclose()

    @pytest.mark.asyncio
    async def test_ttl_is_shorter_than_any_wait_interval(self) -> None:
        """The memo must expire well inside a poll cycle.

        The shortest wait loop re-checks every few seconds; the memo has
        to be gone by then or the re-check learns nothing.
        """
        from dependamerge.merge_manager import DEFAULT_MERGE_RECHECK_INTERVAL

        assert GitHubAsync._BLOCK_REASON_TTL_SECONDS <= DEFAULT_MERGE_RECHECK_INTERVAL

    @pytest.mark.asyncio
    async def test_distinct_prs_do_not_share_a_result(self) -> None:
        c = _client()
        c._analyze_block_reason_uncached = AsyncMock(  # type: ignore[method-assign]
            side_effect=["reason 1", "reason 2"]
        )

        assert await c.analyze_block_reason("o", "r", 1, "sha") == "reason 1"
        assert await c.analyze_block_reason("o", "r", 2, "sha") == "reason 2"
        await c.aclose()

    @pytest.mark.asyncio
    async def test_a_new_head_sha_is_a_new_result(self) -> None:
        """A rebase or force-push invalidates whatever was known."""
        c = _client()
        c._analyze_block_reason_uncached = AsyncMock(  # type: ignore[method-assign]
            side_effect=["before rebase", "after rebase"]
        )

        assert await c.analyze_block_reason("o", "r", 1, "old") == "before rebase"
        assert await c.analyze_block_reason("o", "r", 1, "new") == "after rebase"
        await c.aclose()

    @pytest.mark.asyncio
    async def test_repositories_do_not_collide(self) -> None:
        c = _client()
        c._analyze_block_reason_uncached = AsyncMock(  # type: ignore[method-assign]
            side_effect=["repo a", "repo b"]
        )

        assert await c.analyze_block_reason("o", "a", 1, "sha") == "repo a"
        assert await c.analyze_block_reason("o", "b", 1, "sha") == "repo b"
        await c.aclose()
