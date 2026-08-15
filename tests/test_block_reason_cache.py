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

    @pytest.mark.asyncio
    async def test_base_branches_do_not_collide(self) -> None:
        """The base branch selects which protection rules are read.

        A PR retargeted inside the window, or two callers supplying
        different bases, must not share an answer computed against the
        other's branch.
        """
        c = _client()
        c._analyze_block_reason_uncached = AsyncMock(  # type: ignore[method-assign]
            side_effect=["main rules", "release rules"]
        )

        first = await c.analyze_block_reason("o", "r", 1, "sha", base_branch="main")
        second = await c.analyze_block_reason(
            "o", "r", 1, "sha", base_branch="release-1.0"
        )

        assert first == "main rules"
        assert second == "release rules"
        await c.aclose()

    @pytest.mark.asyncio
    async def test_the_same_base_still_hits_the_memo(self) -> None:
        """Keying on the base must not defeat the memo it belongs to."""
        c = _client()
        c._analyze_block_reason_uncached = AsyncMock(return_value="Blocked by X")  # type: ignore[method-assign]

        for _ in range(5):
            await c.analyze_block_reason("o", "r", 1, "sha", base_branch="main")

        assert c._analyze_block_reason_uncached.await_count == 1
        await c.aclose()


class TestInvalidationOnStateChange:
    """A memo must not outlive the state it describes.

    Approving a PR that reported "requires approval", then failing the
    retry for a different reason, would otherwise replay the stale
    approval message inside the expiry window --- and that message can
    steer the failure down the Dependabot recreate path.
    """

    @pytest.mark.asyncio
    async def test_approving_invalidates_the_memo(self) -> None:
        c = _client()
        c._analyze_block_reason_uncached = AsyncMock(  # type: ignore[method-assign]
            side_effect=["Blocked by branch protection", "Blocked by failing check: X"]
        )
        c.post = AsyncMock(return_value={})  # type: ignore[method-assign]

        assert await c.analyze_block_reason("o", "r", 1, "sha") == (
            "Blocked by branch protection"
        )
        await c.approve_pull_request("o", "r", 1, "lgtm")

        assert await c.analyze_block_reason("o", "r", 1, "sha") == (
            "Blocked by failing check: X"
        )
        await c.aclose()

    @pytest.mark.asyncio
    async def test_a_recovered_approval_also_invalidates(self, monkeypatch) -> None:
        """Reporting success means the approval landed, however it landed.

        A 500 does not mean the review was not created, so a retry that
        finds the approval already present returns success --- and that
        path must clear the memo just as the clean POST does, or the
        caller reads "requires approval" about a PR it has approved.
        """
        import dependamerge.github_async as mod

        monkeypatch.setattr(mod, "_is_transient_server_error", lambda _e: True)
        monkeypatch.setattr(mod, "_APPROVE_RETRY_BASE_DELAY", 0.0)
        c = _client()
        c._analyze_block_reason_uncached = AsyncMock(  # type: ignore[method-assign]
            side_effect=["Blocked by branch protection", "Blocked by failing check: X"]
        )
        # The POST always reports failure; the retry finds the approval
        # already present and returns success without posting again.
        c.post = AsyncMock(side_effect=RuntimeError("500"))  # type: ignore[method-assign]
        c._has_own_approval = AsyncMock(return_value=True)  # type: ignore[method-assign]
        monkeypatch.setattr(c, "_parse_permission_error", lambda *a, **k: None)

        assert await c.analyze_block_reason("o", "r", 1, "sha") == (
            "Blocked by branch protection"
        )
        await c.approve_pull_request("o", "r", 1, "lgtm")

        assert await c.analyze_block_reason("o", "r", 1, "sha") == (
            "Blocked by failing check: X"
        )
        await c.aclose()

    @pytest.mark.asyncio
    async def test_merging_invalidates_the_memo(self) -> None:
        c = _client()
        c._analyze_block_reason_uncached = AsyncMock(  # type: ignore[method-assign]
            side_effect=["Blocked by A", "Blocked by B"]
        )
        c.put = AsyncMock(return_value={"merged": True})  # type: ignore[method-assign]

        assert await c.analyze_block_reason("o", "r", 1, "sha") == "Blocked by A"
        await c.merge_pull_request("o", "r", 1)

        assert await c.analyze_block_reason("o", "r", 1, "sha") == "Blocked by B"
        await c.aclose()

    @pytest.mark.asyncio
    async def test_invalidation_is_scoped_to_one_pr(self) -> None:
        c = _client()
        c._analyze_block_reason_uncached = AsyncMock(return_value="Blocked by A")  # type: ignore[method-assign]

        await c.analyze_block_reason("o", "r", 1, "sha")
        await c.analyze_block_reason("o", "r", 2, "sha")
        c.invalidate_block_reason("o", "r", 1)

        # Only PR 1's entry went.
        assert all(k[2] != 1 for k in c._block_reason_cache)
        assert any(k[2] == 2 for k in c._block_reason_cache)
        await c.aclose()

    @pytest.mark.asyncio
    async def test_invalidating_an_absent_entry_is_harmless(self) -> None:
        c = _client()
        c.invalidate_block_reason("o", "r", 99)
        await c.aclose()


class TestWaitInvalidatesTheMemo:
    """Expiry alone is not enough to survive a short merge timeout.

    The poll cadence is ``min(2.0, recheck_interval)`` and the whole wait
    is bounded by ``--merge-timeout``, so a small timeout can complete a
    wait inside the memo's window. A post-wait reason would then repeat
    the pre-wait one, and a blocker that changed from pending checks to a
    terminal failure would still read as pending.
    """

    @pytest.mark.asyncio
    async def test_a_wait_drops_the_memo(self) -> None:
        import asyncio

        from tests.conftest import make_merge_manager

        mgr, client = make_merge_manager()
        pr_info = _pr_stub()
        # An already-expired deadline: the poll loop does no work, but
        # the wait still completes, which is what must clear the memo.
        deadline = asyncio.get_running_loop().time() - 1.0

        await mgr._wait_for_auto_merge(
            pr_info, "o", "r", continue_states=("blocked",), deadline=deadline
        )

        client.invalidate_block_reason.assert_called_with("o", "r", 1)

    @pytest.mark.asyncio
    async def test_no_wait_mode_has_nothing_to_invalidate(self) -> None:
        """Nothing waited, so nothing can have changed."""
        from tests.conftest import make_merge_manager

        mgr, client = make_merge_manager()
        mgr._no_wait = True

        await mgr._wait_for_auto_merge(
            _pr_stub(), "o", "r", continue_states=("blocked",)
        )

        client.invalidate_block_reason.assert_not_called()


def _pr_stub():
    from dependamerge.models import PullRequestInfo

    return PullRequestInfo(
        number=1,
        title="t",
        body=None,
        author="dependabot[bot]",
        head_sha="a" * 40,
        base_branch="main",
        head_branch="x",
        state="open",
        mergeable=True,
        mergeable_state="blocked",
        behind_by=None,
        files_changed=[],
        repository_full_name="o/r",
        html_url="https://github.com/o/r/pull/1",
        reviews=[],
        review_comments=[],
    )
