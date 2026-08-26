# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Wait sizing and merge-dispatch serialisation on the recovery paths.

Two defects, both on paths reached only after something has already gone
wrong, which is why neither had coverage.

**#436** --- ``_apply_wait_head_start`` sized its skip-ahead from the
``pr_info`` snapshot the run fetched up front.  Owner-wide runs serialise
each repository's pull requests deliberately, and that serialisation is
also what makes the snapshot stale: while PR #1 was merging, PR #2's
checks were running, so by the time #2's worker starts its checks may
have finished while its snapshot still says ``blocked``.  The head start
is sized from the repository's *observed* median check latency, which in
that situation is the latency of a wait that has already elapsed --- so a
pull request that had gone green could sleep up to half its remaining
budget before the first live poll.

**#435** --- two recovery paths dispatched ``merge_pull_request``
**outside** the per-repository dispatch lock, defeating the serialisation
that lock exists to provide.  Both are reached only after a rejected
first attempt, and the consequence is a merge raced against
freshly-propagated branch protection: an intermittent, hard-to-attribute
failure rather than anything pointing at the lock.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

REPO = "lfreleng-actions/slow-repo"

# Any sleep longer than this is a head start rather than a poll interval:
# the tests drive the poll cadence down to a millisecond.
HEAD_START_FLOOR = 1.0


def _pr(mergeable_state: str = "blocked") -> PullRequestInfo:
    return PullRequestInfo(
        number=1,
        title="t",
        body=None,
        author="dependabot[bot]",
        head_sha="a" * 40,
        base_branch="main",
        head_branch="x",
        state="open",
        mergeable=False,
        mergeable_state=mergeable_state,
        behind_by=None,
        files_changed=[],
        repository_full_name=REPO,
        html_url=f"https://github.com/{REPO}/pull/1",
        reviews=[],
        review_comments=[],
    )


def _mgr(**overrides: Any):
    mgr, client = make_merge_manager(**overrides)
    mgr._merge_recheck_interval = 0.001
    mgr._merge_timeout = 600.0
    # One slow repository already observed, so a head start is available.
    mgr._record_wait_duration(REPO, 240.0)
    return mgr, client


class _SleepRecorder:
    """Substitute for ``asyncio.sleep`` that records every delay."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)

    @property
    def took_a_head_start(self) -> bool:
        return any(s > HEAD_START_FLOOR for s in self.slept)


class TestTheHeadStartIsSizedFromLiveState:
    """A skip-ahead is only sound if the state it skips over is current."""

    async def _run_wait(self, mgr, live_state: dict[str, Any]) -> _SleepRecorder:
        recorder = _SleepRecorder()
        mgr._fetch_pr_state = AsyncMock(return_value=live_state)  # type: ignore[method-assign]

        import dependamerge.merge_manager as mod

        original = mod.asyncio.sleep
        mod.asyncio.sleep = recorder  # type: ignore[assignment]
        try:
            await mgr._wait_for_auto_merge(
                _pr(),
                "lfreleng-actions",
                "slow-repo",
                continue_states=("blocked", "unstable"),
                measures_checks=True,
            )
        finally:
            mod.asyncio.sleep = original  # type: ignore[assignment]
        return recorder

    async def _run_wait_sequence(
        self, mgr, states: list[dict[str, Any]]
    ) -> _SleepRecorder:
        """Drive the loop through ``states``, one per poll.

        The substituted ``sleep`` does not advance the clock, so a wait
        that never observes a terminal state would spin against the
        deadline. Ending the sequence on a non-continue state lets the
        loop exit naturally.
        """
        recorder = _SleepRecorder()
        mgr._fetch_pr_state = AsyncMock(side_effect=states)  # type: ignore[method-assign]

        import dependamerge.merge_manager as mod

        original = mod.asyncio.sleep
        mod.asyncio.sleep = recorder  # type: ignore[assignment]
        try:
            await mgr._wait_for_auto_merge(
                _pr(),
                "lfreleng-actions",
                "slow-repo",
                continue_states=("blocked", "unstable"),
                measures_checks=True,
            )
        finally:
            mod.asyncio.sleep = original  # type: ignore[assignment]
        return recorder

    @pytest.mark.asyncio
    async def test_no_head_start_when_the_pr_has_already_gone_green(self) -> None:
        """The defect: the stale snapshot said blocked, so it slept anyway."""
        mgr, _ = _mgr()

        recorder = await self._run_wait(
            mgr, {"state": "open", "mergeable": True, "mergeable_state": "clean"}
        )

        assert not recorder.took_a_head_start, recorder.slept

    @pytest.mark.asyncio
    async def test_no_head_start_when_the_pr_merged_during_the_gap(self) -> None:
        mgr, _ = _mgr()

        recorder = await self._run_wait(
            mgr, {"state": "closed", "merged": True, "merged_at": "2026-08-25"}
        )

        assert not recorder.took_a_head_start, recorder.slept

    @pytest.mark.asyncio
    async def test_a_head_start_is_still_taken_when_it_is_warranted(self) -> None:
        """The optimisation must survive: a genuinely blocked PR skips ahead."""
        mgr, _ = _mgr()

        recorder = await self._run_wait_sequence(
            mgr,
            [
                {"state": "open", "mergeable": False, "mergeable_state": "blocked"},
                {"state": "open", "mergeable": True, "mergeable_state": "clean"},
            ],
        )

        assert recorder.took_a_head_start, recorder.slept

    @pytest.mark.asyncio
    async def test_the_first_poll_precedes_any_head_start(self) -> None:
        """Ordering is the fix: read, then decide whether to skip ahead."""
        mgr, _ = _mgr()
        order: list[str] = []
        recorder = _SleepRecorder()
        states = [
            {"state": "open", "mergeable": False, "mergeable_state": "blocked"},
            {"state": "open", "mergeable": True, "mergeable_state": "clean"},
        ]
        remaining = list(states)

        async def _fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            order.append("poll")
            return remaining.pop(0) if remaining else states[-1]

        async def _sleep(seconds: float) -> None:
            if seconds > HEAD_START_FLOOR:
                order.append("head-start")
            await recorder(seconds)

        mgr._fetch_pr_state = _fetch  # type: ignore[method-assign]

        import dependamerge.merge_manager as mod

        original = mod.asyncio.sleep
        mod.asyncio.sleep = _sleep  # type: ignore[assignment]
        try:
            await mgr._wait_for_auto_merge(
                _pr(),
                "lfreleng-actions",
                "slow-repo",
                continue_states=("blocked", "unstable"),
                measures_checks=True,
            )
        finally:
            mod.asyncio.sleep = original  # type: ignore[assignment]

        assert "head-start" in order, order
        assert order.index("poll") < order.index("head-start"), order


class TestEveryMergeDispatchHoldsTheRepoLock:
    """The lock's guarantee is worthless if a recovery path skips it."""

    @pytest.mark.asyncio
    async def test_approve_on_demand_retry_is_locked(self) -> None:
        mgr, client = _mgr()
        pr = _pr()
        owner, repo = "lfreleng-actions", "slow-repo"
        lock = asyncio.Lock()
        mgr._get_merge_dispatch_lock = AsyncMock(return_value=lock)  # type: ignore[method-assign]
        mgr._last_merge_exception[f"{owner}/{repo}#1"] = Exception(
            "GitHub: At least 1 approving review is required by reviewers "
            "with write access."
        )
        mgr._ensure_pr_approved = AsyncMock(return_value=True)  # type: ignore[method-assign]

        observed: dict[str, bool] = {}

        async def _retry(*_args: Any, **_kwargs: Any) -> bool:
            observed["locked"] = lock.locked()
            return True

        mgr._merge_pr_with_retry = _retry  # type: ignore[method-assign]

        assert await mgr._approve_and_retry_if_review_required(pr, owner, repo) is True
        assert observed["locked"] is True

    @pytest.mark.asyncio
    async def test_the_heuristic_path_retry_is_locked(self) -> None:
        """The same function reaches the retry twice; both must be locked."""
        mgr, client = _mgr()
        pr = _pr()
        owner, repo = "lfreleng-actions", "slow-repo"
        lock = asyncio.Lock()
        mgr._get_merge_dispatch_lock = AsyncMock(return_value=lock)  # type: ignore[method-assign]
        client.analyze_block_reason = AsyncMock(
            return_value="PR requires approval from a code owner"
        )
        mgr._ensure_pr_approved = AsyncMock(return_value=True)  # type: ignore[method-assign]

        observed: dict[str, bool] = {}

        async def _retry(*_args: Any, **_kwargs: Any) -> bool:
            observed["locked"] = lock.locked()
            return True

        mgr._merge_pr_with_retry = _retry  # type: ignore[method-assign]

        assert await mgr._approve_and_retry_if_review_required(pr, owner, repo) is True
        assert observed["locked"] is True

    @pytest.mark.asyncio
    async def test_required_workflow_retry_is_locked(self) -> None:
        mgr, client = _mgr()
        pr = _pr()
        owner, repo = "lfreleng-actions", "slow-repo"
        lock = asyncio.Lock()
        mgr._get_merge_dispatch_lock = AsyncMock(return_value=lock)  # type: ignore[method-assign]
        mgr._wait_for_auto_merge = AsyncMock(return_value=(False, False))  # type: ignore[method-assign]

        observed: dict[str, bool] = {}

        async def _merge(*_args: Any, **_kwargs: Any) -> bool:
            observed["locked"] = lock.locked()
            return True

        client.merge_pull_request = _merge

        assert await mgr._wait_for_required_workflows_and_retry(pr, owner, repo) is True
        assert observed["locked"] is True

    @pytest.mark.asyncio
    async def test_the_lock_is_not_held_across_the_wait(self) -> None:
        """Holding it through the wait would block the whole repository.

        The lock serialises the *moment of merge*, not the waiting that
        precedes it. Holding it across a multi-minute wait would queue
        every sibling in the repository behind this one pull request ---
        the head-of-line blocking the design deliberately avoids.
        """
        mgr, client = _mgr()
        pr = _pr()
        owner, repo = "lfreleng-actions", "slow-repo"
        lock = asyncio.Lock()
        mgr._get_merge_dispatch_lock = AsyncMock(return_value=lock)  # type: ignore[method-assign]

        observed: dict[str, bool] = {}

        async def _wait(*_args: Any, **_kwargs: Any) -> tuple[bool, bool]:
            observed["locked_during_wait"] = lock.locked()
            return False, False

        mgr._wait_for_auto_merge = _wait  # type: ignore[method-assign]
        client.merge_pull_request = AsyncMock(return_value=True)

        await mgr._wait_for_required_workflows_and_retry(pr, owner, repo)

        assert observed["locked_during_wait"] is False


class TestTheHeadStartNeedsAUsableReading:
    """ "Polled" is not the same as "read".

    ``_apply_wait_refresh`` deliberately keeps the previous
    ``mergeable_state`` when GitHub answers ``null``, ``""`` or
    ``"unknown"`` while recomputing, and ``_fetch_pr_state`` can return
    ``None`` without raising when a per-PR fallback fails. In both cases
    ``pr_info`` still holds the *snapshot* value --- so consuming the
    head start there would size it from exactly the stale state this
    change exists to remove.
    """

    async def _run(self, mgr, states: list[Any]) -> _SleepRecorder:
        recorder = _SleepRecorder()
        mgr._fetch_pr_state = AsyncMock(side_effect=states)  # type: ignore[method-assign]

        import dependamerge.merge_manager as mod

        original = mod.asyncio.sleep
        mod.asyncio.sleep = recorder  # type: ignore[assignment]
        try:
            await mgr._wait_for_auto_merge(
                _pr(),
                "lfreleng-actions",
                "slow-repo",
                continue_states=("blocked", "unstable"),
                measures_checks=True,
            )
        finally:
            mod.asyncio.sleep = original  # type: ignore[assignment]
        return recorder

    @pytest.mark.parametrize("unusable", [None, "", "unknown"])
    @pytest.mark.asyncio
    async def test_a_recomputing_state_defers_the_head_start(self, unusable) -> None:
        """The reading is not live, so the head start stays pending."""
        mgr, _ = _mgr()

        recorder = await self._run(
            mgr,
            [
                {"state": "open", "mergeable": False, "mergeable_state": unusable},
                {"state": "open", "mergeable": True, "mergeable_state": "clean"},
            ],
        )

        assert not recorder.took_a_head_start, recorder.slept

    @pytest.mark.asyncio
    async def test_a_missing_payload_defers_the_head_start(self) -> None:
        """``_fetch_pr_state`` returning None must not spend it either."""
        mgr, _ = _mgr()

        recorder = await self._run(
            mgr,
            [
                None,
                {"state": "open", "mergeable": True, "mergeable_state": "clean"},
            ],
        )

        assert not recorder.took_a_head_start, recorder.slept

    @pytest.mark.asyncio
    async def test_it_is_still_taken_once_a_live_state_arrives(self) -> None:
        """Deferred, not discarded: a later usable reading still gets it."""
        mgr, _ = _mgr()

        recorder = await self._run(
            mgr,
            [
                {"state": "open", "mergeable": False, "mergeable_state": "unknown"},
                {"state": "open", "mergeable": False, "mergeable_state": "blocked"},
                {"state": "open", "mergeable": True, "mergeable_state": "clean"},
            ],
        )

        assert recorder.took_a_head_start, recorder.slept


class TestTheCadenceResumesAfterAHeadStart:
    """The head start lands as checks finish, so read promptly after it.

    ``first_poll`` is consumed by the loop's first iteration, so moving
    the head start into the loop left the *next* sleep at the full
    steady-state interval. With the defaults that is 10s rather than the
    2s first-poll cadence --- 8s added to every wait the optimisation
    applies to, which eats into what it saves.
    """

    @pytest.mark.asyncio
    async def test_the_poll_after_a_head_start_is_short(self) -> None:
        mgr, _ = _mgr()
        mgr._merge_recheck_interval = 10.0
        recorder = _SleepRecorder()
        states = [
            {"state": "open", "mergeable": False, "mergeable_state": "blocked"},
            {"state": "open", "mergeable": False, "mergeable_state": "blocked"},
            {"state": "open", "mergeable": True, "mergeable_state": "clean"},
        ]
        mgr._fetch_pr_state = AsyncMock(side_effect=states)  # type: ignore[method-assign]

        import dependamerge.merge_manager as mod

        original = mod.asyncio.sleep
        mod.asyncio.sleep = recorder  # type: ignore[assignment]
        try:
            await mgr._wait_for_auto_merge(
                _pr(),
                "lfreleng-actions",
                "slow-repo",
                continue_states=("blocked", "unstable"),
                measures_checks=True,
            )
        finally:
            mod.asyncio.sleep = original  # type: ignore[assignment]

        # slept: [short first poll, head start, short poll again, ...]
        # The head start is sized from a 240s observation, so it is far
        # larger than any poll interval --- identify it by magnitude
        # rather than a fixed floor, which the 2s first poll exceeds.
        head_start_at = recorder.slept.index(max(recorder.slept))
        assert recorder.slept[head_start_at] > 50.0, recorder.slept
        after = recorder.slept[head_start_at + 1]
        assert after < 10.0, recorder.slept
