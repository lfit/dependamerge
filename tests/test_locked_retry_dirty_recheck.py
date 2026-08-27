# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Locked merge retries must not dispatch against pre-lock state.

#435 brought two recovery paths under the per-repository dispatch lock,
but without the **under-lock dirty recheck** the main dispatch path
performs (``_merge_under_dispatch_lock``). Both paths reach the lock
*after* something has already gone wrong and then wait for it, so a
sibling merge landing in that window leaves them dispatching against
state that predates it --- the classic shared ``uv.lock`` conflict.

The result is a doomed merge that 405s and is reported as a failure,
rather than being routed to conflict recovery where a rebase resolves
it. That presents as an intermittent, hard-to-attribute failure, which
is the same diagnostic problem the locking set out to remove.

``_merge_pr_with_retry`` does not close the gap: its recheck is gated on
``attempt > 0``, so the *first* dispatch still uses the stale snapshot.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

REPO = "lfreleng-actions/some-repo"


def _pr(mergeable_state: str = "clean") -> PullRequestInfo:
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
    mgr._repo_scoped = True
    mgr._merge_recheck_interval = 0.001
    return mgr, client


def _turns_dirty(mgr):
    """Model a sibling merge landing while we waited for the lock."""

    async def _dirty(pr_info, owner, repo):
        pr_info.mergeable_state = "dirty"
        pr_info.mergeable = False
        return True

    mgr._is_pr_dirty_now = _dirty  # type: ignore[method-assign]


class TestTheApproveOnDemandRetryRechecks:
    """Approve-on-demand waits for the lock after a rejected merge."""

    def _armed(self):
        mgr, client = _mgr()
        pr = _pr()
        owner, repo = "lfreleng-actions", "some-repo"
        mgr._last_merge_exception[f"{owner}/{repo}#1"] = Exception(
            "GitHub: At least 1 approving review is required by reviewers "
            "with write access."
        )
        mgr._ensure_pr_approved = AsyncMock(return_value=True)  # type: ignore[method-assign]
        mgr._get_merge_dispatch_lock = AsyncMock(return_value=asyncio.Lock())  # type: ignore[method-assign]
        mgr._merge_pr_with_retry = AsyncMock(return_value=True)  # type: ignore[method-assign]
        return mgr, pr, owner, repo

    @pytest.mark.asyncio
    async def test_a_dirty_pr_is_not_dispatched(self) -> None:
        mgr, pr, owner, repo = self._armed()
        _turns_dirty(mgr)

        merged = await mgr._approve_and_retry_if_review_required(pr, owner, repo)

        assert merged is False
        mgr._merge_pr_with_retry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_snapshot_is_updated_for_routing(self) -> None:
        """The caller routes on ``pr_info``, so it must reflect reality."""
        mgr, pr, owner, repo = self._armed()
        _turns_dirty(mgr)

        await mgr._approve_and_retry_if_review_required(pr, owner, repo)

        assert pr.mergeable_state == "dirty"

    @pytest.mark.asyncio
    async def test_a_clean_pr_still_dispatches(self) -> None:
        mgr, pr, owner, repo = self._armed()
        mgr._is_pr_dirty_now = AsyncMock(return_value=False)  # type: ignore[method-assign]

        merged = await mgr._approve_and_retry_if_review_required(pr, owner, repo)

        assert merged is True
        mgr._merge_pr_with_retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_single_pr_run_skips_the_recheck(self) -> None:
        """Not repo-scoped: no sibling can race, so spend no request."""
        mgr, pr, owner, repo = self._armed()
        mgr._repo_scoped = False
        mgr._is_pr_dirty_now = AsyncMock(return_value=True)  # type: ignore[method-assign]

        merged = await mgr._approve_and_retry_if_review_required(pr, owner, repo)

        assert merged is True
        mgr._is_pr_dirty_now.assert_not_awaited()


class TestTheRequiredWorkflowRetryRechecks:
    """The wider window: the wait can end clean, then a sibling merges."""

    def _armed(self):
        mgr, client = _mgr()
        pr = _pr()
        mgr._get_merge_dispatch_lock = AsyncMock(return_value=asyncio.Lock())  # type: ignore[method-assign]
        mgr._wait_for_auto_merge = AsyncMock(return_value=(False, False))  # type: ignore[method-assign]
        client.merge_pull_request = AsyncMock(return_value=True)
        return mgr, client, pr

    @pytest.mark.asyncio
    async def test_a_dirty_pr_is_not_dispatched(self) -> None:
        mgr, client, pr = self._armed()
        _turns_dirty(mgr)

        merged = await mgr._wait_for_required_workflows_and_retry(
            pr, "lfreleng-actions", "some-repo"
        )

        assert merged is False
        client.merge_pull_request.assert_not_awaited()
        assert pr.mergeable_state == "dirty"

    @pytest.mark.asyncio
    async def test_a_clean_pr_still_dispatches(self) -> None:
        mgr, client, pr = self._armed()
        mgr._is_pr_dirty_now = AsyncMock(return_value=False)  # type: ignore[method-assign]

        merged = await mgr._wait_for_required_workflows_and_retry(
            pr, "lfreleng-actions", "some-repo"
        )

        assert merged is True
        client.merge_pull_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_lock_is_not_held_across_the_wait(self) -> None:
        """Unchanged by the recheck: the wait must stay outside the lock."""
        mgr, client, pr = self._armed()
        lock = asyncio.Lock()
        mgr._get_merge_dispatch_lock = AsyncMock(return_value=lock)  # type: ignore[method-assign]
        mgr._is_pr_dirty_now = AsyncMock(return_value=False)  # type: ignore[method-assign]
        observed: dict[str, bool] = {}

        async def _wait(*_a: Any, **_kw: Any) -> tuple[bool, bool]:
            observed["locked_during_wait"] = lock.locked()
            return False, False

        mgr._wait_for_auto_merge = _wait  # type: ignore[method-assign]

        await mgr._wait_for_required_workflows_and_retry(
            pr, "lfreleng-actions", "some-repo"
        )

        assert observed["locked_during_wait"] is False
