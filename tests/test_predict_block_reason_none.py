# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""A ``None`` block reason must not become an optimistic verdict.

``_lookup_block_reason`` returns whatever the client's
``analyze_block_reason`` probe gives back, and ``_predict_blocked_verdict``
calls ``.lower()`` on it.  A ``None`` would therefore raise
``AttributeError``, which ``_predict_merge_outcome``'s outer handler
converts into "test merge capability failed - assuming mergeable".

That is the wrong direction to fail in: a missing block reason would
silently become a prediction that the PR is fine to merge, and this runs
under preview mode, so it is an answer a user acts on.

The tests pin all three directions --- the ``None`` case, the ordinary
string case that must be unaffected, and the genuine probe failure that
must still reach the optimistic fallback.

``TestTheGuardCoversOnlyTheProbe`` covers the second half of the issue:
the optimistic fallback exists so an unreachable API does not block a
run, and must not also absorb defects in the verdict derivation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

# ``force_level`` defaults to "code-owners", which is itself in the
# bypass list consulted by ``_predict_blocked_verdict``.  Left at the
# default, every assertion below would pass for the wrong reason, so the
# tests opt out of forcing explicitly.
NO_FORCE = "none"

BLOCKED_PR = {
    "mergeable_state": "blocked",
    "mergeable": False,
    "head": {"sha": "abc123"},
    "base": {"ref": "main"},
}

REPO = "lfreleng-actions/some-repo"


def _pr() -> PullRequestInfo:
    """Minimal PR for driving ``_predicted_merge_verdict``."""
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
        mergeable_state="blocked",
        behind_by=None,
        files_changed=[],
        repository_full_name=REPO,
        html_url=f"https://github.com/{REPO}/pull/1",
        reviews=[],
        review_comments=[],
    )


@pytest.mark.asyncio
async def test_none_block_reason_is_not_optimistic() -> None:
    """A ``None`` reason yields a blocked verdict, not "assuming mergeable"."""
    mgr, client = make_merge_manager(force_level=NO_FORCE)
    client.get = AsyncMock(return_value=dict(BLOCKED_PR))
    client.analyze_block_reason = AsyncMock(return_value=None)
    mgr._get_org_settings = AsyncMock(return_value={})  # type: ignore[method-assign]

    can_merge, reason = await mgr._predict_merge_outcome("o", "r", 1, "squash")

    assert can_merge is False
    assert "assuming mergeable" not in reason
    assert "branch protection rules prevent merge" in reason


@pytest.mark.asyncio
async def test_string_block_reason_is_unchanged() -> None:
    """An ordinary reason still routes to the approval branch."""
    mgr, client = make_merge_manager(force_level=NO_FORCE)
    client.get = AsyncMock(return_value=dict(BLOCKED_PR))
    client.analyze_block_reason = AsyncMock(
        return_value="PR requires approval from a code owner"
    )
    mgr._get_org_settings = AsyncMock(return_value={})  # type: ignore[method-assign]

    can_merge, reason = await mgr._predict_merge_outcome("o", "r", 1, "squash")

    assert can_merge is True
    assert "approval" in reason


@pytest.mark.asyncio
async def test_a_genuine_probe_failure_still_falls_back() -> None:
    """The optimistic fallback is preserved for real probe failures.

    Guarding the ``None`` must not also suppress the existing behaviour
    for a failing test-merge probe, which deliberately assumes the PR is
    mergeable rather than blocking a run on an unknown error.
    """
    mgr, client = make_merge_manager(force_level=NO_FORCE)
    client.get = AsyncMock(side_effect=RuntimeError("connection reset"))
    mgr._get_org_settings = AsyncMock(return_value={})  # type: ignore[method-assign]

    can_merge, reason = await mgr._predict_merge_outcome("o", "r", 1, "squash")

    assert can_merge is True
    assert "assuming mergeable" in reason


class TestTheGuardCoversOnlyTheProbe:
    """The optimistic fallback is for unreachable APIs, not for our bugs.

    ``_predict_merge_outcome`` wrapped its whole body in ``except
    Exception``, so a defect anywhere in the verdict derivation was
    converted into "test merge capability failed - assuming mergeable".
    A wrong answer and an unreachable API were indistinguishable, and the
    wrong answer was the optimistic one.

    Narrowing the guard to the probe requests means a derivation bug now
    propagates to ``_predicted_merge_verdict``, whose own handler logs it
    and returns ``None`` --- no verdict, rather than a confident and
    incorrect one.
    """

    @pytest.mark.asyncio
    async def test_a_derivation_bug_is_not_swallowed(self) -> None:
        """A defect below the probe propagates instead of turning optimistic."""
        mgr, client = make_merge_manager(force_level=NO_FORCE)
        client.get = AsyncMock(return_value=dict(BLOCKED_PR))
        boom = AttributeError("'NoneType' object has no attribute 'lower'")
        mgr._predict_from_pr_state = AsyncMock(side_effect=boom)  # type: ignore[method-assign]
        mgr._get_org_settings = AsyncMock(return_value={})  # type: ignore[method-assign]

        with pytest.raises(AttributeError):
            await mgr._predict_merge_outcome("o", "r", 1, "squash")

    @pytest.mark.asyncio
    async def test_the_caller_absorbs_it_as_no_verdict(self) -> None:
        """End to end: the propagated bug becomes "no verdict", not "mergeable".

        This is the property that makes propagating safe. The caller
        already guards the probe, so a derivation bug degrades to an
        absent answer rather than crashing an owner-wide preview.
        """
        mgr, client = make_merge_manager(force_level=NO_FORCE)
        client.get = AsyncMock(return_value=dict(BLOCKED_PR))
        mgr._predict_from_pr_state = AsyncMock(  # type: ignore[method-assign]
            side_effect=AttributeError("boom")
        )
        mgr._get_org_settings = AsyncMock(return_value={})  # type: ignore[method-assign]

        pr = _pr()
        verdict = await mgr._predicted_merge_verdict(pr, "o", "r")

        assert verdict is None

    @pytest.mark.asyncio
    async def test_an_unreachable_probe_still_falls_back(self) -> None:
        """The narrowed guard still covers the request it was written for."""
        mgr, client = make_merge_manager(force_level=NO_FORCE)
        client.get = AsyncMock(side_effect=ConnectionError("network unreachable"))
        mgr._get_org_settings = AsyncMock(return_value={})  # type: ignore[method-assign]

        can_merge, reason = await mgr._predict_merge_outcome("o", "r", 1, "squash")

        assert can_merge is True
        assert "assuming mergeable" in reason
