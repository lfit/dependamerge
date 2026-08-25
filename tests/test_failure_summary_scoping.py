# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Failure summaries must not hide their own conversion defects.

``_blocked_failure_summary`` wrapped far more than the call it meant to
guard: its ``except Exception`` covered the analysis probe *and* the
whole string-conversion chain that follows it.  A genuine analysis
failure and a bug in the conversion therefore produced identical output
--- the generic fallback --- with nothing in the message or the logs to
tell them apart.

That is a diagnostics path, so it cannot cause a wrong merge.  What it
can do is degrade exactly the output operators rely on when a merge is
blocked and they need to know why: the fallback is plausible enough that
a silently swallowed conversion bug could persist indefinitely, with
every blocked PR reporting the generic reason and nobody suspecting the
summary was failing rather than merely being unspecific.

``_unknown_state_failure_summary`` carried the same shape and is fixed
alongside it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

REPO = "lfreleng-actions/some-repo"


def _pr(mergeable: bool = True, mergeable_state: str = "blocked") -> PullRequestInfo:
    return PullRequestInfo(
        number=1,
        title="t",
        body=None,
        author="dependabot[bot]",
        head_sha="a" * 40,
        base_branch="main",
        head_branch="x",
        state="open",
        mergeable=mergeable,
        mergeable_state=mergeable_state,
        behind_by=None,
        files_changed=[],
        repository_full_name=REPO,
        html_url=f"https://github.com/{REPO}/pull/1",
        reviews=[],
        review_comments=[],
    )


class TestOnlyTheProbeIsGuarded:
    """A failed analysis and a broken conversion must look different."""

    @pytest.mark.asyncio
    async def test_a_probe_failure_still_yields_the_fallback(self) -> None:
        """The case the guard was written for is unchanged."""
        mgr, _ = make_merge_manager()
        mgr._analyze_block_reason_async = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("analysis unavailable")
        )

        summary = await mgr._blocked_failure_summary(_pr(mergeable=True))

        assert summary == "branch protection rules prevent merge"

    @pytest.mark.asyncio
    async def test_the_fallback_still_reflects_mergeability(self) -> None:
        mgr, _ = make_merge_manager()
        mgr._analyze_block_reason_async = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("analysis unavailable")
        )

        summary = await mgr._blocked_failure_summary(_pr(mergeable=False))

        assert summary == "blocked by failing status checks"

    @pytest.mark.asyncio
    async def test_a_conversion_defect_is_not_reported_as_the_fallback(self) -> None:
        """A defect below the probe propagates instead of being disguised.

        A ``None`` reason models the shape of a conversion bug: the probe
        succeeded, so any failure past that point belongs to our own
        logic. Previously this surfaced as the generic fallback, making
        it indistinguishable from an unreachable analysis.
        """
        mgr, _ = make_merge_manager()
        mgr._analyze_block_reason_async = AsyncMock(return_value=None)  # type: ignore[method-assign]

        with pytest.raises(AttributeError):
            await mgr._blocked_failure_summary(_pr())

    @pytest.mark.asyncio
    async def test_known_reasons_are_unchanged(self) -> None:
        """Moving the chain out of the guard must not alter its output."""
        mgr, _ = make_merge_manager()
        cases = [
            ("Blocked by failing check: build", "failing check: build"),
            ("Blocked by 2 failing checks", "2 failing checks"),
            (
                "Human reviewer requested changes",
                "human reviewer requested changes",
            ),
            ("Blocked by repository ruleset", "repository ruleset prevents merge"),
            (
                "Blocked for an undetermined reason",
                "blocked for an undetermined reason",
            ),
            (
                "Blocked by branch protection",
                "branch protection rules prevent merge",
            ),
        ]
        for reason, expected in cases:
            mgr._analyze_block_reason_async = AsyncMock(return_value=reason)  # type: ignore[method-assign]
            assert await mgr._blocked_failure_summary(_pr()) == expected, reason


class TestTheUnknownStateSummaryMatches:
    """The same shape sat two functions below, and is scoped the same way."""

    @pytest.mark.asyncio
    async def test_a_probe_failure_still_yields_the_fallback(self) -> None:
        mgr, _ = make_merge_manager()
        mgr._analyze_block_reason_async = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("analysis unavailable")
        )

        summary = await mgr._unknown_state_failure_summary(_pr())

        assert summary == "status checks pending or failed"

    @pytest.mark.asyncio
    async def test_a_conversion_defect_is_not_reported_as_the_fallback(self) -> None:
        mgr, _ = make_merge_manager()
        mgr._analyze_block_reason_async = AsyncMock(return_value=None)  # type: ignore[method-assign]

        with pytest.raises(AttributeError):
            await mgr._unknown_state_failure_summary(_pr())

    @pytest.mark.asyncio
    async def test_a_known_reason_is_unchanged(self) -> None:
        mgr, _ = make_merge_manager()
        mgr._analyze_block_reason_async = AsyncMock(  # type: ignore[method-assign]
            return_value="Blocked by failing check: lint"
        )

        assert await mgr._unknown_state_failure_summary(_pr()) == "failing check: lint"
