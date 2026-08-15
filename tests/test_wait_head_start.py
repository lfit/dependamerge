# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the in-run adaptive first-poll delay.

Polling a repository every ten seconds from t=0 when its checks reliably
take four minutes spends roughly twenty requests learning nothing. Once
one PR in a repository has shown how long its checks take, its siblings
can sleep most of that time before their first poll --- and because the
striped scheduler runs a repository's PRs one after another, the
observation always exists by the second PR.

``docs/BULK_RUN_PERFORMANCE_AUDIT.md`` ties this to the persistent
record in §4, which would carry the figure between runs. This is the
in-run half: it needs no storage and does not pre-commit that design.

The tests weight towards the cases where a head start must *not* be
taken, since sleeping through a resolution is the failure that costs a
merge.
"""

from __future__ import annotations

import pytest

from tests.conftest import make_merge_manager

REPO = "lfreleng-actions/slow-repo"


def _mgr(interval: float = 10.0):
    mgr, client = make_merge_manager()
    mgr._merge_recheck_interval = interval
    return mgr, client


class TestRecordWaitDuration:
    def test_records_a_positive_duration(self) -> None:
        mgr, _ = _mgr()
        mgr._record_wait_duration(REPO, 240.0)
        assert mgr._repo_wait_seconds[REPO] == [240.0]

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_ignores_non_positive(self, value: float) -> None:
        mgr, _ = _mgr()
        mgr._record_wait_duration(REPO, value)
        assert REPO not in mgr._repo_wait_seconds


class TestHeadStart:
    def test_nothing_known_means_no_head_start(self) -> None:
        mgr, _ = _mgr()
        assert mgr._wait_head_start(REPO, budget=300.0) == 0.0

    def test_slow_repository_earns_a_head_start(self) -> None:
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 240.0)

        head_start = mgr._wait_head_start(REPO, budget=300.0)

        # 80% of the observed median, capped at half the budget.
        assert head_start == pytest.approx(min(240.0 * 0.8, 150.0))

    def test_fast_repository_earns_none(self) -> None:
        """Below a few poll intervals the normal cadence is already cheap."""
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 12.0)

        assert mgr._wait_head_start(REPO, budget=300.0) == 0.0

    def test_never_sleeps_more_than_half_the_budget(self) -> None:
        """A repository that has become faster must still be observed."""
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 600.0)

        assert mgr._wait_head_start(REPO, budget=100.0) == pytest.approx(50.0)

    def test_head_start_is_never_negative(self) -> None:
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 240.0)

        assert mgr._wait_head_start(REPO, budget=0.0) == 0.0

    def test_uses_the_median_not_the_worst_case(self) -> None:
        """One slow outlier must not strand every sibling behind it."""
        mgr, _ = _mgr(interval=10.0)
        for value in (60.0, 60.0, 900.0):
            mgr._record_wait_duration(REPO, value)

        # Median is 60, not 900.
        assert mgr._wait_head_start(REPO, budget=1000.0) == pytest.approx(48.0)

    def test_repositories_do_not_share_observations(self) -> None:
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 240.0)

        assert mgr._wait_head_start("lfreleng-actions/other", budget=300.0) == 0.0

    def test_scales_with_the_configured_interval(self) -> None:
        """A slower cadence raises the bar for a head start to be worthwhile."""
        mgr, _ = _mgr(interval=60.0)
        mgr._record_wait_duration(REPO, 100.0)

        # 100s is under 3 x 60s, so no head start despite being slow.
        assert mgr._wait_head_start(REPO, budget=600.0) == 0.0
