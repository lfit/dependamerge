# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Per-repository wait timing, and the head start it earns.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio

from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _WaitBudgetMixin(_MergeManagerBase):
    """Per-repository wait timing, and the head start it earns."""

    # A head start is only worth taking when a repository's checks are
    # slow relative to the poll cadence; below this multiple of the
    # recheck interval the normal rhythm already costs little.
    _HEAD_START_MIN_INTERVALS = 3.0
    # Fraction of the observed median to skip.  Deliberately short of
    # 1.0 so a repository that speeds up is still caught promptly.
    _HEAD_START_FRACTION = 0.8

    def _record_wait_duration(self, repo_full_name: str, seconds: float) -> None:
        """Remember how long a repository's checks took to resolve.

        Only durations from waits that *ended in a result* are recorded;
        a timeout says nothing about latency except that it exceeded the
        budget.
        """
        if seconds <= 0:
            return
        self._repo_wait_seconds.setdefault(repo_full_name, []).append(seconds)

    def _wait_head_start(self, repo_full_name: str, budget: float) -> float:
        """Seconds to skip before the first poll of a wait.

        Polling a repository every ten seconds from t=0 when its checks
        reliably take four minutes spends around twenty requests learning
        nothing.  Once one PR in the repository has shown how long its
        checks take, its siblings can sleep most of that time first ---
        the striped scheduler runs them one after another, so by the
        second PR the observation already exists.

        Returns ``0.0`` when nothing is known, when the repository is
        quick enough that the normal cadence is cheap, or when the
        remaining budget is too small to gamble on a single sleep.
        """
        observations = self._repo_wait_seconds.get(repo_full_name)
        if not observations:
            return 0.0
        ordered = sorted(observations)
        mid = len(ordered) // 2
        # True median: average the middle pair for an even count.  Taking
        # the upper middle instead lets a single slow outlier set the
        # figure --- [60, 900] would yield 900 rather than 480 --- which
        # is exactly what the median is chosen to avoid.
        median = (
            ordered[mid]
            if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2.0
        )
        if median < self._HEAD_START_MIN_INTERVALS * self._merge_recheck_interval:
            return 0.0
        # Never spend more than half the remaining budget asleep: a
        # repository that has become faster must still be observed.
        return max(0.0, min(median * self._HEAD_START_FRACTION, budget / 2.0))

    async def _apply_wait_head_start(
        self,
        pr_info: PullRequestInfo,
        pr_key: str,
        remaining: float,
        continue_states: tuple[str, ...],
        stop_on_clean: bool,
        measures_checks: bool,
    ) -> None:
        """Sleep past the latency this repository has already demonstrated.

        Applies only to waits that measure checks.  The figure describes
        check latency, so using it to skip ahead in a *rebase* wait would
        sleep on an unrelated measurement --- and a rebase often lands in
        seconds, so a check-sized head start could sleep clean through it.

        Skipped, too, unless the PR is currently in a state this wait
        intends to sit through.  The conflict path calls back with
        ``stop_on_clean`` after a rebase may already have left the PR
        ``clean``; sleeping first would burn up to half the shared
        deadline before the loop noticed it should return immediately.

        Extracted from :meth:`_wait_for_auto_merge` to keep that method
        within the complexity budget; see :meth:`_wait_head_start` for
        the sizing decision.
        """
        if not measures_checks:
            return
        if stop_on_clean and pr_info.mergeable_state == "clean":
            return
        if continue_states and pr_info.mergeable_state not in continue_states:
            # Already outside the states this call waits through, so the
            # loop is about to exit; there is nothing to skip ahead to.
            return
        head_start = self._wait_head_start(pr_info.repository_full_name, remaining)
        if head_start <= 0:
            return
        self.log.debug(
            "Head start of %.0fs for %s: this repository's checks have "
            "taken that long already",
            head_start,
            pr_key,
        )
        await asyncio.sleep(head_start)
