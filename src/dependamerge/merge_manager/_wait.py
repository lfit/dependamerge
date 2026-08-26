# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The poll loop that waits for a pending merge to complete.

Also the adaptive head start: repositories whose checks are
consistently slow have the first stretch of the wait skipped, which
saves API calls without materially delaying the outcome.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _AutoMergeWaitMixin(_MergeManagerBase):
    """Waiting for auto-merge, with an adaptive head start."""

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
    ) -> bool:
        """Sleep past the latency this repository has already demonstrated.

        Returns whether it actually slept, so the caller can restore the
        short first-poll cadence: the head start is sized to land as the
        checks finish, and following it with the full steady-state
        interval would add that interval to every wait it applies to.

        Applies only to waits that measure checks.  The figure describes
        check latency, so using it to skip ahead in a *rebase* wait would
        sleep on an unrelated measurement --- and a rebase often lands in
        seconds, so a check-sized head start could sleep clean through it.

        Skipped, too, unless the PR is currently in a state this wait
        intends to sit through.  The conflict path calls back with
        ``stop_on_clean`` after a rebase may already have left the PR
        ``clean``; sleeping first would burn up to half the shared
        deadline before the loop noticed it should return immediately.

        Called **after** the first live poll rather than before it, so
        ``pr_info`` carries state the loop has just read.  Sizing it from
        the run's opening snapshot meant that in a striped run --- where
        a repository's PRs are deliberately serialised --- a sibling
        could sleep up to half its remaining budget having *already* gone
        green, because its snapshot still said ``blocked`` from minutes
        earlier.  The head start exists to save time on slow
        repositories, so spending it on a resolved PR is the exact
        opposite of the intent, and most likely on precisely the
        repositories it was added to help.

        Extracted from :meth:`_wait_for_auto_merge` to keep that method
        within the complexity budget; see :meth:`_wait_head_start` for
        the sizing decision.
        """
        if not measures_checks:
            return False
        if stop_on_clean and pr_info.mergeable_state == "clean":
            return False
        if continue_states and pr_info.mergeable_state not in continue_states:
            # Already outside the states this call waits through, so the
            # loop is about to exit; there is nothing to skip ahead to.
            return False
        head_start = self._wait_head_start(pr_info.repository_full_name, remaining)
        if head_start <= 0:
            return False
        self.log.debug(
            "Head start of %.0fs for %s: this repository's checks have "
            "taken that long already",
            head_start,
            pr_key,
        )
        await asyncio.sleep(head_start)
        return True

    async def _wait_for_auto_merge(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        *,
        continue_states: tuple[str, ...],
        deadline: float | None = None,
        stop_on_clean: bool = True,
        measures_checks: bool = False,
    ) -> tuple[bool, bool]:
        """Poll a PR until it merges, closes, settles, or times out.

        Registers ``owner/repo#N`` in ``_waiting_prs`` so the parallel
        progress ticker can render an aggregate countdown, then polls
        every ``_merge_recheck_interval`` seconds until one of:

          * ``mergeable_state`` becomes ``clean`` (only when
            ``stop_on_clean`` is True — the default; callers that have
            enabled auto-merge and want to observe the PR actually
            close set it False and include ``"clean"`` in
            ``continue_states`` instead),
          * the PR closes (capturing the ``merged`` flag so the caller
            can tell auto-merge success from closed-without-merge),
          * ``mergeable_state`` leaves ``continue_states`` (the caller
            decides what the new state means), or
          * the deadline passes.

        The total wait is bounded by ``merge_timeout`` unless an
        explicit ``deadline`` is supplied — used to share a single
        budget across sequential waits (e.g. the rebase-then-checks
        phases of conflict recovery).

        Mutates ``pr_info`` in place (``mergeable``,
        ``mergeable_state``, ``head_sha``, ``state``).  Returns
        ``(closed_during_wait, merged_during_wait)``.
        """
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        if self._github_client is None:
            return False, False

        # Fire-and-forget (``max_wait == 0``): never block.  Returning
        # "not closed" lets callers fall through to the auto-merge-pending
        # path (Step 6 / the conflict handler arms auto-merge and reports
        # AUTO_MERGE_PENDING).  Auto-merge is armed by the caller before
        # this point, so GitHub still completes the merge later.
        if self._no_wait:
            return False, False

        pr_key = f"{owner}/{repo}#{pr_info.number}"
        loop = asyncio.get_running_loop()
        # Drive the wait off a monotonic deadline so the total is
        # bounded even if a single iteration over-sleeps slightly.
        if deadline is None:
            deadline = loop.time() + self._merge_timeout
        # Clamp to the owner-wide global ceiling (when set) so no single
        # PR's wait can push the whole run past ``max_wait``.
        if self._run_deadline is not None:
            deadline = min(deadline, self._run_deadline)

        async with self._waiting_lock:
            self._waiting_prs[pr_key] = deadline

        # Track whether the PR was closed during the wait and, if so,
        # whether it was actually merged.  The REST payload's
        # ``merged`` boolean distinguishes auto-merge success from
        # closed-without-merge (a human closed it, dependabot
        # superseded it, etc.).
        closed_during_wait = False
        merged_during_wait = False
        first_poll = True
        # Set per iteration: whether that poll supplied a usable live
        # ``mergeable_state``.  Guards the head start below.
        live_state_seen = False
        # Bound before the try so the ``finally`` can always read it,
        # even if the parked block is never entered.
        wait_started = loop.time()
        # Set only where a wait is judged to have measured checks; see
        # the assignment at the end of the poll loop.
        wait_ended: float | None = None
        try:
            # The whole poll loop is a wait on an external event
            # (auto-merge / CI / a rebase), so release this worker's
            # concurrency slot for its duration — a parked PR must
            # never starve runnable PRs (see ``slot_lease.py``).  The
            # polling GETs are paced by the HTTP client's own limits.
            async with _mm.parked():
                # The head start is applied after the first live poll,
                # not here: see ``_apply_wait_head_start``.  Sizing it
                # from the fetch-time snapshot could sleep through a PR
                # that had already gone green.
                head_start_pending = measures_checks
                while loop.time() < deadline:
                    if stop_on_clean and pr_info.mergeable_state == "clean":
                        break
                    # Sleep no longer than the time remaining so we don't
                    # overshoot the deadline.  Clamp to non-negative: the
                    # ``while`` check and this ``time()`` call are not
                    # atomic, so a near-deadline crossing could otherwise
                    # pass ``asyncio.sleep`` a tiny negative value.  The
                    # first poll uses a much shorter delay (see
                    # ``MERGE_WAIT_FIRST_POLL_SECONDS``) so a PR that
                    # resolved the moment the wait started is detected
                    # promptly instead of a full interval late.
                    interval = (
                        min(
                            _mm.MERGE_WAIT_FIRST_POLL_SECONDS,
                            self._merge_recheck_interval,
                        )
                        if first_poll
                        else self._merge_recheck_interval
                    )
                    first_poll = False
                    remaining = max(0.0, deadline - loop.time())
                    await asyncio.sleep(min(interval, remaining))
                    try:
                        refreshed_wait = await self._fetch_pr_state(
                            owner, repo, pr_info.number
                        )
                    except Exception as wait_exc:
                        self.log.debug(
                            "Failed to refresh PR state during auto-merge "
                            "wait for %s: %s",
                            pr_key,
                            wait_exc,
                        )
                        continue
                    if isinstance(refreshed_wait, dict):
                        # ``_apply_wait_refresh`` deliberately keeps the
                        # previous value when GitHub answers ``null`` /
                        # ``""`` / ``"unknown"`` while recomputing, and
                        # ``_fetch_pr_state`` can return ``None`` without
                        # raising when a per-PR fallback fails.  Either
                        # way ``pr_info`` still holds the *snapshot*
                        # value, so the head start must not be sized
                        # from it --- that is the stale read this change
                        # exists to remove.
                        live_state_seen = refreshed_wait.get("mergeable_state") not in (
                            None,
                            "",
                            "unknown",
                        )
                        if self._apply_wait_refresh(pr_info, refreshed_wait):
                            closed_during_wait = True
                            merged_during_wait = bool(
                                refreshed_wait.get("merged", False)
                            )
                            break
                    if (
                        pr_info.mergeable_state == "unstable"
                        and pr_info.mergeable is True
                    ):
                        break
                    # Continue waiting only while the PR is in a state the
                    # caller still considers rescuable; any other value
                    # means it became mergeable, closed, or hit a terminal
                    # state, so exit and let the caller decide.
                    if pr_info.mergeable_state not in continue_states:
                        break

                    # Reaching here means the PR has been read live and
                    # genuinely still needs waiting through: it is not
                    # clean, not closed, and still in a continue state.
                    # Only now is a head start worth taking.  Pending
                    # rather than tied to the first iteration, so a
                    # transient fetch failure --- or a reading GitHub
                    # could not supply --- defers it instead of
                    # discarding it or spending it on a stale value.
                    if head_start_pending and live_state_seen:
                        head_start_pending = False
                        if await self._apply_wait_head_start(
                            pr_info,
                            pr_key,
                            max(0.0, deadline - loop.time()),
                            continue_states,
                            stop_on_clean,
                            measures_checks,
                        ):
                            # The head start is sized to land as the
                            # checks finish, so read promptly afterwards.
                            # Without this the next sleep would be the
                            # full steady-state interval, adding it to
                            # every wait the optimisation applies to.
                            first_poll = True
                # Stop the clock inside the park: leaving it re-acquires
                # this worker's slot, and that queue is the scheduler's
                # time, not the repository's.  Only a wait that polled
                # and saw the checks resolve measured anything.
                if not first_poll and self._checks_resolved(pr_info):
                    wait_ended = loop.time()
        finally:
            async with self._waiting_lock:
                self._waiting_prs.pop(pr_key, None)
            # Whatever was true before the wait is not necessarily true
            # after it: that is the point of waiting.  Drop the memo
            # rather than rely on its expiry, which a short
            # ``--merge-timeout`` can outlast.
            if self._github_client is not None:
                self._github_client.invalidate_block_reason(owner, repo, pr_info.number)
            # Record the latency only when the wait produced a result and
            # actually measured checks (see ``_record_check_wait``).
            if measures_checks:
                self._record_check_wait(pr_info, wait_started, wait_ended, deadline)

        return closed_during_wait, merged_during_wait

    @staticmethod
    def _apply_wait_refresh(
        pr_info: PullRequestInfo, refreshed: dict[str, Any]
    ) -> bool:
        """Fold a polled snapshot into *pr_info*; report whether it closed.

        Extracted from :meth:`_wait_for_auto_merge` to keep that method
        within the complexity budget.

        Each field is overwritten only when present *and* usable.
        GitHub returns ``null`` for ``mergeable`` and ``null`` / ``""``
        / ``"unknown"`` for ``mergeable_state`` while it recomputes
        mergeability, and letting those through would push the state out
        of the caller's ``continue_states`` and end the wait early --- a
        ``blocked`` PR briefly reading ``unknown`` would exit and trigger
        a premature manual merge.

        The head is kept current because it can change mid-wait (rebase,
        force-push), and later block-reason analysis must query the
        commit the PR is actually on.
        """
        if "mergeable" in refreshed:
            refreshed_mergeable = refreshed.get("mergeable")
            if refreshed_mergeable is not None:
                pr_info.mergeable = refreshed_mergeable
        if "mergeable_state" in refreshed:
            refreshed_state = refreshed.get("mergeable_state")
            if refreshed_state not in (None, "", "unknown"):
                pr_info.mergeable_state = refreshed_state
        refreshed_head = (refreshed.get("head") or {}).get("sha")
        if refreshed_head:
            pr_info.head_sha = refreshed_head
        if refreshed.get("state") != "closed":
            return False
        pr_info.state = "closed"
        return True

    @staticmethod
    def _checks_resolved(pr_info: PullRequestInfo) -> bool:
        """Whether the PR's state says its checks have finished.

        Used to decide whether a completed wait measured anything.  The
        loop can also end because the PR turned ``dirty``, was closed,
        or went ``behind`` --- none of which timed a check run, and all
        of which tend to resolve in seconds, so recording them would
        drag the repository's median down and quietly disable the head
        start for its siblings.

        ``unstable`` counts when the PR is mergeable: that is every
        required check finished, with a non-required one failing.
        """
        if pr_info.mergeable_state == "clean":
            return True
        return pr_info.mergeable_state == "unstable" and pr_info.mergeable is True

    def _record_check_wait(
        self,
        pr_info: PullRequestInfo,
        started: float,
        ended: float | None,
        deadline: float,
    ) -> None:
        """Note how long this repository's checks took, for its siblings.

        Only a wait that ended of its own accord counts.  ``ended`` is
        ``None`` when the wait recorded no measurement --- an exception
        cut it short, it never polled, or it ended for a reason that
        timed no check run --- and a wait that reached the deadline
        bounds the latency from below without measuring it.

        ``ended`` is also sampled *before* the caller's parked block
        exits, since leaving it re-acquires a concurrency slot that on a
        busy run queues behind other work; charging that scheduler delay
        to the repository's checks would teach every sibling to sleep
        through it too.

        The caller decides whether its wait measured checks at all:
        :meth:`_wait_for_auto_merge` also waits for dependabot rebases
        and for an armed auto-merge to close, and recording those would
        let a rebase turnaround masquerade as check latency and hand the
        same PR's next phase a head start worth half its remaining
        budget.
        """
        if ended is None or ended >= deadline:
            return
        self._record_wait_duration(pr_info.repository_full_name, ended - started)
