# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Recording and summarising per-pull-request outcomes.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import Any

from ..models import PullRequestInfo
from ..slot_lease import holding_slot
from ._base import _MergeManagerBase
from ._models import (
    MergeResult,
    MergeStatus,
)


class _ResultsMixin(_MergeManagerBase):
    """Recording and summarising per-pull-request outcomes."""

    def _record_terminal_outcome(
        self, pr_info: PullRequestInfo, status: MergeStatus
    ) -> None:
        """Record a PR's terminal outcome on the progress tracker.

        This is the **single** place terminal outcomes reach the
        tracker: every PR ends in exactly one counter (merged /
        failed / skipped / blocked / closed / pending), its
        transitory
        display state (rebasing, waiting, …) is cleared, and the
        PR-level completion percentage advances.  Centralising the
        accounting here closes the historical "result returned but
        tracker never told" and double-count bug classes.
        """
        tracker = self.progress_tracker
        if not tracker:
            return
        pr_key = f"{pr_info.repository_full_name}#{pr_info.number}"
        if status == MergeStatus.MERGED:
            tracker.merge_success(pr_key)
        elif status == MergeStatus.FAILED:
            tracker.merge_failure(pr_key)
        elif status == MergeStatus.SKIPPED:
            tracker.merge_skipped(pr_key)
        elif status == MergeStatus.BLOCKED:
            tracker.merge_blocked(pr_key)
        elif status == MergeStatus.CLOSED:
            tracker.increment_closed(pr_key)
        elif status == MergeStatus.AUTO_MERGE_PENDING:
            tracker.merge_pending(pr_key)
        else:
            # Defensive: an unexpected terminal status still counts
            # toward completion so the percentage reaches 100%.  Clear
            # any transitory display state first so the PR cannot be
            # left stuck in "rebasing"/"waiting" on the live display
            # if a new terminal status is added without a counter
            # mapping here.
            tracker.track_pr_state(pr_key, None)
            tracker.pr_completed()

    def _track_pr_state(self, pr_info: PullRequestInfo, state: str | None) -> None:
        """Move a PR between transitory tracker states (or clear)."""
        tracker = self.progress_tracker
        if not tracker:
            return
        pr_key = f"{pr_info.repository_full_name}#{pr_info.number}"
        tracker.track_pr_state(pr_key, state)

    def _record_rebase(self) -> None:
        """Count one rebase operation on the progress tracker.

        Called wherever the run actually moves a branch onto its base:
        the ``@dependabot rebase`` macro, the local ``git rebase`` +
        force-push, and the REST ``update-branch`` path.  The counter
        is cumulative, so the live display keeps reporting how many
        rebases the run triggered after the PRs have reached their
        terminal outcomes.
        """
        tracker = self.progress_tracker
        if not tracker:
            return
        tracker.record_rebase()

    def _record_retrigger(self) -> None:
        """Count one comment macro on the progress tracker.

        Called after successfully posting ``@dependabot rebase``,
        ``@dependabot recreate`` or ``pre-commit.ci run``.  New macros
        should call this too so the ``Retriggered`` total stays a
        complete record of what the run poked.
        """
        tracker = self.progress_tracker
        if not tracker:
            return
        tracker.record_retrigger()

    def _pr_status(self, message: str, *, level: str = "info") -> None:
        """Emit a per-PR status line to the log.

        Per-PR lines go to the log only — in both preview and real
        runs.  Progress is conveyed by the Rich tracker counters
        ("Mergeable" in preview, "Merged" in real runs) and the
        per-PR reasons are reported in the end-of-run summary
        (:func:`cli._print_failed_pr_details`), so printing one
        console line per PR here would only duplicate the grouped
        PR listing already shown before the run.
        """
        log_func = getattr(self.log, level.lower(), self.log.info)
        log_func(message)

    async def _merge_single_pr_with_semaphore(
        self, pr_info: PullRequestInfo
    ) -> MergeResult:
        """Merge a single PR with concurrency control.

        The slot is leased, not pinned: any wait loop inside
        ``_merge_single_pr`` that wraps itself in ``parked()`` (the
        auto-merge wait, post-rebase polls, recreate waits, …)
        releases the slot for the duration of the wait and re-acquires
        it before resuming active work, so PRs waiting on external
        events (dependabot rebases, CI) never starve runnable PRs.
        See ``slot_lease.py`` and ``docs/MERGE_ENGINE_DESIGN.md``.
        """
        async with holding_slot(self._merge_semaphore):
            result = await self._merge_single_pr(pr_info)
            # Single terminal-accounting point: map the result status
            # onto the tracker counters (see _record_terminal_outcome).
            # Uses the *original* pr_info so the transitory state keyed
            # on it is cleared even when the result carries a
            # recreated PR.
            self._record_terminal_outcome(pr_info, result.status)
            return result

    def get_results_summary(self) -> dict[str, Any]:
        """
        Get a summary of merge results.

        Returns:
            Dictionary with merge statistics
        """
        if not self._results:
            return {
                "total": 0,
                "merged": 0,
                "auto_merge_pending": 0,
                "failed": 0,
                "skipped": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
            }

        total = len(self._results)
        merged = sum(1 for r in self._results if r.status == MergeStatus.MERGED)
        auto_merge_pending = sum(
            1 for r in self._results if r.status == MergeStatus.AUTO_MERGE_PENDING
        )
        failed = sum(1 for r in self._results if r.status == MergeStatus.FAILED)
        skipped = sum(1 for r in self._results if r.status == MergeStatus.SKIPPED)

        success_rate = (merged / total) * 100 if total > 0 else 0.0
        average_duration = (
            sum(r.duration for r in self._results) / total if total > 0 else 0.0
        )

        return {
            "total": total,
            "merged": merged,
            "auto_merge_pending": auto_merge_pending,
            "failed": failed,
            "skipped": skipped,
            "success_rate": success_rate,
            "average_duration": average_duration,
            "results": self._results,
        }

    def get_failed_prs(self) -> list[MergeResult]:
        """
        Get list of failed merge results.

        Returns:
            List of MergeResult objects that failed
        """
        return [r for r in self._results if r.status == MergeStatus.FAILED]

    def get_successful_prs(self) -> list[MergeResult]:
        """
        Get list of successful or auto-merge-pending results.

        "Successful" here covers both PRs that were merged directly
        (``MergeStatus.MERGED``) and PRs where GitHub auto-merge was
        enabled and the PR is expected to merge once all required
        checks pass (``MergeStatus.AUTO_MERGE_PENDING``).

        Returns:
            List of MergeResult objects that were merged successfully
            or have auto-merge pending.
        """
        return [
            r
            for r in self._results
            if r.status in (MergeStatus.MERGED, MergeStatus.AUTO_MERGE_PENDING)
        ]
