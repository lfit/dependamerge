# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Result accumulation and the summaries derived from it.

Every terminal outcome funnels through ``_record_terminal_outcome``
so the live status display, the final summary and the accessors
used by the CLI all read the same record.
"""

from __future__ import annotations

from typing import Any, cast

from ..models import ComparisonResult, PullRequestInfo
from ._base import _MergeManagerBase
from ._types import MergeResult, MergeStatus


class _OutcomeTrackingMixin(_MergeManagerBase):
    """Recording merge outcomes and reporting on them."""

    def _collect_results(
        self,
        pr_list: list[tuple[PullRequestInfo, ComparisonResult | None]],
        results: list[Any],
    ) -> list[MergeResult]:
        """Map gathered task outcomes back to ``MergeResult`` objects.

        Converts any propagated exception (from a per-PR task run with
        ``return_exceptions=True``) into a FAILED result for the matching
        PR, preserving the input ordering.

        A ``BaseException`` that is not an ``Exception`` --- cancellation
        in practice --- is re-raised rather than converted, so an
        interrupted run fails as a cancellation instead of yielding a
        results list with an exception object in it.
        """
        final_results: list[MergeResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException) and not isinstance(result, Exception):
                # ``asyncio.CancelledError`` derives from BaseException,
                # not Exception, so it would otherwise fall through to the
                # ``cast`` below --- a no-op at runtime, which would place
                # the exception object in the results list and surface it
                # much later as an ``AttributeError`` on ``.status``, with
                # nothing tying it back to the cancelled PR.  Re-raising
                # lets cancellation tear the run down here, matching
                # ``_run_striped``, which propagates it explicitly.
                raise result
            if isinstance(result, Exception):
                pr_info = pr_list[i][0]
                error_result = MergeResult(
                    pr_info=pr_info, status=MergeStatus.FAILED, error=str(result)
                )
                final_results.append(error_result)
                # The exception escaped ``_merge_single_pr_with_semaphore``
                # before it could record a terminal outcome, so record
                # the failure here to keep the tracker counters exact.
                self._record_terminal_outcome(pr_info, MergeStatus.FAILED)
                self.log.error(
                    f"Unexpected error merging PR {pr_info.repository_full_name}#{pr_info.number}: {result}"
                )
            else:
                # result is guaranteed to be MergeResult here since it's not an Exception
                final_results.append(cast(MergeResult, result))
        return final_results

    def _record_terminal_outcome(
        self, pr_info: PullRequestInfo, status: MergeStatus
    ) -> None:
        """Record a PR's terminal outcome on the progress tracker.

        This is the **single** place terminal outcomes reach the
        tracker: every PR ends in exactly one counter (merged /
        failed / skipped / blocked / closed / pending / unsettled), its
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
        elif status == MergeStatus.UNSETTLED:
            tracker.merge_unsettled(pr_key)
        else:
            # Defensive: an unexpected terminal status still counts
            # toward completion so the percentage reaches 100%.  Clear
            # any transitory display state first so the PR cannot be
            # left stuck in "rebasing"/"waiting" on the live display
            # if a new terminal status is added without a counter
            # mapping here.
            tracker.track_pr_state(pr_key, None)
            tracker.pr_completed()

    async def _reconcile_reported_failures(
        self, results: list[MergeResult]
    ) -> list[MergeResult]:
        """Re-judge failures once more, after every PR has finished.

        The per-PR confirmation runs the moment a PR's task completes,
        which for an owner-wide run is long before the summary prints.
        A pull request refused at minute two can go green at minute ten
        while its siblings are still being merged, and nothing looked
        again --- so it was reported as failed despite being mergeable by
        the time the operator read the line.  That is the case this whole
        change exists to remove, and the earlier confirmation alone does
        not remove it.

        Costs one GET per PR still reported as failed --- seventeen, in
        the run that prompted this --- and only for those, since
        :meth:`_confirm_failure` returns early for any other status.

        The tracker is corrected alongside the result, so the live counts
        and the closing summary cannot disagree about the same run.
        """
        for result in results:
            if result.status is not MergeStatus.FAILED:
                continue
            before = result.status.value
            await self._confirm_failure(result.pr_info, result)
            after = result.status.value
            if after != before and self.progress_tracker:
                self.progress_tracker.reclassify_outcome(before, after)
        return results

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
                "unsettled": 0,
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
        # Reported apart from ``failed`` for the same reason the console
        # summary separates them: a programmatic consumer counting
        # failures would otherwise either miss these PRs entirely or,
        # seeing them only in ``total``, have no way to tell that they
        # need a re-run rather than a person.
        unsettled = sum(1 for r in self._results if r.status == MergeStatus.UNSETTLED)
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
            "unsettled": unsettled,
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
