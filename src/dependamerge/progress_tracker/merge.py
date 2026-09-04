# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Merge-phase progress tracker.

``MergeProgressTracker`` extends the scan-phase tracker with the
counters a merge or close run needs: transitory per-PR pipeline states,
cumulative activity totals, and terminal outcomes.  Rendering those
counters lives in the sibling ``merge_display`` module.
"""

from __future__ import annotations

import threading
from typing import Any

from .merge_display import generate_merge_display_text
from .scan import ProgressTracker


class MergeProgressTracker(ProgressTracker):
    """Extended progress tracker with merge-specific metrics.

    Three families of counter feed the live display:

    - **Transitory states** — where each PR is *right now* (one of
      the keys in :attr:`_STATE_ORDER`).  Keyed by PR, so a PR
      occupies at most one state at a time and recording a terminal
      outcome removes it from whatever state it was in.
    - **Activity counters** (``rebases_triggered``,
      ``retriggers_issued``) — monotonic totals of *operations* the
      run performed.  Unlike transitory states these never decrease
      when a PR moves on, so the display keeps showing how many
      rebases and comment macros the run issued even after every PR
      has reached ``Merged`` / ``Failed``.
    - **Terminal counters** (merged / failed / skipped / blocked /
      pending / unsettled / closed) — one per PR, recorded exactly once.
    """

    # Transitory display states in pipeline order.  Each entry is
    # ``(state_key, display_label)``; only non-zero states render.
    # ``submitting`` is recorded by the Gerrit submit manager while a
    # change's review + submit round-trips run.
    #
    # NOTE: there is deliberately no ``rebased`` state here.  "A rebase
    # happened" is a completed operation, not a place a PR sits, and
    # keying it per PR made the count vanish the moment the PR moved
    # on; it is tracked by the cumulative ``rebases_triggered`` counter
    # instead.  Post-rebase waiting is reported as ``waiting``.
    _STATE_ORDER: tuple[tuple[str, str], ...] = (
        ("rebasing", "🔄 Rebasing"),
        ("recreating", "♻️ Recreating"),
        ("waiting", "⏳ Waiting"),
        ("submitting", "📤 Submitting"),
    )

    def __init__(
        self,
        organization: str,
        is_close_operation: bool = False,
        operation_label: str | None = None,
        operation_icon: str | None = None,
        preview: bool = False,
        unit_label: str = "PRs",
    ):
        """Initialize merge progress tracker.

        Args:
            organization: Name of the GitHub organization or owner.
            is_close_operation: Whether this tracks a close operation.
            operation_label: Custom heading label for the progress
                display.  When ``None``, defaults to
                ``"Searching for similar PRs"`` (merge) or
                ``"Closing PRs"`` (close).
            operation_icon: Custom emoji icon for the heading.  When
                ``None``, defaults to ``"\U0001f500"`` / ``"\U0001f50d"`` (merge) or
                ``"\U0001f6aa"`` (close) depending on context.
            preview: Whether this tracks a preview (evaluation) run.
                Preview runs perform no merges, so the terminal
                counters render with evaluation labels ("Mergeable" /
                "Would fail") instead of the execution labels
                ("Merged" / "Failed") that would misstate what
                happened.
            unit_label: Noun used for the per-item progress fraction
                (e.g. ``"3/9 PRs"``).  GitHub callers keep the default
                ``"PRs"``; Gerrit callers pass ``"changes"`` so the
                display matches the platform's terminology.
        """
        super().__init__(organization, show_pr_stats=True)
        self.similar_prs_found = 0
        self.prs_merged = 0
        self.prs_failed = 0
        self.prs_skipped = 0
        self.prs_closed = 0
        # PRs left with auto-merge armed when the run ended (GitHub
        # completes the merge server-side once checks pass).
        self.prs_pending = 0
        # PRs that are blocked and cannot be merged by this run.
        self.prs_blocked = 0
        # PRs the run judged before their required checks settled, and
        # that would merge on another run.
        self.prs_unsettled = 0
        # Cumulative activity counters.  These count *operations*, not
        # PRs: a single PR can contribute more than one (e.g. a
        # ``@dependabot rebase`` macro counts as both a rebase and a
        # re-trigger, and a PR rebased twice counts twice).  They are
        # monotonic and survive the PR's transition to a terminal
        # outcome, so the operator can still see what the run did after
        # every PR has landed in Merged / Failed.
        self.rebases_triggered = 0
        self.retriggers_issued = 0
        self.is_close_operation = is_close_operation
        self.preview = preview
        self._custom_label = operation_label
        self._custom_icon = operation_icon
        self.unit_label = unit_label
        # PR-level progress (used for repo-scoped operations)
        self.total_prs = 0
        self.completed_prs = 0
        # Transitory per-PR display states: pr_key -> state key from
        # ``_STATE_ORDER``.  Terminal outcomes remove the entry.
        self._pr_states: dict[str, str] = {}
        # Counter and state mutations may arrive from worker threads
        # (the Gerrit submit manager records outcomes from a
        # ThreadPoolExecutor) while Rich's refresh thread renders the
        # display, so both mutation and the render-time snapshot are
        # guarded by this re-entrant lock.
        self._state_lock = threading.RLock()

    def found_similar_pr(self, count: int = 1) -> None:
        """Update count of similar PRs found."""
        with self._state_lock:
            self.similar_prs_found += count
        self._refresh_display()

    def set_total_prs(self, total: int) -> None:
        """Set the total number of PRs to process.

        When set, the progress display switches from repo-level
        to PR-level progress (e.g. ``3/9 PRs, 33%``).
        """
        with self._state_lock:
            self.total_prs = total
        self._refresh_display()

    def record_rebase(self, count: int = 1) -> None:
        """Count a rebase operation triggered against a PR branch.

        Covers every mechanism that brings a branch up to date: a
        local ``git rebase`` + force-push, the REST ``update-branch``
        call, and the ``@dependabot rebase`` comment macro.  The
        counter is cumulative and never cleared — it records what the
        run *did*, independently of where each PR ended up.

        ``count`` must be positive; a zero or negative value is
        ignored so the monotonic contract holds no matter what a
        caller passes.  A display counter must never abort a merge
        run, so a bad argument is dropped rather than raised.
        """
        if count <= 0:
            return
        with self._state_lock:
            self.rebases_triggered += count
        self._refresh_display()

    def record_retrigger(self, count: int = 1) -> None:
        """Count a comment macro posted to re-trigger external tooling.

        Covers ``@dependabot rebase``, ``@dependabot recreate``,
        ``pre-commit.ci run`` and any macro added later.  Like
        :meth:`record_rebase` this is cumulative and never cleared.

        A ``@dependabot rebase`` deliberately increments *both*
        counters: it is one comment macro and one rebase request, and
        the two counters answer different questions ("how many
        branches did we move?" vs "how many times did we poke an
        external bot?").

        ``count`` is validated exactly as in :meth:`record_rebase`:
        zero and negative values are ignored.
        """
        if count <= 0:
            return
        with self._state_lock:
            self.retriggers_issued += count
        self._refresh_display()

    def track_pr_state(self, pr_key: str, state: str | None) -> None:
        """Move a PR between transitory display states.

        ``state`` is one of the keys in ``_STATE_ORDER`` (e.g.
        ``"rebasing"``, ``"waiting"``) or ``None`` to clear the PR's
        entry when an operation finishes without reaching a terminal
        outcome.  Terminal outcomes are recorded via ``merge_success``
        / ``merge_failure`` / ``merge_skipped`` / ``merge_blocked`` /
        ``merge_pending``, which also clear the transitory entry when
        given the PR key.
        """
        if state is None:
            with self._state_lock:
                self._pr_states.pop(pr_key, None)
        else:
            with self._state_lock:
                self._pr_states[pr_key] = state
        self._refresh_display()

    def _finish_pr(self, pr_key: str | None) -> None:
        """Shared terminal-outcome bookkeeping.

        Clears the PR's transitory state (when a key is supplied) and
        advances PR-level completion progress.  Callers must hold
        ``_state_lock``.
        """
        if pr_key is not None:
            self._pr_states.pop(pr_key, None)
        if self.total_prs > 0:
            self.completed_prs += 1

    def merge_success(self, pr_key: str | None = None) -> None:
        """Record a successful merge."""
        with self._state_lock:
            self._finish_pr(pr_key)
            self.prs_merged += 1
        self._refresh_display()

    def merge_failure(self, pr_key: str | None = None) -> None:
        """Record a failed merge."""
        with self._state_lock:
            self._finish_pr(pr_key)
            self.prs_failed += 1
        self._refresh_display()

    def merge_skipped(self, pr_key: str | None = None) -> None:
        """Record a PR skipped because it was merged externally.

        Distinct from ``merge_failure`` because the operator does
        not need to follow up: the PR is already merged, just not
        by us.  Tracked separately so the final summary can show
        a non-zero ⏭️ Skipped count alongside Merged / Failed.
        """
        with self._state_lock:
            self._finish_pr(pr_key)
            self.prs_skipped += 1
        self._refresh_display()

    def merge_blocked(self, pr_key: str | None = None) -> None:
        """Record a PR that is blocked and cannot merge in this run."""
        with self._state_lock:
            self._finish_pr(pr_key)
            self.prs_blocked += 1
        self._refresh_display()

    def merge_pending(self, pr_key: str | None = None) -> None:
        """Record a PR left with auto-merge armed at run end.

        GitHub merges the PR server-side once its required checks
        pass; from this run's perspective the PR is terminal but
        neither merged nor failed.
        """
        with self._state_lock:
            self._finish_pr(pr_key)
            self.prs_pending += 1
        self._refresh_display()

    def merge_unsettled(self, pr_key: str | None = None) -> None:
        """Record a PR whose refusal stopped applying too late.

        Nothing merged it and nothing is wrong with it: whatever GitHub
        refused the merge over has since cleared.  Counted apart from
        ``merge_failure`` so the display does not send an operator
        looking for a cause that no longer exists, and apart from
        ``merge_pending`` because no auto-merge is armed --- this one
        needs another run.
        """
        with self._state_lock:
            self._finish_pr(pr_key)
            self.prs_unsettled += 1
        self._refresh_display()

    def increment_closed(self, pr_key: str | None = None) -> None:
        """Record a successful close."""
        with self._state_lock:
            self._finish_pr(pr_key)
            self.prs_closed += 1
        self._refresh_display()

    #: Terminal status value -> the counter that records it.  Used to
    #: move a PR between counters when a late reconciliation changes its
    #: outcome after the per-PR accounting has already run.
    _COUNTER_FIELDS = {
        "merged": "prs_merged",
        "failed": "prs_failed",
        "skipped": "prs_skipped",
        "blocked": "prs_blocked",
        "auto_merge_pending": "prs_pending",
        "unsettled": "prs_unsettled",
        "closed": "prs_closed",
    }

    def reclassify_outcome(self, from_value: str, to_value: str) -> None:
        """Move one PR between terminal counters.

        The end-of-run reconciliation can find that a failure recorded
        early in the run has since cleared.  Its counter was incremented
        when the PR finished, so correcting the outcome means correcting
        the tally too --- otherwise the live display and the closing
        summary disagree about the same run.

        ``completed_prs`` is deliberately untouched: the PR finished
        once, and only *which* counter holds it has changed.
        """
        source = self._COUNTER_FIELDS.get(from_value)
        target = self._COUNTER_FIELDS.get(to_value)
        with self._state_lock:
            if source is not None and getattr(self, source, 0) > 0:
                setattr(self, source, getattr(self, source) - 1)
            if target is not None:
                setattr(self, target, getattr(self, target) + 1)
        self._refresh_display()

    def pr_completed(self) -> None:
        """Record a PR as processed without changing status counters.

        Use this for terminal outcomes that bypass the dedicated
        counter methods.  Those methods already increment
        ``completed_prs``; this one exists solely to keep the
        progress percentage accurate for terminal states that no
        counter method covers.
        """
        with self._state_lock:
            if self.total_prs > 0:
                self.completed_prs += 1
        self._refresh_display()

    def _generate_display_text(self) -> Any:
        """Generate merge-specific display text."""
        return generate_merge_display_text(self)

    def get_summary(self) -> dict[str, Any]:
        """Get merge-specific summary."""
        base = super().get_summary()
        base.update(
            {
                "similar_prs_found": self.similar_prs_found,
                "prs_merged": self.prs_merged,
                "prs_failed": self.prs_failed,
                "prs_skipped": self.prs_skipped,
                "prs_blocked": self.prs_blocked,
                "prs_pending": self.prs_pending,
                "prs_unsettled": self.prs_unsettled,
                "prs_closed": self.prs_closed,
                "rebases_triggered": self.rebases_triggered,
                "retriggers_issued": self.retriggers_issued,
                "total_prs": self.total_prs,
                "completed_prs": self.completed_prs,
            }
        )
        return base
