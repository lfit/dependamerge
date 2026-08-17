# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Rendering of the merge-phase live display.

``MergeProgressTracker`` tracks three families of counter and paints
them as a single Rich ``Text`` block.  The painting is a self-contained
read-only view over the tracker, so it lives here as a handful of
focused helpers rather than one long method on the tracker itself.

Every lock-guarded counter is read once, up front, into
:class:`_MergeCounters`; the helpers below then render from that
snapshot so a repaint can never observe a half-updated tracker.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple

from .rich_compat import Text

if TYPE_CHECKING:
    from .merge import MergeProgressTracker


class _MergeCounters(NamedTuple):
    """A consistent, single-acquisition snapshot of the merge counters."""

    total_prs: int
    completed_prs: int
    similar_prs_found: int
    prs_merged: int
    prs_pending: int
    prs_closed: int
    prs_failed: int
    prs_skipped: int
    prs_blocked: int
    rebases_triggered: int
    retriggers_issued: int
    pr_states: list[str]


def _snapshot_counters(tracker: MergeProgressTracker) -> _MergeCounters:
    """Read every counter guarded by ``_state_lock`` in one acquisition.

    Rendering runs on Rich's refresh thread while worker threads mutate
    these fields, so reading them individually could otherwise observe a
    mixed snapshot (or, worst case, see ``total_prs`` flip to 0 between
    the guard and the division and raise ``ZeroDivisionError``).
    """
    with tracker._state_lock:
        return _MergeCounters(
            total_prs=tracker.total_prs,
            completed_prs=tracker.completed_prs,
            similar_prs_found=tracker.similar_prs_found,
            prs_merged=tracker.prs_merged,
            prs_pending=tracker.prs_pending,
            prs_closed=tracker.prs_closed,
            prs_failed=tracker.prs_failed,
            prs_skipped=tracker.prs_skipped,
            prs_blocked=tracker.prs_blocked,
            rebases_triggered=tracker.rebases_triggered,
            retriggers_issued=tracker.retriggers_issued,
            pr_states=list(tracker._pr_states.values()),
        )


def _append_heading(
    text: Any,
    tracker: MergeProgressTracker,
    counters: _MergeCounters,
    label: str,
) -> None:
    """Append the main progress line for merge/close operations.

    PR-level progress takes priority over repo-level progress so
    repo-scoped merges show "3/9 PRs" instead of "0/1 repos".
    """
    if counters.total_prs > 0:
        progress_pct = (counters.completed_prs / counters.total_prs) * 100
        default_icon = "🚪" if tracker.is_close_operation else "🔀"
        icon = tracker._custom_icon or default_icon
        text.append(f"{icon} {label} in ", style="bold blue")
        text.append(f"{tracker.organization} ", style="bold cyan")
        text.append(
            f"({counters.completed_prs}/{counters.total_prs} {tracker.unit_label}, ",
            style="dim",
        )
        text.append(f"{progress_pct:.0f}%", style="bold green")
        text.append(")", style="dim")
    elif tracker.total_repositories > 0:
        progress_pct = (
            tracker.completed_repositories / tracker.total_repositories
        ) * 100
        default_icon = "🚪" if tracker.is_close_operation else "🔀"
        icon = tracker._custom_icon or default_icon
        text.append(f"{icon} {label} in ", style="bold blue")
        text.append(f"{tracker.organization} ", style="bold cyan")
        text.append(
            f"({tracker.completed_repositories}/{tracker.total_repositories} repos, ",
            style="dim",
        )
        text.append(f"{progress_pct:.0f}%", style="bold green")
        text.append(")", style="dim")
    else:
        default_icon = "🚪" if tracker.is_close_operation else "🔍"
        icon = tracker._custom_icon or default_icon
        text.append(f"{icon} {label} in ", style="bold blue")
        text.append(f"{tracker.organization} ", style="bold cyan")


def _build_stats_parts(
    tracker: MergeProgressTracker, counters: _MergeCounters
) -> list[str]:
    """Build the merge stats segments in pipeline order.

    Transitory pipeline states come first (in flow order), then the
    cumulative activity totals, then terminal outcomes.
    """
    stats_parts: list[str] = []
    if counters.similar_prs_found > 0:
        stats_parts.append(f"🔁 Similar: {counters.similar_prs_found}")
    state_counts: dict[str, int] = {}
    for pr_state in counters.pr_states:
        state_counts[pr_state] = state_counts.get(pr_state, 0) + 1
    for state_key, label in tracker._STATE_ORDER:
        count = state_counts.get(state_key, 0)
        if count > 0:
            stats_parts.append(f"{label}: {count}")
    # Defensive: render unknown states too (sorted for stable
    # output) so a new caller-supplied state is never silently
    # dropped from the display.
    known_states = {key for key, _ in tracker._STATE_ORDER}
    for state_key in sorted(state_counts):
        if state_key not in known_states:
            stats_parts.append(f"{state_key.capitalize()}: {state_counts[state_key]}")
    # Cumulative activity totals.  Rendered between the transitory
    # states and the terminal outcomes so the line reads in
    # pipeline order, and kept visible for the rest of the run
    # once non-zero.
    if counters.rebases_triggered > 0:
        stats_parts.append(f"⬆️ Rebased: {counters.rebases_triggered}")
    if counters.retriggers_issued > 0:
        stats_parts.append(f"📣 Retriggered: {counters.retriggers_issued}")
    if counters.prs_merged > 0:
        # A preview run merges nothing — the counter records PRs
        # judged mergeable, so label it accordingly.
        merged_label = "✅ Mergeable" if tracker.preview else "✅ Merged"
        stats_parts.append(f"{merged_label}: {counters.prs_merged}")
    if counters.prs_pending > 0:
        stats_parts.append(f"\U0001f916 Pending: {counters.prs_pending}")
    if counters.prs_closed > 0:
        stats_parts.append(f"\U0001f6aa Closed: {counters.prs_closed}")
    if counters.prs_failed > 0:
        failed_label = "❌ Would fail" if tracker.preview else "❌ Failed"
        stats_parts.append(f"{failed_label}: {counters.prs_failed}")
    if counters.prs_skipped > 0:
        stats_parts.append(f"⏭️ Skipped: {counters.prs_skipped}")
    if counters.prs_blocked > 0:
        stats_parts.append(f"🛑 Blocked: {counters.prs_blocked}")
    return stats_parts


def _append_status_lines(text: Any, tracker: MergeProgressTracker) -> None:
    """Append the metrics, error, rate-limit and elapsed-time lines."""
    # Metrics line
    if tracker.metrics_concurrency is not None or tracker.metrics_rps is not None:
        parts: list[str] = []
        if tracker.metrics_concurrency is not None:
            parts.append(f"concurrency={tracker.metrics_concurrency}")
        if tracker.metrics_rps is not None:
            parts.append(f"rps={tracker.metrics_rps:.1f}")
        text.append(f"\n   ⚡ {', '.join(parts)}", style="dim")

    # Error count
    if tracker.errors_count > 0:
        text.append(f"\n   ❌ Errors: {tracker.errors_count}", style="bold red")

    # Rate limit indicator
    if tracker.rate_limited:
        text.append("\n   ⏳ Rate limited", style="bold yellow")

    # Elapsed time
    elapsed = datetime.now() - tracker.start_time
    text.append(f"\n   ⏱️ Elapsed: {tracker._format_duration(elapsed)}", style="dim")


def generate_merge_display_text(tracker: MergeProgressTracker) -> Any:
    """Generate merge-specific display text."""
    if not tracker.rich_available:
        return Text()

    text = Text()

    # Resolve label and icon — use custom values when provided,
    # otherwise fall back to the default close/merge text.
    default_label = (
        "Closing PRs" if tracker.is_close_operation else "Searching for similar PRs"
    )
    label = tracker._custom_label or default_label

    counters = _snapshot_counters(tracker)
    _append_heading(text, tracker, counters, label)

    # Current operation
    if tracker.current_operation:
        text.append(f"\n   {tracker.current_operation}", style="dim")

    # Merge stats — the per-PR states and counters were snapshotted
    # under ``_state_lock`` above.
    stats_parts = _build_stats_parts(tracker, counters)
    if stats_parts:
        text.append(f"\n   {' | '.join(stats_parts)}", style="dim")

    _append_status_lines(text, tracker)

    return text
