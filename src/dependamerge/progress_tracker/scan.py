# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Scan-phase progress tracker.

``ProgressTracker`` drives the live display for the long-running
organization-wide scans (``blocked``, ``status`` and the similar-PR
search), and is the base class every other tracker in this package
extends.  Terminal detection and logging quieting come from
``_TerminalLoggingMixin``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from .rich_compat import RICH_AVAILABLE, Console, Live, Text
from .terminal import _TerminalLoggingMixin


class ProgressTracker(_TerminalLoggingMixin):
    """Real-time progress tracker for organization blocked PR checking operations."""

    def __init__(self, organization: str, show_pr_stats: bool = True):
        """Initialize progress tracker for an organization blocked PR check.

        Args:
            organization: Name of the GitHub organization being checked
            show_pr_stats: Whether to show PR analysis statistics (default True)
        """
        self.organization = organization
        self.start_time = datetime.now()
        self.console: Any = Console() if RICH_AVAILABLE else None

        # Progress counters
        self.total_repositories = 0
        self.completed_repositories = 0
        self.current_repository = ""
        self.total_prs_analyzed = 0
        self.unmergeable_prs_found = 0
        self.current_operation = "Initializing..."
        self.errors_count = 0

        # Configuration
        self.show_pr_stats = show_pr_stats

        # Rate limiting tracking
        self.rate_limited = False
        self.rate_limit_reset_time: datetime | None = None

        # Rich Live display
        self.live: Any = None
        self.rich_available = RICH_AVAILABLE
        self.paused = False
        # Metrics (optional; displayed when provided)
        self.metrics_concurrency: int | None = None
        self.metrics_rps: float | None = None

        # Fallback for when Rich is not available
        self._last_display = ""
        # Terminal-bound logging handlers silenced while the live
        # display is on screen; each entry is ``(handler, saved_level)``
        # so the original level can be restored on stop.
        self._quieted_handlers: list[tuple[logging.Handler, int]] = []

    def start(self) -> None:
        """Start the live progress display."""
        if not self.rich_available:
            return

        try:
            # Pass a callable rather than a static renderable: Rich
            # re-invokes it on every auto-refresh tick, so the elapsed
            # clock keeps advancing even when no progress events fire
            # (long silent API sequences used to freeze the display).
            self.live = Live(
                get_renderable=self._generate_display_text,
                console=self.console,
                refresh_per_second=2,
                transient=False,
            )
            if self.live:
                # Quiet terminal logging *before* starting Live: if quieting
                # raised after the start, the ``except`` path would drop the
                # reference without stopping the already-started display,
                # leaving it orphaned. Ordering it first also closes the
                # window where a log could slip past Rich right after start.
                self._quiet_terminal_logging()
                self.live.start()
        except Exception:
            # Fallback if Rich display fails (e.g., unsupported terminal)
            self._restore_terminal_logging()
            self.live = None
            self.rich_available = False

    def stop(self) -> None:
        """Stop the live progress display."""
        # Stop the live display *before* restoring terminal logging so the
        # teardown itself stays quiet even if ``live.stop()`` raises;
        # restore always runs via ``finally``.
        try:
            if self.live:
                try:
                    self.live.stop()
                except Exception:
                    # Best-effort teardown: ignore errors from Rich when
                    # the terminal no longer accepts control sequences.
                    pass
            else:
                # Non-Rich fallback: emit a final newline so the shell
                # prompt doesn't appear mid-line after carriage-return
                # in-place updates.
                if self._last_display and self._stdout_is_tty():
                    # aislop-ignore-next-line ai-slop/python-print-debug -- terminal newline on teardown
                    print(flush=True)
        finally:
            self._restore_terminal_logging()
        self.live = None
        self.paused = False

    def suspend(self) -> None:
        """Temporarily suspend the live display (e.g. for interactive prompts)."""
        if self.live:
            # Stop the live display first, then restore logging in
            # ``finally`` so teardown stays quiet even if ``live.stop()``
            # raises and handlers are always restored for the prompt.
            try:
                self.live.stop()
            except Exception:
                # Best-effort suspend: ignore Rich teardown errors so an
                # interactive prompt can still take over the terminal.
                pass
            finally:
                self._restore_terminal_logging()
            self.paused = True

    def resume(self) -> None:
        """Resume the live display after it was suspended."""
        if self.rich_available and self.paused:
            try:
                self.live = Live(
                    get_renderable=self._generate_display_text,
                    console=self.console,
                    refresh_per_second=2,
                    transient=False,
                )
                if self.live:
                    # Quiet before starting Live for the same reason as
                    # ``start()``: avoid orphaning a started display if
                    # quieting raises, and close the log-slip window.
                    self._quiet_terminal_logging()
                    self.live.start()
            except Exception:
                self._restore_terminal_logging()
                self.live = None
                self.rich_available = False
            self.paused = False

    def update_metrics(self, concurrency: int, rps: float) -> None:
        self.metrics_concurrency = concurrency
        self.metrics_rps = rps
        self._refresh_display()

    def clear_metrics(self) -> None:
        self.metrics_concurrency = None
        self.metrics_rps = None
        self._refresh_display()

    def update_total_repositories(self, total: int) -> None:
        """Update the total number of repositories to scan."""
        self.total_repositories = total
        self._refresh_display()

    def start_repository(self, repo_name: str) -> None:
        """Mark the start of scanning a repository."""
        self.current_repository = repo_name
        self.current_operation = f"Scanning {repo_name}..."
        self._refresh_display()

    def complete_repository(self, unmergeable_count: int = 0) -> None:
        """Mark completion of a repository check."""
        self.completed_repositories += 1
        self.unmergeable_prs_found += unmergeable_count
        self.current_operation = ""
        self.current_repository = ""
        self._refresh_display()

    def update_operation(self, operation: str) -> None:
        """Update the current operation description."""
        self.current_operation = operation
        self._refresh_display()

    def analyze_pr(self, pr_number: int, repo_name: str = "") -> None:
        """Mark the start of analyzing a specific PR."""
        self.total_prs_analyzed += 1
        if repo_name:
            self.current_operation = f"Analyzing PR #{pr_number} in {repo_name}"
        else:
            self.current_operation = f"Analyzing PR #{pr_number}..."
        self._refresh_display()

    def add_error(self) -> None:
        """Increment the error counter."""
        self.errors_count += 1
        self._refresh_display()

    def set_rate_limited(self, reset_time: datetime | None = None) -> None:
        """Mark that we're rate limited."""
        self.rate_limited = True
        self.rate_limit_reset_time = reset_time
        self._refresh_display()

    def clear_rate_limited(self) -> None:
        """Clear rate limit state."""
        self.rate_limited = False
        self.rate_limit_reset_time = None
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Repaint the live display with current progress.

        The Live instance renders via ``get_renderable``, so a plain
        ``refresh()`` repaints with current state immediately instead
        of waiting for the next auto-refresh tick.
        """
        if self.live and self.rich_available and not self.paused:
            try:
                self.live.refresh()
            except Exception:
                # If Rich display fails, fall back to simple print
                self._fallback_display()
        elif not self.rich_available:
            self._fallback_display()

    def _generate_display_text(self) -> Any:
        """Generate the current progress display text."""
        if not self.rich_available:
            return Text()

        text = Text()

        # Main progress line
        if self.total_repositories > 0:
            progress_pct = (self.completed_repositories / self.total_repositories) * 100
            text.append("🔍 Checking ", style="bold blue")
            text.append(f"{self.organization} ", style="bold cyan")
            text.append(
                f"({self.completed_repositories}/{self.total_repositories} repos, ",
                style="dim",
            )
            text.append(f"{progress_pct:.0f}%", style="bold green")
            text.append(")", style="dim")
        else:
            text.append("🔍 Checking ", style="bold blue")
            text.append(f"{self.organization} ", style="bold cyan")
            text.append("(initializing...)", style="dim")

        # Current operation
        if self.current_operation:
            text.append(f"\n   {self.current_operation}", style="dim")

        # Stats line (optional)
        if self.show_pr_stats and self.total_prs_analyzed > 0:
            text.append("\n   📊 PRs analyzed: ", style="dim")
            text.append(str(self.total_prs_analyzed), style="bold")
            if self.unmergeable_prs_found > 0:
                text.append(" | ⚠️ Unmergeable: ", style="dim")
                text.append(str(self.unmergeable_prs_found), style="bold yellow")

        # Metrics line (concurrency / requests-per-second)
        if self.metrics_concurrency is not None or self.metrics_rps is not None:
            parts: list[str] = []
            if self.metrics_concurrency is not None:
                parts.append(f"concurrency={self.metrics_concurrency}")
            if self.metrics_rps is not None:
                parts.append(f"rps={self.metrics_rps:.1f}")
            text.append(f"\n   ⚡ {', '.join(parts)}", style="dim")

        # Error count
        if self.errors_count > 0:
            text.append(f"\n   ❌ Errors: {self.errors_count}", style="bold red")

        # Rate limit indicator
        if self.rate_limited:
            text.append("\n   ⏳ Rate limited", style="bold yellow")
            if self.rate_limit_reset_time:
                remaining = self.rate_limit_reset_time - datetime.now()
                if remaining.total_seconds() > 0:
                    text.append(
                        f" (resets in {self._format_duration(remaining)})",
                        style="yellow",
                    )

        # Elapsed time
        elapsed = datetime.now() - self.start_time
        text.append(f"\n   ⏱️ Elapsed: {self._format_duration(elapsed)}", style="dim")

        return text

    def _fallback_display(self) -> None:
        """Simple text fallback when Rich is not available."""
        if self.total_repositories > 0:
            progress_pct = (self.completed_repositories / self.total_repositories) * 100
            display = (
                f"Progress: {self.completed_repositories}/{self.total_repositories} "
                f"repos ({progress_pct:.0f}%)"
            )
        else:
            display = "Initializing..."

        if self.current_operation:
            display += f" | {self.current_operation}"

        if self.total_prs_analyzed > 0:
            display += f" | PRs: {self.total_prs_analyzed}"

        if self.errors_count > 0:
            display += f" | Errors: {self.errors_count}"

        # Only print if display has changed
        if display != self._last_display:
            if self._stdout_is_tty():
                # \033[K clears from cursor to end-of-line so shorter
                # updates don't leave trailing characters from the
                # previous render.
                # aislop-ignore-next-line ai-slop/python-print-debug -- progress render to stdout
                print(f"\r{display}\033[K", end="", flush=True)
            else:
                # aislop-ignore-next-line ai-slop/python-print-debug -- progress render to stdout
                print(display)
            self._last_display = display

    def _format_duration(self, td: timedelta) -> str:
        """Format a timedelta as a human-readable duration string."""
        total_seconds = int(td.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        if minutes < 60:
            return f"{minutes}m {seconds}s"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m {seconds}s"

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the progress tracking."""
        elapsed = datetime.now() - self.start_time
        formatted = self._format_duration(elapsed)
        return {
            "organization": self.organization,
            "total_repositories": self.total_repositories,
            "completed_repositories": self.completed_repositories,
            "total_prs_analyzed": self.total_prs_analyzed,
            "unmergeable_prs_found": self.unmergeable_prs_found,
            "errors_count": self.errors_count,
            "elapsed_seconds": elapsed.total_seconds(),
            "elapsed_formatted": formatted,
            "elapsed_time": formatted,
        }
