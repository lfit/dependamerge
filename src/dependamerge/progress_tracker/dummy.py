# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
No-op progress tracker.

``DummyProgressTracker`` stands in for either real tracker when
progress display is disabled, so callers never have to guard their
tracker calls.  It mirrors the full surface of both trackers and does
nothing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .scan import ProgressTracker


class DummyProgressTracker(ProgressTracker):
    """A no-op progress tracker for when progress display is disabled."""

    def __init__(self) -> None:
        # Initialize the base tracker, then neutralize Rich so this
        # stand-in performs no terminal output.
        super().__init__("", show_pr_stats=False)
        self.console = None
        self.rich_available = False
        self.current_operation = ""
        # MergeProgressTracker fields (Dummy stands in for either tracker)
        self.similar_prs_found = 0
        self.prs_merged = 0
        self.prs_failed = 0
        self.prs_skipped = 0
        self.prs_blocked = 0
        self.prs_pending = 0
        self.prs_unsettled = 0
        self.prs_closed = 0
        self.rebases_triggered = 0
        self.retriggers_issued = 0
        self.is_close_operation = False
        self.preview = False
        self._custom_label: str | None = None
        self._custom_icon: str | None = None
        self.unit_label = "PRs"
        self.total_prs = 0
        self.completed_prs = 0
        self._pr_states: dict[str, str] = {}

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def update_total_repositories(self, total: int) -> None:
        pass

    def start_repository(self, repo_name: str) -> None:
        pass

    def complete_repository(self, unmergeable_count: int = 0) -> None:
        pass

    def update_operation(self, operation: str) -> None:
        pass

    def analyze_pr(self, pr_number: int, repo_name: str = "") -> None:
        pass

    def add_error(self) -> None:
        pass

    def set_rate_limited(self, reset_time: datetime | None = None) -> None:
        pass

    def clear_rate_limited(self) -> None:
        pass

    def set_total_prs(self, total: int) -> None:
        pass

    def pr_completed(self) -> None:
        pass

    def found_similar_pr(self, count: int = 1) -> None:
        pass

    def track_pr_state(self, pr_key: str, state: str | None) -> None:
        pass

    def record_rebase(self, count: int = 1) -> None:
        pass

    def record_retrigger(self, count: int = 1) -> None:
        pass

    def merge_success(self, pr_key: str | None = None) -> None:
        pass

    def merge_failure(self, pr_key: str | None = None) -> None:
        pass

    def merge_skipped(self, pr_key: str | None = None) -> None:
        pass

    def merge_blocked(self, pr_key: str | None = None) -> None:
        pass

    def merge_pending(self, pr_key: str | None = None) -> None:
        pass

    def merge_unsettled(self, pr_key: str | None = None) -> None:
        pass

    def reclassify_outcome(self, from_value: str, to_value: str) -> None:
        pass

    def increment_closed(self, pr_key: str | None = None) -> None:
        pass

    def _refresh_display(self) -> None:
        pass

    def _fallback_display(self) -> None:
        pass

    def get_summary(self) -> dict[str, Any]:
        return {}
