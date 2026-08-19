# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Real-time progress tracking for long-running dependamerge operations.

Three trackers share one surface:

- ``ProgressTracker`` — the scan-phase tracker used by the
  organization-wide ``blocked`` / ``status`` scans and the similar-PR
  search.
- ``MergeProgressTracker`` — adds the merge/close counters (transitory
  per-PR states, cumulative activity totals, terminal outcomes).
- ``DummyProgressTracker`` — a no-op stand-in for either of the above
  when progress display is disabled.

``Live``, ``Text``, ``Console`` and ``RICH_AVAILABLE`` are re-exported
here because they are part of this package's historical module surface.
They are not the substitution point: ``scan.py`` binds ``Live`` from
``rich_compat``, so a call site resolves it from that module rather than
from this one. Tests that swap the class patch
``dependamerge.progress_tracker.scan.Live``.
"""

from __future__ import annotations

from .dummy import DummyProgressTracker
from .merge import MergeProgressTracker
from .merge_display import generate_merge_display_text
from .rich_compat import RICH_AVAILABLE, Console, Live, Text
from .scan import ProgressTracker
from .terminal import _TerminalLoggingMixin

__all__ = [
    "RICH_AVAILABLE",
    "Console",
    "DummyProgressTracker",
    "Live",
    "MergeProgressTracker",
    "ProgressTracker",
    "Text",
    "_TerminalLoggingMixin",
    "generate_merge_display_text",
]
