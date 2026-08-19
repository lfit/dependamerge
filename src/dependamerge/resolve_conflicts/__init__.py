# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
resolve_conflicts: Orchestrate interactive rebase flows to fix simple PR merge conflicts.

This package provides:
- Data models for selecting PRs to fix and controlling behavior
- A FixOrchestrator that:
  * Fetches PR details (head/base repo/branches, fork status, permissions)
  * Prepares secure temporary workspaces and clones/fetches repos
  * Runs an interactive rebase flow (manual resolution via user's editor/mergetool)
  * Amends commit when appropriate and force-pushes the updated branch
  * Cleans up temp workspaces by default (unless keep_temp is requested)
- An InteractiveResolver that guides a user through conflict resolution loops

The orchestrator design allows swapping the resolver for a future automated variant
that can run in parallel. The current interactive flow runs one PR at a time to keep
terminal interaction clean.
"""

from __future__ import annotations

from .models import (
    FixOptions,
    FixResult,
    PRContext,
    PRSelection,
)
from .orchestrator import FixOrchestrator
from .preparation import (
    _LOG,
)
from .resolver import InteractiveResolver

__all__ = [
    "FixOptions",
    "FixOrchestrator",
    "FixResult",
    "InteractiveResolver",
    "PRContext",
    "PRSelection",
]
