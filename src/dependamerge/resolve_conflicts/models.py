# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Data models for the conflict-resolution fix workflow.

These carry the inputs, options and outcomes that flow between
:class:`~dependamerge.resolve_conflicts.FixOrchestrator` and
:class:`~dependamerge.resolve_conflicts.InteractiveResolver`: which PRs to
fix, how to fix them, everything cloning and rebasing needs to know about a
PR, and what happened to each one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PRSelection:
    """Minimal PR selector: repository 'owner/repo' and PR number."""

    repository: str
    pr_number: int


@dataclass
class FixOptions:
    """
    Options controlling the fix workflow.

    Attributes:
        workdir: Base directory for workspaces. If None, a secure temp directory is created.
        keep_temp: If True, workspaces are not removed on completion (default False).
        prefetch: Number of concurrent workspace preparations (clone/fetch).
        editor: Override command to edit conflicted files. If None, use $VISUAL or $EDITOR.
        mergetool: If True, try 'git mergetool' for conflicts; otherwise open in editor.
        interactive: If True, attach git commands to TTY where useful for user feedback.
        logger: Optional logger callable for informational messages (redacted).
    """

    workdir: str | None = None
    keep_temp: bool = False
    prefetch: int = 6
    editor: str | None = None
    mergetool: bool = False
    interactive: bool = True
    logger: Callable[[str], None] | None = None


@dataclass
class FixResult:
    """Outcome of attempting to fix a single PR."""

    selection: PRSelection
    success: bool
    message: str
    workspace: str | None = None


@dataclass
class PRContext:
    """Detailed PR information required for cloning/rebasing/pushing."""

    owner: str
    repo: str
    pr_number: int
    base_branch: str
    head_branch: str
    base_repo_full_name: str
    base_repo_clone_url: str
    head_repo_full_name: str
    head_repo_clone_url: str
    is_fork: bool
    maintainer_can_modify: bool

    @property
    def selection(self) -> PRSelection:
        return PRSelection(
            repository=f"{self.owner}/{self.repo}", pr_number=self.pr_number
        )
