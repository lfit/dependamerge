# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The ``blocked --fix`` interactive rebase workflow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..error_codes import (
    ExitCode,
    exit_for_configuration_error,
    exit_with_error,
)
from ..progress_tracker import ProgressTracker
from ..resolve_conflicts import FixOptions, FixOrchestrator, PRSelection
from ..system_utils import get_default_workers
from ._app import console


@dataclass(frozen=True)
class _FixRequest:
    """Workspace and editor knobs for the ``--fix`` rebase workflow."""

    workdir: str | None
    keep_temp: bool
    prefetch: int | None
    editor: str | None
    mergetool: bool
    interactive: bool


def _run_blocked_fix(
    scan_result,
    token: str | None,
    progress_tracker: ProgressTracker | None,
    reason: str | None,
    limit: int | None,
    request: _FixRequest,
) -> None:
    """Rebase the selected blocked PRs interactively."""
    allowed_default = {"merge_conflict", "behind_base"}
    reasons_to_attempt = allowed_default if not reason else {reason.strip().lower()}

    selections: list[PRSelection] = []
    for pr in scan_result.unmergeable_prs:
        pr_reason_types = {r.type for r in pr.reasons}
        if pr_reason_types & reasons_to_attempt:
            selections.append(
                PRSelection(repository=pr.repository, pr_number=pr.pr_number)
            )

    if limit is not None and limit > 0:
        selections = selections[:limit]

    if not selections:
        console.print("No eligible PRs to fix based on the selected reasons.")
        return

    token_to_use = token or os.getenv("GITHUB_TOKEN")
    if not token_to_use:
        exit_for_configuration_error(
            message="❌ GitHub token required for --fix option",
            details="Provide --token or set GITHUB_TOKEN environment variable",
        )

    console.print(f"Starting interactive fix for {len(selections)} PR(s)...")
    try:
        orchestrator = FixOrchestrator(
            token_to_use,
            progress_tracker=progress_tracker,
            logger=lambda m: console.print(m),
        )
        fix_options = FixOptions(
            workdir=request.workdir,
            keep_temp=request.keep_temp,
            prefetch=request.prefetch
            if request.prefetch is not None
            else get_default_workers(),
            editor=request.editor,
            mergetool=request.mergetool,
            interactive=request.interactive,
            logger=lambda m: console.print(m),
        )
        results = orchestrator.run(selections, fix_options)
        success_count = sum(1 for r in results if r.success)
        console.print(f"✅ Fix complete: {success_count}/{len(selections)} succeeded")
    except Exception as e:
        exit_with_error(
            ExitCode.GENERAL_ERROR,
            message="❌ Error during fix workflow",
            details=str(e),
            exception=e,
        )
