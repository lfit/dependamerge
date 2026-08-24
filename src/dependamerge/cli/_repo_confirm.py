# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Preview-then-confirm handoff for repository-scoped merges.

A repository-scoped run evaluates every open PR before merging any of
them, then asks the operator to echo a token derived from the repository
and the mergeable count.  Only the PRs that the evaluation pass judged
mergeable are re-run for real.
"""

import hashlib

import typer

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
import dependamerge.cli as _pkg

from ..merge_manager import (
    MergeResult,
)
from ..models import ComparisonResult, PullRequestInfo
from ..progress_tracker import MergeProgressTracker
from ._app import console
from ._context import _MergeContext
from ._merge_report import (
    _print_failed_pr_details,
    _print_final_merge_summary,
)


def _handle_repo_preview_confirmation(
    ctx: _MergeContext,
    merge_results: list[MergeResult],
    all_prs_to_merge: list[tuple[PullRequestInfo, ComparisonResult | None]],
    merged_count: int,
    total_to_merge: int,
) -> None:
    """Handle preview-then-confirm for repository-scoped merges.

    Similar to _handle_preview_confirmation but does not require a
    source PR for SHA generation — it uses the repository name instead.
    """
    console.print(f"\nMergeable {merged_count}/{total_to_merge} PRs")
    # Per-PR preview lines are no longer printed during the run, so
    # report the not-mergeable PRs (and why) here before prompting.
    _print_failed_pr_details(merge_results)

    if merged_count == 0:
        console.print("\n\U0001f4a1 No PRs are mergeable at this time.")
        return

    # Generate a confirmation token from the repo context
    combined = f"repo-merge:{ctx.owner}/{ctx.repo_name}:{merged_count}"
    confirm_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    console.print()
    console.print(f"To proceed with merging enter: {confirm_hash}")

    try:
        user_input = typer.prompt(
            "Enter the string above to continue (or press Enter to cancel)",
            default="",
            show_default=False,
        ).strip()

        if user_input == confirm_hash:
            _execute_repo_confirmed_merge(ctx, merge_results, all_prs_to_merge)
        elif user_input == "":
            console.print("❌ Merge cancelled by user.")
        else:
            console.print("❌ Invalid input. Merge cancelled.")
    except (KeyboardInterrupt, EOFError, typer.Abort):
        console.print("\n❌ Merge cancelled by user.")


def _execute_repo_confirmed_merge(
    ctx: _MergeContext,
    preview_results: list[MergeResult],
    all_prs_to_merge: list[tuple[PullRequestInfo, ComparisonResult | None]],
) -> None:
    """Run the real merge after user confirmation (repo mode)."""
    mergeable_prs = [
        all_prs_to_merge[i]
        for i, result in enumerate(preview_results)
        if result.status.value == "merged"
    ]

    if ctx.show_progress:
        ctx.progress_tracker = MergeProgressTracker(
            f"{ctx.owner}/{ctx.repo_name}",
            operation_label="Merging PRs",
            operation_icon="▶️",
        )
        ctx.progress_tracker.set_total_prs(len(mergeable_prs))
        ctx.progress_tracker.start()

    try:
        real_results = _pkg._run_parallel_merge(
            ctx,
            mergeable_prs,
            preview=False,
            # Per-repo merge dispatch lock makes parallel workers
            # safe; cap by PR count.
            concurrency=min(5, len(mergeable_prs)) or 1,
            # Single-repo batch — refresh live merge state before each
            # dispatch (a sibling merge can introduce a conflict).
            repo_scoped=True,
        )
    finally:
        if ctx.show_progress and ctx.progress_tracker:
            ctx.progress_tracker.stop()

    _print_final_merge_summary(real_results)
