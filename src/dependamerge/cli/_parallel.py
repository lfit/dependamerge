# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Parallel merge execution and the single-PR preview/confirm flow.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING

from ..merge_manager import (
    AsyncMergeManager,
    MergeResult,
)
from ..models import ComparisonResult, PullRequestInfo
from ..progress_tracker import MergeProgressTracker

if TYPE_CHECKING:
    pass

from ._app import MAX_RETRIES, console
from ._context import (
    _MergeContext,
)
from ._results import (
    _print_failed_pr_details,
    _print_final_merge_summary,
)
from ._sha import (
    _generate_continue_sha,
)


def _run_parallel_merge(
    ctx: _MergeContext,
    prs_to_merge: list[tuple[PullRequestInfo, ComparisonResult | None]],
    preview: bool,
    concurrency: int = 10,
    leading_blank: bool = True,
    repo_scoped: bool = False,
    stripe: bool = False,
) -> list[MergeResult]:
    """Execute a parallel merge (preview or real) and return results.

    Args:
        ctx: Shared merge context.
        prs_to_merge: PRs to process.
        preview: If True, run in preview mode without side effects.
        concurrency: Maximum number of concurrent merge workers.
            For org-wide merges (PRs spread across repos) the default
            of 10 is fine.  For repo-scoped merges (all PRs in the
            same repo) ``AsyncMergeManager`` serialises the actual
            ``merge_pull_request`` API call via a per-repo dispatch
            lock, so a value > 1 is safe and lets the worker pool
            keep processing other PRs while one PR sits in Step 5.5's
            auto-merge wait loop — see ``_get_merge_dispatch_lock``.
        leading_blank: When True (default), the ``🚀 Merging ...``
            banner is preceded by a blank line to separate it from a
            preceding similar-PR list.  Callers pass False when no
            list was printed (e.g. no similar PRs were found) so the
            output stays compact.
        repo_scoped: When True, each worker refreshes its PR's live
            merge state just before dispatching the merge, because a
            sibling merge in the same repository can make a PR ``dirty``
            / ``behind`` mid-batch.  Enabled for both repo-scoped merges
            (all PRs in one repository) and owner-wide striped merges
            (which interleave repositories but serialise PRs within each
            repository).  See
            ``AsyncMergeManager._refresh_pr_mergeability``.
        stripe: When True, the merge is scheduled with the striped
            scheduler (one serial worker per repository, distinct repos
            concurrent) so owner-wide batches never stack two merges on
            the same repository at once.  See
            ``AsyncMergeManager._run_striped``.
    """

    async def _do_merge():
        async with AsyncMergeManager(
            token=ctx.token,  # pyright: ignore[reportArgumentType]
            merge_method=ctx.merge_method,
            max_retries=MAX_RETRIES,
            concurrency=concurrency,
            fix_out_of_date=not ctx.no_fix,
            fix_semantic_title=ctx.fix_semantic_title,
            merge_timeout=ctx.merge_timeout,
            progress_tracker=ctx.progress_tracker,
            preview_mode=preview,
            dismiss_copilot=ctx.dismiss_copilot,
            force_level=ctx.force,
            github2gerrit_mode=ctx.github2gerrit_mode,
            no_netrc=ctx.no_netrc,
            netrc_file=ctx.netrc_file,
            rebase_local=ctx.rebase_local,
            repo_scoped=repo_scoped,
            # The owner-wide global wait ceiling applies to striped
            # (owner/user-wide) runs; single-PR and single-repository
            # merges keep the uncapped per-PR ``merge_timeout`` behaviour.
            max_wait=ctx.max_wait if stripe else None,
        ) as merge_manager:
            if not preview:
                prefix = "\n" if leading_blank else ""
                console.print(
                    f"{prefix}🚀 Merging {len(prs_to_merge)} pull requests..."
                )
            return await merge_manager.merge_prs_parallel(prs_to_merge, stripe=stripe)

    return asyncio.run(_do_merge())


def _restart_merge_progress_tracker(ctx: _MergeContext, total_prs: int) -> None:
    """Stand up a fresh, started progress tracker for the merge phase.

    The org-wide scan (``_scan_and_find_similar``) calls ``stop()`` on
    the tracker it used, tearing down its Rich ``Live`` and leaving
    ``ctx.progress_tracker`` in a stopped state.  Reusing that stopped
    tracker for the real merge means the background wait-status ticker
    pushes ``update_operation`` calls into a dead ``Live`` — silently
    dropped by ``ProgressTracker._refresh_display`` — so the user sees
    no countdown while PRs sit in the Step 5.5 auto-merge wait.

    Mirror the repo-scoped path (``_handle_repo_merge`` /
    ``_execute_repo_confirmed_merge``) by replacing the tracker with a
    fresh one dedicated to the merge phase and starting it.  No-op when
    progress display is disabled (``--no-progress``), where the plain
    text ticker already provides feedback.
    """
    if not ctx.show_progress:
        return
    ctx.progress_tracker = MergeProgressTracker(
        ctx.owner,
        operation_label="Merging PRs",
        operation_icon="▶️",
    )
    ctx.progress_tracker.set_total_prs(total_prs)
    ctx.progress_tracker.start()


def _handle_preview_confirmation(
    ctx: _MergeContext,
    merge_results: list[MergeResult],
    all_prs_to_merge: list[tuple[PullRequestInfo, ComparisonResult | None]],
    merged_count: int,
    total_to_merge: int,
) -> None:
    """Handle the interactive preview-then-confirm flow.

    Prompts the user for a continuation SHA and, if confirmed,
    executes the real merge.
    """
    assert ctx.github_client is not None
    assert ctx.source_pr is not None

    console.print(f"\nMergeable {merged_count}/{total_to_merge} PRs")
    # Per-PR preview lines are no longer printed during the run, so
    # report the not-mergeable PRs (and why) here before prompting.
    _print_failed_pr_details(merge_results)

    if merged_count == 0:
        console.print("\n\U0001f4a1 No PRs are mergeable at this time.")
        return

    commit_messages = ctx.github_client.get_pull_request_commits(
        ctx.owner, ctx.repo_name, ctx.pr_number
    )
    first_commit_line = commit_messages[0].split("\n")[0] if commit_messages else ""
    continue_sha_hash = _generate_continue_sha(ctx.source_pr, first_commit_line)
    console.print()
    console.print(f"To proceed with merging enter: {continue_sha_hash}")

    try:
        if "pytest" in sys.modules or os.getenv("TESTING"):
            console.print("⚠️ Test mode detected - skipping interactive prompt")
            return

        user_input = input(
            "Enter the string above to continue (or press Enter to cancel): "
        ).strip()

        if user_input == continue_sha_hash:
            _execute_confirmed_merge(ctx, merge_results, all_prs_to_merge)
        elif user_input == "":
            console.print("❌ Merge cancelled by user.")
        else:
            console.print("❌ Invalid input. Merge cancelled.")
    except KeyboardInterrupt:
        console.print("\n❌ Merge cancelled by user.")
    except EOFError:
        console.print("\n❌ Merge cancelled.")


def _execute_confirmed_merge(
    ctx: _MergeContext,
    preview_results: list[MergeResult],
    all_prs_to_merge: list[tuple[PullRequestInfo, ComparisonResult | None]],
) -> None:
    """Run the real merge after user confirmation."""
    mergeable_prs = [
        all_prs_to_merge[i]
        for i, result in enumerate(preview_results)
        if result.status.value == "merged"
    ]
    merged_count = len(mergeable_prs)

    _restart_merge_progress_tracker(ctx, merged_count)
    try:
        real_results = _run_parallel_merge(ctx, mergeable_prs, preview=False)
    finally:
        if ctx.show_progress and ctx.progress_tracker:
            ctx.progress_tracker.stop()

    _print_final_merge_summary(real_results)
