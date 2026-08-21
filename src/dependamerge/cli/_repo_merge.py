# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Repository-scoped bulk merge: fetch, partition, preview, confirm.
"""

from __future__ import annotations

import asyncio
import hashlib

import typer

from ..bot_identity import is_automation_author
from ..merge_manager import (
    MergeResult,
)
from ..models import ComparisonResult, PullRequestInfo
from ..progress_tracker import MergeProgressTracker
from ..url_parser import (
    ParsedRepoUrl,
)
from . import _deps, _parallel
from ._app import console
from ._context import (
    _MergeContext,
)
from ._permissions import _maybe_check_merge_permissions
from ._results import (
    _display_merge_results,
    _print_failed_pr_details,
    _print_final_merge_summary,
    _repo_merge_order,
)


def _prepare_repo_merge(
    parsed_repo: ParsedRepoUrl,
    ctx: _MergeContext,
) -> None:
    """Bind the client to *ctx* and start the PR-fetch progress tracker."""
    from ..url_parser import _host_matches

    if not _host_matches(parsed_repo.host, "github.com"):
        console.print(
            "❌ Repository-scoped merge is currently only supported "
            f"for github.com (got host: {parsed_repo.host}).\n"
            "   GitHub Enterprise support requires API base URL "
            "configuration — use a direct PR URL instead."
        )
        raise typer.Exit(code=1)

    ctx.github_client = _deps.GitHubClient(ctx.token)
    assert ctx.github_client.token is not None
    ctx.token = ctx.github_client.token
    ctx.owner = parsed_repo.owner
    ctx.repo_name = parsed_repo.repo

    console.print(f"🔍 Repository mode: fetching open PRs in {parsed_repo.project}...")

    _maybe_check_merge_permissions(ctx)

    if ctx.show_progress:
        ctx.progress_tracker = MergeProgressTracker(
            f"{ctx.owner}/{ctx.repo_name}",
            operation_label="Fetching open PRs",
            operation_icon="🔍",
        )
        # Repo-scoped runs operate on a single repository, so the
        # ``X/Y repos`` progress fraction is meaningless.  Skip
        # ``update_total_repositories`` so the tracker falls through
        # to the no-progress display branch and renders cleanly as
        # ``🔍 Fetching open PRs in <owner>/<repo>``.
        ctx.progress_tracker.start()


def _fetch_repo_prs(
    ctx: _MergeContext,
    only_automation: bool,
) -> list[PullRequestInfo]:
    """Fetch the repository's open PRs, stopping the tracker either way."""
    from ..github_service import GitHubService

    async def _fetch_prs() -> list[PullRequestInfo]:
        svc = GitHubService(
            token=ctx.token,
            progress_tracker=ctx.progress_tracker,
        )
        try:
            return await svc.fetch_repo_open_prs(
                ctx.owner,
                ctx.repo_name,
                only_automation=only_automation,
            )
        finally:
            await svc.close()

    try:
        repo_prs = asyncio.run(_fetch_prs())
    except Exception:
        if ctx.progress_tracker:
            ctx.progress_tracker.stop()
        raise

    if ctx.progress_tracker:
        ctx.progress_tracker.stop()

    return repo_prs


def _select_repo_prs(
    parsed_repo: ParsedRepoUrl,
    ctx: _MergeContext,
    repo_prs: list[PullRequestInfo],
) -> list[PullRequestInfo] | None:
    """Report the fetched PRs and settle whether human ones stay in scope.

    Returns the PRs to merge, or ``None`` when the run should stop.
    """
    automation_prs: list[PullRequestInfo] = []
    human_prs: list[PullRequestInfo] = []

    for pr in repo_prs:
        # is_automation_author normalizes REST and GraphQL login forms
        # (e.g. "dependabot[bot]" vs "dependabot") so they classify
        # identically.
        if is_automation_author(pr.author):
            automation_prs.append(pr)
        else:
            human_prs.append(pr)

    console.print(f"\n📊 Found {len(repo_prs)} open PR(s) in {parsed_repo.project}")
    if automation_prs:
        console.print(f"🤖 Automation PRs: {len(automation_prs)}")
    if human_prs:
        console.print(f"👤 Human PRs: {len(human_prs)}")

    # List PRs that will be processed. The trailing ``(by {author})``
    # already reveals whether each PR is automation or human, so a
    # per-row icon would only duplicate the summary counts above.
    for pr in repo_prs:
        console.print(f"  #{pr.number} {pr.title} (by {pr.author})")

    # Only prompt when human PRs are actually in scope, not merely
    # because --include-human-prs was supplied.  A dry run never prompts:
    # it keeps opted-in human PRs in the preview, so the output mirrors
    # what a real --include-human-prs run would attempt (see below).
    needs_human_confirm = bool(human_prs) and not ctx.no_confirm and not ctx.dry_run
    if needs_human_confirm:
        console.print("\n⚠️ Human-authored PRs are included in this merge operation.")
        console.print("   Review the list above carefully before proceeding.")
        try:
            user_input = (
                typer.prompt(
                    "Type 'yes' to include human PRs, or press Enter to skip them",
                    default="",
                    show_default=False,
                )
                .strip()
                .lower()
            )
            if user_input != "yes":
                console.print("ℹ️ Excluding human PRs from merge.")
                repo_prs = automation_prs
                if not repo_prs:
                    console.print("❌ No automation PRs remain to merge.")
                    return None
        except (KeyboardInterrupt, EOFError, typer.Abort):
            console.print("\n❌ Merge cancelled by user.")
            return None
    elif human_prs and ctx.dry_run:
        # Human PRs only reach this branch when --include-human-prs was
        # supplied (otherwise they are filtered out at fetch time).  A dry
        # run performs no writes, so keep them in the preview to faithfully
        # mirror what a real --include-human-prs run would attempt; a real
        # run would prompt for confirmation first (unless --no-confirm).
        console.print(
            "\nℹ️ Dry run: human-authored PRs are kept in this preview "
            "(a real run would prompt before merging them)."
        )

    return repo_prs


def _handle_repo_merge(
    parsed_repo: ParsedRepoUrl,
    ctx: _MergeContext,
) -> None:
    """Handle merge operation for a repository-scoped URL.

    Instead of scanning an entire org for similar PRs, this fetches all
    open PRs in a single repository and merges the automation ones (or
    all of them when --include-human-prs is given).

    Args:
        parsed_repo: Parsed repository URL with owner and repo.
        ctx: Shared merge context populated with CLI parameters.
    """
    _prepare_repo_merge(parsed_repo, ctx)

    only_automation = not ctx.include_human_prs
    repo_prs = _fetch_repo_prs(ctx, only_automation)

    if not repo_prs:
        label = "automation " if only_automation else ""
        console.print(f"❌ No open {label}PRs found in {parsed_repo.project}")
        return

    # The GraphQL fetch returns PRs newest-first (CREATED_AT DESC), but
    # merging within a repository must drain oldest-first.  Merging the
    # newest PR ahead of an older sibling leaves the older one behind the
    # base branch, forcing automation (e.g. dependabot) into an avoidable
    # rebase-and-revalidate cycle that can block the batch.  Sorting here
    # mirrors the within-repository key applied owner-wide by
    # ``_owner_merge_order`` so both schemes sequence a repository's PRs
    # identically; the preview list below is derived from this order too.
    repo_prs = _repo_merge_order(repo_prs)

    selected = _select_repo_prs(parsed_repo, ctx, repo_prs)
    if selected is None:
        return
    repo_prs = selected

    all_prs_to_merge: list[tuple[PullRequestInfo, ComparisonResult | None]] = [
        (pr, None) for pr in repo_prs
    ]

    # Without --no-confirm this first pass is a preview (evaluation)
    # run — label the tracker accordingly so its counters ("Mergeable")
    # don't claim merges that never happened.
    preview_run = ctx.dry_run or not ctx.no_confirm
    if ctx.show_progress:
        ctx.progress_tracker = MergeProgressTracker(
            f"{ctx.owner}/{ctx.repo_name}",
            operation_label="Evaluating PRs" if preview_run else "Merging PRs",
            operation_icon="\U0001f50d" if preview_run else "\u25b6\ufe0f",
            preview=preview_run,
        )
        ctx.progress_tracker.set_total_prs(len(all_prs_to_merge))
        ctx.progress_tracker.start()

    try:
        # Per-repo merge dispatch is serialised inside
        # ``AsyncMergeManager`` (see ``_get_merge_dispatch_lock``),
        # so it is now safe to run multiple workers against PRs
        # that target the same repository — only the actual
        # ``merge_pull_request`` call queues, while approve,
        # rebase, and the Step 5.5 auto-merge wait run in
        # parallel.
        merge_results = _parallel._run_parallel_merge(
            ctx,
            all_prs_to_merge,
            preview=preview_run,
            # Allow parallel workers; the merge dispatch itself is
            # serialised per repo by ``AsyncMergeManager`` so PRs
            # parked in Step 5.5's wait loop no longer block other
            # PRs in the batch.  Cap by PR count so we don't spawn
            # more workers than there is work.
            concurrency=min(5, len(all_prs_to_merge)) or 1,
            # All PRs target the same repo, so a sibling merge can make
            # a queued PR ``dirty`` / ``behind`` mid-batch; refresh
            # live state before each merge dispatch.
            repo_scoped=True,
        )
    finally:
        if ctx.show_progress and ctx.progress_tracker:
            ctx.progress_tracker.stop()

    if not merge_results:
        console.print("❌ No PRs were processed")
        return

    merged_count = sum(1 for r in merge_results if r.status.value == "merged")

    # Dry run: report the preview and stop before any prompt or merge.
    if ctx.dry_run:
        _display_merge_results(merge_results, no_confirm=False)
        return

    if not ctx.no_confirm:
        # In preview mode, show what would happen, then prompt
        # for confirmation via an override-style SHA.
        _handle_repo_preview_confirmation(
            ctx,
            merge_results,
            all_prs_to_merge,
            merged_count,
            len(merge_results),
        )
        return

    _display_merge_results(merge_results, ctx.no_confirm)


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
        real_results = _parallel._run_parallel_merge(
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
