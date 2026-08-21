# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Merging every open PR in a single repository.

A repository-scoped URL skips similar-PR matching entirely: every open PR
in the repository is a candidate, drained oldest-first so automation is
not forced into avoidable rebases.
"""

import asyncio

import typer

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
import dependamerge.cli as _pkg

from ..bot_identity import is_automation_author
from ..merge_manager import (
    MergeResult,
)
from ..models import ComparisonResult, PullRequestInfo
from ..progress_tracker import MergeProgressTracker
from ..url_parser import (
    ParsedRepoUrl,
)
from ._app import console
from ._context import _MergeContext
from ._merge_order import _repo_merge_order
from ._merge_permissions import _maybe_check_merge_permissions
from ._merge_report import (
    _display_merge_results,
)
from ._repo_confirm import _handle_repo_preview_confirmation


def _init_repo_merge_client(
    parsed_repo: ParsedRepoUrl,
    ctx: _MergeContext,
) -> None:
    """Point the merge context at one repository and check access.

    Args:
        parsed_repo: Parsed repository URL with owner and repo.
        ctx: Shared merge context populated with CLI parameters.
    """
    from ..url_parser import _host_matches

    if not _host_matches(parsed_repo.host, "github.com"):
        console.print(
            "❌ Repository-scoped merge is currently only supported "
            f"for github.com (got host: {parsed_repo.host}).\n"
            "   GitHub Enterprise support requires API base URL "
            "configuration — use a direct PR URL instead."
        )
        raise typer.Exit(code=1)

    ctx.github_client = _pkg.GitHubClient(ctx.token)
    assert ctx.github_client.token is not None
    ctx.token = ctx.github_client.token
    ctx.owner = parsed_repo.owner
    ctx.repo_name = parsed_repo.repo

    console.print(f"🔍 Repository mode: fetching open PRs in {parsed_repo.project}...")

    _maybe_check_merge_permissions(ctx)


def _fetch_repo_prs(
    ctx: _MergeContext,
    *,
    only_automation: bool,
) -> list[PullRequestInfo]:
    """Fetch the open PRs of the repository named by the context.

    Args:
        ctx: Shared merge context, already pointed at the repository.
        only_automation: Restrict the fetch to automation-authored PRs.

    Returns:
        The repository's open PRs, newest first.
    """
    from ..github_service import GitHubService

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


def _partition_repo_prs(
    parsed_repo: ParsedRepoUrl,
    repo_prs: list[PullRequestInfo],
) -> tuple[list[PullRequestInfo], list[PullRequestInfo]]:
    """Split the fetched PRs by authorship and print the repository summary.

    Args:
        parsed_repo: Parsed repository URL, used for the summary line.
        repo_prs: The repository's open PRs, in merge order.

    Returns:
        The automation-authored PRs and the human-authored PRs.
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

    return automation_prs, human_prs


def _confirm_repo_human_prs(
    ctx: _MergeContext,
    repo_prs: list[PullRequestInfo],
    automation_prs: list[PullRequestInfo],
    human_prs: list[PullRequestInfo],
) -> list[PullRequestInfo] | None:
    """Resolve which PRs stay in scope when human-authored PRs are present.

    Args:
        ctx: Shared merge context populated with CLI parameters.
        repo_prs: Every open PR in the repository, in merge order.
        automation_prs: The automation-authored subset.
        human_prs: The human-authored subset.

    Returns:
        The PRs to merge, or ``None`` when the operator cancelled or no
        automation PRs remain after excluding the human ones.
    """
    # Only prompt when human PRs are actually in scope, not merely
    # because --include-human-prs was supplied.  A dry run never prompts;
    # it mirrors the safe default (exclude human PRs) so the preview
    # matches a real run where the operator presses Enter at the prompt.
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


def _run_repo_merge_pass(
    ctx: _MergeContext,
    all_prs_to_merge: list[tuple[PullRequestInfo, ComparisonResult | None]],
    *,
    preview_run: bool,
) -> list[MergeResult]:
    """Run one evaluation or merge pass over a repository's PRs.

    Args:
        ctx: Shared merge context populated with CLI parameters.
        all_prs_to_merge: The PRs to process, each without a comparison.
        preview_run: Evaluate without merging rather than merging.

    Returns:
        One result per processed PR.
    """
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
        merge_results = _pkg._run_parallel_merge(
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

    return merge_results


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
    _init_repo_merge_client(parsed_repo, ctx)

    only_automation = not ctx.include_human_prs
    repo_prs = _fetch_repo_prs(ctx, only_automation=only_automation)

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

    automation_prs, human_prs = _partition_repo_prs(parsed_repo, repo_prs)

    selected = _confirm_repo_human_prs(ctx, repo_prs, automation_prs, human_prs)
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
    merge_results = _run_repo_merge_pass(ctx, all_prs_to_merge, preview_run=preview_run)

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
