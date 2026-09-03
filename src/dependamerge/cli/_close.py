# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The steps behind the ``close`` command.

Authorises the request, finds the PRs similar to the source, reports what
was found, and closes them --- as a dry run, interactively, or straight
away.
"""

import asyncio
import os
import sys
from dataclasses import dataclass

import typer

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
import dependamerge.cli as _pkg

from ..close_manager import CloseResult
from ..models import ComparisonResult, PullRequestInfo
from ..progress_tracker import MergeProgressTracker
from ._app import console
from ._sha import _generate_continue_sha, _generate_override_sha
from ._similarity import _format_condensed_similarity


@dataclass
class _CloseContext:
    """Shared state for the ``close`` command pipeline."""

    token: str
    github_client: _pkg.GitHubClient
    # The GitHub host this run addresses, taken from the pull request
    # URL.  A GitHub Enterprise Server install serves its API from
    # different base URLs, so a run that loses the host silently
    # addresses github.com instead.
    host: str
    owner: str
    repo_name: str
    pr_number: int
    source_pr: PullRequestInfo
    progress_tracker: MergeProgressTracker | None


def _run_close_parallel(
    ctx: _CloseContext,
    prs: list[PullRequestInfo],
    preview_mode: bool,
) -> list[CloseResult]:
    """Close (or preview-close) ``prs`` in parallel."""

    async def _close_parallel() -> list[CloseResult]:
        close_manager = _pkg.AsyncCloseManager(
            token=ctx.token,
            host=ctx.host,
            progress_tracker=ctx.progress_tracker,
            preview_mode=preview_mode,
        )
        async with close_manager:
            # Convert to list of tuples (PR, None) for consistency
            pr_tuples: list[tuple[PullRequestInfo, ComparisonResult | None]] = [
                (pr, None) for pr in prs
            ]
            return await close_manager.close_prs_parallel(pr_tuples)

    return asyncio.run(_close_parallel())


def _print_close_debug_matching(
    ctx: _CloseContext,
    comparator: _pkg.PRComparator,
    similarity_threshold: float,
) -> None:
    """Show detailed matching diagnostics for the source PR."""
    source_pr = ctx.source_pr
    console.print("\n🔍 Debug Matching Information")
    console.print(
        "   Source PR automation status: "
        f"{ctx.github_client.is_automation_author(source_pr.author)}"
    )
    console.print(
        f"   Extracted package: '{comparator._extract_package_name(source_pr.title)}'"
    )
    console.print(f"   Similarity threshold: {similarity_threshold}")
    if source_pr.body:
        console.print(f"   Body preview: {source_pr.body[:100]}...")
        console.print(
            f"   Is dependabot body: {comparator._is_dependabot_body(source_pr.body)}"
        )
    else:
        console.print("   ⚠️ Source PR has no body")
    console.print()


def _validate_close_authorization(
    ctx: _CloseContext,
    override: str | None,
) -> bool:
    """Gate non-automation source PRs behind the override SHA.

    Returns True when the close may proceed; False when the caller
    should stop (the reason has already been printed).
    """
    if ctx.github_client.is_automation_author(ctx.source_pr.author):
        return True

    commit_messages = ctx.github_client.get_pull_request_commits(
        ctx.owner, ctx.repo_name, ctx.pr_number
    )
    first_commit_line = commit_messages[0].split("\n")[0] if commit_messages else ""

    # Generate expected SHA
    expected_sha = _generate_override_sha(ctx.source_pr, first_commit_line)

    if not override:
        console.print("Source PR is not from a recognized automation tool.")
        console.print(
            f"To close this and similar PRs, run again with: --override {expected_sha}"
        )
        console.print(
            f"This SHA is based on the author '{ctx.source_pr.author}' and commit message '{first_commit_line[:50]}...'",
            style="dim",
        )
        return False

    if override != expected_sha:
        console.print(
            f"Error: Invalid override SHA. Expected: {expected_sha}",
            style="bold red",
        )
        console.print(
            "This prevents accidental bulk operations on non-automation PRs.",
            style="dim",
        )
        return False

    console.print("Override SHA validated. Proceeding with non-automation PR close.")
    return True


def _find_similar_prs_for_close(
    ctx: _CloseContext,
    comparator: _pkg.PRComparator,
    debug_matching: bool,
) -> list[tuple[PullRequestInfo, ComparisonResult]]:
    """Search the owner for PRs similar to the source PR."""
    if ctx.progress_tracker:
        console.print()
    else:
        console.print(f"\nChecking owner: {ctx.owner}")

    # Use GitHubService for async PR finding
    from ..github_service import GitHubService

    if ctx.progress_tracker:
        ctx.progress_tracker.update_operation("Listing repositories...")

    async def _find_similar():
        svc = GitHubService(
            token=ctx.token,
            host=ctx.host,
            progress_tracker=ctx.progress_tracker,
            debug_matching=debug_matching,
        )
        try:
            only_automation = ctx.github_client.is_automation_author(
                ctx.source_pr.author
            )
            return await svc.find_similar_prs(
                ctx.owner,
                ctx.source_pr,
                comparator,
                only_automation=only_automation,
            )
        finally:
            await svc.close()

    return asyncio.run(_find_similar())


def _print_close_analysis_summary(
    ctx: _CloseContext,
    all_similar_prs: list[tuple[PullRequestInfo, ComparisonResult]],
) -> None:
    """Report the search outcome and list the matching PRs."""
    if ctx.progress_tracker:
        ctx.progress_tracker.stop()
        summary = ctx.progress_tracker.get_summary()
        elapsed_time = summary.get("elapsed_time")
        total_prs_analyzed = summary.get("total_prs_analyzed")
        completed_repositories = summary.get("completed_repositories")
        similar_prs_found = summary.get("similar_prs_found")
        errors_count = summary.get("errors_count", 0)
        console.print(f"\n✅ Analysis completed in {elapsed_time}")
        console.print(
            f"📊 Analyzed {total_prs_analyzed} PRs across {completed_repositories} repositories"
        )
        console.print(f"🔍 Found {similar_prs_found} similar PRs")
        if errors_count > 0:
            console.print(f"⚠️ {errors_count} errors encountered during analysis")
        console.print()
    else:
        console.print(f"\n🔍 Found {len(all_similar_prs)} similar PRs")

    if not all_similar_prs:
        # Not a failure: the supplied PR is simply the only one to
        # close.  Use a neutral glyph rather than ❌ so the output
        # does not read like an error.
        console.print("⏩ No similar PRs found for this owner")

    for target_pr, comparison in all_similar_prs:
        console.print(f"  • {target_pr.repository_full_name} #{target_pr.number}")
        console.print(f"    {_format_condensed_similarity(comparison)}")


def _run_close_dry_run(
    ctx: _CloseContext,
    all_prs_to_close: list[PullRequestInfo],
) -> None:
    """Evaluate in preview mode and report what *would* be closed."""
    if ctx.progress_tracker:
        ctx.progress_tracker.start()
        console.print()
    else:
        console.print(
            f"\n🧪 Dry run: evaluating {len(all_prs_to_close)} pull requests..."
        )

    close_results = _run_close_parallel(ctx, all_prs_to_close, True)

    if ctx.progress_tracker:
        ctx.progress_tracker.stop()
        console.print()

    closeable_count = sum(1 for r in close_results if r.status.value == "closed")
    console.print(
        f"\n🧪 Dry run: would close {closeable_count}/"
        f"{len(all_prs_to_close)} PRs (no changes made)"
    )


def _run_interactive_close(
    ctx: _CloseContext,
    all_prs_to_close: list[PullRequestInfo],
) -> None:
    """Preview the close, then prompt for SHA-gated confirmation."""
    if ctx.progress_tracker:
        ctx.progress_tracker.start()
        console.print()
    else:
        console.print(f"\n🚀 Evaluating {len(all_prs_to_close)} pull requests...")

    close_results = _run_close_parallel(ctx, all_prs_to_close, True)

    if ctx.progress_tracker:
        ctx.progress_tracker.stop()
        console.print()

    # Count closeable PRs
    closed_count = sum(1 for r in close_results if r.status.value == "closed")
    total_to_close = len(all_prs_to_close)

    console.print(f"\nCloseable {closed_count}/{total_to_close} PRs")

    if closed_count == 0:
        console.print("\n❌ No PRs are eligible for closing")
        return

    # Generate continuation SHA and prompt user
    commit_messages = ctx.github_client.get_pull_request_commits(
        ctx.owner, ctx.repo_name, ctx.pr_number
    )
    first_commit_line = commit_messages[0].split("\n")[0] if commit_messages else ""
    continue_sha_hash = _generate_continue_sha(ctx.source_pr, first_commit_line)
    console.print()
    console.print(f"To proceed with closing enter: {continue_sha_hash}")

    # Check if in test mode (don't prompt during tests)
    if "pytest" in sys.modules or os.getenv("TESTING"):
        console.print("⚠️ Test mode detected - skipping interactive prompt")
        return

    user_input = typer.prompt(
        "\nEnter the string above to continue (or press Enter to cancel)"
    ).strip()

    if user_input != continue_sha_hash:
        console.print("\n❌ Operation cancelled by user")
        return

    # Run actual close on closeable PRs only
    console.print(f"\n🔨 Closing {closed_count} closeable pull requests...")
    closeable_prs = [
        all_prs_to_close[i]
        for i, result in enumerate(close_results)
        # These were preview "closed"
        if result.status.value == "closed"
    ]

    if ctx.progress_tracker:
        ctx.progress_tracker.start()

    final_results = _run_close_parallel(ctx, closeable_prs, False)

    if ctx.progress_tracker:
        ctx.progress_tracker.stop()

    # Count final results
    final_closed = sum(1 for r in final_results if r.status.value == "closed")
    final_failed = sum(1 for r in final_results if r.status.value == "failed")

    console.print(f"\n🚀 Final Results: {final_closed} closed, {final_failed} failed")


def _run_immediate_close(
    ctx: _CloseContext,
    all_prs_to_close: list[PullRequestInfo],
) -> None:
    """Close all candidate PRs without confirmation."""
    if ctx.progress_tracker:
        ctx.progress_tracker.start()
    console.print(f"\n🚀 Closing {len(all_prs_to_close)} pull requests...")

    close_results = _run_close_parallel(ctx, all_prs_to_close, False)

    if ctx.progress_tracker:
        ctx.progress_tracker.stop()

    # Count results
    closed_count = sum(1 for r in close_results if r.status.value == "closed")
    failed_count = sum(1 for r in close_results if r.status.value == "failed")

    console.print(f"\n🚀 Final Results: {closed_count} closed, {failed_count} failed")
