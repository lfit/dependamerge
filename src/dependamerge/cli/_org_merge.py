# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Owner-scoped bulk merge: enumerate, partition, preview, confirm.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

import typer

from ..bot_identity import is_automation_author
from ..merge_manager import (
    MergeResult,
)
from ..models import ComparisonResult, PullRequestInfo
from ..progress_tracker import MergeProgressTracker
from ..url_parser import (
    ParsedOrgUrl,
)

if TYPE_CHECKING:
    pass

from . import _deps, _parallel
from ._app import console
from ._context import (
    _MergeContext,
)
from ._permissions import _maybe_check_merge_permissions
from ._results import (
    _display_merge_results,
    _MergePreview,
    _owner_merge_order,
    _print_failed_pr_details,
    _print_final_merge_summary,
    _print_prs_grouped_by_repo,
)


def _prepare_org_merge(
    parsed_org: ParsedOrgUrl,
    ctx: _MergeContext,
) -> None:
    """Bind the client to *ctx* and start the enumeration tracker."""
    # The host here is guaranteed to *match* github.com: ``parse_org_url``
    # is the single github.com-only choke point (it rejects every other
    # host, including GHE, before a ``ParsedOrgUrl`` can reach this
    # handler).  "Matches" is deliberate — ``parse_org_url`` accepts
    # github.com and its subdomains (e.g. ``api.github.com``) via
    # ``_host_matches``, so this is not an exact ``== "github.com"``
    # guarantee.  Enabling GHE (#343) means relaxing that one parser
    # guard and threading ``derive_api_urls(host)`` through the service
    # stack — deliberately not re-checked here so there is no second
    # guard to drift out of sync.

    ctx.github_client = _deps.GitHubClient(ctx.token)
    assert ctx.github_client.token is not None
    ctx.token = ctx.github_client.token
    ctx.owner = parsed_org.owner
    ctx.repo_name = ""

    console.print(f"🔍 Owner mode: scanning {parsed_org.owner} for automation PRs...")

    # NOTE: the token permission check is deferred until *after*
    # enumeration (see below).  ``_check_merge_permissions`` probes a
    # concrete repository, and ``check_token_permissions`` reports every
    # operation as missing when no repo is supplied — running it here
    # with an empty ``ctx.repo_name`` would abort every owner-wide run
    # unconditionally.

    if ctx.show_progress:
        ctx.progress_tracker = MergeProgressTracker(
            parsed_org.owner,
            operation_label="Scanning repositories for automation PRs",
            operation_icon="🔍",
        )
        ctx.progress_tracker.start()


def _fetch_owner_prs(
    parsed_org: ParsedOrgUrl,
    ctx: _MergeContext,
    only_automation: bool,
) -> tuple[list[PullRequestInfo], list[str]]:
    """Enumerate the owner's open PRs, stopping the tracker either way."""
    from ..github_service import GitHubService

    async def _fetch_prs() -> tuple[list[PullRequestInfo], list[str]]:
        svc = GitHubService(
            token=ctx.token,
            progress_tracker=ctx.progress_tracker,
        )
        try:
            return await svc.fetch_owner_open_prs(
                parsed_org.owner,
                only_automation=only_automation,
            )
        finally:
            await svc.close()

    try:
        owner_prs, scan_errors = asyncio.run(_fetch_prs())
    except Exception:
        if ctx.progress_tracker:
            ctx.progress_tracker.stop()
        raise

    if ctx.progress_tracker:
        ctx.progress_tracker.stop()

    return owner_prs, scan_errors


def _select_owner_prs(
    parsed_org: ParsedOrgUrl,
    ctx: _MergeContext,
    owner_prs: list[PullRequestInfo],
) -> list[PullRequestInfo] | None:
    """Report the enumerated PRs and settle whether human ones stay in scope.

    Returns the PRs to merge, or ``None`` when the run should stop.
    """
    automation_prs: list[PullRequestInfo] = []
    human_prs: list[PullRequestInfo] = []
    for pr in owner_prs:
        if is_automation_author(pr.author):
            automation_prs.append(pr)
        else:
            human_prs.append(pr)

    distinct_repos = {pr.repository_full_name for pr in owner_prs}
    repo_count = len(distinct_repos)
    repo_noun = "repository" if repo_count == 1 else "repositories"
    console.print(
        f"\n📊 Found {len(owner_prs)} open PR(s) across "
        f"{repo_count} {repo_noun} in {parsed_org.owner}"
    )
    if automation_prs:
        console.print(f"🤖 Automation PRs: {len(automation_prs)}")
    if human_prs:
        console.print(f"👤 Human PRs: {len(human_prs)}")

    # Grouped-by-repository listing keeps a large owner-wide list scannable.
    _print_prs_grouped_by_repo(owner_prs)

    needs_human_confirm = bool(human_prs) and not ctx.no_confirm and not ctx.dry_run
    if needs_human_confirm:
        console.print(
            "\n⚠️ Human-authored PRs across the entire owner "
            f"'{parsed_org.owner}' are included in this merge operation."
        )
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
                owner_prs = automation_prs
                if not owner_prs:
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

    return owner_prs


def _handle_org_merge(
    parsed_org: ParsedOrgUrl,
    ctx: _MergeContext,
) -> None:
    """Handle merge operation for an owner-scoped (org/user) URL.

    Enumerates every in-scope automation PR across all of the owner's
    non-archived, non-fork repositories, then bulk merges them with the
    striped scheduler so no two merges stack on the same repository at
    once.  The owner may be an organization or a personal user account;
    the account type is detected at runtime during enumeration.

    Args:
        parsed_org: Parsed owner URL with the org/user login.
        ctx: Shared merge context populated with CLI parameters.
    """
    _prepare_org_merge(parsed_org, ctx)

    only_automation = not ctx.include_human_prs
    owner_prs, scan_errors = _fetch_owner_prs(parsed_org, ctx, only_automation)

    if scan_errors:
        error_count = len(scan_errors)
        repo_noun = "repository" if error_count == 1 else "repositories"
        console.print(f"\n⚠️ {error_count} {repo_noun} could not be scanned:")
        for err in scan_errors:
            console.print(f"   - {err}")

    if not owner_prs:
        label = "automation " if only_automation else ""
        console.print(f"❌ No open {label}PRs found in {parsed_org.owner}")
        return

    # Order for striped merging: repositories with the most PRs first,
    # ascending PR number within each repository (see _owner_merge_order).
    # Both the grouped listing and the merge list derive from this order.
    owner_prs = _owner_merge_order(owner_prs)

    # Deferred from before enumeration: the helper needs a concrete repo
    # to probe, so point it at the first in-scope PR's repository as a
    # representative sample.  The common failure modes (expired/invalid
    # token, no write access) fail identically on any repo; genuine
    # per-repo permission variance with fine-grained tokens is caught at
    # merge time by the scheduler's per-repository error isolation.
    ctx.repo_name = owner_prs[0].repository_full_name.split("/", 1)[-1]
    _maybe_check_merge_permissions(ctx)

    selected = _select_owner_prs(parsed_org, ctx, owner_prs)
    if selected is None:
        return
    owner_prs = selected

    all_prs_to_merge: list[tuple[PullRequestInfo, ComparisonResult | None]] = [
        (pr, None) for pr in owner_prs
    ]

    # Global concurrency 10, bounded by the distinct-repo count: the
    # striped scheduler runs at most one PR per repo, so more workers
    # than repositories cannot help.
    final_distinct_repos = {pr.repository_full_name for pr in owner_prs}
    concurrency = min(10, len(final_distinct_repos)) or 1

    # Without --no-confirm this first pass is a preview (evaluation)
    # run — label the tracker accordingly so its counters ("Mergeable")
    # don't claim merges that never happened.
    preview_run = ctx.dry_run or not ctx.no_confirm
    if ctx.show_progress:
        ctx.progress_tracker = MergeProgressTracker(
            parsed_org.owner,
            operation_label="Evaluating PRs" if preview_run else "Merging PRs",
            operation_icon="\U0001f50d" if preview_run else "\u25b6\ufe0f",
            preview=preview_run,
        )
        ctx.progress_tracker.set_total_prs(len(all_prs_to_merge))
        ctx.progress_tracker.start()

    try:
        merge_results = _parallel._run_parallel_merge(
            ctx,
            all_prs_to_merge,
            preview=preview_run,
            concurrency=concurrency,
            # Owner-wide batches mix many repositories and often contain
            # multiple PRs per repository; stripe to avoid stacking and
            # repo_scoped to refresh live merge state before each dispatch.
            repo_scoped=True,
            stripe=True,
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
        _handle_org_preview_confirmation(
            ctx,
            parsed_org,
            _MergePreview(
                merge_results=merge_results,
                all_prs_to_merge=all_prs_to_merge,
                merged_count=merged_count,
                total_to_merge=len(merge_results),
            ),
            concurrency,
        )
        return

    _display_merge_results(merge_results, ctx.no_confirm)


def _handle_org_preview_confirmation(
    ctx: _MergeContext,
    parsed_org: ParsedOrgUrl,
    preview: _MergePreview,
    concurrency: int,
) -> None:
    """Handle preview-then-confirm for owner-scoped merges.

    Mirrors :func:`_handle_repo_preview_confirmation` but derives the
    confirmation token from the owner login instead of a repository.
    """
    merge_results = preview.merge_results
    merged_count = preview.merged_count
    console.print(f"\nMergeable {merged_count}/{preview.total_to_merge} PRs")
    # Per-PR preview lines are no longer printed during the run, so
    # report the not-mergeable PRs (and why) here before prompting.
    _print_failed_pr_details(merge_results)

    if merged_count == 0:
        console.print("\n\U0001f4a1 No PRs are mergeable at this time.")
        return

    combined = f"org-merge:{parsed_org.owner}:{merged_count}"
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
            _execute_org_confirmed_merge(
                ctx, merge_results, preview.all_prs_to_merge, concurrency
            )
        elif user_input == "":
            console.print("❌ Merge cancelled by user.")
        else:
            console.print("❌ Invalid input. Merge cancelled.")
    except (KeyboardInterrupt, EOFError, typer.Abort):
        console.print("\n❌ Merge cancelled by user.")


def _execute_org_confirmed_merge(
    ctx: _MergeContext,
    preview_results: list[MergeResult],
    all_prs_to_merge: list[tuple[PullRequestInfo, ComparisonResult | None]],
    concurrency: int,
) -> None:
    """Run the real owner-wide merge after user confirmation."""
    mergeable_prs = [
        all_prs_to_merge[i]
        for i, result in enumerate(preview_results)
        if result.status.value == "merged"
    ]

    if ctx.show_progress:
        ctx.progress_tracker = MergeProgressTracker(
            ctx.owner,
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
            concurrency=concurrency,
            repo_scoped=True,
            stripe=True,
        )
    finally:
        if ctx.show_progress and ctx.progress_tracker:
            ctx.progress_tracker.stop()

    _print_final_merge_summary(real_results)
