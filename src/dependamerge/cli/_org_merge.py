# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Merging every open automation PR across one owner.

Enumerates the owner's non-archived, non-fork repositories, then hands
the combined batch to the striped scheduler so no two merges stack on the
same repository at once.
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
    ParsedOrgUrl,
)
from ._app import console
from ._context import _MergeContext
from ._merge_order import _owner_merge_order, _print_prs_grouped_by_repo
from ._merge_permissions import _maybe_check_merge_permissions
from ._merge_report import (
    _display_merge_results,
)
from ._org_confirm import _handle_org_preview_confirmation


def _init_org_merge_client(
    parsed_org: ParsedOrgUrl,
    ctx: _MergeContext,
) -> None:
    """Point the merge context at an owner and announce the scan.

    Args:
        parsed_org: Parsed owner URL with the org/user login.
        ctx: Shared merge context populated with CLI parameters.
    """
    # ``parse_org_url`` is the single choke point that decides which
    # hosts are permitted: github.com and its subdomains always, and a
    # GitHub Enterprise Server host once the operator has declared it.
    # The host reaches the client from ``ctx`` so an Enterprise run
    # addresses the right API base URLs.  Deliberately not re-checked
    # here --- a second guard would only drift out of sync with the
    # first, which is exactly what happened in the repository handler.

    ctx.github_client = _pkg.GitHubClient(ctx.token, host=ctx.host)
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


def _fetch_owner_prs(
    parsed_org: ParsedOrgUrl,
    ctx: _MergeContext,
    *,
    only_automation: bool,
) -> tuple[list[PullRequestInfo], list[str]]:
    """Enumerate the owner's open PRs across every in-scope repository.

    Args:
        parsed_org: Parsed owner URL with the org/user login.
        ctx: Shared merge context populated with CLI parameters.
        only_automation: Restrict the scan to automation-authored PRs.

    Returns:
        The open PRs found, and one message per repository that could not
        be scanned.
    """
    from ..github_service import GitHubService

    if ctx.show_progress:
        ctx.progress_tracker = MergeProgressTracker(
            parsed_org.owner,
            operation_label="Scanning repositories for automation PRs",
            operation_icon="🔍",
        )
        ctx.progress_tracker.start()

    async def _fetch_prs() -> tuple[list[PullRequestInfo], list[str]]:
        svc = GitHubService(
            token=ctx.token,
            host=ctx.host,
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

    if scan_errors:
        error_count = len(scan_errors)
        repo_noun = "repository" if error_count == 1 else "repositories"
        console.print(f"\n⚠️ {error_count} {repo_noun} could not be scanned:")
        for err in scan_errors:
            console.print(f"   - {err}")

    return owner_prs, scan_errors


def _partition_owner_prs(
    parsed_org: ParsedOrgUrl,
    owner_prs: list[PullRequestInfo],
) -> tuple[list[PullRequestInfo], list[PullRequestInfo]]:
    """Split the enumerated PRs by authorship and print the owner summary.

    Args:
        parsed_org: Parsed owner URL, used for the summary line.
        owner_prs: Every in-scope open PR, in merge order.

    Returns:
        The automation-authored PRs and the human-authored PRs.
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

    return automation_prs, human_prs


def _confirm_owner_human_prs(
    ctx: _MergeContext,
    parsed_org: ParsedOrgUrl,
    owner_prs: list[PullRequestInfo],
    automation_prs: list[PullRequestInfo],
    human_prs: list[PullRequestInfo],
) -> list[PullRequestInfo] | None:
    """Resolve which PRs stay in scope when human-authored PRs are present.

    Args:
        ctx: Shared merge context populated with CLI parameters.
        parsed_org: Parsed owner URL, named in the warning.
        owner_prs: Every in-scope open PR, in merge order.
        automation_prs: The automation-authored subset.
        human_prs: The human-authored subset.

    Returns:
        The PRs to merge, or ``None`` when the operator cancelled or no
        automation PRs remain after excluding the human ones.
    """
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


def _run_owner_merge_pass(
    ctx: _MergeContext,
    parsed_org: ParsedOrgUrl,
    all_prs_to_merge: list[tuple[PullRequestInfo, ComparisonResult | None]],
    *,
    preview_run: bool,
) -> list[MergeResult]:
    """Run one evaluation or merge pass over an owner's PRs.

    Args:
        ctx: Shared merge context, carrying the derived concurrency.
        parsed_org: Parsed owner URL, used to label the progress tracker.
        all_prs_to_merge: The PRs to process, each without a comparison.
        preview_run: Evaluate without merging rather than merging.

    Returns:
        One result per processed PR.
    """
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
        merge_results = _pkg._run_parallel_merge(
            ctx,
            all_prs_to_merge,
            preview=preview_run,
            concurrency=ctx.merge_concurrency,
            # Owner-wide batches mix many repositories and often contain
            # multiple PRs per repository; stripe to avoid stacking and
            # repo_scoped to refresh live merge state before each dispatch.
            repo_scoped=True,
            stripe=True,
        )
    finally:
        if ctx.show_progress and ctx.progress_tracker:
            ctx.progress_tracker.stop()

    return merge_results


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
    _init_org_merge_client(parsed_org, ctx)

    only_automation = not ctx.include_human_prs
    owner_prs, _scan_errors = _fetch_owner_prs(
        parsed_org, ctx, only_automation=only_automation
    )

    if not owner_prs:
        label = "automation " if only_automation else ""
        console.print(f"❌ No open {label}PRs found in {parsed_org.owner}")
        return

    # Order for striped merging: repositories with the most PRs first,
    # ascending PR number within each repository (see _owner_merge_order).
    # Both the grouped listing and the merge list derive from this order.
    owner_prs = _owner_merge_order(owner_prs)

    automation_prs, human_prs = _partition_owner_prs(parsed_org, owner_prs)

    selected = _confirm_owner_human_prs(
        ctx, parsed_org, owner_prs, automation_prs, human_prs
    )
    if selected is None:
        return
    owner_prs = selected

    # Deferred until the final set is known: the helper needs a concrete
    # repo to probe, so point it at the first PR the run will actually
    # merge as a representative sample.  Probing before the prompt would
    # sample the unfiltered set, and under --include-human-prs the first
    # PR there can belong to a repository holding only human-authored
    # PRs --- which the operator then drops by pressing Enter.  A token
    # scoped to exactly the automation repositories would be rejected on
    # the strength of a repository the run never touches.
    #
    # Sampling one repo remains sufficient: the common failure modes
    # (expired/invalid token, no write access) fail identically on any
    # repo, and genuine per-repo permission variance with fine-grained
    # tokens is caught at merge time by the scheduler's per-repository
    # error isolation.  ``owner_prs`` is non-empty here --- it was
    # checked above, and _confirm_owner_human_prs returns None rather
    # than an empty selection.
    ctx.repo_name = owner_prs[0].repository_full_name.split("/", 1)[-1]
    _maybe_check_merge_permissions(ctx)

    all_prs_to_merge: list[tuple[PullRequestInfo, ComparisonResult | None]] = [
        (pr, None) for pr in owner_prs
    ]

    # Global concurrency 10, bounded by the distinct-repo count: the
    # striped scheduler runs at most one PR per repo, so more workers
    # than repositories cannot help.
    final_distinct_repos = {pr.repository_full_name for pr in owner_prs}
    ctx.merge_concurrency = min(10, len(final_distinct_repos)) or 1

    # Without --no-confirm this first pass is a preview (evaluation)
    # run — label the tracker accordingly so its counters ("Mergeable")
    # don't claim merges that never happened.
    preview_run = ctx.dry_run or not ctx.no_confirm
    merge_results = _run_owner_merge_pass(
        ctx, parsed_org, all_prs_to_merge, preview_run=preview_run
    )

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
            merge_results,
            all_prs_to_merge,
            merged_count,
            len(merge_results),
        )
        return

    _display_merge_results(merge_results, ctx.no_confirm)
