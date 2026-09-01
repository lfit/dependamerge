# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The ``blocked`` command and its options.

Also holds the scan, the summary, and the optional ``--fix`` workflow
that rebases the PRs the scan found blocked.
"""

import asyncio
import os

import typer

from ..error_codes import (
    DependamergeError,
    ExitCode,
    convert_git_error,
    convert_github_api_error,
    convert_network_error,
    exit_for_configuration_error,
    exit_for_github_api_error,
    exit_with_error,
    is_github_api_permission_error,
    is_network_error,
)
from ..git_ops import GitError
from ..github_async import (
    GraphQLError,
    RateLimitError,
    SecondaryRateLimitError,
)
from ..progress_tracker import ProgressTracker
from ..resolve_conflicts import FixOptions, FixOrchestrator, PRSelection
from ..system_utils import get_default_workers
from ..url_parser import (
    UrlParseError,
    parse_owner_target,
)
from ._app import app, console
from ._reports import _display_blocked_results


def _resolve_blocked_owner(org_input: str) -> tuple[str, str]:
    """Resolve the owner login the blocked report should scan.

    Args:
        org_input: A bare login or any GitHub owner URL form.

    Returns:
        The owner login.

    Raises:
        typer.Exit: The input names no recognisable owner.
    """
    # Parse owner login from input (handles a bare login plus every
    # GitHub owner URL form, including /orgs/owner/repositories).
    try:
        organization, owner_host = parse_owner_target(org_input)
    except UrlParseError:
        organization = ""
        owner_host = ""
    if not organization:
        console.print("❌ Invalid GitHub owner name or URL")
        console.print(
            "   Expected an organization or user account, e.g. "
            "'owner-name' or 'https://github.com/owner-name/'"
        )
        raise typer.Exit(1)
    return (organization, owner_host)


def _scan_for_blocked_prs(
    organization: str,
    token: str | None,
    include_drafts: bool,
    progress_tracker: ProgressTracker | None,
    host: str = "",
):
    """Scan every repository of an owner for unmergeable pull requests.

    Args:
        organization: The owner login to scan.
        token: GitHub token, or ``None`` to fall back to the environment.
        include_drafts: Include draft pull requests in the report.
        progress_tracker: Live progress display, when one is running.
        host: The GitHub host the owner lives on.  Carried from the
            parsed argument, so an Enterprise owner URL scans the
            server it named rather than github.com.

    Returns:
        The scan result reported by the GitHub service.
    """
    from ..github_service import GitHubService

    async def _run_blocked_check():
        svc = GitHubService(
            token=token,
            host=host,
            progress_tracker=progress_tracker,
        )
        try:
            return await svc.scan_organization(
                organization, include_drafts=include_drafts
            )
        finally:
            await svc.close()

    return asyncio.run(_run_blocked_check())


def _print_blocked_scan_summary(progress_tracker: ProgressTracker | None) -> None:
    """Stop the progress display and report what the scan covered.

    Args:
        progress_tracker: Live progress display, when one is running.
    """
    if not progress_tracker:
        return

    progress_tracker.stop()
    if progress_tracker.rich_available:
        console.print()  # Add blank line after progress display
    else:
        console.print()  # Clear the fallback display line

    # Show scan summary
    summary = progress_tracker.get_summary()
    elapsed_time = summary.get("elapsed_time")
    total_prs_analyzed = summary.get("total_prs_analyzed")
    completed_repositories = summary.get("completed_repositories")
    errors_count = summary.get("errors_count", 0)
    console.print(f"✅ Check completed in {elapsed_time}")
    console.print(
        f"📊 Analyzed {total_prs_analyzed} PRs across {completed_repositories} repositories"
    )
    if errors_count > 0:
        console.print(f"⚠️ {errors_count} errors encountered during check")
    console.print()  # Add blank line before results


def _run_blocked_fix(
    scan_result,
    progress_tracker: ProgressTracker | None,
    token: str | None,
    reason: str | None,
    limit: int | None,
    *,
    host: str = "",
    workdir: str | None = None,
    keep_temp: bool = False,
    prefetch: int | None = None,
    editor: str | None = None,
    mergetool: bool = False,
    interactive: bool = True,
) -> None:
    """Rebase the blocked PRs that the operator asked ``--fix`` to repair.

    Args:
        scan_result: The completed blocked-PR scan.
        progress_tracker: Live progress display, when one is running.
        token: GitHub token, or ``None`` to fall back to the environment.
        reason: Restrict fixing to PRs blocked for this reason.
        limit: Maximum number of PRs to attempt.
        workdir: Base directory for workspaces.
        keep_temp: Keep the temporary workspace after completion.
        prefetch: Repositories to prepare in parallel.
        editor: Editor command for resolving conflicts.
        mergetool: Use ``git mergetool`` when available.
        interactive: Attach the rebase to the terminal.
    """
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
            host=host,
            progress_tracker=progress_tracker,
            logger=lambda m: console.print(m),
        )
        fix_options = FixOptions(
            workdir=workdir,
            keep_temp=keep_temp,
            prefetch=prefetch if prefetch is not None else get_default_workers(),
            editor=editor,
            mergetool=mergetool,
            interactive=interactive,
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


@app.command()
def blocked(
    org_input: str = typer.Argument(
        ...,
        help="GitHub owner (organization or user) name or URL (e.g., 'lfreleng-actions' or 'https://github.com/lfreleng-actions/')",
    ),
    token: str | None = typer.Option(
        None, "--token", help="GitHub token (or set GITHUB_TOKEN env var)"
    ),
    output_format: str = typer.Option(
        "table", "--format", help="Output format: table, json"
    ),
    include_drafts: bool = typer.Option(
        False,
        "--include-drafts",
        help="Include draft pull requests in the blocked PRs report",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Interactively rebase to resolve conflicts and force-push updates",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Maximum number of PRs to attempt fixing"
    ),
    reason: str | None = typer.Option(
        None,
        "--reason",
        help="Only fix PRs with this blocking reason (e.g., merge_conflict, behind_base)",
    ),
    workdir: str | None = typer.Option(
        None,
        "--workdir",
        help="Base directory for workspaces (defaults to a secure temp dir)",
    ),
    keep_temp: bool = typer.Option(
        False,
        "--keep-temp",
        help="Keep the temporary workspace for inspection after completion",
    ),
    prefetch: int | None = typer.Option(
        None,
        "--prefetch",
        help="Number of repositories to prepare in parallel (auto-detects CPU cores if not specified)",
    ),
    editor: str | None = typer.Option(
        None,
        "--editor",
        help="Editor command to use for resolving conflicts (defaults to $VISUAL or $EDITOR)",
    ),
    mergetool: bool = typer.Option(
        False,
        "--mergetool",
        help="Use 'git mergetool' for resolving conflicts when available",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Attach rebase to the terminal for interactive resolution",
    ),
    show_progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show real-time progress updates"
    ),
):
    """
    Reports blocked pull requests in a GitHub organization or user account.

    This command will:
    1. Check all repositories owned by the organization or user
    2. Identify pull requests that cannot be merged
    3. Report blocking reasons (conflicts, failing checks, etc.)
    4. Count unresolved Copilot feedback comments

    Standard code review requirements are not considered blocking.
    """
    organization, owner_host = _resolve_blocked_owner(org_input)

    progress_tracker = None

    try:
        if show_progress:
            progress_tracker = ProgressTracker(organization)
            progress_tracker.start()
            # Check if Rich display is available
            if not progress_tracker.rich_available:
                console.print(f"🔍 Checking owner: {organization}")
                console.print("Progress updates will be shown as simple text...")
        else:
            console.print(f"🔍 Checking owner: {organization}")
            console.print(
                "This may take a few minutes for owners with many repositories..."
            )

        # Perform the scan
        scan_result = _scan_for_blocked_prs(
            organization, token, include_drafts, progress_tracker, owner_host
        )

        _print_blocked_scan_summary(progress_tracker)

        # Display results
        _display_blocked_results(scan_result, output_format)

        # Optional fix workflow
        if fix:
            _run_blocked_fix(
                scan_result,
                progress_tracker,
                token,
                reason,
                limit,
                host=owner_host,
                workdir=workdir,
                keep_temp=keep_temp,
                prefetch=prefetch,
                editor=editor,
                mergetool=mergetool,
                interactive=interactive,
            )

    except DependamergeError as exc:
        # Our structured errors handle display and exit themselves
        if progress_tracker:
            progress_tracker.stop()
        exc.display_and_exit()
    except (KeyboardInterrupt, SystemExit):
        # Don't catch system interrupts or exits
        if progress_tracker:
            progress_tracker.stop()
        raise
    except typer.Exit as e:
        if progress_tracker:
            progress_tracker.stop()
        raise e
    except (GitError, RateLimitError, SecondaryRateLimitError, GraphQLError) as exc:
        # Convert known errors to centralized error handling
        if progress_tracker:
            progress_tracker.stop()
        if isinstance(exc, GitError):
            converted_error = convert_git_error(exc)
        else:  # GitHub API errors
            converted_error = convert_github_api_error(exc)
        converted_error.display_and_exit()
    except Exception as e:
        # Ensure progress tracker is stopped even if an error occurs
        if progress_tracker:
            progress_tracker.stop()

        # Try to categorize the error
        if is_github_api_permission_error(e):
            exit_for_github_api_error(exception=e)
        elif is_network_error(e):
            converted_error = convert_network_error(e)
            converted_error.display_and_exit()
        else:
            exit_with_error(
                ExitCode.GENERAL_ERROR,
                message="❌ Error during owner scan",
                details=str(e),
                exception=e,
            )
