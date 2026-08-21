# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The ``blocked`` Typer command and its report rendering.
"""

from __future__ import annotations

import asyncio

import typer
from rich.table import Table

from ..error_codes import (
    DependamergeError,
    ExitCode,
    convert_git_error,
    convert_github_api_error,
    convert_network_error,
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
from ..url_parser import (
    UrlParseError,
    parse_owner_arg,
)
from ._app import app, console
from ._blocked_fix import _FixRequest, _run_blocked_fix


def _resolve_blocked_owner(org_input: str) -> str:
    """Parse the owner login, exiting with guidance when it is unusable."""
    try:
        organization = parse_owner_arg(org_input)
    except UrlParseError:
        organization = ""
    if not organization:
        console.print("❌ Invalid GitHub owner name or URL")
        console.print(
            "   Expected an organization or user account, e.g. "
            "'owner-name' or 'https://github.com/owner-name/'"
        )
        raise typer.Exit(1)
    return organization


def _run_blocked_scan(
    organization: str,
    token: str | None,
    include_drafts: bool,
    progress_tracker: ProgressTracker | None,
):
    """Scan the owner for blocked PRs and report the scan summary."""
    from ..github_service import GitHubService

    async def _run_blocked_check():
        svc = GitHubService(token=token, progress_tracker=progress_tracker)
        try:
            return await svc.scan_organization(
                organization, include_drafts=include_drafts
            )
        finally:
            await svc.close()

    scan_result = asyncio.run(_run_blocked_check())

    if progress_tracker:
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

    return scan_result


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
    # Parse owner login from input (handles a bare login plus every
    # GitHub owner URL form, including /orgs/owner/repositories).
    organization = _resolve_blocked_owner(org_input)

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

        scan_result = _run_blocked_scan(
            organization, token, include_drafts, progress_tracker
        )

        # Display results
        _display_blocked_results(scan_result, output_format)

        # Optional fix workflow
        if fix:
            _run_blocked_fix(
                scan_result,
                token,
                progress_tracker,
                reason,
                limit,
                _FixRequest(
                    workdir=workdir,
                    keep_temp=keep_temp,
                    prefetch=prefetch,
                    editor=editor,
                    mergetool=mergetool,
                    interactive=interactive,
                ),
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


def _display_blocked_results(scan_result, output_format: str):
    """Display the organization blocked PR results."""

    if output_format == "json":
        import json

        console.print(json.dumps(scan_result.model_dump(), indent=2, default=str))
        return

    # Table format
    if not scan_result.unmergeable_prs:
        console.print("🎉 No unmergeable pull requests found!")
        return

    pr_table = Table(title=f"Blocked Pull Requests: {scan_result.organization}")
    pr_table.add_column("Repository", style="cyan")
    pr_table.add_column("PR", style="white")
    pr_table.add_column("Title", style="white", max_width=40)
    pr_table.add_column("Author", style="white")
    pr_table.add_column("Blocking Reasons", style="yellow")

    # Only show Copilot column if there are any copilot comments
    show_copilot_col = any(
        p.copilot_comments_count > 0 for p in scan_result.unmergeable_prs
    )
    if show_copilot_col:
        pr_table.add_column("Copilot", style="blue")

    for pr in scan_result.unmergeable_prs:
        reasons = [reason.description for reason in pr.reasons]
        reasons_text = "\n".join(reasons) if reasons else "Unknown"

        row_data = [
            pr.repository.split("/", 1)[1] if "/" in pr.repository else pr.repository,
            f"#{pr.pr_number}",
            pr.title,
            pr.author,
            reasons_text,
        ]

        # Add Copilot count if column is shown
        if show_copilot_col:
            row_data.append(str(pr.copilot_comments_count))

        pr_table.add_row(*row_data)

    console.print(pr_table)
    console.print()

    # Create summary table (moved to bottom)
    summary_table = Table()
    summary_table.add_column("Summary", style="cyan")
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Total Repositories", str(scan_result.total_repositories))
    summary_table.add_row("Checked Repositories", str(scan_result.scanned_repositories))
    summary_table.add_row("Total Open PRs", str(scan_result.total_prs))
    summary_table.add_row("Unmergeable PRs", str(len(scan_result.unmergeable_prs)))

    if scan_result.errors:
        summary_table.add_row("Errors", str(len(scan_result.errors)), style="red")

    console.print(summary_table)

    # Show errors if any
    if scan_result.errors:
        console.print()
        error_table = Table(title="Errors Encountered During Check")
        error_table.add_column("Error", style="red")

        for error in scan_result.errors:
            error_table.add_row(error)

        console.print(error_table)
