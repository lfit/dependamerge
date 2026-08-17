# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The ``status`` Typer command and its organization status report.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import typer
from rich.table import Table

from ..github_service import AUTOMATION_TOOLS
from ..progress_tracker import ProgressTracker
from ..url_parser import (
    UrlParseError,
    parse_owner_arg,
)

if TYPE_CHECKING:
    pass

from ._app import app, console


@app.command()
def status(
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
    show_progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show real-time progress updates"
    ),
):
    """
    Reports repository statistics for tags, releases and pull requests.

    This command will:
    1. Scan all repositories owned by the organization or user
    2. Gather tag and release information
    3. Count open and merged pull requests
    4. Identify PRs affecting actions or workflows

    Automation tools supported: Dependabot, Renovate, pre-commit.ci,
    GitHub Actions, GitHub Copilot, and any other [bot] account.
    """
    # Parse owner login from input (handles a bare login plus every
    # GitHub owner URL form, including /orgs/owner/repositories).
    try:
        org_name = parse_owner_arg(org_input)
    except UrlParseError:
        org_name = ""
    if not org_name:
        console.print("❌ Invalid GitHub owner name or URL")
        console.print(
            "   Expected an organization or user account, e.g. "
            "'owner-name' or 'https://github.com/owner-name/'"
        )
        raise typer.Exit(1)

    # Initialize progress tracker (disable PR stats for status command)
    progress_tracker = None

    try:
        if show_progress:
            progress_tracker = ProgressTracker(org_name, show_pr_stats=False)
            progress_tracker.start()
            if not progress_tracker.rich_available:
                console.print(f"🔍 Scanning owner: {org_name}")
                console.print("Progress updates will be shown as simple text...")
        else:
            console.print(f"🔍 Scanning owner: {org_name}")
            console.print(
                "This may take a few minutes for owners with many repositories..."
            )

        # Perform the scan
        from ..github_service import GitHubService

        async def _run_status_check():
            svc = GitHubService(token=token, progress_tracker=progress_tracker)
            try:
                return await svc.gather_organization_status(org_name)
            finally:
                await svc.close()

        status_result = asyncio.run(_run_status_check())

        if progress_tracker:
            progress_tracker.stop()
            if progress_tracker.rich_available:
                console.print()
            else:
                console.print()

            # Show scan summary
            summary = progress_tracker.get_summary()
            elapsed_time = summary.get("elapsed_time")
            console.print(f"\n✅ Scan completed in {elapsed_time}")
            console.print()

        # Display results
        _display_status_results(status_result, output_format)

    except KeyboardInterrupt:
        if progress_tracker:
            progress_tracker.stop()
        console.print("\n⚠️ Scan interrupted by user")
        raise typer.Exit(130) from None
    except Exception as e:
        if progress_tracker:
            progress_tracker.stop()
        console.print(f"❌ Error during scan: {e}")
        raise typer.Exit(1) from e


def _display_status_results(status_result, output_format: str):
    """Display the organization status results."""

    if output_format == "json":
        import json

        console.print(json.dumps(status_result.model_dump(), indent=2, default=str))
        return

    # Table format
    if not status_result.repository_statuses:
        console.print("❌ No repositories found in organization!")
        return

    status_table = Table(title=f"Organization: {status_result.organization}")
    status_table.add_column("Repository", style="cyan")
    status_table.add_column("Tag", style="white")
    status_table.add_column("Date", style="white")
    status_table.add_column("PRs Open", style="white")
    status_table.add_column("PRs Merged", style="white")
    status_table.add_column("Action", style="white")
    status_table.add_column("Workflows", style="white")

    for repo in status_result.repository_statuses:
        # Format tag with icon
        tag_display = "—"
        if repo.latest_tag:
            tag_display = f"{repo.status_icon} {repo.latest_tag}"

        # Format date
        date_display = repo.tag_date or repo.release_date or "—"

        # Format PR counts
        open_prs = f"{repo.open_prs_human} / {repo.open_prs_automation}"
        merged_prs = f"{repo.merged_prs_human} / {repo.merged_prs_automation}"
        action_prs = f"{repo.action_prs_human} / {repo.action_prs_automation}"
        workflow_prs = f"{repo.workflow_prs_human} / {repo.workflow_prs_automation}"

        status_table.add_row(
            repo.repository_name,
            tag_display,
            date_display,
            open_prs,
            merged_prs,
            action_prs,
            workflow_prs,
        )

    console.print(status_table)
    console.print()
    console.print("PR counts are for human/automation")
    console.print("\nAutomation tools supported:")
    special_tool_labels = {
        "[bot]": "Any other [bot] account",
        "pre-commit": "pre-commit.ci",
        "github-actions": "GitHub Actions",
        "copilot": "GitHub Copilot",
    }
    for tool in AUTOMATION_TOOLS:
        label = special_tool_labels.get(tool, tool.capitalize())
        console.print(f"  • {label}")
    console.print()

    summary_table = Table()
    summary_table.add_column("Summary", style="cyan")
    summary_table.add_column("Value", style="white")

    # Aggregate open-PR counts across all scanned repositories.  The
    # per-repository "PRs Open" column shows human / automation, so the
    # summary totals those same open-PR figures and reports the split
    # before the combined total.
    total_automation_prs = sum(
        repo.open_prs_automation for repo in status_result.repository_statuses
    )
    total_human_prs = sum(
        repo.open_prs_human for repo in status_result.repository_statuses
    )

    summary_table.add_row("🤖 Automation PRs", str(total_automation_prs))
    summary_table.add_row("🤷 Human      PRs", str(total_human_prs))
    summary_table.add_section()
    summary_table.add_row("Total PRs", str(total_automation_prs + total_human_prs))
    summary_table.add_row("Total Repositories", str(status_result.total_repositories))

    # Only show Scanned Repositories if it differs from Total
    if status_result.scanned_repositories != status_result.total_repositories:
        summary_table.add_row(
            "Scanned Repositories", str(status_result.scanned_repositories)
        )

    if status_result.errors:
        summary_table.add_row("Errors", str(len(status_result.errors)), style="red")

    console.print(summary_table)

    # Show errors if any
    if status_result.errors:
        console.print()
        error_table = Table(title="Errors Encountered During Scan")
        error_table.add_column("Error", style="red")

        for error in status_result.errors:
            error_table.add_row(error)

        console.print(error_table)
