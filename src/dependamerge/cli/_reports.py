# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Table and JSON rendering for the reporting commands.

``blocked`` and ``status`` both emit either a Rich table or JSON; the
formatting for each lives here so the commands stay declarations of their
options.
"""

from rich.table import Table

from ..github_service import AUTOMATION_TOOLS
from ._app import console


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
