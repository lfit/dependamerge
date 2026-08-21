# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Similarity formatting and Rich table rendering for PRs and changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from ..gerrit import (
    GerritChangeInfo,
    GerritComparisonResult,
)
from ..models import PullRequestInfo
from ..progress_tracker import ProgressTracker

if TYPE_CHECKING:
    from ..github_client import GitHubClient

from ._app import console


def _format_condensed_similarity(comparison) -> str:
    """Format similarity comparison result in condensed format."""
    reasons = comparison.reasons

    # Check if same author is present
    has_same_author = any("Same automation author" in reason for reason in reasons)

    score_parts = []
    for reason in reasons:
        if "Similar titles (score:" in reason:
            score = reason.split("score: ")[1].replace(")", "")
            score_parts.append(f"title {score}")
        elif "Similar PR descriptions (score:" in reason:
            score = reason.split("score: ")[1].replace(")", "")
            score_parts.append(f"descriptions {score}")
        elif "Similar file changes (score:" in reason:
            score = reason.split("score: ")[1].replace(")", "")
            score_parts.append(f"changes {score}")

    if has_same_author:
        author_text = "Same author; "
    else:
        author_text = ""

    total_score = f"total score: {comparison.confidence_score:.2f}"

    if score_parts:
        breakdown = f" [{', '.join(score_parts)}]"
    else:
        breakdown = ""

    return f"{author_text}{total_score}{breakdown}"


def _display_change_info(
    change: GerritChangeInfo,
    title: str = "",
    console: Console = console,
    auth_method: str | None = None,
) -> None:
    """Display Gerrit change information in a formatted table.

    Args:
        change: The Gerrit change info to display.
        title: Optional title for the table.
        console: Rich console for output.
        auth_method: Description of authentication method used (e.g., ".netrc file").
    """

    table = Table(title=title if title else None)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    # Map Gerrit status to user-friendly description
    status_map = {
        "NEW": "Open (awaiting review)",
        "MERGED": "Merged",
        "ABANDONED": "Abandoned",
    }
    status_display = status_map.get(change.status, change.status)

    # Check if change is submittable (check merge conflicts first!)
    if change.status == "NEW":
        if change.mergeable is False:
            status_display = "Has merge conflicts"
        elif change.submittable:
            status_display = "Ready to submit"

    table.add_row("Project", change.project)
    table.add_row("Change Number", str(change.number))
    table.add_row("Subject", change.subject)
    table.add_row("Owner", change.owner)
    table.add_row("Branch", change.branch)
    table.add_row("State", change.status)
    table.add_row("Status", status_display)
    if change.files_changed:
        table.add_row("Files Changed", str(len(change.files_changed)))
    if change.url:
        table.add_row("URL", change.url)
    if auth_method:
        table.add_row("Auth Method", auth_method)

    console.print(table)


def _format_gerrit_similarity(comparison: GerritComparisonResult) -> str:
    """Format Gerrit comparison result in condensed format."""
    reasons = comparison.reasons

    # Check if same author is present
    has_same_author = any("Same automation author" in reason for reason in reasons)

    if has_same_author:
        author_text = "Same author; "
    else:
        author_text = ""

    total_score = f"total score: {comparison.confidence_score:.2f}"

    score_parts = []
    for reason in reasons:
        if "Similar subjects" in reason and "score:" in reason:
            score = reason.split("score: ")[1].replace(")", "")
            score_parts.append(f"subject {score}")
        elif "Similar files" in reason and "score:" in reason:
            score = reason.split("score: ")[1].replace(")", "")
            score_parts.append(f"files {score}")

    if score_parts:
        breakdown = f" [{', '.join(score_parts)}]"
    else:
        breakdown = ""

    return f"{author_text}{total_score}{breakdown}"


def _display_pr_info(
    pr: PullRequestInfo,
    title: str,
    github_client: GitHubClient,
    progress_tracker: ProgressTracker | None = None,
) -> None:
    """Display pull request information in a formatted table."""
    table = Table(title=title)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    status = github_client.get_pr_status_details(pr)

    table.add_row("Repository", pr.repository_full_name)
    table.add_row("PR Number", str(pr.number))
    table.add_row("Title", pr.title)
    table.add_row("Author", pr.author)
    table.add_row("State", pr.state)
    table.add_row("Status", status)
    table.add_row("Files Changed", str(len(pr.files_changed)))
    table.add_row("URL", pr.html_url)

    if progress_tracker:
        progress_tracker.suspend()
    console.print(table)
    if progress_tracker:
        progress_tracker.resume()
