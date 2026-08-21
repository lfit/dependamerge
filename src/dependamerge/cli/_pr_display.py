# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Pull request detail rendering.

Prints the source PR's identifying fields, and the per-field matching
breakdown that ``--debug-matching`` asks for.
"""

from rich.table import Table

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
import dependamerge.cli as _pkg

from ..models import PullRequestInfo
from ..progress_tracker import ProgressTracker
from ._app import console
from ._context import _MergeContext


def _print_debug_matching(ctx: _MergeContext) -> None:
    """Print debug matching information for the source PR."""
    assert ctx.github_client is not None
    assert ctx.source_pr is not None
    assert ctx.comparator is not None

    console.print("\n🔍 Debug Matching Information")
    console.print(
        "   Source PR automation status: "
        f"{ctx.github_client.is_automation_author(ctx.source_pr.author)}"
    )
    console.print(
        "   Extracted package: "
        f"'{ctx.comparator._extract_package_name(ctx.source_pr.title)}'"
    )
    console.print(f"   Similarity threshold: {ctx.similarity_threshold}")
    if ctx.source_pr.body:
        console.print(f"   Body preview: {ctx.source_pr.body[:100]}...")
        console.print(
            "   Is dependabot body: "
            f"{ctx.comparator._is_dependabot_body(ctx.source_pr.body)}"
        )
    else:
        console.print("   ⚠️ Source PR has no body")
    console.print()


def _display_pr_info(
    pr: PullRequestInfo,
    title: str,
    github_client: _pkg.GitHubClient,
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
