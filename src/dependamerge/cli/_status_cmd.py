# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The ``status`` command and its options.
"""

import asyncio

import typer

from ..progress_tracker import ProgressTracker
from ..url_parser import (
    UrlParseError,
    parse_owner_target,
)
from ._app import app, console
from ._github_host import apply_github_host
from ._reports import _display_status_results


@app.command()
def status(
    org_input: str = typer.Argument(
        ...,
        help="GitHub owner (organization or user) name or URL (e.g., 'lfreleng-actions' or 'https://github.com/lfreleng-actions/')",
    ),
    token: str | None = typer.Option(
        None, "--token", help="GitHub token (or set GITHUB_TOKEN env var)"
    ),
    github_host: str | None = typer.Option(
        None,
        "--github-host",
        help=(
            "GitHub host to address, e.g. a GitHub Enterprise Server "
            "install. Takes priority over DEPENDAMERGE_GITHUB_HOST and "
            "GH_HOST, and declares the host as permitted."
        ),
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
    # Applied before anything parses a target: the flag sets both
    # the host a shorthand resolves against and the set of hosts
    # permitted at all.
    apply_github_host(github_host)

    # Parse owner login from input (handles a bare login plus every
    # GitHub owner URL form, including /orgs/owner/repositories).
    try:
        org_name, owner_host = parse_owner_target(org_input)
    except UrlParseError as exc:
        # The parser's own message carries the remedy --- which
        # host to declare, and how.  Swallowing it left an
        # Enterprise operator with only "invalid owner".
        console.print(f"❌ {exc}")
        raise typer.Exit(1) from None
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
            svc = GitHubService(
                token=token,
                host=owner_host,
                progress_tracker=progress_tracker,
            )
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
