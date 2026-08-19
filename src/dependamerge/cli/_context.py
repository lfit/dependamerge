# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Shared merge context plus input validation and source-PR fetch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import requests
import typer
import urllib3.exceptions

from ..error_codes import (
    ExitCode,
    exit_for_github_api_error,
    exit_for_pr_state_error,
    exit_with_error,
    is_github_api_permission_error,
    is_network_error,
)
from ..models import ComparisonResult, PullRequestInfo
from ..progress_tracker import MergeProgressTracker

if TYPE_CHECKING:
    from ..github_client import GitHubClient
    from ..pr_comparator import PRComparator

from . import _deps
from ._app import DEFAULT_MAX_WAIT, console
from ._display import (
    _display_pr_info,
)


@dataclass
class _MergeContext:
    """Shared state threaded through the merge sub-routines."""

    # CLI parameters
    pr_url: str
    no_confirm: bool
    similarity_threshold: float
    merge_method: str
    token: str | None
    override: str | None
    no_fix: bool
    merge_timeout: float
    show_progress: bool
    debug_matching: bool
    dismiss_copilot: bool
    force: str
    verbose: bool
    no_netrc: bool
    netrc_file: Path | None
    netrc_optional: bool
    github2gerrit_mode: str
    include_human_prs: bool = False
    # Optional behaviour flag; defaulted so a context can be built
    # without every construction site opting in explicitly.
    fix_semantic_title: bool = True
    rebase_local: bool = True
    # Dry-run: perform the full analysis and preview but never merge,
    # approve, rebase, or close anything.  Because no write occurs, the
    # write-permission pre-flight check is skipped so the command can run
    # under a read-only token (e.g. in CI).  Implies preview-only and
    # suppresses the interactive confirmation prompt.
    dry_run: bool = False
    # Owner-wide global wait ceiling (seconds).  Default 900 (15 min);
    # 0 = fire-and-forget (arm auto-merge, report pending, never block).
    # Applies to owner/user-wide runs; ignored for single-PR and
    # single-repository merges.
    max_wait: float = DEFAULT_MAX_WAIT

    # Derived / mutable state
    github_client: GitHubClient | None = None
    owner: str = ""
    repo_name: str = ""
    pr_number: int = 0
    comparator: PRComparator | None = None
    source_pr: PullRequestInfo | None = None
    progress_tracker: MergeProgressTracker | None = None
    all_similar_prs: list[tuple[PullRequestInfo, ComparisonResult]] = field(
        default_factory=list
    )


def _validate_merge_inputs(
    submit_gerrit_changes: bool,
    skip_gerrit_changes: bool,
    ignore_github2gerrit: bool,
    force: str,
    verbose: bool,
) -> str:
    """Validate CLI flags and configure logging.

    Returns the effective github2gerrit_mode string.

    Raises:
        typer.Exit: On mutually exclusive flags or invalid force level.
    """
    g2g_flags_set = sum(
        [submit_gerrit_changes, skip_gerrit_changes, ignore_github2gerrit]
    )
    if g2g_flags_set > 1:
        console.print(
            "❌ Error: --submit-gerrit-changes, --skip-gerrit-changes, and "
            "--ignore-github2gerrit are mutually exclusive."
        )
        raise typer.Exit(1)

    if skip_gerrit_changes:
        github2gerrit_mode = "skip"
    elif ignore_github2gerrit:
        github2gerrit_mode = "ignore"
    else:
        github2gerrit_mode = "submit"

    if verbose:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.getLogger("dependamerge").setLevel(logging.DEBUG)
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s - %(message)s",
        )

    valid_force_levels = [
        "none",
        "code-owners",
        "protection-rules",
        "all",
    ]
    if force not in valid_force_levels:
        console.print(
            f"Error: Invalid --force level '{force}'. "
            f"Must be one of: {', '.join(valid_force_levels)}"
        )
        raise typer.Exit(1)

    if force == "all":
        console.print("⚠️ Warning: Using --force=all will bypass most safety checks.")
        console.print("   This may attempt merges that will fail at GitHub API level.")

    return github2gerrit_mode


def _init_github_merge(ctx: _MergeContext) -> None:
    """Initialise the GitHub client, progress tracker, and comparator.

    Populates *ctx* in-place with the resolved objects.
    """
    ctx.github_client = _deps.GitHubClient(ctx.token)
    assert ctx.github_client.token is not None
    ctx.token = ctx.github_client.token
    ctx.owner, ctx.repo_name, ctx.pr_number = ctx.github_client.parse_pr_url(ctx.pr_url)

    if ctx.show_progress:
        # Create the tracker but do NOT start it yet — it will be
        # started in _scan_and_find_similar() when the long-running
        # org scan begins.  The early phases (PR fetch, permissions
        # check) are fast and use plain console output instead.
        ctx.progress_tracker = MergeProgressTracker(ctx.owner)

    console.print(f"🔍 Examining source pull request in {ctx.owner}...")

    ctx.comparator = _deps.PRComparator(ctx.similarity_threshold)


def _fetch_and_validate_source_pr(ctx: _MergeContext) -> None:
    """Fetch the source PR and validate that it is open.

    Populates *ctx.source_pr*.
    """
    assert ctx.github_client is not None

    try:
        ctx.source_pr = ctx.github_client.get_pull_request_info(
            ctx.owner, ctx.repo_name, ctx.pr_number
        )
        if ctx.source_pr.state != "open":
            if ctx.progress_tracker:
                ctx.progress_tracker.stop()
            exit_for_pr_state_error(
                ctx.pr_number,
                "closed",
                details="Pull request has been closed",
            )
    except (
        urllib3.exceptions.NameResolutionError,
        urllib3.exceptions.MaxRetryError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.RequestException,
    ) as e:
        if is_network_error(e):
            exit_with_error(
                ExitCode.NETWORK_ERROR,
                details="Failed to fetch PR details from GitHub API",
                exception=e,
            )
        elif is_github_api_permission_error(e):
            exit_for_github_api_error(details="Failed to fetch PR details", exception=e)
        else:
            exit_with_error(
                ExitCode.GENERAL_ERROR,
                message="❌ Failed to fetch PR details",
                details=str(e),
                exception=e,
            )

    assert ctx.source_pr is not None

    _display_pr_info(
        ctx.source_pr,
        "",
        ctx.github_client,
    )


def _source_pr_modifies_workflows(ctx: _MergeContext) -> bool:
    """Return True when the source PR changes GitHub Actions workflow files.

    Merging such a PR through the REST API requires the classic
    ``workflow`` token scope (or the fine-grained ``Workflows: Read and
    write`` permission), which is a *separate* gate from plain repository
    write access.  Detecting this up-front lets the pre-flight check verify
    the scope instead of failing only at merge time.
    """
    source_pr = ctx.source_pr
    if source_pr is None:
        return False
    for change in source_pr.files_changed:
        path = getattr(change, "filename", "") or ""
        if path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml")):
            return True
    return False


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
