# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The ``close`` command and its options.
"""

from urllib.parse import urlparse

import typer

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
import dependamerge.cli as _pkg

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
from ..progress_tracker import MergeProgressTracker
from ..url_parser import (
    normalize_target,
    set_github_host,  # noqa: F401
)
from ._app import app, console
from ._close import (
    _CloseContext,
    _find_similar_prs_for_close,
    _print_close_analysis_summary,
    _print_close_debug_matching,
    _run_close_dry_run,
    _run_immediate_close,
    _run_interactive_close,
    _validate_close_authorization,
)
from ._pr_display import _display_pr_info


@app.command()
def close(
    pr_url: str = typer.Argument(..., help="GitHub pull request URL"),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help="Skip confirmation prompt and close immediately without preview",
    ),
    similarity_threshold: float = typer.Option(
        0.8, "--threshold", help="Similarity threshold for matching PRs (0.0-1.0)"
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
    override: str | None = typer.Option(
        None, "--override", help="SHA hash to override non-automation PR restriction"
    ),
    show_progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show real-time progress updates"
    ),
    debug_matching: bool = typer.Option(
        False,
        "--debug-matching",
        help="Show detailed scoring information for PR matching",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Analyze and preview only: never close anything. Suppresses "
            "the confirmation prompt so it runs unattended under a "
            "read-only token (e.g. in CI)."
        ),
    ),
):
    """
    Bulk close pull requests across a GitHub owner (organization or user
    account).

    By default, runs in interactive mode showing what changes will apply,
    then prompts to proceed with closing. Use --no-confirm to close immediately.

    This command will:

    1. Analyze the provided PR

    2. Find similar PRs across the owner (organization or user account)

    3. Close matching PRs

    Closes similar PRs from the same automation tool (dependabot, pre-commit.ci).

    For user generated bulk PRs, use the --override flag with SHA hash.
    """
    # Applied before anything parses a target: the flag sets both
    # the host a shorthand resolves against and the set of hosts
    # permitted at all.
    set_github_host(github_host)

    progress_tracker = None

    try:
        # Derive the host from the URL before building the client: a
        # GitHub Enterprise Server install serves its API from
        # different base URLs, and the client bakes those in at
        # construction.  Normalise first, or a scheme-less target is
        # read as a bare path and yields no host at all.
        github_client = _pkg.GitHubClient(
            token,
            host=(urlparse(normalize_target(pr_url)).hostname or "").lower() or None,
        )
        # GitHubClient resolves None -> GITHUB_TOKEN env var (raises if missing)
        assert github_client.token is not None
        token = github_client.token
        owner, repo_name, pr_number = github_client.parse_pr_url(pr_url)

        if show_progress:
            progress_tracker = MergeProgressTracker(owner, is_close_operation=True)
            progress_tracker.start()
            # Check if Rich display is available
            if not progress_tracker.rich_available:
                console.print(f"🔍 Examining source pull request in {owner}...")
                console.print("Progress updates will be shown as simple text...")
        else:
            console.print(f"🔍 Examining source pull request in {owner}...")

        source_pr = github_client.get_pull_request_info(owner, repo_name, pr_number)

        # Display source PR info
        _display_pr_info(
            source_pr, "", github_client, progress_tracker=progress_tracker
        )

        comparator = _pkg.PRComparator(similarity_threshold)

        ctx = _CloseContext(
            token=token,
            github_client=github_client,
            host=github_client.host,
            owner=owner,
            repo_name=repo_name,
            pr_number=pr_number,
            source_pr=source_pr,
            progress_tracker=progress_tracker,
        )

        # Debug matching info for source PR
        if debug_matching:
            _print_close_debug_matching(ctx, comparator, similarity_threshold)

        # Check if source PR is from automation or has valid override
        if not _validate_close_authorization(ctx, override):
            return

        # Find similar PRs across the owner (organization or user)
        all_similar_prs = _find_similar_prs_for_close(ctx, comparator, debug_matching)
        _print_close_analysis_summary(ctx, all_similar_prs)

        if not no_confirm:
            # IMPORTANT: Each PR must produce exactly ONE line of output in this section
            console.print("\n🔍 Dependamerge Evaluation\n")

        # Determine which PRs to close
        all_prs_to_close = [source_pr] + [pr for pr, _ in all_similar_prs]

        if dry_run:
            # Dry run: evaluate in preview mode and report what *would*
            # be closed, then stop.  No prompt, no actual close — safe to
            # run unattended under a read-only token.
            _run_close_dry_run(ctx, all_prs_to_close)
            return

        if no_confirm:
            # No confirmation - close immediately
            _run_immediate_close(ctx, all_prs_to_close)
        else:
            _run_interactive_close(ctx, all_prs_to_close)

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
    except typer.Exit:
        if progress_tracker:
            progress_tracker.stop()
        # Re-raise without additional error messages
        raise
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
        # Ensure progress tracker is stopped even if an unexpected error occurs
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
                message="❌ Error during close operation",
                details=str(e),
                exception=e,
            )
