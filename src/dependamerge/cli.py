# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

import asyncio
import hashlib
import logging
import os
import sys

import requests
import typer
import urllib3.exceptions
from rich.console import Console
from rich.table import Table

from ._version import __version__
from .close_manager import AsyncCloseManager
from .error_codes import (
    DependamergeError,
    ExitCode,
    convert_git_error,
    convert_github_api_error,
    convert_network_error,
    exit_for_configuration_error,
    exit_for_github_api_error,
    exit_for_pr_state_error,
    exit_with_error,
    is_github_api_permission_error,
    is_network_error,
)
from .gerrit import (
    GerritAuthError,
    GerritChangeInfo,
    GerritComparisonResult,
    GerritRestError,
    create_gerrit_comparator,
    create_gerrit_service,
    create_submit_manager,
)
from .git_ops import GitError
from .github_async import (
    GitHubAsync,
    GraphQLError,
    RateLimitError,
    SecondaryRateLimitError,
)
from .github_async import (
    PermissionError as GitHubPermissionError,
)
from .github_client import GitHubClient
from .github_service import AUTOMATION_TOOLS
from .merge_manager import AsyncMergeManager
from .models import PullRequestInfo
from .pr_comparator import PRComparator
from .progress_tracker import MergeProgressTracker, ProgressTracker
from .resolve_conflicts import FixOptions, FixOrchestrator, PRSelection
from .system_utils import get_default_workers
from .url_parser import ParsedUrl, UrlParseError, parse_change_url

# Constants
MAX_RETRIES = 2


def version_callback(value: bool):
    """Callback to show version and exit."""
    if value:
        console.print(f"🏷️  dependamerge version {__version__}")
        raise typer.Exit()


class CustomTyper(typer.Typer):
    """Custom Typer class that shows version in help."""

    def __call__(self, *args, **kwargs):
        # Check if help is being requested
        if "--help" in sys.argv or "-h" in sys.argv:
            console.print(f"🏷️  dependamerge version {__version__}")
        return super().__call__(*args, **kwargs)


app = CustomTyper(
    help="Find blocked PRs in GitHub organizations and automatically merge pull requests"
)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """
    Dependamerge command line interface.
    """
    # The actual handling is done via the version_callback.
    # This callback exists only to expose --version at the top level.
    pass


console = Console(markup=False)


def _generate_override_sha(
    pr_info: PullRequestInfo, commit_message_first_line: str
) -> str:
    """
    Generate a SHA hash based on PR author info and commit message.

    Args:
        pr_info: Pull request information containing author details
        commit_message_first_line: First line of the commit message to use as salt

    Returns:
        SHA256 hash string
    """
    # Create a string combining author info and commit message first line
    combined_data = f"{pr_info.author}:{commit_message_first_line.strip()}"

    # Generate SHA256 hash
    sha_hash = hashlib.sha256(combined_data.encode("utf-8")).hexdigest()

    # Return first 16 characters for readability
    return sha_hash[:16]


def _validate_override_sha(
    provided_sha: str, pr_info: PullRequestInfo, commit_message_first_line: str
) -> bool:
    """
    Validate that the provided SHA matches the expected one for this PR.

    Args:
        provided_sha: SHA provided by user via --override flag
        pr_info: Pull request information
        commit_message_first_line: First line of commit message

    Returns:
        True if SHA is valid, False otherwise
    """
    expected_sha = _generate_override_sha(pr_info, commit_message_first_line)
    return provided_sha == expected_sha


def _generate_continue_sha(
    pr_info: PullRequestInfo, commit_message_first_line: str
) -> str:
    """
    Generate a SHA hash for continuing after preview evaluation.

    Args:
        pr_info: Source pull request information
        commit_message_first_line: First line of the commit message

    Returns:
        SHA256 hash string for continuation
    """
    # Create a string combining source PR info for preview continuation
    combined_data = f"continue:{pr_info.repository_full_name}#{pr_info.number}:{commit_message_first_line.strip()}"

    # Generate SHA256 hash
    sha_hash = hashlib.sha256(combined_data.encode("utf-8")).hexdigest()

    # Return first 16 characters for readability
    return sha_hash[:16]


def _format_condensed_similarity(comparison) -> str:
    """Format similarity comparison result in condensed format."""
    reasons = comparison.reasons

    # Check if same author is present
    has_same_author = any("Same automation author" in reason for reason in reasons)

    # Extract individual scores from reasons
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

    # Build condensed format
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
) -> None:
    """Display Gerrit change information in a formatted table."""

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

    console.print(table)


def _format_gerrit_similarity(comparison: GerritComparisonResult) -> str:
    """Format Gerrit comparison result in condensed format."""
    reasons = comparison.reasons

    # Check if same author is present
    has_same_author = any("Same automation author" in reason for reason in reasons)

    # Build condensed format
    if has_same_author:
        author_text = "Same author; "
    else:
        author_text = ""

    total_score = f"total score: {comparison.confidence_score:.2f}"

    # Extract individual scores from reasons
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


def _handle_gerrit_merge(
    parsed_url: ParsedUrl,
    no_confirm: bool,
    similarity_threshold: float,
    verbose: bool,
    console: Console,
) -> None:
    """
    Handle merge operation for a Gerrit change URL.

    Args:
        parsed_url: Parsed Gerrit URL with host, project, and change number.
        no_confirm: If True, skip confirmation prompt.
        similarity_threshold: Threshold for matching similar changes.
        verbose: Enable verbose output.
        console: Rich console for output.
    """
    # Get Gerrit credentials from environment
    gerrit_username = os.getenv("GERRIT_USERNAME")
    gerrit_password = os.getenv("GERRIT_PASSWORD")

    if not gerrit_username or not gerrit_password:
        console.print("❌ Gerrit credentials not found in environment.")
        console.print("   Set GERRIT_USERNAME and GERRIT_PASSWORD environment variables.")
        console.print("   Tip: Source your .secrets.gerrit file and run use_lf or use_onap")
        raise typer.Exit(1)

    console.print(f"🔍 Examining Gerrit change on {parsed_url.host}...")

    try:
        # Create Gerrit service
        service = create_gerrit_service(
            host=parsed_url.host,
            base_path=parsed_url.base_path,
            username=gerrit_username,
            password=gerrit_password,
        )

        if not service.is_authenticated:
            console.print("⚠️  Warning: Service created but may not be authenticated")

        # Get the source change info
        console.print(f"📋 Fetching change {parsed_url.change_number}...")
        source_change = service.get_change_info(parsed_url.change_number)

        if source_change is None:
            console.print(f"❌ Change {parsed_url.change_number} not found")
            raise typer.Exit(1)

        # Display source change info using Rich table (same style as GitHub)
        _display_change_info(source_change, console=console)

        if source_change.status == "MERGED":
            console.print("\n✅ Change is already merged.")
            raise typer.Exit(0)

        if source_change.status == "ABANDONED":
            console.print("\n❌ Change has been abandoned.")
            raise typer.Exit(1)

        # Check for merge conflicts and attempt rebase if needed
        if source_change.mergeable is False:
            console.print("\n⚠️  Change has merge conflicts. Attempting to rebase...")
            rebase_result = service.rebase_change(source_change.number)

            if rebase_result["success"]:
                console.print("✅ Rebase successful! Refreshing change info...")
                # Refresh the change info after successful rebase
                source_change = service.get_change_info(parsed_url.change_number)
                _display_change_info(source_change, console=console)
            elif rebase_result["conflict"]:
                console.print("\n❌ Rebase failed due to merge conflicts:")
                if rebase_result["conflicting_files"]:
                    console.print("\n   Conflicting files:")
                    for file_path in rebase_result["conflicting_files"]:
                        console.print(f"   • {file_path}")
                console.print("\n💡 To resolve: manually rebase the change locally and push a new patchset.")
                console.print(f"   git review -d {source_change.number}")
                console.print(f"   git rebase origin/{source_change.branch}")
                console.print("   # resolve conflicts, then:")
                console.print("   git review")
                raise typer.Exit(1)
            else:
                console.print(f"\n❌ Rebase failed: {rebase_result['error']}")
                raise typer.Exit(1)

        # Create comparator and find similar changes
        comparator = create_gerrit_comparator(similarity_threshold=similarity_threshold)

        console.print(f"\n🔍 Searching for similar changes on {parsed_url.host}...")
        similar_changes = service.find_similar_changes(
            source_change,
            comparator,
        )

        console.print(f"Found {len(similar_changes)} similar changes:")

        if similar_changes:
            for change, comparison in similar_changes:
                console.print(f"  • {change.project} #{change.number}: {change.subject}")
                console.print(f"    {_format_gerrit_similarity(comparison)}")

        # Prepare list of changes to submit (similar + source)
        all_changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]] = (
            similar_changes + [(source_change, None)]
        )

        if not no_confirm:
            # Preview mode
            console.print(f"\n📊 Preview: {len(all_changes)} changes would be reviewed and submitted")
            console.print("\nTo proceed, run with --no-confirm flag")
            return

        # Create submit manager and submit changes
        console.print(f"\n🚀 Submitting {len(all_changes)} changes...")

        submit_manager = create_submit_manager(
            host=parsed_url.host,
            base_path=parsed_url.base_path,
            username=gerrit_username,
            password=gerrit_password,
        )

        # Pass the tuples directly (submit_changes expects list of tuples)
        results = submit_manager.submit_changes(all_changes)

        # Display results (GerritSubmitResult has success/submitted/error fields)
        submitted_count = sum(1 for r in results if r.submitted)
        reviewed_count = sum(1 for r in results if r.reviewed and not r.submitted)
        failed_count = sum(1 for r in results if not r.success)

        console.print(f"\n📈 Results:")
        console.print(f"   ✅ Submitted: {submitted_count}")
        if reviewed_count > 0:
            console.print(f"   📝 Reviewed (not submitted): {reviewed_count}")
        if failed_count > 0:
            console.print(f"   ❌ Failed: {failed_count}")

        # Show details for failed submissions
        for result in results:
            if not result.success:
                console.print(f"\n   ❌ {result.project} #{result.change_number}: {result.error}")

    except typer.Exit:
        # Re-raise typer.Exit without treating it as an error
        raise
    except GerritAuthError as e:
        console.print(f"❌ Gerrit authentication failed: {e}")
        console.print("   Check your GERRIT_USERNAME and GERRIT_PASSWORD")
        raise typer.Exit(1)
    except GerritRestError as e:
        console.print(f"❌ Gerrit API error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Error during Gerrit merge operation: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def merge(
    pr_url: str = typer.Argument(..., help="GitHub PR URL or Gerrit change URL"),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help="Skip confirmation prompt and merge immediately",
    ),
    similarity_threshold: float = typer.Option(
        0.8, "--threshold", help="Similarity threshold for matching PRs (0.0-1.0)"
    ),
    merge_method: str = typer.Option(
        "merge", "--merge-method", help="Merge method: merge, squash, or rebase"
    ),
    token: str | None = typer.Option(
        None, "--token", help="GitHub token (or set GITHUB_TOKEN env var)"
    ),
    override: str | None = typer.Option(
        None, "--override", help="SHA hash to override non-automation PR restriction"
    ),
    no_fix: bool = typer.Option(
        False,
        "--no-fix",
        help="Do not attempt to automatically fix out-of-date branches",
    ),
    show_progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show real-time progress updates"
    ),
    debug_matching: bool = typer.Option(
        False,
        "--debug-matching",
        help="Show detailed scoring information for PR matching",
    ),
    dismiss_copilot: bool = typer.Option(
        False,
        "--dismiss-copilot",
        help="Automatically dismiss unresolved GitHub Copilot review comments",
    ),
    force: str = typer.Option(
        "code-owners",
        "--force",
        help="Override level: 'none', 'code-owners', 'protection-rules', 'all' (default: code-owners)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose debug logging",
    ),
):
    """
    Bulk approve/merge pull requests or Gerrit changes.

    Supports both GitHub PRs and Gerrit Code Review changes.

    By default, runs in interactive mode showing what changes will apply,
    then prompts to proceed with merge. Use --no-confirm to merge immediately.

    For GitHub PRs, this command will:

    1. Analyze the provided PR

    2. Find similar PRs in the organization

    3. Approve and merge matching PRs

    4. Automatically fix out-of-date branches (use --no-fix to disable)

    For Gerrit changes, this command will:

    1. Analyze the provided change

    2. Find similar open changes on the server

    3. Review (+2 Code-Review) and submit matching changes

    Merges similar PRs/changes from the same automation tool.

    For user generated bulk PRs, use the --override flag with SHA hash.

    GitHub Force levels:
    - none: Respect all protections
    - code-owners: Bypass code owner review requirements (default)
    - protection-rules: Bypass branch protection checks (requires permissions)
    - all: Attempt merge despite most warnings (not recommended)

    Environment Variables (Gerrit):
    - GERRIT_USERNAME: HTTP username for Gerrit authentication
    - GERRIT_PASSWORD: HTTP password for Gerrit authentication
    """
    # Configure logging
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s - %(message)s",
        )
        # Suppress noisy HTTP request logs from httpx and httpcore unless verbose
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("hpack").setLevel(logging.WARNING)

    # Validate force level
    valid_force_levels = ["none", "code-owners", "protection-rules", "all"]
    if force not in valid_force_levels:
        console.print(
            f"Error: Invalid --force level '{force}'. Must be one of: {', '.join(valid_force_levels)}"
        )
        raise typer.Exit(1)

    # Warn user about force levels
    if force == "all":
        console.print("⚠️  Warning: Using --force=all will bypass most safety checks.")
        console.print("   This may attempt merges that will fail at GitHub API level.")

    # Initialize progress tracker
    progress_tracker = None

    try:
        # Parse URL to detect platform (GitHub or Gerrit)
        try:
            parsed_url = parse_change_url(pr_url)
        except UrlParseError as e:
            console.print(f"❌ Invalid URL: {e}")
            raise typer.Exit(1)

        # Route to appropriate handler based on platform
        if parsed_url.is_gerrit:
            # Handle Gerrit change
            _handle_gerrit_merge(
                parsed_url=parsed_url,
                no_confirm=no_confirm,
                similarity_threshold=similarity_threshold,
                verbose=verbose,
                console=console,
            )
            return

        # GitHub flow continues below
        # Parse PR URL to get organization info
        github_client = GitHubClient(token)
        owner, repo_name, pr_number = github_client.parse_pr_url(pr_url)

        # Initialize progress tracker with organization name
        if show_progress:
            progress_tracker = MergeProgressTracker(owner)
            progress_tracker.start()
            # Check if Rich display is available
            if not progress_tracker.rich_available:
                console.print(f"🔍 Examining source pull request in {owner}...")
                console.print("Progress updates will be shown as simple text...")
            progress_tracker.update_operation("Getting source PR details...")
        else:
            console.print(f"🔍 Examining source pull request in {owner}...")

        # Initialize comparator
        comparator = PRComparator(similarity_threshold)

        if progress_tracker:
            progress_tracker.update_operation("Getting source PR details...")

        try:
            source_pr: PullRequestInfo = github_client.get_pull_request_info(
                owner, repo_name, pr_number
            )

            # Skip closed PRs early
            if source_pr.state != "open":
                if progress_tracker:
                    progress_tracker.stop()
                exit_for_pr_state_error(
                    pr_number, "closed", details="Pull request has been closed"
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
                exit_for_github_api_error(
                    details="Failed to fetch PR details", exception=e
                )
            else:
                exit_with_error(
                    ExitCode.GENERAL_ERROR,
                    message="❌ Failed to fetch PR details",
                    details=str(e),
                    exception=e,
                )

        # Display source PR info
        _display_pr_info(
            source_pr, "", github_client, progress_tracker=progress_tracker
        )

        # Stop progress tracker before permissions check to avoid Rich display interference
        if progress_tracker:
            progress_tracker.stop()

        # Pre-flight permission check (for all modes, to fail fast before expensive org scan)
        console.print("\n🔍 Checking token permissions...")

        # Check permissions on the source repository
        async def _check_permissions():
            async with GitHubAsync(token=token) as client:
                operations = ["approve", "merge"]
                if not no_fix:
                    operations.append("update_branch")
                return await client.check_token_permissions(
                    operations, owner, repo_name
                )

        try:
            perm_results = asyncio.run(_check_permissions())

            # Check if any required permissions are missing
            missing_perms = [
                op
                for op, result in perm_results.items()
                if not result["has_permission"]
            ]

            if missing_perms:
                console.print("\n❌ Token Permission Check Failed:\n")
                for op in missing_perms:
                    result = perm_results[op]
                    console.print(f"   • {op}: {result['error']}")
                    if result.get("guidance"):
                        console.print(
                            f"     Classic: {result['guidance'].get('classic', 'N/A')}"
                        )
                        console.print(
                            f"     Fine-grained: {result['guidance'].get('fine_grained', 'N/A')}"
                        )
                console.print("\n💡 Update your token permissions and try again.")
                raise typer.Exit(code=3)

            console.print("✅ Token has required permissions\n")
        except GitHubPermissionError as e:
            console.print(f"\n❌ Permission check failed: {e}")
            raise typer.Exit(code=3) from e
        except Exception as e:
            # Don't fail on permission check errors, just warn
            console.print(f"⚠️  Could not verify permissions: {e}")
            console.print("   Continuing anyway...\n")

        # Restart progress tracker after permissions check
        if show_progress and progress_tracker:
            progress_tracker.start()

        # Debug matching info for source PR
        if debug_matching:
            console.print("\n🔍 Debug Matching Information")
            console.print(
                f"   Source PR automation status: {github_client.is_automation_author(source_pr.author)}"
            )
            console.print(
                f"   Extracted package: '{comparator._extract_package_name(source_pr.title)}'"
            )
            console.print(f"   Similarity threshold: {similarity_threshold}")
            if source_pr.body:
                console.print(f"   Body preview: {source_pr.body[:100]}...")
                console.print(
                    f"   Is dependabot body: {comparator._is_dependabot_body(source_pr.body)}"
                )
            else:
                console.print("   ⚠️  Source PR has no body")
            console.print()

        # Check if source PR is from automation or has valid override
        if not github_client.is_automation_author(source_pr.author):
            # Get commit messages to generate SHA
            commit_messages = github_client.get_pull_request_commits(
                owner, repo_name, pr_number
            )
            first_commit_line = (
                commit_messages[0].split("\n")[0] if commit_messages else ""
            )

            # Generate expected SHA for this PR
            expected_sha = _generate_override_sha(source_pr, first_commit_line)

            if not override:
                console.print("Source PR is not from a recognized automation tool.")
                console.print(
                    f"To merge this and similar PRs, run again with: --override {expected_sha}"
                )
                console.print(
                    f"This SHA is based on the author '{source_pr.author}' and commit message '{first_commit_line[:50]}...'",
                    style="dim",
                )
                return

            # Validate provided override SHA
            if not _validate_override_sha(override, source_pr, first_commit_line):
                # Use the already generated expected_sha for error message
                exit_with_error(
                    ExitCode.VALIDATION_ERROR,
                    message="❌ Invalid override SHA provided",
                    details=f"Expected SHA for this PR and author: --override {expected_sha}",
                )

            console.print(
                "Override SHA validated. Proceeding with non-automation PR merge."
            )

        # Get organization repositories
        if progress_tracker:
            progress_tracker.update_operation("Getting organization repositories...")
        else:
            console.print(f"\nChecking organization: {owner}")

        try:
            repositories: list[str] = github_client.get_organization_repositories(owner)
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
                    details="Failed to fetch organization repositories from GitHub API",
                    exception=e,
                )
            elif is_github_api_permission_error(e):
                exit_for_github_api_error(
                    details="Failed to fetch organization repositories", exception=e
                )
            else:
                exit_with_error(
                    ExitCode.GENERAL_ERROR,
                    message="❌ Failed to fetch organization repositories",
                    details=str(e),
                    exception=e,
                )
        console.print(f"Found {len(repositories)} repositories")
        #     progress.update(task, description=f"Found {len(repositories)} repositories")

        # Find similar PRs
        # similar_prs: List[Tuple[PullRequestInfo, ComparisonResult]] = []

        if progress_tracker:
            progress_tracker.update_operation("Listing repositories...")

        repositories = github_client.get_organization_repositories(owner)
        total_repos = len(repositories)

        if progress_tracker:
            progress_tracker.update_total_repositories(total_repos)
        else:
            console.print(f"Found {total_repos} repositories")

        # Find matching PRs across all repositories
        all_similar_prs = []

        from .github_service import GitHubService

        if progress_tracker:
            progress_tracker.update_operation("Listing repositories...")

        async def _find_similar():
            svc = GitHubService(
                token=token,
                progress_tracker=progress_tracker,
                debug_matching=debug_matching,
            )
            try:
                only_automation = github_client.is_automation_author(source_pr.author)
                return await svc.find_similar_prs(
                    owner,
                    source_pr,
                    comparator,
                    only_automation=only_automation,
                )
            finally:
                await svc.close()

        all_similar_prs = asyncio.run(_find_similar())

        # Stop progress tracker and show results
        if progress_tracker:
            progress_tracker.stop()
            summary = progress_tracker.get_summary()
            elapsed_time = summary.get("elapsed_time")
            total_prs_analyzed = summary.get("total_prs_analyzed")
            completed_repositories = summary.get("completed_repositories")
            similar_prs_found = summary.get("similar_prs_found")
            errors_count = summary.get("errors_count", 0)
            console.print(f"\n✅ Analysis completed in {elapsed_time}")
            console.print(
                f"📊 Analyzed {total_prs_analyzed} PRs across {completed_repositories} repositories"
            )
            console.print(f"🔍 Found {similar_prs_found} similar PRs")
            if errors_count > 0:
                console.print(f"⚠️  {errors_count} errors encountered during analysis")
            console.print()

        if not all_similar_prs:
            console.print("❌ No similar PRs found in the organization")

        console.print(f"Found {len(all_similar_prs)} similar PRs:")

        for target_pr, comparison in all_similar_prs:
            console.print(f"  • {target_pr.repository_full_name} #{target_pr.number}")
            console.print(f"    {_format_condensed_similarity(comparison)}")

        if not no_confirm:
            # IMPORTANT: Each PR must produce exactly ONE line of output in this section
            # This ensures clean, consistent evaluation reporting format
            console.print("\n🔍 Dependamerge Evaluation\n")

        # Add source PR to the list for parallel processing
        all_prs_to_merge = all_similar_prs + [(source_pr, None)]

        # Merge PRs in parallel using async merge manager
        async def _merge_parallel():
            async with AsyncMergeManager(
                token=token,
                merge_method=merge_method,
                max_retries=MAX_RETRIES,
                concurrency=10,  # Process up to 10 PRs concurrently
                fix_out_of_date=not no_fix,  # Fix is default, --no-fix disables it
                progress_tracker=progress_tracker,
                preview_mode=not no_confirm,
                dismiss_copilot=dismiss_copilot,
                force_level=force,
            ) as merge_manager:
                if not no_confirm:
                    pass  # No merge message in preview mode
                else:
                    console.print(
                        f"\n🚀 Merging {len(all_prs_to_merge)} pull requests..."
                    )
                results = await merge_manager.merge_prs_parallel(all_prs_to_merge)
                return results

        # Run the parallel merge process
        merge_results = asyncio.run(_merge_parallel())

        # Display results
        if merge_results:
            # Create a simple summary from results
            merged_count = sum(1 for r in merge_results if r.status.value == "merged")
            failed_count = sum(1 for r in merge_results if r.status.value == "failed")
            skipped_count = sum(1 for r in merge_results if r.status.value == "skipped")
            blocked_count = sum(1 for r in merge_results if r.status.value == "blocked")
            total_to_merge = len(merge_results)
            if not no_confirm:
                console.print(f"\nMergeable {merged_count}/{total_to_merge} PRs")

                # Generate continuation SHA and prompt user
                if merged_count > 0:
                    # Get commit message for SHA generation
                    commit_messages = github_client.get_pull_request_commits(
                        owner, repo_name, pr_number
                    )
                    first_commit_line = (
                        commit_messages[0].split("\n")[0] if commit_messages else ""
                    )
                    continue_sha_hash = _generate_continue_sha(
                        source_pr, first_commit_line
                    )
                    console.print()
                    console.print(f"To proceed with merging enter: {continue_sha_hash}")

                    try:
                        # Skip interactive prompt in test mode
                        if "pytest" in sys.modules or os.getenv("TESTING"):
                            console.print(
                                "⚠️  Test mode detected - skipping interactive prompt"
                            )
                            return

                        user_input = input(
                            "Enter the string above to continue (or press Enter to cancel): "
                        ).strip()
                        if user_input == continue_sha_hash:
                            # Run actual merge on mergeable PRs only
                            console.print(
                                f"\n🔨 Merging {merged_count} mergeable pull requests..."
                            )
                            mergeable_prs = []
                            for i, result in enumerate(merge_results):
                                if (
                                    result.status.value == "merged"
                                ):  # These were preview "merged"
                                    mergeable_prs.append(all_prs_to_merge[i])

                            # Define async function for real merge
                            async def _real_merge():
                                async with AsyncMergeManager(
                                    token=token,
                                    merge_method=merge_method,
                                    max_retries=MAX_RETRIES,
                                    concurrency=10,
                                    fix_out_of_date=not no_fix,
                                    progress_tracker=progress_tracker,
                                    preview_mode=False,  # Execute merge
                                    dismiss_copilot=dismiss_copilot,
                                    force_level=force,
                                ) as real_merge_manager:
                                    return await real_merge_manager.merge_prs_parallel(
                                        mergeable_prs
                                    )

                            # Run the real merge
                            real_results = asyncio.run(_real_merge())

                            # Display final results
                            final_merged = sum(
                                1 for r in real_results if r.status.value == "merged"
                            )
                            final_failed = sum(
                                1 for r in real_results if r.status.value == "failed"
                            )
                            final_skipped = sum(
                                1 for r in real_results if r.status.value == "skipped"
                            )
                            final_blocked = sum(
                                1 for r in real_results if r.status.value == "blocked"
                            )

                            console.print(
                                f"\n🚀 Final Results: {final_merged} merged, {final_failed} failed"
                            )
                            if final_skipped > 0:
                                console.print(f"⏭️  Skipped {final_skipped} PRs")
                            if final_blocked > 0:
                                console.print(f"🛑 Blocked {final_blocked} PRs")
                        elif user_input == "":
                            console.print("❌ Merge cancelled by user.")
                        else:
                            console.print("❌ Invalid input. Merge cancelled.")
                    except KeyboardInterrupt:
                        console.print("\n❌ Merge cancelled by user.")
                    except EOFError:
                        console.print("\n❌ Merge cancelled.")

                    return  # Exit after handling preview continuation
                else:
                    console.print("\n💡 No PRs are mergeable at this time.")

            if failed_count > 0:
                if not no_confirm:
                    console.print(f"❌ Would fail to merge {failed_count} PRs")
                else:
                    console.print(f"❌ Failed {failed_count} PRs")
            if skipped_count > 0:
                console.print(f"⏭️  Skipped {skipped_count} PRs")
            if blocked_count > 0:
                console.print(f"🛑 Blocked {blocked_count} PRs")

            if no_confirm:
                console.print(
                    f"📈 Final Results: {merged_count} merged, {failed_count} failed"
                )

        else:
            console.print("❌ No PRs were processed")

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
        # Handle typer exits (like closed PR errors) gracefully - already printed message
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
                message="❌ Error during merge operation",
                details=str(e),
                exception=e,
            )


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

    # Get proper status instead of raw mergeable field
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
):
    """
    Bulk close pull requests across a GitHub organization.

    By default, runs in interactive mode showing what changes will apply,
    then prompts to proceed with closing. Use --no-confirm to close immediately.

    This command will:

    1. Analyze the provided PR

    2. Find similar PRs in the organization

    3. Close matching PRs

    Closes similar PRs from the same automation tool (dependabot, pre-commit.ci).

    For user generated bulk PRs, use the --override flag with SHA hash.
    """
    # Initialize progress tracker
    progress_tracker = None

    try:
        # Parse PR URL first to get organization info
        github_client = GitHubClient(token)
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

        # Get source PR details
        source_pr = github_client.get_pull_request_info(owner, repo_name, pr_number)

        # Display source PR info
        _display_pr_info(
            source_pr, "", github_client, progress_tracker=progress_tracker
        )

        # Initialize comparator
        comparator = PRComparator(similarity_threshold)

        # Debug matching info for source PR
        if debug_matching:
            console.print("\n🔍 Debug Matching Information")
            console.print(
                f"   Source PR automation status: {github_client.is_automation_author(source_pr.author)}"
            )
            console.print(
                f"   Extracted package: '{comparator._extract_package_name(source_pr.title)}'"
            )
            console.print(f"   Similarity threshold: {similarity_threshold}")
            if source_pr.body:
                console.print(f"   Body preview: {source_pr.body[:100]}...")
                console.print(
                    f"   Is dependabot body: {comparator._is_dependabot_body(source_pr.body)}"
                )
            else:
                console.print("   ⚠️  Source PR has no body")
            console.print()

        # Check if source PR is from automation or has valid override
        is_automation = github_client.is_automation_author(source_pr.author)
        override_valid = False

        if not is_automation:
            # Get first commit message for SHA generation
            commit_messages = github_client.get_pull_request_commits(
                owner, repo_name, pr_number
            )
            first_commit_line = (
                commit_messages[0].split("\n")[0] if commit_messages else ""
            )

            # Generate expected SHA
            expected_sha = _generate_override_sha(source_pr, first_commit_line)

            # Check if override matches
            if override == expected_sha:
                override_valid = True

            if not override:
                console.print("Source PR is not from a recognized automation tool.")
                console.print(
                    f"To close this and similar PRs, run again with: --override {expected_sha}"
                )
                console.print(
                    f"This SHA is based on the author '{source_pr.author}' and commit message '{first_commit_line[:50]}...'",
                    style="dim",
                )
                return

            if not override_valid:
                console.print(
                    f"Error: Invalid override SHA. Expected: {expected_sha}",
                    style="bold red",
                )
                console.print(
                    "This prevents accidental bulk operations on non-automation PRs.",
                    style="dim",
                )
                return

            console.print(
                "Override SHA validated. Proceeding with non-automation PR close."
            )

        # Find similar PRs in the organization
        if progress_tracker:
            console.print()
        else:
            console.print(f"\nChecking organization: {owner}")

        # Use GitHubService for async PR finding
        from .github_service import GitHubService

        if progress_tracker:
            progress_tracker.update_operation("Listing repositories...")

        async def _find_similar():
            svc = GitHubService(
                token=token,
                progress_tracker=progress_tracker,
                debug_matching=debug_matching,
            )
            try:
                only_automation = github_client.is_automation_author(source_pr.author)
                return await svc.find_similar_prs(
                    owner,
                    source_pr,
                    comparator,
                    only_automation=only_automation,
                )
            finally:
                await svc.close()

        all_similar_prs = asyncio.run(_find_similar())

        # Stop progress tracker before displaying results
        if progress_tracker:
            progress_tracker.stop()
            summary = progress_tracker.get_summary()
            elapsed_time = summary.get("elapsed_time")
            total_prs_analyzed = summary.get("total_prs_analyzed")
            completed_repositories = summary.get("completed_repositories")
            similar_prs_found = summary.get("similar_prs_found")
            errors_count = summary.get("errors_count", 0)
            console.print(f"\n✅ Analysis completed in {elapsed_time}")
            console.print(
                f"📊 Analyzed {total_prs_analyzed} PRs across {completed_repositories} repositories"
            )
            console.print(f"🔍 Found {similar_prs_found} similar PRs")
            if errors_count > 0:
                console.print(f"⚠️  {errors_count} errors encountered during analysis")
            console.print()

        if not all_similar_prs:
            console.print("❌ No similar PRs found in the organization")

        console.print(f"Found {len(all_similar_prs)} similar PRs:")

        for target_pr, comparison in all_similar_prs:
            console.print(f"  • {target_pr.repository_full_name} #{target_pr.number}")
            console.print(f"    {_format_condensed_similarity(comparison)}")

        if not no_confirm:
            # IMPORTANT: Each PR must produce exactly ONE line of output in this section
            console.print("\n🔍 Dependamerge Evaluation\n")

        # Determine which PRs to close
        all_prs_to_close = [source_pr] + [pr for pr, _ in all_similar_prs]

        # Perform preview close operation
        async def _close_parallel(prs, preview_mode):
            close_manager = AsyncCloseManager(
                token=token,
                progress_tracker=progress_tracker,
                preview_mode=preview_mode,
            )
            async with close_manager:
                # Convert to list of tuples (PR, None) for consistency
                pr_tuples = [(pr, None) for pr in prs]
                return await close_manager.close_prs_parallel(pr_tuples)

        # Perform preview to check which PRs can be closed
        if not no_confirm:
            if progress_tracker:
                progress_tracker.start()
                console.print()
            else:
                console.print(
                    f"\n🚀 Evaluating {len(all_prs_to_close)} pull requests..."
                )

            close_results = asyncio.run(_close_parallel(all_prs_to_close, True))

            if progress_tracker:
                progress_tracker.stop()
                console.print()

            # Count closeable PRs
            closed_count = sum(1 for r in close_results if r.status.value == "closed")
            total_to_close = len(all_prs_to_close)

            if not no_confirm:
                console.print(f"\nCloseable {closed_count}/{total_to_close} PRs")

                # Generate continuation SHA and prompt user
                if closed_count > 0:
                    # Get commit message for SHA generation
                    commit_messages = github_client.get_pull_request_commits(
                        owner, repo_name, pr_number
                    )
                    first_commit_line = (
                        commit_messages[0].split("\n")[0] if commit_messages else ""
                    )
                    continue_sha_hash = _generate_continue_sha(
                        source_pr, first_commit_line
                    )
                    console.print()
                    console.print(f"To proceed with closing enter: {continue_sha_hash}")

                    # Check if in test mode (don't prompt during tests)
                    if "pytest" in sys.modules or os.getenv("TESTING"):
                        console.print(
                            "⚠️  Test mode detected - skipping interactive prompt"
                        )
                        return

                    user_input = typer.prompt(
                        "\nEnter the string above to continue (or press Enter to cancel)"
                    ).strip()

                    if user_input == continue_sha_hash:
                        # Run actual close on closeable PRs only
                        console.print(
                            f"\n🔨 Closing {closed_count} closeable pull requests..."
                        )
                        closeable_prs = []
                        for i, result in enumerate(close_results):
                            if (
                                result.status.value == "closed"
                            ):  # These were preview "closed"
                                closeable_prs.append(all_prs_to_close[i])

                        if progress_tracker:
                            progress_tracker.start()

                        final_results = asyncio.run(
                            _close_parallel(closeable_prs, False)
                        )

                        if progress_tracker:
                            progress_tracker.stop()

                        # Count final results
                        final_closed = sum(
                            1 for r in final_results if r.status.value == "closed"
                        )
                        final_failed = sum(
                            1 for r in final_results if r.status.value == "failed"
                        )

                        console.print(
                            f"\n🚀 Final Results: {final_closed} closed, {final_failed} failed"
                        )

                    else:
                        console.print("\n❌ Operation cancelled by user")
                        return
                else:
                    console.print("\n❌ No PRs are eligible for closing")
                    return
        else:
            # No confirmation - close immediately
            if progress_tracker:
                progress_tracker.start()
            console.print(f"\n🚀 Closing {len(all_prs_to_close)} pull requests...")

            close_results = asyncio.run(_close_parallel(all_prs_to_close, False))

            if progress_tracker:
                progress_tracker.stop()

            # Count results
            closed_count = sum(1 for r in close_results if r.status.value == "closed")
            failed_count = sum(1 for r in close_results if r.status.value == "failed")

            console.print(
                f"\n🚀 Final Results: {closed_count} closed, {failed_count} failed"
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
    except typer.Exit:
        # Handle typer exits gracefully - already printed message
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


@app.command()
def status(
    org_input: str = typer.Argument(
        ...,
        help="GitHub organization name or URL (e.g., 'lfreleng-actions' or 'https://github.com/lfreleng-actions/')",
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
    1. Scan all repositories in the organization
    2. Gather tag and release information
    3. Count open and merged pull requests
    4. Identify PRs affecting actions or workflows

    Automation tools supported: dependabot, pre-commit.ci
    """
    # Parse organization name from input (handle both URL and plain name)
    org_name = org_input.rstrip("/").split("/")[-1]
    if not org_name:
        console.print("❌ Invalid GitHub organization name or URL")
        console.print(
            "   Expected: 'organization-name' or 'https://github.com/organization-name/'"
        )
        raise typer.Exit(1)

    # Initialize progress tracker (disable PR stats for status command)
    progress_tracker = None

    try:
        if show_progress:
            progress_tracker = ProgressTracker(org_name, show_pr_stats=False)
            progress_tracker.start()
            if not progress_tracker.rich_available:
                console.print(f"🔍 Scanning organization: {org_name}")
                console.print("Progress updates will be shown as simple text...")
        else:
            console.print(f"🔍 Scanning organization: {org_name}")
            console.print("This may take a few minutes for large organizations...")

        # Perform the scan
        from .github_service import GitHubService

        async def _run_status_check():
            svc = GitHubService(token=token, progress_tracker=progress_tracker)
            try:
                return await svc.gather_organization_status(org_name)
            finally:
                await svc.close()

        status_result = asyncio.run(_run_status_check())

        # Stop progress tracker before displaying results
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
        console.print("\n⚠️  Scan interrupted by user")
        raise typer.Exit(130) from None
    except Exception as e:
        if progress_tracker:
            progress_tracker.stop()
        console.print(f"❌ Error during scan: {e}")
        raise typer.Exit(1) from e


@app.command()
def blocked(
    org_input: str = typer.Argument(
        ...,
        help="GitHub organization name or URL (e.g., 'lfreleng-actions' or 'https://github.com/lfreleng-actions/')",
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
    Reports blocked pull requests in a GitHub organization.

    This command will:
    1. Check all repositories in the organization
    2. Identify pull requests that cannot be merged
    3. Report blocking reasons (conflicts, failing checks, etc.)
    4. Count unresolved Copilot feedback comments

    Standard code review requirements are not considered blocking.
    """
    # Parse organization name from input (handle both URL and plain name)
    organization = org_input.rstrip("/").split("/")[-1]
    if not organization:
        console.print("❌ Invalid GitHub organization name or URL")
        console.print(
            "   Expected: 'organization-name' or 'https://github.com/organization-name/'"
        )
        raise typer.Exit(1)

    # Initialize progress tracker
    progress_tracker = None

    try:
        if show_progress:
            progress_tracker = ProgressTracker(organization)
            progress_tracker.start()
            # Check if Rich display is available
            if not progress_tracker.rich_available:
                console.print(f"🔍 Checking organization: {organization}")
                console.print("Progress updates will be shown as simple text...")
        else:
            console.print(f"🔍 Checking organization: {organization}")
            console.print("This may take a few minutes for large organizations...")

        # Perform the scan
        from .github_service import GitHubService

        async def _run_blocked_check():
            svc = GitHubService(token=token, progress_tracker=progress_tracker)
            try:
                return await svc.scan_organization(
                    organization, include_drafts=include_drafts
                )
            finally:
                await svc.close()

        scan_result = asyncio.run(_run_blocked_check())

        # Stop progress tracker before displaying results
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
                console.print(f"⚠️  {errors_count} errors encountered during check")
            console.print()  # Add blank line before results

        # Display results
        _display_blocked_results(scan_result, output_format)

        # Optional fix workflow
        if fix:
            # Build candidate list based on reasons
            allowed_default = {"merge_conflict", "behind_base"}
            reasons_to_attempt = (
                allowed_default if not reason else {reason.strip().lower()}
            )

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
                    progress_tracker=progress_tracker,
                    logger=lambda m: console.print(m),
                )
                fix_options = FixOptions(
                    workdir=workdir,
                    keep_temp=keep_temp,
                    prefetch=prefetch
                    if prefetch is not None
                    else get_default_workers(),
                    editor=editor,
                    mergetool=mergetool,
                    interactive=interactive,
                    logger=lambda m: console.print(m),
                )
                results = orchestrator.run(selections, fix_options)
                success_count = sum(1 for r in results if r.success)
                console.print(
                    f"✅ Fix complete: {success_count}/{len(selections)} succeeded"
                )
            except Exception as e:
                exit_with_error(
                    ExitCode.GENERAL_ERROR,
                    message="❌ Error during fix workflow",
                    details=str(e),
                    exception=e,
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
        # Handle typer exits gracefully
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
                message="❌ Error during organization scan",
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

    # Create detailed blocked PRs table
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

    # Create status table
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
    for tool in AUTOMATION_TOOLS:
        # Format tool names nicely
        if tool == "[bot]":
            console.print("  • Any bot account")
        elif tool == "pre-commit":
            console.print("  • pre-commit.ci")
        elif tool == "github-actions":
            console.print("  • GitHub Actions")
        else:
            console.print(f"  • {tool.capitalize()}")
    console.print()

    # Create summary table
    summary_table = Table()
    summary_table.add_column("Summary", style="cyan")
    summary_table.add_column("Value", style="white")

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


if __name__ == "__main__":
    app()
