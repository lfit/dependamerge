# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

import hashlib
from typing import List, Optional, Tuple

import requests
import typer
import urllib3.exceptions
from github.Repository import Repository
from rich.console import Console
from rich.table import Table

from .github_client import GitHubClient
from .models import ComparisonResult, PullRequestInfo
from .pr_comparator import PRComparator

app = typer.Typer(
    help="Automatically merge pull requests created by automation tools across GitHub organizations"
)
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


@app.command()
def merge(
    pr_url: str = typer.Argument(..., help="GitHub pull request URL"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what changes will apply without making them"
    ),
    similarity_threshold: float = typer.Option(
        0.8, "--threshold", help="Similarity threshold for matching PRs (0.0-1.0)"
    ),
    merge_method: str = typer.Option(
        "merge", "--merge-method", help="Merge method: merge, squash, or rebase"
    ),
    token: Optional[str] = typer.Option(
        None, "--token", help="GitHub token (or set GITHUB_TOKEN env var)"
    ),
    override: Optional[str] = typer.Option(
        None, "--override", help="SHA hash to override non-automation PR restriction"
    ),
):
    """
    Merge automation pull requests across an organization.

    This command will:
    1. Analyze the provided PR
    2. Find similar PRs in the organization
    3. Approve and merge matching PRs

    For automation PRs (dependabot, pre-commit-ci, etc.):
    - Merges similar PRs from the same automation tool

    For non-automation PRs:
    - Requires --override flag with SHA hash
    - Only merges PRs from the same author
    - SHA is generated from author + commit message
    """
    try:
        # Initialize clients
        github_client = GitHubClient(token)
        comparator = PRComparator(similarity_threshold)

        console.print(f"Analyzing PR: {pr_url}")

        # Parse PR URL and get info
        owner, repo_name, pr_number = github_client.parse_pr_url(pr_url)

        try:
            source_pr: PullRequestInfo = github_client.get_pull_request_info(
                owner, repo_name, pr_number
            )
        except (
            urllib3.exceptions.NameResolutionError,
            urllib3.exceptions.MaxRetryError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            console.print(
                "[red]Network Error: Failed to connect to GitHub API while fetching source PR.[/red]"
            )
            console.print(f"[red]Details: {e}[/red]")
            console.print(
                "[red]Please check your internet connection and try again.[/red]"
            )
            raise typer.Exit(1) from e

        # Display source PR info
        _display_pr_info(source_pr, "Source PR", github_client)

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
                console.print(
                    "[yellow]Source PR is not from a recognized automation tool.[/yellow]"
                )
                console.print(
                    f"[yellow]To merge this and similar PRs, run again with: --override {expected_sha}[/yellow]"
                )
                console.print(
                    f"[dim]This SHA is based on the author '{source_pr.author}' and commit message '{first_commit_line[:50]}...'[/dim]"
                )
                return

            # Validate provided override SHA
            if not _validate_override_sha(override, source_pr, first_commit_line):
                console.print(
                    "[red]Error: Invalid override SHA. Expected SHA for this PR and author is:[/red]"
                )
                console.print(f"[red]--override {expected_sha}[/red]")
                raise typer.Exit(1)

            console.print(
                "[green]Override SHA validated. Proceeding with non-automation PR merge.[/green]"
            )

        # Get organization repositories
        console.print(f"\nScanning organization: {owner}")

        # TEMP: Remove progress for debugging
        # with Progress(
        #     SpinnerColumn(),
        #     TextColumn("[progress.description]{task.description}"),
        #     console=console,
        # ) as progress:
        #     task = progress.add_task("Fetching repositories...", total=None)
        try:
            repositories: List[Repository] = (
                github_client.get_organization_repositories(owner)
            )
        except (
            urllib3.exceptions.NameResolutionError,
            urllib3.exceptions.MaxRetryError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            console.print(
                "[red]Network Error: Failed to connect to GitHub API while fetching organization repositories.[/red]"
            )
            console.print(f"[red]Details: {e}[/red]")
            console.print(
                "[red]Please check your internet connection and try again.[/red]"
            )
            raise typer.Exit(1) from e
        console.print(f"Found {len(repositories)} repositories")
        #     progress.update(task, description=f"Found {len(repositories)} repositories")

        # Find similar PRs
        similar_prs: List[Tuple[PullRequestInfo, ComparisonResult]] = []

        # TEMP: Remove progress for debugging
        # with Progress(
        #     SpinnerColumn(),
        #     TextColumn("[progress.description]{task.description}"),
        #     console=console,
        # ) as progress:
        #     task = progress.add_task("Analyzing PRs...", total=len(repositories))

        for repo in repositories:
            if repo.full_name == source_pr.repository_full_name:
                # progress.advance(task)
                continue

            open_prs = github_client.get_open_pull_requests(repo)

            for pr in open_prs:
                # Check if PR should be considered based on override status
                is_automation = github_client.is_automation_author(pr.user.login)

                # If source PR is automation, only consider automation PRs
                # If source PR is non-automation with override, only consider non-automation PRs from same author
                source_is_automation = github_client.is_automation_author(
                    source_pr.author
                )

                if source_is_automation:
                    if not is_automation:
                        continue
                else:
                    if is_automation or pr.user.login != source_pr.author:
                        continue

                try:
                    target_pr = github_client.get_pull_request_info(
                        repo.owner.login, repo.name, pr.number
                    )

                    comparison = comparator.compare_pull_requests(source_pr, target_pr)

                    if comparison.is_similar:
                        similar_prs.append((target_pr, comparison))

                except (
                    urllib3.exceptions.NameResolutionError,
                    urllib3.exceptions.MaxRetryError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException,
                ) as e:
                    console.print(
                        "[red]Network Error: Failed to connect to GitHub API while analyzing PRs.[/red]"
                    )
                    console.print(f"[red]Details: {e}[/red]")
                    console.print(
                        "[red]Please check your internet connection and try again.[/red]"
                    )
                    raise typer.Exit(1) from e
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Failed to analyze PR {pr.number} in {repo.full_name}: {e}[/yellow]"
                    )

            # progress.advance(task)

        # Display results
        if not similar_prs:
            console.print("\nNo similar PRs found in the organization")
        else:
            console.print(f"\nFound {len(similar_prs)} similar PR(s)")

            # Display similar PRs table
            table = Table(title="Similar Pull Requests")
            table.add_column("Repository", style="cyan")
            table.add_column("PR #", style="magenta")
            table.add_column("Title", style="green")
            table.add_column("Confidence", style="yellow")
            table.add_column("Status", style="blue")

            for pr_info, comparison in similar_prs:
                # Strip organization name from repository full name
                repo_name = pr_info.repository_full_name.split("/")[-1]

                # Get detailed status information
                status = github_client.get_pr_status_details(pr_info)

                # Display title with proper truncation
                display_title = (
                    pr_info.title[:50] + "..."
                    if len(pr_info.title) > 50
                    else pr_info.title
                )

                table.add_row(
                    repo_name,
                    str(pr_info.number),
                    display_title,
                    f"{comparison.confidence_score:.2f}",
                    status,
                )

            console.print(table)

        # Merge PRs
        if dry_run:
            console.print("\nDry run mode - no changes will be made")
            return

        success_count = 0
        # Merge similar PRs if any were found
        for pr_info, _comparison in similar_prs:
            if _merge_single_pr(pr_info, github_client, merge_method, console):
                success_count += 1

        # Always merge source PR (whether similar PRs were found or not)
        console.print(f"\nMerging source PR {source_pr.number}")
        source_pr_merged = _merge_single_pr(
            source_pr, github_client, merge_method, console
        )
        if source_pr_merged:
            success_count += 1

        total_prs = len(similar_prs) + 1  # similar PRs + source PR
        console.print(
            f"\nSuccessfully merged {success_count}/{total_prs} PRs (including source PR)"
        )

    except Exception as e:
        console.print(f"Error: {e}")
        raise typer.Exit(1) from e


def _display_pr_info(pr: PullRequestInfo, title: str, github_client: GitHubClient):
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

    console.print(table)


def _merge_single_pr(
    pr_info: PullRequestInfo,
    github_client: GitHubClient,
    merge_method: str,
    console: Console,
) -> bool:
    """
    Merge a single pull request.

    Returns True if successfully merged, False otherwise.
    """
    repo_owner, repo_name = pr_info.repository_full_name.split("/")

    # Get initial status
    status = github_client.get_pr_status_details(pr_info)

    # Handle different types of blocks intelligently
    if pr_info.mergeable_state == "blocked" and pr_info.mergeable is True:
        # This is likely blocked by branch protection (review required, etc.)
        # Don't show "attempting anyway" message since this is expected and handleable
        pass
    elif pr_info.mergeable_state == "blocked" and pr_info.mergeable is False:
        console.print(
            f"PR {pr_info.number} is blocked by failing checks - attempting merge anyway"
        )
    elif not pr_info.mergeable:
        console.print(
            f"Skipping unmergeable PR {pr_info.number} in {pr_info.repository_full_name} ({status})"
        )
        return False

    # Approve PR
    console.print(f"Approving PR {pr_info.number} in {pr_info.repository_full_name}")
    if not github_client.approve_pull_request(repo_owner, repo_name, pr_info.number):
        console.print(f"Failed to approve PR {pr_info.number} ❌")
        return False

    # Attempt merge with retry logic for different failure conditions
    max_retries = 2
    for attempt in range(max_retries + 1):
        console.print(
            f"Merging PR {pr_info.number} in {pr_info.repository_full_name} (attempt {attempt + 1})"
        )

        merge_result = github_client.merge_pull_request(
            repo_owner, repo_name, pr_info.number, merge_method
        )

        if merge_result:
            console.print(f"Successfully merged PR {pr_info.number} ✅")
            return True

        # If merge failed, check if we can fix the issue and retry
        if attempt < max_retries:
            # Refresh PR info to get latest status
            try:
                updated_pr_info = github_client.get_pull_request_info(
                    repo_owner, repo_name, pr_info.number
                )

                # Check if branch is out of date and can be fixed
                if updated_pr_info.mergeable_state == "behind":
                    console.print(
                        f"PR {pr_info.number} is out of date - updating branch and retrying"
                    )
                    if github_client.fix_out_of_date_pr(
                        repo_owner, repo_name, pr_info.number
                    ):
                        console.print(
                            f"Successfully updated PR {pr_info.number} branch ✅"
                        )
                        pr_info = updated_pr_info  # Update for next attempt
                        continue
                    else:
                        console.print(f"Failed to update PR {pr_info.number} branch ❌")
                        break
                else:
                    # Other types of merge failures - no point in retrying
                    break
            except Exception as e:
                console.print(f"Warning: Failed to refresh PR info for retry: {e}")
                break

    console.print(
        f"Failed to merge PR {pr_info.number} after {max_retries + 1} attempts ❌"
    )
    return False


if __name__ == "__main__":
    app()
