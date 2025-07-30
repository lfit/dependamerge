# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

import os
from typing import Dict, List, Optional

import requests
import urllib3.exceptions
from github import Github, GithubException
from github.CommitStatus import CommitStatus
from github.PullRequest import PullRequest
from github.Repository import Repository

from .models import FileChange, PullRequestInfo


class GitHubClient:
    """GitHub API client for managing pull requests."""

    def __init__(self, token: Optional[str] = None):
        """Initialize GitHub client with token."""
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError(
                "GitHub token is required. Set GITHUB_TOKEN environment variable."
            )
        self.github = Github(self.token)

    def parse_pr_url(self, url: str) -> tuple[str, str, int]:
        """Parse GitHub PR URL to extract owner, repo, and PR number."""
        # Expected format: https://github.com/owner/repo/pull/123[/files|/commits|etc]
        parts = url.rstrip("/").split("/")
        if len(parts) < 7 or "github.com" not in url or "pull" not in parts:
            raise ValueError(f"Invalid GitHub PR URL: {url}")

        # Find the 'pull' segment and get the PR number from the next segment
        try:
            pull_index = parts.index("pull")
            if pull_index + 1 >= len(parts):
                raise ValueError("PR number not found after 'pull'")

            owner = parts[pull_index - 2]
            repo = parts[pull_index - 1]
            pr_number = int(parts[pull_index + 1])

            return owner, repo, pr_number
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid GitHub PR URL: {url}") from e

    def get_pull_request_info(
        self, owner: str, repo: str, pr_number: int
    ) -> PullRequestInfo:
        """Get detailed information about a pull request."""
        try:
            repository = self.github.get_repo(f"{owner}/{repo}")
            pr = repository.get_pull(pr_number)

            # Get file changes
            files_changed = []
            for file in pr.get_files():
                files_changed.append(
                    FileChange(
                        filename=file.filename,
                        additions=file.additions,
                        deletions=file.deletions,
                        changes=file.changes,
                        status=file.status,
                    )
                )

            return PullRequestInfo(
                number=pr.number,
                title=pr.title,
                body=pr.body,
                author=pr.user.login,
                head_sha=pr.head.sha,
                base_branch=pr.base.ref,
                head_branch=pr.head.ref,
                state=pr.state,
                mergeable=pr.mergeable,
                mergeable_state=pr.mergeable_state,
                behind_by=getattr(pr, "behind_by", None),
                files_changed=files_changed,
                repository_full_name=repository.full_name,
                html_url=pr.html_url,
            )
        except (
            urllib3.exceptions.NameResolutionError,
            urllib3.exceptions.MaxRetryError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            raise RuntimeError(f"Network error while fetching PR info: {e}") from e
        except GithubException as e:
            raise RuntimeError(f"Failed to fetch PR info: {e}") from e

    def get_pull_request_commits(
        self, owner: str, repo: str, pr_number: int
    ) -> List[str]:
        """Get commit messages from a pull request."""
        try:
            repository = self.github.get_repo(f"{owner}/{repo}")
            pr = repository.get_pull(pr_number)

            commits = pr.get_commits()
            commit_messages = []

            for commit in commits:
                if commit.commit.message:
                    commit_messages.append(commit.commit.message)

            return commit_messages
        except (
            urllib3.exceptions.NameResolutionError,
            urllib3.exceptions.MaxRetryError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            raise RuntimeError(f"Network error while fetching PR commits: {e}") from e
        except GithubException as e:
            raise RuntimeError(f"Failed to fetch PR commits: {e}") from e

    def get_organization_repositories(self, org_name: str) -> List[Repository]:
        """Get all repositories in an organization."""
        try:
            org = self.github.get_organization(org_name)
            return list(org.get_repos())
        except (
            urllib3.exceptions.NameResolutionError,
            urllib3.exceptions.MaxRetryError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            raise RuntimeError(
                f"Network error while fetching organization repositories: {e}"
            ) from e
        except GithubException as e:
            raise RuntimeError(f"Failed to fetch organization repositories: {e}") from e

    def get_open_pull_requests(self, repository: Repository) -> List[PullRequest]:
        """Get all open pull requests for a repository."""
        try:
            return list(repository.get_pulls(state="open"))
        except (
            urllib3.exceptions.NameResolutionError,
            urllib3.exceptions.MaxRetryError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            raise RuntimeError(
                f"Network error while fetching PRs for {repository.full_name}: {e}"
            ) from e
        except GithubException as e:
            print(f"Warning: Failed to fetch PRs for {repository.full_name}: {e}")
            return []

    def approve_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        message: str = "Auto-approved by dependamerge",
    ) -> bool:
        """Approve a pull request."""
        try:
            repository = self.github.get_repo(f"{owner}/{repo}")
            pr = repository.get_pull(pr_number)
            pr.create_review(body=message, event="APPROVE")
            return True
        except GithubException as e:
            print(f"Failed to approve PR {pr_number}: {e}")
            return False

    def merge_pull_request(
        self, owner: str, repo: str, pr_number: int, merge_method: str = "merge"
    ) -> bool:
        """Merge a pull request with detailed error handling."""
        try:
            repository = self.github.get_repo(f"{owner}/{repo}")
            pr = repository.get_pull(pr_number)

            # Attempt merge even if mergeable is False for blocked PRs
            # (branch protection might resolve after approval)
            result = pr.merge(merge_method=merge_method)
            return bool(result.merged)
        except GithubException as e:
            print(f"Failed to merge PR {pr_number}: {e}")
            return False

    def is_automation_author(self, author: str) -> bool:
        """Check if the author is a known automation tool."""
        automation_authors = {
            "dependabot[bot]",
            "pre-commit-ci[bot]",
            "renovate[bot]",
            "github-actions[bot]",
            "allcontributors[bot]",
        }
        return author in automation_authors

    def get_pr_status_details(self, pr_info: PullRequestInfo) -> str:
        """Get detailed status information for a PR."""
        if pr_info.state != "open":
            return f"Closed ({pr_info.state})"

        # Check for draft status first
        if pr_info.mergeable_state == "draft":
            return "Draft PR"

        # Handle blocked state - need to determine why it's blocked
        if pr_info.mergeable_state == "blocked" and pr_info.mergeable is True:
            # This means technically mergeable but blocked by branch protection
            # We need to check what's blocking it to provide intelligent status
            block_reason = self._analyze_block_reason(pr_info)
            return block_reason

        if pr_info.mergeable is False:
            # Check for specific reasons why it's not mergeable
            if pr_info.mergeable_state == "dirty":
                return "Merge conflicts"
            elif pr_info.mergeable_state == "behind":
                return "Rebase required"
            elif pr_info.mergeable_state == "blocked":
                return "Blocked by checks"
            else:
                return f"Not mergeable ({pr_info.mergeable_state or 'unknown'})"

        if pr_info.mergeable_state == "behind":
            return "Rebase required"

        # If mergeable is True and mergeable_state is clean, it's ready
        if pr_info.mergeable is True and pr_info.mergeable_state == "clean":
            return "Ready to merge"

        # For any other combination where mergeable is True but state is unclear
        if pr_info.mergeable is True:
            return "Ready to merge"

        # Fallback for unclear states
        return f"Status unclear ({pr_info.mergeable_state or 'unknown'})"

    def _analyze_block_reason(self, pr_info: PullRequestInfo) -> str:
        """Analyze why a PR is blocked and return appropriate status."""
        try:
            repo_owner, repo_name = pr_info.repository_full_name.split("/")
            repository = self.github.get_repo(f"{repo_owner}/{repo_name}")
            pr = repository.get_pull(pr_info.number)

            # Check if there are any reviews
            reviews = list(pr.get_reviews())
            approved_reviews = [r for r in reviews if r.state == "APPROVED"]

            # Check commit status and check runs
            commit = repository.get_commit(pr.head.sha)

            # Get commit statuses (legacy status API) and get latest per context
            all_statuses = list(commit.get_statuses())
            latest_statuses: Dict[str, CommitStatus] = {}
            for status in all_statuses:
                context = status.context
                if (
                    context not in latest_statuses
                    or status.updated_at > latest_statuses[context].updated_at
                ):
                    latest_statuses[context] = status

            failing_statuses = [
                s for s in latest_statuses.values() if s.state in ["failure", "error"]
            ]
            pending_statuses = [
                s for s in latest_statuses.values() if s.state == "pending"
            ]

            # Get check runs (GitHub Actions, etc.)
            check_runs = list(commit.get_check_runs())
            failing_checks = [
                c
                for c in check_runs
                if c.conclusion in ["failure", "cancelled", "timed_out"]
            ]
            pending_checks = [
                c for c in check_runs if c.status in ["queued", "in_progress"]
            ]

            # Determine the primary blocking reason - prioritize review requirements
            if failing_statuses or failing_checks:
                return "Blocked by failing checks"
            elif not approved_reviews:
                # Check if there are any real pending checks (not just stale statuses)
                if (
                    pending_checks
                ):  # Only consider pending check runs, not stale statuses
                    return "Blocked by pending checks"
                else:
                    # All checks passed but no approval - needs review
                    return "Requires approval"
            elif pending_statuses or pending_checks:
                return "Blocked by pending checks"
            else:
                # Has approvals but still blocked - might be other branch protection rules
                return "Blocked by branch protection"

        except Exception:
            # Fallback if we can't analyze the specific reason
            return "Blocked"

    def fix_out_of_date_pr(self, owner: str, repo: str, pr_number: int) -> bool:
        """Fix an out-of-date PR by updating the branch."""
        try:
            repository = self.github.get_repo(f"{owner}/{repo}")
            pr = repository.get_pull(pr_number)

            if pr.mergeable_state != "behind":
                print(f"PR {pr_number} is not behind the base branch")
                return False

            # Update the branch using GitHub's update branch API
            pr.update_branch()
            return True
        except GithubException as e:
            print(f"Failed to update PR {pr_number}: {e}")
            return False
