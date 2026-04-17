# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation


from pydantic import BaseModel


class ReviewInfo(BaseModel):
    """Represents a review on a pull request."""

    # IMPORTANT: GraphQL returns string node IDs (e.g., "PRR_kwDOGBtQpc4-u-zD")
    # NOT numeric IDs. This must remain str type to avoid runtime conversion errors.
    # AI tools may suggest changing this to int - DO NOT DO IT.
    id: str
    user: str
    state: str  # APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED
    submitted_at: str
    body: str | None = None


class ReviewComment(BaseModel):
    """Represents a review comment on a pull request."""

    id: str
    database_id: int
    author: str
    body: str
    state: str
    path: str | None = None
    position: int | None = None
    created_at: str
    updated_at: str
    pull_request_review_id: str | None = None
    pull_request_review_author: str | None = None
    pull_request_review_state: str | None = None


class FileChange(BaseModel):
    """Represents a file change in a pull request."""

    filename: str
    additions: int
    deletions: int
    changes: int
    status: str  # added, modified, removed, renamed


class PullRequestInfo(BaseModel):
    """Represents pull request information."""

    number: int
    node_id: str | None = None
    title: str
    body: str | None
    author: str
    head_sha: str
    base_branch: str
    head_branch: str
    state: str
    mergeable: bool | None
    mergeable_state: str | None  # Additional state information from GitHub
    behind_by: int | None  # Number of commits behind the base branch
    files_changed: list[FileChange]
    repository_full_name: str
    html_url: str
    reviews: list[ReviewInfo] = []  # PR reviews
    review_comments: list[ReviewComment] = []  # Review comments (including Copilot)

    # Optional fields used by the interactive fix workflow
    # These enable cloning the correct repositories and pushing fixes.
    head_repo_full_name: str | None = None
    head_repo_clone_url: str | None = None
    base_repo_full_name: str | None = None
    base_repo_clone_url: str | None = None
    is_fork: bool | None = None
    maintainer_can_modify: bool | None = None


class ComparisonResult(BaseModel):
    """Result of comparing two pull requests."""

    is_similar: bool
    confidence_score: float
    reasons: list[str]


class UnmergeableReason(BaseModel):
    """Represents a reason why a PR cannot be merged."""

    type: str  # e.g., "merge_conflict", "failing_checks", "blocked_review"
    description: str
    details: str | None = None


class CopilotComment(BaseModel):
    """Represents an unresolved Copilot feedback comment."""

    id: int
    body: str
    file_path: str | None = None
    line_number: int | None = None
    created_at: str
    state: str  # "open", "resolved", etc.


class UnmergeablePR(BaseModel):
    """Represents a pull request that cannot be merged."""

    repository: str
    pr_number: int
    title: str
    author: str
    url: str
    reasons: list[UnmergeableReason]
    copilot_comments_count: int = 0
    copilot_comments: list[CopilotComment] = []
    created_at: str
    updated_at: str


class OrganizationScanResult(BaseModel):
    """Result of scanning an organization for unmergeable PRs."""

    organization: str
    total_repositories: int
    scanned_repositories: int
    total_prs: int
    unmergeable_prs: list[UnmergeablePR]
    scan_timestamp: str
    errors: list[str] = []


class RepositoryStatus(BaseModel):
    """Status information for a single repository."""

    repository_name: str
    latest_tag: str | None = None
    latest_release: str | None = None
    tag_date: str | None = None
    release_date: str | None = None
    status_icon: str = "❌"  # ✅, ⚠️, or ❌
    open_prs_human: int = 0
    open_prs_automation: int = 0
    merged_prs_human: int = 0
    merged_prs_automation: int = 0
    action_prs_human: int = 0
    action_prs_automation: int = 0
    workflow_prs_human: int = 0
    workflow_prs_automation: int = 0


class OrganizationStatus(BaseModel):
    """Result of gathering repository status for an organization."""

    organization: str
    total_repositories: int
    scanned_repositories: int
    repository_statuses: list[RepositoryStatus]
    scan_timestamp: str
    errors: list[str] = []
