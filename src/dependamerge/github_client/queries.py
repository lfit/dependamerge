# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The read-only GitHub API calls behind ``GitHubClient``.

:class:`_GitHubQueryMixin` carries the operations that only fetch:
pull request detail, commit messages, an organization's repositories and
the organization-wide unmergeable-PR scan.  Each one runs the async
client for the duration of a single call via :func:`asyncio.run`, which
is what makes them usable from the synchronous CLI.

The three module-level helpers below carry the pieces of
:meth:`_GitHubQueryMixin.get_pull_request_info` that are independent of
the client: fetching the changed files, fetching the reviews, and
assembling the :class:`PullRequestInfo` from the raw REST payload.

``GitHubAsync`` and ``GitHubService`` are imported inside the methods
that use them, not at module scope, so that patching them in their own
modules stays effective.  Every attribute this mixin reads is
established by ``GitHubClient.__init__``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..models import (
    FileChange,
    OrganizationScanResult,
    PullRequestInfo,
    ReviewInfo,
)

if TYPE_CHECKING:
    from ..github_async import GitHubAsync
    from ..progress_tracker import ProgressTracker


async def _fetch_pr_files(
    api: GitHubAsync, owner: str, repo: str, pr_number: int
) -> list[FileChange]:
    """Fetch a pull request's changed files, tolerating pagination failure."""
    files_changed: list[FileChange] = []
    try:
        async for page in api.get_paginated(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/files", per_page=100
        ):
            for f in page:
                file_data = f
                assert isinstance(file_data, dict)
                files_changed.append(
                    FileChange(
                        filename=file_data.get("filename", ""),
                        additions=int(file_data.get("additions", 0)),
                        deletions=int(file_data.get("deletions", 0)),
                        changes=int(
                            file_data.get(
                                "changes",
                                (file_data.get("additions", 0) or 0)
                                + (file_data.get("deletions", 0) or 0),
                            )
                        ),
                        status=file_data.get("status", "modified"),
                    )
                )
    except Exception as exc:
        # If pagination of files fails, continue with what we have
        api.log.debug(
            f"File pagination failed for PR #{pr_number}: {exc}",
            exc_info=True,
        )
    return files_changed


async def _fetch_pr_reviews(
    api: GitHubAsync, owner: str, repo: str, pr_number: int
) -> list[ReviewInfo]:
    """Fetch a pull request's reviews, tolerating a failed review call."""
    reviews: list[ReviewInfo] = []
    try:
        review_response = await api.get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        )
        assert isinstance(review_response, list), "reviews must be a list"
        review_data = review_response
        # review_data is a list of review dictionaries
        for review in review_data:
            if not isinstance(review, dict):
                continue
            if review.get("user") and review.get("state"):
                reviews.append(
                    ReviewInfo(
                        # NOTE: REST API returns string IDs that look numeric but may be node IDs
                        # Do not convert to int() - keep as string to match GraphQL behavior
                        id=review.get("id", ""),
                        user=(review.get("user") or {}).get("login", ""),
                        state=review.get("state", ""),
                        submitted_at=review.get("submitted_at", ""),
                        body=review.get("body"),
                    )
                )
    except Exception as exc:
        # If review fetching fails, continue without reviews
        api.log.debug(
            f"Review fetch failed for PR #{pr_number}: {exc}",
            exc_info=True,
        )
    return reviews


def _build_pull_request_info(
    pr: dict[str, Any],
    owner: str,
    repo: str,
    pr_number: int,
    files_changed: list[FileChange],
    reviews: list[ReviewInfo],
) -> PullRequestInfo:
    """Assemble a ``PullRequestInfo`` from a REST pull request payload."""
    return PullRequestInfo(
        number=int(pr.get("number", pr_number)),
        node_id=pr.get("node_id"),  # REST API uses "node_id" key
        title=pr.get("title") or "",
        body=pr.get("body"),
        author=((pr.get("user") or {}).get("login") or ""),
        head_sha=((pr.get("head") or {}).get("sha") or ""),
        base_branch=((pr.get("base") or {}).get("ref") or ""),
        head_branch=((pr.get("head") or {}).get("ref") or ""),
        state=pr.get("state") or "open",
        mergeable=pr.get("mergeable"),
        mergeable_state=pr.get("mergeable_state"),
        behind_by=None,
        files_changed=files_changed,
        repository_full_name=f"{owner}/{repo}",
        html_url=pr.get("html_url") or "",
        reviews=reviews,
        # Populate head/base repo identity so the
        # signature-preserving local-rebase path can
        # tell whether the PR is from a fork (and
        # which remote to push to).  Without these,
        # ``rebase.local_rebase_pr()`` fails closed
        # to avoid pushing to the wrong repository.
        head_repo_full_name=(
            ((pr.get("head") or {}).get("repo") or {}).get("full_name")
        ),
        head_repo_clone_url=(
            ((pr.get("head") or {}).get("repo") or {}).get("clone_url")
        ),
        base_repo_full_name=(
            ((pr.get("base") or {}).get("repo") or {}).get("full_name")
        ),
        base_repo_clone_url=(
            ((pr.get("base") or {}).get("repo") or {}).get("clone_url")
        ),
        is_fork=(((pr.get("head") or {}).get("repo") or {}).get("fork")),
    )


class _GitHubQueryMixin:
    """Read-only GitHub operations shared into ``GitHubClient``."""

    # Established by GitHubClient.__init__.
    token: str
    host: str
    # Provided by GitHubClient: builds a transport client aimed at
    # ``host``, so an Enterprise run does not silently fall back to
    # github.com.  Annotated rather than defined, to avoid shadowing
    # the real method through the MRO.
    _new_async: Callable[..., GitHubAsync]

    def get_pull_request_info(
        self, owner: str, repo: str, pr_number: int
    ) -> PullRequestInfo:
        """Get detailed information about a pull request using the async REST client."""

        async def _run() -> PullRequestInfo:
            async with self._new_async() as api:
                pr_response = await api.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
                assert isinstance(pr_response, dict), "PR endpoint must return a dict"
                pr: dict[str, Any] = pr_response
                files_changed = await _fetch_pr_files(api, owner, repo, pr_number)
                reviews = await _fetch_pr_reviews(api, owner, repo, pr_number)
                return _build_pull_request_info(
                    pr, owner, repo, pr_number, files_changed, reviews
                )

        return asyncio.run(_run())  # type: ignore[no-any-return]

    def get_pull_request_commits(
        self, owner: str, repo: str, pr_number: int
    ) -> list[str]:
        """Get commit messages from a pull request using the async REST client."""

        async def _run() -> list[str]:
            messages: list[str] = []
            async with self._new_async() as api:
                async for page in api.get_paginated(
                    f"/repos/{owner}/{repo}/pulls/{pr_number}/commits", per_page=100
                ):
                    for c in page:
                        commit_data = c
                        assert isinstance(commit_data, dict)
                        msg = (commit_data.get("commit") or {}).get("message") or ""
                        if msg:
                            messages.append(msg)
            return messages

        return asyncio.run(_run())

    def get_organization_repositories(self, org_name: str) -> list[str]:
        """Get all repositories in an organization using REST API. Returns list of full_name strings."""

        async def _run() -> list[str]:
            repos: list[str] = []
            async with self._new_async() as api:
                try:
                    async for page in api.get_paginated(
                        f"/orgs/{org_name}/repos", per_page=100
                    ):
                        for r in page:
                            repo_data = r
                            assert isinstance(repo_data, dict)
                            full = repo_data.get("full_name")
                            if full:
                                repos.append(full)
                except Exception as exc:
                    # Fall back to empty list on pagination issues
                    api.log.debug(
                        f"Repo pagination failed for org {org_name}: {exc}",
                        exc_info=True,
                    )
            return repos

        return asyncio.run(_run())

    def get_open_pull_requests(self, repository) -> list[Any]:
        """Legacy method not supported in async-only client. Use async service for PR enumeration."""
        return []

    def scan_organization_for_unmergeable_prs(
        self,
        org_name: str,
        progress_tracker: ProgressTracker | None = None,
        include_drafts: bool = False,
    ) -> OrganizationScanResult:
        """Scan an entire GitHub organization for unmergeable pull requests using the async service.

        Args:
            org_name: The organization name to scan.
            progress_tracker: Optional progress tracker for UI updates.
            include_drafts: If True, include draft PRs in results. If False (default),
                          filter out PRs that are only blocked due to draft status.
        """
        scan_timestamp = datetime.now().isoformat()
        from ..github_service import GitHubService

        async def _run():
            svc = GitHubService(
                token=self.token,
                host=self.host,
                progress_tracker=progress_tracker,
            )
            try:
                return await svc.scan_organization(
                    org_name, include_drafts=include_drafts
                )
            finally:
                await svc.close()

        try:
            return asyncio.run(_run())  # type: ignore[no-any-return]
        except Exception as e:
            return OrganizationScanResult(
                organization=org_name,
                total_repositories=0,
                scanned_repositories=0,
                total_prs=0,
                unmergeable_prs=[],
                scan_timestamp=scan_timestamp,
                errors=[f"{e}"],
            )
