# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Shared attribute and method declarations for the ``GitHubService`` mixins.

``GitHubService`` is too large for one reviewable module, so its methods
live in ``_XxxMixin`` classes that are mixed back together in
``dependamerge.github_service._service``.  Each mixin reads state that
``GitHubService.__init__`` establishes and calls methods implemented by
its siblings; this base declares both, and declares nothing at runtime,
so the real implementations always win.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from ..github_async import GitHubAsync
from ..models import (
    CopilotComment,
    FileChange,
    PullRequestInfo,
    ReviewInfo,
    UnmergeablePR,
)


class _GitHubServiceBase:
    """Declarations shared by the ``GitHubService`` mixins."""

    # Established by GitHubService.__init__.
    log: logging.Logger
    _api: GitHubAsync
    _owns_api: bool
    _callbacks_attached: bool
    _progress: Any | None
    _max_repo_tasks: int
    _max_page_tasks: int
    _repo_semaphore: asyncio.Semaphore
    _page_semaphore: asyncio.Semaphore
    _rate_limited: bool
    _debug_matching: bool
    _branch_protection_cache: dict[str, dict[str, Any] | None]
    _owner_root_cache: dict[str, tuple[str, str]]

    # Implemented by sibling mixins; declared here (type-checking only,
    # so nothing is defined at runtime) for cross-mixin calls.
    if TYPE_CHECKING:

        def _split_owner_repo(self, full_name: str) -> tuple[str, str]: ...

        def _iter_org_repositories(self, org: str) -> AsyncIterator[dict[str, Any]]: ...

        def _iter_org_repositories_with_open_prs(
            self, org: str
        ) -> AsyncIterator[dict[str, Any]]: ...

        def _iter_owner_repositories(
            self, owner: str, *, skip_forks: bool = True
        ) -> AsyncIterator[dict[str, Any]]: ...

        def _iter_repo_open_prs_pages(
            self, owner: str, name: str, cursor: str | None
        ) -> AsyncIterator[dict[str, Any]]: ...

        async def _fetch_repo_prs_first_page(
            self, owner: str, name: str
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...

        async def _analyze_pr_node(
            self,
            repo_full_name: str,
            pr: dict[str, Any],
            include_drafts: bool = False,
        ) -> UnmergeablePR | None: ...

        def to_pull_request_info(
            self, repo_full_name: str, pr: dict[str, Any]
        ) -> PullRequestInfo: ...

        def _map_mergeable_enum(self, value: str | None) -> bool | None: ...

        def _safe_get_merge_state(
            self, merge_state_status: str | None
        ) -> str | None: ...

        def _extract_file_changes(self, pr: dict[str, Any]) -> list[FileChange]: ...

        def _extract_reviews(self, pr: dict[str, Any]) -> list[ReviewInfo]: ...

        def _extract_copilot_comments(
            self, pr: dict[str, Any]
        ) -> list[CopilotComment]: ...

        @staticmethod
        def _extract_failing_checks(pr: dict[str, Any]) -> list[str]: ...

        async def _collect_repo_open_prs(
            self, owner: str, repo: str, *, only_automation: bool
        ) -> list[PullRequestInfo]: ...

        def _is_automation_author(self, author: str) -> bool: ...

        def _affects_action_files(self, files: list[dict[str, Any]]) -> bool: ...

        def _affects_workflow_files(self, files: list[dict[str, Any]]) -> bool: ...

        async def _get_latest_tag(
            self, owner: str, name: str
        ) -> tuple[str | None, str | None]: ...

        async def _get_latest_release(
            self, owner: str, name: str
        ) -> tuple[str | None, str | None]: ...

        def _determine_status_icon(
            self,
            latest_tag: str | None,
            latest_release: str | None,
            tag_date: str | None,
            release_date: str | None,
        ) -> str: ...

        async def _gather_pr_statistics(
            self, owner: str, name: str, since_date: str | None
        ) -> dict[str, int]: ...

        async def _get_merged_prs_since(
            self, owner: str, name: str, since_date: str
        ) -> list[dict[str, Any]]: ...
