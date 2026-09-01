# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Mergeability interpretation behind ``GitHubClient``.

:class:`_GitHubStatusMixin` turns GitHub's ``mergeable`` /
``mergeable_state`` pair into the human-readable status the CLI prints,
decides whether a merge is worth attempting at all, and — only when a PR
reports ``blocked`` despite being mergeable — asks the API why.

``GitHubAsync`` is imported inside :meth:`_GitHubStatusMixin._analyze_block_reason`,
not at module scope, so that patching it in its own module stays
effective.  Every attribute this mixin reads is established by
``GitHubClient.__init__``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from collections.abc import Callable

from ..bot_identity import is_automation_author
from ..github_async import GitHubAsync
from ..models import PullRequestInfo


class _GitHubStatusMixin:
    """Mergeability interpretation shared into ``GitHubClient``."""

    # Established by GitHubClient.__init__.
    token: str
    host: str
    # Provided by GitHubClient: builds a transport client aimed at
    # ``host``, so an Enterprise run does not silently fall back to
    # github.com.  Annotated rather than defined, to avoid shadowing
    # the real method through the MRO.
    _new_async: Callable[..., GitHubAsync]

    def is_automation_author(self, author: str) -> bool:
        """Check if the author is a known automation tool.

        Delegates to the shared :func:`bot_identity.is_automation_author`
        so REST and GraphQL login forms are classified identically.
        """
        return is_automation_author(author)

    def get_pr_status_details(self, pr_info: PullRequestInfo) -> str:
        """Get detailed status information for a PR."""
        if pr_info.state != "open":
            return f"Closed ({pr_info.state})"

        if pr_info.mergeable_state == "draft":
            return "Draft PR"

        # Handle blocked state - need to determine why it's blocked
        if pr_info.mergeable_state == "blocked" and pr_info.mergeable is True:
            # This means technically mergeable but blocked by branch protection
            # We need to check what's blocking it to provide intelligent status
            block_reason = self._analyze_block_reason(pr_info)
            return block_reason

        if pr_info.mergeable is False:
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

        # Handle unstable state - this usually means CI is running but PR is mergeable
        if pr_info.mergeable is True and pr_info.mergeable_state == "unstable":
            return "Ready to merge"

        # For any other combination where mergeable is True but state is unclear
        if pr_info.mergeable is True:
            return "Ready to merge"

        # Fallback for unclear states
        return f"Status unclear ({pr_info.mergeable_state or 'unknown'})"

    def _analyze_block_reason(self, pr_info: PullRequestInfo) -> str:
        """Analyze why a PR is blocked and return appropriate status using REST."""
        try:
            repo_owner, repo_name = pr_info.repository_full_name.split("/")

            # Check if we're already in an event loop
            try:
                asyncio.get_running_loop()
                # We're in an async context - can't use asyncio.run()
                # Return a basic status message to avoid the coroutine warning
                # The caller should use the async version instead
                return "Blocked by branch protection"
            except RuntimeError:
                # No event loop running - safe to use asyncio.run()
                pass

            async def _run():
                async with self._new_async() as api:
                    return await api.analyze_block_reason(
                        repo_owner, repo_name, pr_info.number, pr_info.head_sha
                    )

            return asyncio.run(_run())  # type: ignore[no-any-return]
        except Exception:
            return "Blocked"

    def _should_attempt_merge(self, pr) -> bool:
        """
        Determine if we should attempt to merge a PR based on its mergeable state.

        Returns True if merge should be attempted, False otherwise.
        """
        # If mergeable is explicitly False, only attempt merge for blocked state
        # where branch protection might resolve after approval
        if pr.mergeable is False:
            # For blocked state, we can attempt merge as approval might resolve the block
            # For other states (dirty, behind), don't attempt as they need manual fixes
            return bool(pr.mergeable_state == "blocked")

        # If mergeable is None, GitHub is still calculating - be conservative
        if pr.mergeable is None:
            # Only attempt if state suggests it might work
            return bool(pr.mergeable_state in ["clean", "blocked"])

        # If mergeable is True, attempt merge for most states except draft
        if pr.mergeable is True:
            return bool(pr.mergeable_state != "draft")

        # Fallback to False for any unexpected cases
        return False
