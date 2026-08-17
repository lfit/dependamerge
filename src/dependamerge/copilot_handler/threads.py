# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Review-thread inspection and resolution for Copilot feedback.

:class:`_CopilotThreadMixin` carries the thread-level half of
``CopilotCommentHandler``.  GitHub refuses to dismiss a COMMENTED review,
so the only way to clear that feedback is to walk the pull request's
review threads, decide which ones are Copilot's and safe to close, and
resolve them individually.

It is a mixin rather than a separate collaborator so the handler's method
surface stays exactly as it was before this module existed.  Every
attribute it reads is established by ``CopilotCommentHandler.__init__``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import logging
from typing import Any

from ..bot_identity import is_copilot


class _CopilotThreadMixin:
    """Thread-level Copilot handling shared into ``CopilotCommentHandler``."""

    # Established by CopilotCommentHandler.__init__.
    github_client: Any
    preview_mode: bool
    log: logging.Logger

    async def get_pr_review_threads(
        self, owner: str, repo: str, pr_number: int
    ) -> list[dict[str, Any]]:
        """
        Get all review threads for a pull request.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            List of review thread data
        """
        from .github_graphql import GET_PR_REVIEW_THREADS

        threads = []
        cursor = None
        has_next = True

        while has_next:
            variables = {
                "owner": owner,
                "name": repo,
                "number": pr_number,
                "cursor": cursor,
            }

            result = await self.github_client.graphql(GET_PR_REVIEW_THREADS, variables)

            if (
                not result
                or not result.get("repository")
                or not result["repository"].get("pullRequest")
            ):
                self.log.error(
                    f"❌ Invalid GraphQL response structure for threads: {result}"
                )
                break

            pr_data = result["repository"]["pullRequest"]

            review_threads = pr_data["reviewThreads"]
            nodes = review_threads["nodes"]
            threads.extend(nodes)

            page_info = review_threads["pageInfo"]
            has_next = page_info["hasNextPage"]
            cursor = page_info["endCursor"]

        return threads

    def is_copilot_thread(self, thread: dict[str, Any]) -> bool:
        """
        Check if a review thread contains Copilot comments.

        Args:
            thread: Review thread data

        Returns:
            True if thread contains Copilot comments
        """
        comments = (thread.get("comments") or {}).get("nodes", [])

        for comment in comments:
            author = comment.get("author", {})
            if author and is_copilot(author.get("login")):
                return True

            # Also check comment body for Copilot patterns
            body = comment.get("body", "").lower()
            if any(
                pattern in body
                for pattern in ["github copilot", "copilot suggestion", "🤖"]
            ):
                return True

        return False

    def is_safe_copilot_thread_to_resolve(self, thread: dict[str, Any]) -> bool:
        """
        Check if a Copilot thread is safe to auto-resolve.

        Args:
            thread: Review thread data

        Returns:
            True if safe to resolve automatically
        """
        if thread.get("isResolved", False):
            return False  # Already resolved

        if thread.get("isOutdated", False):
            return True  # Outdated threads are usually safe to resolve

        # Check if this is a common/safe Copilot suggestion
        comments = (thread.get("comments") or {}).get("nodes", [])

        for comment in comments:
            body = comment.get("body", "").lower()

            # Safe patterns that are typically automation suggestions
            safe_patterns = [
                "use: ubuntu-24.04",
                "consider using",
                "you might want to",
                "suggestion:",
                "performance:",
                "style:",
                "formatting",
                "indentation",
                "whitespace",
            ]

            if any(pattern in body for pattern in safe_patterns):
                return True

            # Unsafe patterns that might need human attention
            unsafe_patterns = [
                "security",
                "vulnerability",
                "critical",
                "error",
                "bug",
                "broken",
                "incorrect",
            ]

            if any(pattern in body for pattern in unsafe_patterns):
                return False

        # Default to safe for general Copilot suggestions
        return True

    async def resolve_review_thread(self, thread_id: str, pr_context: str = "") -> bool:
        """
        Resolve a specific review thread.

        Args:
            thread_id: GraphQL ID of the thread to resolve

        Returns:
            True if successfully resolved
        """
        if self.preview_mode:
            context = f" for {pr_context}" if pr_context else ""
            self.log.info(
                f"🔍 PREVIEW: Would resolve review thread {thread_id}{context}"
            )
            return True

        from .github_graphql import RESOLVE_REVIEW_THREAD

        try:
            variables = {"threadId": thread_id}
            result = await self.github_client.graphql(RESOLVE_REVIEW_THREAD, variables)

            if result and result.get("resolveReviewThread"):
                thread_data = result["resolveReviewThread"]["thread"]
                if thread_data.get("isResolved"):
                    context = f" for {pr_context}" if pr_context else ""
                    self.log.info(f"✅ Resolved review thread {thread_id}{context}")
                    return True
                else:
                    context = f" for {pr_context}" if pr_context else ""
                    self.log.error(
                        f"❌ Thread {thread_id}{context} not marked as resolved in response: {thread_data}"
                    )

            context = f" for {pr_context}" if pr_context else ""
            self.log.error(
                f"❌ Failed to resolve review thread {thread_id}{context}. Full response: {result}"
            )
            if result and result.get("errors"):
                self.log.error(
                    f"❌ GraphQL errors for {thread_id}{context}: {result['errors']}"
                )
            return False

        except Exception as e:
            context = f" for {pr_context}" if pr_context else ""
            self.log.error(
                f"❌ Error resolving review thread {thread_id}{context}: {e}"
            )
            return False

    async def resolve_copilot_threads_for_commented_review(
        self, owner: str, repo: str, pr_number: int, review_id: str
    ) -> tuple[int, int]:
        """
        For COMMENTED reviews that can't be dismissed, try to resolve individual threads.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            review_id: Review ID (for logging context)

        Returns:
            Tuple of (resolved_count, total_copilot_threads)
        """
        self.log.info(
            f"🧵 Attempting thread-level resolution for COMMENTED review {review_id}"
        )

        all_threads = await self.get_pr_review_threads(owner, repo, pr_number)

        # Filter for unresolved Copilot threads that are safe to resolve
        copilot_threads = []
        for thread in all_threads:
            if (
                self.is_copilot_thread(thread)
                and not thread.get("isResolved", False)
                and self.is_safe_copilot_thread_to_resolve(thread)
            ):
                copilot_threads.append(thread)

        if not copilot_threads:
            self.log.warning(
                f"⚠️ Failed to resolve comment/review thread {review_id} in {owner}/{repo}#{pr_number} (no resolvable Copilot threads)"
            )
            return 0, len(all_threads)

        self.log.info(
            f"🎯 Found {len(copilot_threads)} resolvable Copilot threads out of {len(all_threads)} total for {owner}/{repo}#{pr_number}"
        )

        resolved_count = 0
        for i, thread in enumerate(copilot_threads, 1):
            thread_id = thread["id"]

            path = thread.get("path", "unknown")
            line = thread.get("line", "unknown")
            self.log.info(
                f"🔍 Resolving thread {i}/{len(copilot_threads)} in {owner}/{repo}#{pr_number}: {thread_id} on {path}:{line}"
            )

            if await self.resolve_review_thread(
                thread_id, f"{owner}/{repo}#{pr_number}"
            ):
                resolved_count += 1
                self.log.info(
                    f"✅ Successfully resolved thread {i}/{len(copilot_threads)} in {owner}/{repo}#{pr_number}"
                )
            else:
                self.log.error(
                    f"❌ Failed to resolve thread {i}/{len(copilot_threads)} in {owner}/{repo}#{pr_number}: {thread_id}"
                )

        self.log.info(
            f"📊 Resolved {resolved_count}/{len(copilot_threads)} Copilot threads for review {review_id} in {owner}/{repo}#{pr_number}"
        )
        return resolved_count, len(copilot_threads)
