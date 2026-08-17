# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The ``CopilotCommentHandler`` entry point.

This module holds the handler's construction and its one orchestrating
operation, :meth:`CopilotCommentHandler.dismiss_copilot_comments_for_pr`,
which drives both halves of the behaviour: straight dismissal for reviews
GitHub allows us to dismiss, and thread-by-thread resolution for the
COMMENTED reviews it does not.

The two halves themselves live in :mod:`.reviews` and :mod:`.threads` and
are mixed in here, so the handler exposes exactly the method surface it
always did.
"""

from __future__ import annotations

import logging

from ..models import PullRequestInfo
from .reviews import _CopilotReviewMixin
from .threads import _CopilotThreadMixin

logger = logging.getLogger("dependamerge.copilot_handler")

# Common Copilot comment patterns that are often safe to dismiss
COMMON_COPILOT_PATTERNS = [
    r"use:\s+ubuntu-24\.04",  # Ubuntu version suggestions
    r"consider using.*instead of",  # Generic suggestions
    r"you might want to",  # Soft suggestions
    r"this could be improved by",  # Improvement suggestions
]


class CopilotCommentHandler(_CopilotReviewMixin, _CopilotThreadMixin):
    """Handler for managing GitHub Copilot review comments."""

    def __init__(self, github_client, preview_mode: bool = False, debug: bool = False):
        """
        Initialize the Copilot comment handler.

        Args:
            github_client: Async GitHub client for API operations
            preview_mode: If True, only simulate dismissal operations
            debug: Enable debug logging
        """
        self.github_client = github_client
        self.preview_mode = preview_mode
        self.debug = debug
        self.log = logging.getLogger("dependamerge.copilot_handler")

    async def dismiss_copilot_comments_for_pr(
        self, pr_info: PullRequestInfo
    ) -> tuple[int, int]:
        """
        Dismiss all unresolved Copilot reviews and comments for a pull request.

        Args:
            pr_info: Pull request information

        Returns:
            Tuple of (successful_dismissals, total_items)
        """
        owner, repo = pr_info.repository_full_name.split("/")

        unresolved_reviews = self.get_unresolved_copilot_reviews(pr_info)
        review_comments = await self._get_copilot_review_comments(
            owner, repo, pr_info.number
        )

        total_items = len(unresolved_reviews) + len(review_comments)

        if total_items == 0:
            self.log.info(
                f"✅ No unresolved Copilot feedback found for PR {pr_info.number}"
            )
            return 0, 0

        self.log.info(
            f"🤖 Found {len(unresolved_reviews)} Copilot reviews and {len(review_comments)} Copilot comments for PR {pr_info.number}"
        )

        successful_dismissals = 0
        thread_resolutions = 0

        for review in unresolved_reviews:
            if self.debug:
                self.log.info(
                    f"🔍 Processing Copilot review {review.id} (state: {review.state})"
                )

            if review.state == "COMMENTED":
                # For COMMENTED reviews, use thread resolution fallback
                self.log.info(
                    f"🧵 Using thread resolution for COMMENTED review {review.id}"
                )
                (
                    resolved_threads,
                    total_threads,
                ) = await self.resolve_copilot_threads_for_commented_review(
                    owner, repo, pr_info.number, review.id
                )
                if resolved_threads > 0:
                    successful_dismissals += (
                        1  # Count as success if we resolved threads
                    )
                    thread_resolutions += resolved_threads
                    if self.preview_mode:
                        self.log.info(
                            f"🔍 PREVIEW: Would resolve {resolved_threads}/{total_threads} threads in review {review.id}"
                        )
                    else:
                        self.log.info(
                            f"🧵 Resolved {resolved_threads}/{total_threads} threads in review {review.id}"
                        )
                else:
                    # No threads resolved - this is a failure, not success
                    # Logging is already handled in resolve_copilot_threads_for_commented_review
                    # Don't increment successful_dismissals - this is a failure
                    pass
            else:
                # For APPROVED/CHANGES_REQUESTED reviews, use standard dismissal
                success = await self.resolve_copilot_review(
                    owner, repo, review.id, review.state
                )
                if success:
                    successful_dismissals += 1

        # Handle individual review comments (deprecated in favor of thread resolution)
        for comment in review_comments:
            if self.debug:
                self.log.info(
                    f"🔍 Processing Copilot comment {comment.get('id')} on {comment.get('path', 'unknown file')}"
                )
                self.log.info(f"   Content: {comment.get('body', '')[:100]}...")

            # For review comments, we need to resolve the thread rather than dismiss
            success = await self._resolve_review_comment_thread(comment)
            if success:
                successful_dismissals += 1

        # Provide comprehensive reporting
        commented_reviews = len(
            [r for r in unresolved_reviews if r.state == "COMMENTED"]
        )
        dismissed_reviews = len(
            [r for r in unresolved_reviews if r.state != "COMMENTED"]
        )

        if commented_reviews > 0 and thread_resolutions > 0:
            self.log.info(
                f"📊 Processed {successful_dismissals}/{total_items} Copilot items for PR {pr_info.number}"
            )
            self.log.info(
                f"   └─ {dismissed_reviews} reviews dismissed, {commented_reviews} COMMENTED reviews processed via {thread_resolutions} thread resolutions"
            )
        elif commented_reviews > 0:
            self.log.info(
                f"📊 Processed {successful_dismissals}/{total_items} Copilot items for PR {pr_info.number}"
            )
            self.log.info(
                f"   └─ {dismissed_reviews} reviews dismissed, {commented_reviews} COMMENTED reviews processed (no resolvable threads)"
            )
        else:
            self.log.info(
                f"📊 Processed {successful_dismissals}/{total_items} Copilot items for PR {pr_info.number} (all via dismissal)"
            )

        return successful_dismissals, total_items
