# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Copilot review discovery, dismissibility analysis and review dismissal.

:class:`_CopilotReviewMixin` carries the review-level half of
``CopilotCommentHandler``: recognising which of a pull request's reviews
came from Copilot, classifying which of those GitHub will actually let us
dismiss, dismissing them, and pulling the matching REST review comments.

It is a mixin rather than a separate collaborator so the handler's method
surface stays exactly as it was before this module existed.  Every
attribute it reads is established by ``CopilotCommentHandler.__init__``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import logging
from typing import Any

from ..bot_identity import is_copilot
from ..models import PullRequestInfo, ReviewInfo


class _CopilotReviewMixin:
    """Review-level Copilot handling shared into ``CopilotCommentHandler``."""

    # Established by CopilotCommentHandler.__init__.
    github_client: Any
    preview_mode: bool
    debug: bool
    log: logging.Logger

    def is_copilot_review(self, review: ReviewInfo) -> bool:
        """
        Determine if a review is from GitHub Copilot.

        Args:
            review: Review to check

        Returns:
            True if review is from Copilot, False otherwise
        """
        if not review.user:
            return False

        # Route author matching through the shared identity predicate so
        # every Copilot login form (REST and GraphQL) is recognised.
        return is_copilot(review.user)

    def get_copilot_reviews(self, pr_info: PullRequestInfo) -> list[ReviewInfo]:
        """
        Extract all Copilot reviews from a pull request.

        Args:
            pr_info: Pull request information

        Returns:
            List of Copilot reviews
        """
        copilot_reviews = []

        for review in pr_info.reviews:
            if self.is_copilot_review(review):
                copilot_reviews.append(review)
                if self.debug:
                    self.log.info(
                        f"🤖 Found Copilot review: {review.id} - {review.state}"
                    )

        return copilot_reviews

    def get_unresolved_copilot_reviews(
        self, pr_info: PullRequestInfo
    ) -> list[ReviewInfo]:
        """
        Get unresolved Copilot reviews that may be blocking the merge.

        Args:
            pr_info: Pull request information

        Returns:
            List of unresolved Copilot reviews
        """
        copilot_reviews = self.get_copilot_reviews(pr_info)

        # Filter for reviews that are blocking (CHANGES_REQUESTED or COMMENTED)
        # Note: COMMENTED reviews cannot be dismissed but we include them for reporting
        unresolved = []
        for review in copilot_reviews:
            if review.state in ["CHANGES_REQUESTED", "COMMENTED", "PENDING"]:
                unresolved.append(review)
                if self.debug:
                    dismissible = (
                        "dismissible"
                        if review.state != "COMMENTED"
                        else "non-dismissible"
                    )
                    self.log.info(
                        f"🚫 Unresolved Copilot review: {review.id} (state: {review.state}, {dismissible})"
                    )

        return unresolved

    def analyze_copilot_review_dismissibility(
        self, pr_info: PullRequestInfo
    ) -> dict[str, int]:
        """
        Analyze which Copilot reviews can and cannot be dismissed.

        Args:
            pr_info: Pull request information

        Returns:
            Dictionary with counts of dismissible vs non-dismissible reviews
        """
        copilot_reviews = self.get_copilot_reviews(pr_info)

        dismissible_states = ["APPROVED", "CHANGES_REQUESTED"]
        non_dismissible_states = ["COMMENTED"]

        analysis = {
            "total": len(copilot_reviews),
            "dismissible": len(
                [r for r in copilot_reviews if r.state in dismissible_states]
            ),
            "non_dismissible": len(
                [r for r in copilot_reviews if r.state in non_dismissible_states]
            ),
            "pending": len([r for r in copilot_reviews if r.state == "PENDING"]),
            "other": len(
                [
                    r
                    for r in copilot_reviews
                    if r.state
                    not in dismissible_states + non_dismissible_states + ["PENDING"]
                ]
            ),
        }

        if self.debug:
            self.log.info(f"📊 Copilot review analysis for PR {pr_info.number}:")
            self.log.info(
                f"   Total: {analysis['total']}, Dismissible: {analysis['dismissible']}, Non-dismissible: {analysis['non_dismissible']}"
            )

        return analysis

    async def resolve_copilot_review(
        self, owner: str, repo: str, review_id: str, review_state: str | None = None
    ) -> bool:
        """
        Resolve a Copilot review by dismissing it (if possible).

        Args:
            owner: Repository owner
            repo: Repository name
            review_id: GraphQL ID of the review to dismiss
            review_state: Current state of the review (APPROVED, CHANGES_REQUESTED, COMMENTED)

        Returns:
            True if successfully resolved, False otherwise
        """
        if self.preview_mode:
            if review_state == "COMMENTED":
                self.log.info(
                    f"🔍 PREVIEW: Would skip COMMENTED Copilot review {review_id} (cannot be dismissed)"
                )
            else:
                self.log.info(
                    f"🔍 PREVIEW: Would dismiss Copilot review {review_id} (state: {review_state})"
                )
            return True

        # Skip COMMENTED reviews as they cannot be dismissed via GitHub API
        if review_state == "COMMENTED":
            self.log.info(
                f"⏭️ Skipping COMMENTED Copilot review {review_id} (GitHub API limitation)"
            )
            return True  # Return True as this is expected behavior, not a failure

        try:
            # Use GraphQL mutation to dismiss the pull request review
            mutation = """
            mutation DismissPullRequestReview($reviewId: ID!, $message: String!) {
              dismissPullRequestReview(input: {
                pullRequestReviewId: $reviewId
                message: $message
              }) {
                pullRequestReview {
                  id
                  state
                  author { login }
                }
              }
            }
            """

            variables = {
                "reviewId": review_id,
                "message": "Auto-dismissed by dependamerge: Copilot feedback resolved",
            }

            result = await self.github_client.graphql(mutation, variables)

            if result and (result.get("data") or {}).get("dismissPullRequestReview"):
                self.log.info(
                    f"✅ Successfully dismissed Copilot review {review_id} (state: {review_state})"
                )
                return True
            else:
                # Check if this is the known "commented review" error
                errors = result.get("errors", [])
                if any(
                    "Can not dismiss a commented pull request review" in str(error)
                    for error in errors
                ):
                    self.log.info(
                        f"⏭️ Cannot dismiss COMMENTED review {review_id} - this is a GitHub API limitation"
                    )
                    return True  # Treat as success since this is expected
                else:
                    self.log.error(
                        f"❌ Failed to dismiss Copilot review {review_id}: {result}"
                    )
                    return False

        except Exception as e:
            # Check if the error message contains the "commented review" limitation
            if "Can not dismiss a commented pull request review" in str(e):
                self.log.info(
                    f"⏭️ Cannot dismiss COMMENTED review {review_id} - this is a GitHub API limitation"
                )
                return True  # Treat as success since this is expected
            else:
                self.log.error(f"❌ Error dismissing Copilot review {review_id}: {e}")
                return False

    def has_blocking_copilot_comments(self, pr_info: PullRequestInfo) -> bool:
        """
        Check if a PR has unresolved Copilot reviews that might block merging.
        Note: This only checks reviews, not individual comments (which require async call).

        Args:
            pr_info: Pull request information

        Returns:
            True if there are blocking Copilot reviews, False otherwise
        """
        unresolved_reviews = self.get_unresolved_copilot_reviews(pr_info)
        return len(unresolved_reviews) > 0

    async def _get_copilot_review_comments(
        self, owner: str, repo: str, pr_number: int
    ) -> list[dict[str, Any]]:
        """
        Get Copilot review comments from REST API.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            List of Copilot review comments
        """
        try:
            all_comments = await self.github_client.get_pull_request_review_comments(
                owner, repo, pr_number
            )
            copilot_comments = []

            for comment in all_comments:
                author = (comment.get("user") or {}).get("login", "")
                # Check if comment is from Copilot
                if is_copilot(author):
                    copilot_comments.append(comment)
                    if self.debug:
                        self.log.info(
                            f"🤖 Found Copilot review comment: {comment.get('id')} on {comment.get('path', 'unknown')}"
                        )

            return copilot_comments

        except Exception as e:
            self.log.warning(
                f"⚠️ Could not fetch review comments for PR {pr_number}: {e}"
            )
            return []

    async def _resolve_review_comment_thread(self, comment: dict[str, Any]) -> bool:
        """
        Resolve a review comment thread by marking it as resolved.

        Args:
            comment: Review comment dictionary from REST API

        Returns:
            True if successfully resolved, False otherwise
        """
        if self.preview_mode:
            self.log.info(
                f"🔍 PREVIEW: Would resolve Copilot comment thread {comment.get('id')}"
            )
            return True

        try:
            self.log.info(
                f"ℹ️ Individual comment {comment.get('id')} handled via comprehensive thread resolution"
            )
            return True

        except Exception as e:
            self.log.error(f"❌ Error processing comment {comment.get('id')}: {e}")
            return False
