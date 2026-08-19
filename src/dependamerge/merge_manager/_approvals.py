# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Deciding whether to approve a pull request, and doing so.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import Any

from ..github_async import PermissionError as GitHubPermissionError
from ._base import _MergeManagerBase


class _ApprovalsMixin(_MergeManagerBase):
    """Deciding whether to approve a pull request, and doing so."""

    @staticmethod
    def _already_sufficiently_approved(
        pr_data: dict[str, Any],
        reviews_data: list[Any],
        current_user: str,
    ) -> tuple[bool, str | None]:
        """Return ``(skip, approvers)`` when the PR needs no new approval.

        ``skip`` is True when the current user has already approved, or when
        other reviewers have approved a ``clean`` PR.  ``approvers`` names the
        relevant approver(s) for the debug log.
        """
        for review in reviews_data:
            if not isinstance(review, dict):
                continue
            if (review.get("user") or {}).get("login") == current_user and review.get(
                "state"
            ) == "APPROVED":
                return True, current_user

        approved_reviews = [
            review
            for review in reviews_data
            if isinstance(review, dict)
            and review.get("state") == "APPROVED"
            and (review.get("user") or {}).get("login") != current_user
        ]
        if approved_reviews and pr_data.get("mergeable_state") == "clean":
            # A review may carry ``"user": null`` or ``{"login": null}``;
            # coerce both to a string so join() cannot see a None login.
            approvers = [
                (review.get("user") or {}).get("login") or "unknown"
                for review in approved_reviews
            ]
            return True, ", ".join(approvers)
        return False, None

    async def _should_skip_approval(
        self, owner: str, repo: str, pr_number: int
    ) -> tuple[bool, str | None]:
        """Return ``(skip, approvers)`` when the PR already has adequate approval."""
        if self._github_client is None:
            raise RuntimeError("GitHub client not initialized")
        pr_data = await self._github_client.get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}"
        )
        if not isinstance(pr_data, dict):
            return False, None
        # Get current user login (cached on the client after the first
        # call — the login is session-constant, so this costs one
        # round-trip per run instead of one per PR).
        current_user = await self._github_client.get_authenticated_user_login()
        if not current_user:
            return False, None
        reviews_data = await self._github_client.get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        )
        if not isinstance(reviews_data, list):
            return False, None
        return self._already_sufficiently_approved(pr_data, reviews_data, current_user)

    async def _approve_pr(self, owner: str, repo: str, pr_number: int) -> bool:
        """
        Approve a pull request if not already approved by the current user or sufficiently approved.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            True if approval was added, False if already approved/sufficient

        Raises:
            Exception: If approval fails
        """
        if not self._github_client:
            raise RuntimeError("GitHub client not initialized")

        try:
            skip, approvers = await self._should_skip_approval(owner, repo, pr_number)
            if skip:
                self.log.debug(
                    f"⏩ Already approved: {owner}/{repo}#{pr_number} [{approvers}]"
                )
                return False

            await self._github_client.approve_pull_request(
                owner,
                repo,
                pr_number,
                "🤖 Dependamerge\nApproved this pull request ✅",
            )
            return True
        except GitHubPermissionError:
            # Let typed permission errors propagate to the caller's
            # dedicated handler in ``_merge_single_pr``.  Wrapping
            # them in a generic ``RuntimeError`` (as the old broad
            # ``except Exception`` below did) hid them from that
            # handler and routed the failure through the catch-all
            # path, which dumps a full stack trace to stderr on
            # every PR in the batch.
            raise
        except Exception as e:
            error_str = str(e)

            # Check for 403 Forbidden - missing pull request review permissions
            if "403" in error_str and "Forbidden" in error_str:
                raise RuntimeError(
                    f"Failed to approve PR {owner}/{repo}#{pr_number}: Missing 'Pull requests: Read and write' permission. "
                    f"For fine-grained tokens, enable 'Pull requests: Read and write' access. "
                    f"For classic tokens, ensure 'repo' scope is enabled."
                ) from e
            elif "422" in error_str and "Unprocessable Entity" in error_str:
                # This usually means the PR can't be approved (e.g., already approved by user, or other restrictions)
                self.log.debug(
                    f"⏩ Already approved: {owner}/{repo}#{pr_number} [cannot approve - already approved or restricted]"
                )
                return False
            else:
                raise RuntimeError(
                    f"Failed to approve PR {owner}/{repo}#{pr_number}: {e}"
                ) from e
