# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Posting comments to a pull request, with retry.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio

from ..github_async import PermissionError as GitHubPermissionError
from ._base import _MergeManagerBase


class _CommentsMixin(_MergeManagerBase):
    """Posting comments to a pull request, with retry."""

    async def _post_pr_comment_with_retry(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        html_url: str,
        body: str,
    ) -> bool:
        """Post a PR comment with one retry after a 5s pause.

        Used for the auto-merge audit-trail comment so the PR
        conversation reflects that dependamerge enabled auto-merge.
        Approval comments take a different path: the approval body
        is passed directly to ``approve_pull_request()``, which
        creates a review (not an issue comment), so this helper is
        not used there. If both attempts fail, emit a single
        user-visible warning to the console rather than silently
        failing.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.
            html_url: Full PR URL, used for the warning message.
            body: Markdown body of the comment.

        Returns:
            True if the comment posted successfully (first or
            second attempt), False otherwise.
        """
        if not self._github_client:
            return False

        for attempt in (1, 2):
            try:
                await self._github_client.post_issue_comment(
                    owner, repo, pr_number, body
                )
                return True
            except GitHubPermissionError as exc:
                # Permission errors (typically HTTP 403) are not
                # transient — the token lacks the required scope or
                # the repo's branch protection forbids comments.
                # Skip the retry to avoid a pointless 5s delay per
                # PR and surface the failure right away.
                self.log.debug(
                    "Audit comment post denied (permission) for %s: %s",
                    html_url,
                    exc,
                )
                break
            except Exception as exc:
                # Heuristic: treat 4xx (other than 408/429) as
                # permanent and skip the retry. 5xx, 429 (rate
                # limit), 408 (timeout), and network/transport
                # errors get one retry after a short pause.
                #
                # We check several attribute paths so the
                # heuristic works across the various exception
                # shapes we may see:
                #   * ``exc.status_code`` — some custom wrappers
                #   * ``exc.status`` — ``aiohttp.ClientResponseError``
                #   * ``exc.response.status_code`` — ``httpx`` raises
                #     ``HTTPStatusError`` whose status lives on the
                #     attached ``Response`` object (this is what
                #     ``httpx.Response.raise_for_status()`` produces).
                response = getattr(exc, "response", None)
                status_code = (
                    getattr(exc, "status_code", None)
                    or getattr(exc, "status", None)
                    or getattr(response, "status_code", None)
                    or getattr(response, "status", None)
                )
                permanent = (
                    isinstance(status_code, int)
                    and 400 <= status_code < 500
                    and status_code not in (408, 429)
                )
                self.log.debug(
                    "Audit comment post attempt %d failed for %s: %s"
                    " (status=%r, permanent=%s)",
                    attempt,
                    html_url,
                    exc,
                    status_code,
                    permanent,
                )
                if permanent:
                    break
                if attempt == 1:
                    await asyncio.sleep(5.0)

        # Both attempts failed — surface a single line so the
        # user knows the PR-side audit trail is incomplete.
        self._pr_status(
            f"⚠️ Unable to add pull request comment: {html_url}",
            level="warning",
        )
        return False
