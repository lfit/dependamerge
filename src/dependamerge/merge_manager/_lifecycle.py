# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Async context management and presentation helpers.

Construction of the shared GitHub client and its collaborators on
entry, their orderly shutdown on exit, and the mergeable-state icon
lookup the status output uses.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import dependamerge.merge_manager as _pkg

from ..copilot_handler import CopilotCommentHandler
from ..github_service import GitHubService
from ..pr_poller import PullRequestStatePoller
from ._base import _MergeManagerBase
from ._constants import (
    _MERGEABILITY_ICON_AND_STYLE,
)


class _LifecycleMixin(_MergeManagerBase):
    """Async context management and presentation helpers."""

    def __repr__(self) -> str:
        """Safe repr that never exposes the token value."""
        return "AsyncMergeManager(token=***)"

    def _get_mergeability_icon_and_style(
        self, mergeable_state: str | None
    ) -> tuple[str, str | None]:
        """Get appropriate icon and style for mergeable state."""
        return _MERGEABILITY_ICON_AND_STYLE.get(mergeable_state, ("🔍", None))

    async def __aenter__(self):
        """Async context manager entry."""
        self._github_client = _pkg.GitHubAsync(token=self.token)
        await self._github_client.__aenter__()

        # Coalesces the wait loops' per-PR state reads into batched
        # GraphQL queries.  See ``pr_poller`` for why: unbatched, polling
        # costs 6 requests/min per parked PR, so ~14 parked PRs consume
        # the entire REST budget doing nothing but asking for status.
        self._pr_poller = PullRequestStatePoller(self._github_client, log=self.log)

        # Share the one client.  Rate limiting, concurrency and adaptive
        # throttling are all per-instance, so a second client doubled the
        # effective ceiling --- 40 concurrent / 16 rps rather than the
        # 20 / 8 reported --- against a budget GitHub shares between
        # them, and left each half blind to the pressure the other was
        # causing.  See ``docs/BULK_RUN_PERFORMANCE_AUDIT.md`` §2.4.
        self._github_service = GitHubService(
            token=self.token, client=self._github_client
        )

        # Initialize Copilot handler if dismissal is enabled
        if self.dismiss_copilot:
            self._copilot_handler = CopilotCommentHandler(
                self._github_client, preview_mode=self.preview_mode, debug=True
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._github_service:
            await self._github_service.close()
        if self._github_client:
            await self._github_client.__aexit__(exc_type, exc_val, exc_tb)
        # Drop the poller and client together: the poller holds the now
        # closed client, so leaving it in place would route any later
        # ``_fetch_pr_state`` call into a use-after-close instead of the
        # direct-read fallback that method documents.
        self._pr_poller = None
        self._github_client = None
