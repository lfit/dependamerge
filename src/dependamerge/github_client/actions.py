# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The state-changing GitHub API calls behind ``GitHubClient``.

:class:`_GitHubActionMixin` carries the three operations that write:
approving a pull request, merging one, and updating an out-of-date branch.
All three share the same contract — they swallow failures, log a warning
and report a boolean — so that a bulk run is never derailed by one
repository refusing an operation.

``GitHubAsync`` is imported inside each method, not at module scope, so
that patching it in its own module stays effective.  Every attribute this
mixin reads is established by ``GitHubClient.__init__``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("dependamerge.github_client")


class _GitHubActionMixin:
    """State-changing GitHub operations shared into ``GitHubClient``."""

    # Established by GitHubClient.__init__.
    token: str

    def approve_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        message: str = "Auto-approved by dependamerge",
    ) -> bool:
        """Approve a pull request using the async REST client."""
        try:
            from ..github_async import GitHubAsync

            async def _run():
                async with GitHubAsync(token=self.token) as api:
                    await api.approve_pull_request(owner, repo, pr_number, message)
                    return True

            return bool(asyncio.run(_run()))
        except Exception as e:
            logger.warning("Failed to approve PR %s: %s", pr_number, e, exc_info=True)
            return False

    def merge_pull_request(
        self, owner: str, repo: str, pr_number: int, merge_method: str = "merge"
    ) -> bool:
        """Merge a pull request using the async REST client."""
        try:
            from ..github_async import GitHubAsync

            async def _run():
                async with GitHubAsync(token=self.token) as api:
                    return await api.merge_pull_request(
                        owner, repo, pr_number, merge_method
                    )

            return bool(asyncio.run(_run()))
        except Exception as e:
            logger.warning("Failed to merge PR %s: %s", pr_number, e, exc_info=True)
            return False

    def fix_out_of_date_pr(self, owner: str, repo: str, pr_number: int) -> bool:
        """Fix an out-of-date PR by updating the branch."""
        try:
            from ..github_async import GitHubAsync

            async def _run():
                async with GitHubAsync(token=self.token) as api:
                    await api.update_branch(owner, repo, pr_number)
                    return True

            return bool(asyncio.run(_run()))
        except Exception as e:
            logger.warning("Failed to update PR %s: %s", pr_number, e, exc_info=True)
            return False
