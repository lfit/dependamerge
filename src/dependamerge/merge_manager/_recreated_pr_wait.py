# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Waiting for the checks on a dependabot-recreated pull request.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from typing import Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _RecreatedPrWaitMixin(_MergeManagerBase):
    """Waiting for the checks on a dependabot-recreated pull request."""

    async def _wait_for_recreated_pr_checks(
        self,
        repo_owner: str,
        repo_name: str,
        new_number: int,
        pr_data: dict[str, Any],
    ) -> PullRequestInfo | None:
        """
        Wait for the recreated PR's status checks to complete.

        Polls the new PR using the shared merge timeout settings. The
        total wait here is controlled by ``self._merge_poll_max_attempts
        * self._merge_recheck_interval`` (default: ~5 minutes), so
        ``--merge-timeout`` also affects this loop.

        Args:
            repo_owner: Repository owner.
            repo_name: Repository name.
            new_number: The PR number of the recreated pull request.
            pr_data: The initial PR data dict from the GitHub API.

        Returns:
            A ``PullRequestInfo`` if the PR became mergeable, None on timeout.
        """
        if not self._github_client:
            return None

        full_name = f"{repo_owner}/{repo_name}"
        html_url = pr_data.get(
            "html_url", f"https://github.com/{full_name}/pull/{new_number}"
        )

        self._pr_status(
            f"🔍 Found recreated PR, waiting for checks: {html_url}",
            level="info",
        )

        # Enable auto-merge on the recreated PR so it merges
        # even if we time out waiting for status checks.
        if pr_data.get("node_id"):
            # We don't have a full PullRequestInfo yet, but we can
            # construct a minimal one for the auto-merge helper.
            _tmp_pr = PullRequestInfo(
                number=new_number,
                node_id=pr_data.get("node_id"),
                title=pr_data.get("title", ""),
                body=pr_data.get("body"),
                author=((pr_data.get("user") or {}).get("login", "")),
                head_sha=((pr_data.get("head") or {}).get("sha", "")),
                base_branch=((pr_data.get("base") or {}).get("ref", "")),
                head_branch=((pr_data.get("head") or {}).get("ref", "")),
                state="open",
                mergeable=None,
                mergeable_state=None,
                behind_by=None,
                files_changed=[],
                repository_full_name=full_name,
                html_url=html_url,
            )
            await self._enable_auto_merge_for_pr(_tmp_pr, repo_owner, repo_name)

        # Poll for the new PR to become mergeable
        max_check_polls = self._merge_poll_max_attempts
        for check_attempt in range(max_check_polls):
            await asyncio.sleep(self._merge_recheck_interval)
            try:
                refreshed = await self._github_client.get(
                    f"/repos/{repo_owner}/{repo_name}/pulls/{new_number}"
                )
                if not isinstance(refreshed, dict):
                    continue

                mergeable = refreshed.get("mergeable")
                mergeable_state = refreshed.get("mergeable_state")

                if mergeable_state == "clean" or (
                    mergeable is True and mergeable_state in ("clean", "unstable")
                ):
                    self._pr_status(
                        f"✅ Recreated PR is ready to merge: {html_url}",
                        level="info",
                    )
                    from ..models import FileChange

                    files_changed: list[FileChange] = []
                    try:
                        async for files_data in self._github_client.get_paginated(
                            f"/repos/{repo_owner}/{repo_name}/pulls/{new_number}/files",
                            per_page=100,
                        ):
                            if isinstance(files_data, list):
                                for f in files_data:
                                    if isinstance(f, dict):
                                        files_changed.append(
                                            FileChange(
                                                filename=f.get("filename", ""),
                                                additions=int(f.get("additions", 0)),
                                                deletions=int(f.get("deletions", 0)),
                                                changes=int(f.get("changes", 0)),
                                                status=f.get("status", "modified"),
                                            )
                                        )
                    except Exception as e:
                        self.log.debug(
                            "Failed to fetch files for recreated PR %s#%s: %s",
                            f"{repo_owner}/{repo_name}",
                            new_number,
                            e,
                        )

                    return PullRequestInfo(
                        number=new_number,
                        node_id=refreshed.get("node_id"),
                        title=refreshed.get("title", ""),
                        body=refreshed.get("body"),
                        author=((refreshed.get("user") or {}).get("login", "")),
                        head_sha=((refreshed.get("head") or {}).get("sha", "")),
                        base_branch=((refreshed.get("base") or {}).get("ref", "")),
                        head_branch=((refreshed.get("head") or {}).get("ref", "")),
                        state=refreshed.get("state", "open"),
                        mergeable=mergeable,
                        mergeable_state=mergeable_state,
                        behind_by=None,
                        files_changed=files_changed,
                        repository_full_name=full_name,
                        html_url=html_url,
                    )

                if mergeable_state == "dirty":
                    self.log.warning(
                        "Recreated PR %s#%s has merge conflicts; aborting wait.",
                        full_name,
                        new_number,
                    )
                    return None

                # blocked / behind / unknown — keep polling
                if check_attempt % 3 == 2:
                    self.log.debug(
                        "Waiting for checks on recreated PR %s#%s "
                        "(state=%s, %.0fs elapsed)",
                        full_name,
                        new_number,
                        mergeable_state,
                        (check_attempt + 1) * self._merge_recheck_interval,
                    )

            except Exception as e:
                self.log.debug(
                    "Error polling recreated PR %s#%s: %s",
                    full_name,
                    new_number,
                    e,
                )

        self.log.warning(
            "Timed out waiting for checks on recreated PR %s#%s",
            full_name,
            new_number,
        )
        return None
