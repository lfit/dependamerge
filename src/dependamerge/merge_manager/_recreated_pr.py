# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Waiting for a freshly recreated pull request to become ready.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase

if TYPE_CHECKING:
    from ..models import FileChange


def _collect_file_changes(files_data: Any, into: list[FileChange]) -> None:
    """Append one page of the files API response to ``into``.

    Appends in place so a failure part-way through pagination keeps
    whatever was already gathered, as the unrefactored loop did.
    """
    from ..models import FileChange

    if not isinstance(files_data, list):
        return
    for f in files_data:
        if isinstance(f, dict):
            into.append(
                FileChange(
                    filename=f.get("filename", ""),
                    additions=int(f.get("additions", 0)),
                    deletions=int(f.get("deletions", 0)),
                    changes=int(f.get("changes", 0)),
                    status=f.get("status", "modified"),
                )
            )


class _RecreatedPullRequestMixin(_MergeManagerBase):
    """Polling a recreated pull request until its checks settle."""

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

        await self._auto_merge_recreated_pr(repo_owner, repo_name, new_number, pr_data)

        # Poll for the new PR to become mergeable
        max_check_polls = self._merge_poll_max_attempts
        for check_attempt in range(max_check_polls):
            await asyncio.sleep(self._merge_recheck_interval)
            settled, ready = await self._poll_recreated_pr(
                full_name, new_number, html_url, check_attempt
            )
            if settled:
                return ready

        self.log.warning(
            "Timed out waiting for checks on recreated PR %s#%s",
            full_name,
            new_number,
        )
        return None

    async def _auto_merge_recreated_pr(
        self,
        repo_owner: str,
        repo_name: str,
        new_number: int,
        pr_data: dict[str, Any],
    ) -> None:
        """Enable auto-merge on the recreated PR.

        Doing so means it still merges even if we time out waiting for
        status checks.
        """
        if not pr_data.get("node_id"):
            return

        full_name = f"{repo_owner}/{repo_name}"
        html_url = pr_data.get(
            "html_url", f"https://github.com/{full_name}/pull/{new_number}"
        )

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

    async def _poll_recreated_pr(
        self,
        full_name: str,
        new_number: int,
        html_url: str,
        check_attempt: int,
    ) -> tuple[bool, PullRequestInfo | None]:
        """Take one reading of the recreated PR's mergeability.

        Returns a ``(settled, pr_info)`` pair.  ``settled`` is True once
        the wait is over: either the PR is ready, in which case
        ``pr_info`` describes it, or it is conflicted and the caller
        should give up.
        """
        if not self._github_client:
            return False, None
        try:
            refreshed = await self._github_client.get(
                f"/repos/{full_name}/pulls/{new_number}"
            )
            if not isinstance(refreshed, dict):
                return False, None

            mergeable = refreshed.get("mergeable")
            mergeable_state = refreshed.get("mergeable_state")

            if mergeable_state == "clean" or (
                mergeable is True and mergeable_state in ("clean", "unstable")
            ):
                self._pr_status(
                    f"✅ Recreated PR is ready to merge: {html_url}",
                    level="info",
                )
                files_changed = await self._recreated_pr_files(full_name, new_number)
                return True, PullRequestInfo(
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
                return True, None

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
        return False, None

    async def _recreated_pr_files(
        self, full_name: str, new_number: int
    ) -> list[FileChange]:
        """Gather the recreated PR's changed files, tolerating failure."""
        files_changed: list[FileChange] = []
        if not self._github_client:
            return files_changed
        try:
            async for files_data in self._github_client.get_paginated(
                f"/repos/{full_name}/pulls/{new_number}/files",
                per_page=100,
            ):
                _collect_file_changes(files_data, files_changed)
        except Exception as e:
            self.log.debug(
                "Failed to fetch files for recreated PR %s#%s: %s",
                full_name,
                new_number,
                e,
            )
        return files_changed
