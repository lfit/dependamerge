# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Waiting for the checks on a dependabot-recreated pull request.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase

if TYPE_CHECKING:
    from ..github_async import GitHubAsync
    from ..models import FileChange


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
        client = self._github_client
        if not client:
            return None

        full_name = f"{repo_owner}/{repo_name}"
        html_url = pr_data.get(
            "html_url", f"https://github.com/{full_name}/pull/{new_number}"
        )

        self._pr_status(
            f"🔍 Found recreated PR, waiting for checks: {html_url}",
            level="info",
        )

        await self._auto_merge_recreated_pr(
            repo_owner, repo_name, new_number, pr_data, html_url
        )

        # Poll for the new PR to become mergeable
        max_check_polls = self._merge_poll_max_attempts
        for check_attempt in range(max_check_polls):
            await asyncio.sleep(self._merge_recheck_interval)
            try:
                refreshed = await client.get(
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
                    files_changed = await self._recreated_pr_files(
                        client, repo_owner, repo_name, new_number
                    )
                    return self._recreated_pr_info(
                        refreshed, new_number, full_name, html_url, files_changed
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

    async def _auto_merge_recreated_pr(
        self,
        repo_owner: str,
        repo_name: str,
        new_number: int,
        pr_data: dict[str, Any],
        html_url: str,
    ) -> None:
        """Arm auto-merge on the recreated PR before the wait begins.

        Auto-merge is enabled up front so the replacement still merges if
        the poll loop below gives up before the checks finish.  A full
        ``PullRequestInfo`` does not exist yet, so a minimal one is built
        from the search payload purely to satisfy the auto-merge helper;
        keeping that scaffolding here leaves the caller reading as the
        sequence of phases it is.  Without a node ID there is nothing for
        the GraphQL mutation to address, so the step is skipped.
        """
        if not pr_data.get("node_id"):
            return

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
            repository_full_name=f"{repo_owner}/{repo_name}",
            html_url=html_url,
        )
        await self._enable_auto_merge_for_pr(_tmp_pr, repo_owner, repo_name)

    async def _recreated_pr_files(
        self,
        client: GitHubAsync,
        repo_owner: str,
        repo_name: str,
        new_number: int,
    ) -> list[FileChange]:
        """Fetch the recreated PR's changed files, tolerating failure.

        The file list is decoration on a PR that has already been judged
        mergeable, so a pagination error is logged and swallowed and
        whatever was collected so far is returned rather than losing the
        PR.  Separated from the poll loop so that error handling sits
        next to the fetch it covers instead of nested inside the loop.
        """
        files_changed: list[FileChange] = []
        try:
            async for files_data in client.get_paginated(
                f"/repos/{repo_owner}/{repo_name}/pulls/{new_number}/files",
                per_page=100,
            ):
                self._collect_file_changes(files_data, files_changed)
        except Exception as e:
            self.log.debug(
                "Failed to fetch files for recreated PR %s#%s: %s",
                f"{repo_owner}/{repo_name}",
                new_number,
                e,
            )
        return files_changed

    @staticmethod
    def _collect_file_changes(files_data: Any, collected: list[FileChange]) -> None:
        """Append one page of the files response to ``collected``.

        Takes the destination list rather than returning a new one so
        that a malformed entry part-way through a page leaves the
        entries already converted in the caller's hands, exactly as the
        inline loop did: the caller logs the failure and keeps what it
        has.  A page that is not a list, or an entry that is not an
        object, is skipped rather than trusted.
        """
        from ..models import FileChange

        if not isinstance(files_data, list):
            return
        for f in files_data:
            if not isinstance(f, dict):
                continue
            collected.append(
                FileChange(
                    filename=f.get("filename", ""),
                    additions=int(f.get("additions", 0)),
                    deletions=int(f.get("deletions", 0)),
                    changes=int(f.get("changes", 0)),
                    status=f.get("status", "modified"),
                )
            )

    @staticmethod
    def _recreated_pr_info(
        refreshed: dict[str, Any],
        new_number: int,
        full_name: str,
        html_url: str,
        files_changed: list[FileChange],
    ) -> PullRequestInfo:
        """Build the result from the poll that found the PR mergeable.

        ``refreshed`` is the payload of the successful poll; the number
        and URL come from the caller instead, since the search that
        found the replacement is what identifies it.  Kept apart so the
        poll loop shows the decision rather than the field-by-field
        translation that follows it.
        """
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
            mergeable=refreshed.get("mergeable"),
            mergeable_state=refreshed.get("mergeable_state"),
            behind_by=None,
            files_changed=files_changed,
            repository_full_name=full_name,
            html_url=html_url,
        )
