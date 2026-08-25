# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Reading a recreated pull request's state and classifying it.

Split from :mod:`._recreated_pr`, which orchestrates the wait; this
module holds the per-poll reading, the terminal classification of a
closed replacement, and the changed-files fetch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._types import RecreateOutcome, RecreateResult, _merged_from_payload

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


class _RecreatedPollMixin(_MergeManagerBase):
    """Taking and classifying readings of a recreated pull request."""

    async def _poll_recreated_pr(
        self,
        full_name: str,
        new_number: int,
        html_url: str,
        check_attempt: int,
    ) -> RecreateResult | None:
        """Take one reading of the recreated PR's state.

        Returns ``None`` while the wait should continue, or a
        :class:`RecreateResult` once it has reached a terminal state.
        """
        if not self._github_client:
            return None
        try:
            refreshed = await self._github_client.get(
                f"/repos/{full_name}/pulls/{new_number}"
            )
            if not isinstance(refreshed, dict):
                return None

            mergeable = refreshed.get("mergeable")
            mergeable_state = refreshed.get("mergeable_state")

            # Checked first: auto-merge was armed before this loop began,
            # so the replacement can close between polls.  A closed
            # payload satisfies neither the ready test nor the "dirty"
            # test below, so without this branch the loop would run to
            # timeout and the caller would report a failure for a PR
            # that had in fact merged.
            if refreshed.get("state") == "closed":
                return self._closed_recreated_pr(
                    full_name, new_number, html_url, refreshed
                )

            # ``mergeable_state == "clean"`` alone is enough; the second
            # test covers "unstable", where non-required checks are still
            # running but the PR can merge.
            if mergeable_state == "clean" or (
                mergeable is True and mergeable_state == "unstable"
            ):
                self._pr_status(
                    f"✅ Recreated PR is ready to merge: {html_url}",
                    level="info",
                )
                files_changed = await self._recreated_pr_files(full_name, new_number)
                return RecreateResult(
                    RecreateOutcome.READY,
                    PullRequestInfo(
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
                    ),
                )

            if mergeable_state == "dirty":
                self.log.warning(
                    "Recreated PR %s#%s has merge conflicts; aborting wait.",
                    full_name,
                    new_number,
                )
                # Carry the replacement, as the closed branch does: the
                # caller reports ABANDONED against this PR, and the
                # original is already closed by dependabot, so without
                # it the operator would be pointed at the wrong PR.
                repo_owner, repo_name = full_name.split("/", 1)
                return RecreateResult(
                    RecreateOutcome.ABANDONED,
                    self._recreated_pr_stub(
                        repo_owner, repo_name, new_number, refreshed
                    ),
                )

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
        return None

    def _closed_recreated_pr(
        self,
        full_name: str,
        new_number: int,
        html_url: str,
        refreshed: dict[str, Any],
    ) -> RecreateResult:
        """Classify a replacement PR that has closed.

        Closed-and-merged is a success; closed-unmerged is a resolved
        failure.  Neither should keep polling, and neither should be
        merged again by the caller.
        """
        merged = _merged_from_payload(refreshed)
        pr_info = PullRequestInfo(
            number=new_number,
            node_id=refreshed.get("node_id"),
            title=refreshed.get("title", ""),
            body=refreshed.get("body"),
            author=((refreshed.get("user") or {}).get("login", "")),
            head_sha=((refreshed.get("head") or {}).get("sha", "")),
            base_branch=((refreshed.get("base") or {}).get("ref", "")),
            head_branch=((refreshed.get("head") or {}).get("ref", "")),
            state="closed",
            mergeable=refreshed.get("mergeable"),
            mergeable_state=refreshed.get("mergeable_state"),
            behind_by=None,
            files_changed=[],
            repository_full_name=full_name,
            html_url=html_url,
        )

        if merged:
            self._pr_status(
                f"✅ Recreated PR merged while waiting: {html_url}",
                level="info",
            )
            return RecreateResult(RecreateOutcome.MERGED, pr_info)

        # ``merged`` may be None for a payload carrying neither field.
        # Treat that as unmerged: the PR is closed either way, so there
        # is nothing left to wait for, and claiming a merge we cannot
        # evidence would be the wrong direction to guess in.
        self.log.warning(
            "Recreated PR %s#%s closed without merging; aborting wait.",
            full_name,
            new_number,
        )
        return RecreateResult(RecreateOutcome.ABANDONED, pr_info)

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
