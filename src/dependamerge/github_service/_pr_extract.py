# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""Field-level extraction from PR GraphQL nodes."""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import Any

from ..check_runs import failing_check_names
from ..models import (
    CopilotComment,
    FileChange,
    ReviewInfo,
)
from ._base import _GitHubServiceBase


class _PullRequestExtractMixin(_GitHubServiceBase):
    """PR-node field extraction for ``GitHubService``."""

    def _map_mergeable_enum(self, value: str | None) -> bool | None:
        # GraphQL mergeable: "MERGEABLE" | "CONFLICTING" | "UNKNOWN"
        self.log.debug(f"Mapping mergeable enum: '{value}'")
        if not value:
            self.log.debug("mergeable value is falsy (None, empty, etc.)")
            return None
        v = value.upper()
        if v == "MERGEABLE":
            self.log.debug("Mapped to True (mergeable)")
            return True
        if v == "CONFLICTING":
            self.log.debug("Mapped to False (conflicting)")
            return False
        if v == "UNKNOWN":
            # GitHub is still calculating - treat as potentially mergeable
            self.log.debug("Mapped UNKNOWN to None (still calculating)")
            return None
        self.log.warning(f"Unexpected mergeable value from GraphQL: {value}")
        return None

    def _safe_get_merge_state(self, merge_state_status: str | None) -> str | None:
        """Safely extract and normalize mergeStateStatus from GraphQL."""
        if not merge_state_status:
            # Log when we get null/missing mergeStateStatus for debugging
            self.log.debug("GraphQL mergeStateStatus is null or missing")
            return None

        normalized = merge_state_status.lower().strip()
        if not normalized:
            self.log.debug("GraphQL mergeStateStatus is empty string")
            return None

        # Valid states: clean, dirty, blocked, behind, draft, unstable, unknown
        valid_states = {
            "clean",
            "dirty",
            "blocked",
            "behind",
            "draft",
            "unstable",
            "unknown",
        }
        if normalized not in valid_states:
            self.log.warning(
                f"Unexpected mergeStateStatus from GraphQL: {merge_state_status}"
            )

        return normalized

    def _extract_file_changes(self, pr: dict[str, Any]) -> list[FileChange]:
        files = (pr.get("files") or {}).get("nodes", []) or []
        result: list[FileChange] = []
        for f in files:
            additions = int(f.get("additions") or 0)
            deletions = int(f.get("deletions") or 0)
            result.append(
                FileChange(
                    filename=f.get("path") or "",
                    additions=additions,
                    deletions=deletions,
                    changes=additions + deletions,
                    status="modified",  # GraphQL 'files' doesn't include a status; best-effort
                )
            )
        return result

    def _extract_reviews(self, pr: dict[str, Any]) -> list[ReviewInfo]:
        """Extract PR reviews from GraphQL node."""
        reviews = (pr.get("reviews") or {}).get("nodes", []) or []
        result: list[ReviewInfo] = []

        for review in reviews:
            author = (review.get("author") or {}).get("login") or "unknown"
            result.append(
                ReviewInfo(
                    # NOTE: GraphQL returns string node IDs (e.g., "PRR_kwDOGBtQpc4-u-zD")
                    # NOT numeric IDs. Do not convert to int() - it will cause runtime errors.
                    id=review.get("id", ""),
                    user=author,
                    state=review.get("state") or "",
                    submitted_at=review.get("createdAt") or "",
                    body=review.get("body"),
                )
            )
        return result

    def _extract_copilot_comments(self, pr: dict[str, Any]) -> list[CopilotComment]:
        comments = (pr.get("comments") or {}).get("nodes", []) or []
        result: list[CopilotComment] = []
        for c in comments:
            author = ((c.get("author") or {}).get("login") or "").lower()
            if author in ("github-copilot[bot]", "copilot"):
                result.append(
                    CopilotComment(
                        id=0,  # GraphQL doesn't provide numeric IDs in this selection; not critical for reporting
                        body=c.get("body") or "",
                        created_at=c.get("createdAt") or "",
                        state="open",
                    )
                )
        return result

    @staticmethod
    def _extract_failing_checks(pr: dict[str, Any]) -> list[str]:
        """
        Extract failing checks from the statusCheckRollup on the latest commit.

        A commit can carry several check runs sharing one name, typically
        when a duplicate workflow event causes ``concurrency`` to cancel a
        superseded run.  Only the latest run per name decides the outcome,
        matching GitHub's own merge protection; see ``check_runs``.
        """
        failing: list[str] = []

        commits = (pr.get("commits") or {}).get("nodes", []) or []
        if not commits:
            return failing

        commit = (commits[0] or {}).get("commit") or {}
        rollup = commit.get("statusCheckRollup") or {}
        contexts = (rollup.get("contexts") or {}).get("nodes", []) or []

        check_runs: list[dict[str, Any]] = []

        for ctx in contexts:
            typ = ctx.get("__typename")
            if typ == "CheckRun":
                # Collected for deduplication rather than judged here.
                check_runs.append(ctx)
            elif typ == "StatusContext":
                # Commit statuses are already latest-per-context: GitHub
                # collapses repeated posts to the same context itself.
                state = (ctx.get("state") or "").upper()
                if state in ("FAILURE", "ERROR"):
                    name = ctx.get("context") or ""
                    if name:
                        failing.append(name)

        return failing_check_names(check_runs) + failing
