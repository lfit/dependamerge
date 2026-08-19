# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""Turning PR GraphQL nodes into ``UnmergeablePR`` / ``PullRequestInfo``."""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from typing import Any

from ..bot_identity import canonical_bot_login
from ..models import (
    PullRequestInfo,
    UnmergeablePR,
    UnmergeableReason,
)
from ._base import _GitHubServiceBase
from ._helpers import (
    _bool_or_none,
    _clone_url_with_git_suffix,
    _str_or_none,
)


class _PullRequestNodeMixin(_GitHubServiceBase):
    """PR-node analysis and conversion for ``GitHubService``."""

    async def _analyze_pr_node(
        self, repo_full_name: str, pr: dict[str, Any], include_drafts: bool = False
    ) -> UnmergeablePR | None:
        """
        Analyze a PR GraphQL node and produce UnmergeablePR if any blocking reasons
        are detected. Returns None if mergeable or if insufficient data.

        Args:
            repo_full_name: The full name of the repository (owner/repo).
            pr: The PR GraphQL node data.
            include_drafts: If True, include draft PRs in the results. If False (default),
                          return None for PRs that are only blocked due to draft status.

        This applies code-owners level bypass logic by default (matching merge command behavior).
        PRs that can be merged with standard permissions are not reported as blocked.
        """
        if self._progress:
            try:
                self._progress.analyze_pr(pr.get("number", 0), repo_full_name)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Progress display is best-effort; ignore UI errors.
                self.log.debug(f"Progress analyze_pr failed: {exc}", exc_info=True)

        reasons: list[UnmergeableReason] = []

        # Draft status
        if pr.get("isDraft") is True:
            reasons.append(
                UnmergeableReason(
                    type="draft",
                    description="Pull request is in draft state",
                )
            )

        # Mergeability
        mergeable = (
            pr.get("mergeable") or ""
        ).upper()  # MERGEABLE | CONFLICTING | UNKNOWN
        merge_state = (
            pr.get("mergeStateStatus") or ""
        ).lower()  # clean, behind, blocked, draft, dirty, unknown

        if mergeable == "CONFLICTING" or merge_state == "dirty":
            reasons.append(
                UnmergeableReason(
                    type="merge_conflict",
                    description="Pull request has merge conflicts",
                    details="Branch cannot be automatically merged due to conflicts",
                )
            )

        if merge_state == "behind":
            reasons.append(
                UnmergeableReason(
                    type="behind_base",
                    description="Pull request is behind the base branch",
                    details="Branch needs to be updated with latest changes",
                )
            )

        # Status check rollup
        failing_checks = self._extract_failing_checks(pr)
        if failing_checks:
            reasons.append(
                UnmergeableReason(
                    type="failing_checks",
                    description="Required status checks are failing",
                    details=f"Failing checks: {', '.join(sorted(set(failing_checks)))}",
                )
            )

        if not reasons:
            return None

        # Filter out PRs that are only blocked due to draft status if include_drafts is False
        if not include_drafts:
            # Check if draft is the only blocking reason
            if len(reasons) == 1 and reasons[0].type == "draft":
                return None
            # Remove draft reason from the list if there are other blocking reasons
            reasons = [r for r in reasons if r.type != "draft"]
            # If after filtering there are no reasons left, return None
            if not reasons:
                return None

        copilot_comments = self._extract_copilot_comments(pr)
        # File change extraction not required for UnmergeablePR summary here

        return UnmergeablePR(
            repository=repo_full_name,
            pr_number=int(pr.get("number", 0)),
            title=pr.get("title") or "",
            author=canonical_bot_login(
                (pr.get("author") or {}).get("login"),
                (pr.get("author") or {}).get("__typename"),
            ),
            url=pr.get("url") or "",
            reasons=reasons,
            copilot_comments_count=len(copilot_comments),
            copilot_comments=copilot_comments,
            created_at=pr.get("createdAt") or "",
            updated_at=pr.get("updatedAt") or "",
        )

    def to_pull_request_info(
        self, repo_full_name: str, pr: dict[str, Any]
    ) -> PullRequestInfo:
        """
        Convert a PR GraphQL node to PullRequestInfo (for merge workflows).
        """
        files = self._extract_file_changes(pr)
        reviews = self._extract_reviews(pr)

        # Debug logging to see actual GraphQL values
        mergeable_raw = pr.get("mergeable")
        merge_state_raw = pr.get("mergeStateStatus")
        self.log.debug(
            f"GraphQL raw values for PR {pr.get('number', 'unknown')}: "
            f"mergeable='{mergeable_raw}', mergeStateStatus='{merge_state_raw}'"
        )

        return PullRequestInfo(
            number=int(pr.get("number", 0)),
            node_id=pr.get("id"),  # GraphQL node ID for mutations
            title=pr.get("title") or "",
            body=(pr.get("body") or None),
            author=canonical_bot_login(
                (pr.get("author") or {}).get("login"),
                (pr.get("author") or {}).get("__typename"),
            ),
            head_sha=pr.get("headRefOid") or "",
            base_branch=pr.get("baseRefName") or "",
            head_branch=pr.get("headRefName") or "",
            state="open",  # GraphQL query filters for OPEN PRs only, so all results are open
            mergeable=self._map_mergeable_enum(pr.get("mergeable")),
            mergeable_state=self._safe_get_merge_state(pr.get("mergeStateStatus")),
            behind_by=None,  # Not included in GraphQL; could be computed if needed
            files_changed=files,
            repository_full_name=repo_full_name,
            html_url=pr.get("url") or "",
            reviews=reviews,
            # Populate head/base repo identity from the GraphQL
            # ``headRepository`` / ``baseRepository`` fields so the
            # signature-preserving local-rebase path can tell
            # whether the PR is from a fork (and which remote to
            # push to).  Without these, ``rebase.local_rebase_pr()``
            # fails closed to avoid pushing to the wrong repository.
            # GraphQL returns the HTTPS URL via ``url`` (without the
            # ``.git`` suffix), so we synthesise the canonical
            # ``clone_url`` form for parity with REST.
            head_repo_full_name=_str_or_none(
                (pr.get("headRepository") or {}).get("nameWithOwner")
            ),
            head_repo_clone_url=_clone_url_with_git_suffix(
                (pr.get("headRepository") or {}).get("url")
            ),
            base_repo_full_name=_str_or_none(
                (pr.get("baseRepository") or {}).get("nameWithOwner")
            ),
            base_repo_clone_url=_clone_url_with_git_suffix(
                (pr.get("baseRepository") or {}).get("url")
            ),
            is_fork=_bool_or_none((pr.get("headRepository") or {}).get("isFork")),
        )
