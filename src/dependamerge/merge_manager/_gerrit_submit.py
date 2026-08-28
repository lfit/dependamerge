# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Submission of GitHub PRs that are really Gerrit changes.

Repositories replicated by github2gerrit carry the real review on a
Gerrit server; the GitHub PR is a mirror.  These methods detect that
arrangement, submit the change on Gerrit and close the mirror PR.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from ..gerrit import GerritAuthError, GerritRestError, GerritServiceError
from ..github2gerrit_detector import (
    GitHub2GerritDetectionResult,
    GitHub2GerritMapping,
    build_gerrit_change_url_from_mapping,
    build_gerrit_submission_comment,
    detect_github2gerrit_comments,
    fetch_gitreview_from_github,
)
from ..models import PullRequestInfo
from ._gerrit_submit_steps import _GerritSubmitStepsMixin

# Annotation-only, so nothing is bound at run time that could shadow a
# substitution made on the package (see tests/test_patch_targets.py).
if TYPE_CHECKING:
    from ..netrc import GerritCredentials


class _GerritSubmitMixin(_GerritSubmitStepsMixin):
    """Detecting and submitting github2gerrit-backed changes."""

    async def _detect_github2gerrit(
        self,
        repo_owner: str,
        repo_name: str,
        pr_number: int,
    ) -> GitHub2GerritDetectionResult:
        """
        Fetch issue comments for a PR and check for GitHub2Gerrit mapping.

        Args:
            repo_owner: Repository owner.
            repo_name: Repository name.
            pr_number: Pull request number.

        Returns:
            Detection result with mapping data if found.
        """
        try:
            if self._github_client is None:
                raise RuntimeError("GitHub client not initialized")

            # Fetch issue comments (not review comments) via REST API
            comments = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/issues/{pr_number}/comments"
            )

            if not isinstance(comments, list):
                return GitHub2GerritDetectionResult()

            return detect_github2gerrit_comments(comments)

        except Exception as exc:
            self.log.debug(
                "Failed to check GitHub2Gerrit comments for %s/%s#%d: %s",
                repo_owner,
                repo_name,
                pr_number,
                exc,
            )
            return GitHub2GerritDetectionResult()

    async def _submit_gerrit_change(
        self,
        mapping: GitHub2GerritMapping,
        pr_info: PullRequestInfo,
        repo_owner: str,
        repo_name: str,
    ) -> bool:
        """
        Submit the corresponding Gerrit change for a GitHub2Gerrit PR.

        Resolves Gerrit credentials, looks up the change by Change-ID,
        applies +2 Code-Review, and submits it.

        Args:
            mapping: The parsed GitHub2Gerrit mapping.
            pr_info: The GitHub pull request info.
            repo_owner: Repository owner (org or user).
            repo_name: Repository name.

        Returns:
            True if the Gerrit change was successfully submitted.
        """
        # We need to figure out the Gerrit host.  The mapping's topic name
        # follows the pattern "GH-<repo>-<number>" which doesn't embed the
        # host.  We look for a Gerrit change URL in the mapping comment body,
        # or fall back to well-known hosts.
        gerrit_host, gerrit_base_path = await self._resolve_gerrit_host(
            mapping, repo_owner, repo_name
        )

        if not gerrit_host:
            self.log.warning(
                "Cannot determine Gerrit host for GitHub2Gerrit mapping "
                "(topic: %s). Skipping Gerrit submission.",
                mapping.topic,
            )
            return False

        gerrit_target: tuple[str, str | None] = (gerrit_host, gerrit_base_path)

        credentials = self._resolve_gerrit_submit_credentials(gerrit_host, mapping)
        if credentials is None:
            return False

        try:
            return await self._submit_resolved_gerrit_change(
                mapping, pr_info, gerrit_target, credentials
            )

        except (GerritAuthError, GerritRestError, GerritServiceError) as exc:
            self.log.warning(
                "Gerrit error submitting change for topic %s: %s",
                mapping.topic,
                exc,
            )
            return False
        except Exception as exc:
            self.log.warning(
                "Unexpected error submitting Gerrit change for topic %s: %s",
                mapping.topic,
                exc,
            )
            return False

    async def _submit_resolved_gerrit_change(
        self,
        mapping: GitHub2GerritMapping,
        pr_info: PullRequestInfo,
        gerrit_target: tuple[str, str | None],
        credentials: GerritCredentials,
    ) -> bool:
        """
        Find, submit and report on the Gerrit change behind a PR.

        Runs once credentials and the Gerrit host are known; Gerrit
        transport errors are left to the caller to translate.
        """
        gerrit_host, gerrit_base_path = gerrit_target

        gerrit_change = self._find_gerrit_change(
            gerrit_target, credentials, mapping.primary_change_id
        )
        if gerrit_change is None:
            return False

        results = self._run_gerrit_submit(gerrit_target, credentials, gerrit_change)

        if results and results[0].submitted:
            self.log.info(
                "Successfully submitted Gerrit change %s #%d",
                gerrit_change.project,
                gerrit_change.number,
            )

            # Post a comment on the GitHub PR and close it
            gerrit_url = build_gerrit_change_url_from_mapping(
                mapping, gerrit_host, gerrit_base_path
            )
            await self._close_github_pr_after_gerrit_submit(
                pr_info, mapping, gerrit_url
            )

            return True

        if results and results[0].success and self.preview_mode:
            # Dry-run succeeded
            return True

        error_msg = results[0].error if results else "Unknown error"
        self.log.warning(
            "Failed to submit Gerrit change %s #%d: %s",
            gerrit_change.project,
            gerrit_change.number,
            error_msg,
        )
        return False

    async def _close_github_pr_after_gerrit_submit(
        self,
        pr_info: PullRequestInfo,
        mapping: GitHub2GerritMapping,
        gerrit_url: str,
    ) -> None:
        """
        Close the GitHub PR and post a comment after Gerrit submission.

        Args:
            pr_info: The GitHub pull request.
            mapping: The parsed mapping.
            gerrit_url: URL of the submitted Gerrit change.
        """
        if self.preview_mode:
            return

        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)

        try:
            if self._github_client is None:
                raise RuntimeError("GitHub client not initialized")

            # Post comment following GitHub2Gerrit conventions
            comment_body = build_gerrit_submission_comment(mapping, gerrit_url)
            await self._github_client.post_issue_comment(
                repo_owner, repo_name, pr_info.number, comment_body
            )

            # Close the PR
            await self._github_client.close_pull_request(
                repo_owner, repo_name, pr_info.number
            )

            self.log.info(
                "Closed GitHub PR %s#%d after Gerrit submission",
                pr_info.repository_full_name,
                pr_info.number,
            )
        except Exception as exc:
            self.log.warning(
                "Failed to close GitHub PR %s#%d after Gerrit submission: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )

    async def _resolve_gerrit_host(
        self,
        mapping: GitHub2GerritMapping,
        repo_owner: str,
        repo_name: str,
    ) -> tuple[str | None, str | None]:
        """
        Determine the Gerrit host and base path for a GitHub2Gerrit PR.

        Resolution priority (highest first):

        1. ``.gitreview`` file in the repository (canonical source of truth)
        2. ``GERRIT_HOST`` / ``GERRIT_BASE_PATH`` environment variables
        3. Gerrit URL embedded in the mapping comment body
        4. Well-known host conventions (e.g. ``lfit`` → LF Gerrit)
        5. ``GERRIT_URL`` environment variable

        The ``.gitreview`` file is treated as definitive because every
        repository that uses GitHub2Gerrit is required to have one, and it
        records the exact Gerrit host, port, and project path.

        Args:
            mapping: The parsed GitHub2Gerrit mapping from the PR comment.
            repo_owner: Repository owner (org or user).
            repo_name: Repository name.

        Returns:
            Tuple of (host, base_path) or (None, None) if unresolvable.
        """
        # 1. .gitreview file — highest priority / source of truth
        if self._github_client is not None:
            gitreview_info = await fetch_gitreview_from_github(
                self._github_client, repo_owner, repo_name
            )
            if gitreview_info and gitreview_info.is_valid:
                self.log.info(
                    "Resolved Gerrit host from .gitreview in %s/%s: %s (base_path=%s)",
                    repo_owner,
                    repo_name,
                    gitreview_info.host,
                    gitreview_info.base_path,
                )
                return gitreview_info.host, gitreview_info.base_path

        # 2. Explicit environment variables
        env_host = os.getenv("GERRIT_HOST", "").strip()
        env_base_path = os.getenv("GERRIT_BASE_PATH", "").strip() or None
        if env_host:
            return env_host, env_base_path

        # 3. Gerrit URL embedded in the mapping comment body
        if mapping.raw_comment_body:
            gerrit_url_match = re.search(
                r"https?://([^/\s]+)(?:/([\w-]+))?/c/",
                mapping.raw_comment_body,
            )
            if gerrit_url_match:
                host = gerrit_url_match.group(1)
                base_path = (
                    gerrit_url_match.group(2) if gerrit_url_match.group(2) else None
                )
                return host, base_path

        # 4. Well-known LF Gerrit host
        if (
            mapping.pr_url and "github.com/lfit/" in mapping.pr_url
        ) or repo_owner == "lfit":
            return "gerrit.linuxfoundation.org", "infra"

        # 5. GERRIT_URL environment variable (catch-all)
        gerrit_url = os.getenv("GERRIT_URL", "").strip()
        if gerrit_url:
            url_match = re.match(r"https?://([^/]+)(?:/([\w-]+))?/?$", gerrit_url)
            if url_match:
                return url_match.group(1), url_match.group(2) if url_match.group(
                    2
                ) else None

        return None, None
