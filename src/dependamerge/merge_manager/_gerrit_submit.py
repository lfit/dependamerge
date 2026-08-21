# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Submission of GitHub pull requests that mirror Gerrit changes.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import TYPE_CHECKING

import dependamerge.merge_manager as _pkg

from ..gerrit import (
    GerritAuthError,
    GerritRestError,
)
from ..github2gerrit_detector import (
    GitHub2GerritDetectionResult,
    GitHub2GerritMapping,
    build_gerrit_change_url_from_mapping,
    build_gerrit_submission_comment,
    detect_github2gerrit_comments,
)
from ..models import PullRequestInfo
from ..netrc import NetrcParseError
from ._base import _MergeManagerBase

if TYPE_CHECKING:
    from ..gerrit import GerritChangeInfo
    from ..netrc import GerritCredentials


class _GerritSubmitMixin(_MergeManagerBase):
    """Submission of GitHub pull requests that mirror Gerrit changes."""

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

        credentials = self._resolve_gerrit_submit_credentials(gerrit_host, mapping)
        if credentials is None:
            return False

        try:
            return await self._submit_with_gerrit_credentials(
                mapping, pr_info, gerrit_host, gerrit_base_path, credentials
            )
        except (GerritAuthError, GerritRestError) as exc:
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

    def _resolve_gerrit_submit_credentials(
        self,
        gerrit_host: str,
        mapping: GitHub2GerritMapping,
    ) -> GerritCredentials | None:
        """
        Resolve usable Gerrit credentials for a submission, or report why not.

        Kept apart from the submission itself because it is the one phase
        that must run *outside* the caller's Gerrit error handler: a
        malformed .netrc is a local configuration fault rather than a
        remote one, and is reported as its own warning.

        Args:
            gerrit_host: Gerrit host the credentials must cover.
            mapping: The parsed mapping, used only for the topic in logs.

        Returns:
            Valid credentials, or None if none could be resolved.
        """
        try:
            credentials = _pkg.resolve_gerrit_credentials(
                host=gerrit_host,
                use_netrc=not self.no_netrc,
                netrc_file=self.netrc_file,
            )
        except NetrcParseError as exc:
            self.log.warning("Error parsing .netrc for Gerrit: %s", exc)
            credentials = None

        if credentials is None or not credentials.is_valid:
            self.log.warning(
                "No Gerrit credentials found for %s. Cannot submit "
                "GitHub2Gerrit change (topic: %s).",
                gerrit_host,
                mapping.topic,
            )
            return None

        return credentials

    async def _submit_with_gerrit_credentials(
        self,
        mapping: GitHub2GerritMapping,
        pr_info: PullRequestInfo,
        gerrit_host: str,
        gerrit_base_path: str | None,
        credentials: GerritCredentials,
    ) -> bool:
        """
        Submit the mapped Gerrit change and act on the result.

        Holds every step that talks to Gerrit once the host and
        credentials are known, so the caller can wrap the whole exchange
        in a single error handler rather than repeating one per step.

        Args:
            mapping: The parsed GitHub2Gerrit mapping.
            pr_info: The GitHub pull request info.
            gerrit_host: Resolved Gerrit host.
            gerrit_base_path: Resolved Gerrit base path, if any.
            credentials: Validated Gerrit credentials.

        Returns:
            True if the Gerrit change was submitted (or, in preview mode,
            would have been).
        """
        gerrit_change = self._find_open_gerrit_change(
            mapping, gerrit_host, gerrit_base_path, credentials
        )
        if gerrit_change is None:
            return False

        submit_manager = _pkg.create_submit_manager(
            host=gerrit_host,
            base_path=gerrit_base_path,
            username=credentials.username,
            password=credentials.password,
        )

        results = submit_manager.submit_changes(
            [(gerrit_change, None)],
            review_labels={"Code-Review": 2},
            dry_run=self.preview_mode,
        )

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

    def _find_open_gerrit_change(
        self,
        mapping: GitHub2GerritMapping,
        gerrit_host: str,
        gerrit_base_path: str | None,
        credentials: GerritCredentials,
    ) -> GerritChangeInfo | None:
        """
        Look up the open Gerrit change the mapping's Change-Id names.

        Separate from the submission because the lookup needs its own
        Gerrit service, distinct from the submit manager, and because
        "no such open change" is an ordinary outcome rather than an error.

        Args:
            mapping: The parsed GitHub2Gerrit mapping.
            gerrit_host: Resolved Gerrit host.
            gerrit_base_path: Resolved Gerrit base path, if any.
            credentials: Validated Gerrit credentials.

        Returns:
            The first matching open change, or None if there is none.
        """
        service = _pkg.create_gerrit_service(
            host=gerrit_host,
            base_path=gerrit_base_path,
            username=credentials.username,
            password=credentials.password,
        )

        # Query Gerrit for the change using the primary Change-ID
        change_id = mapping.primary_change_id
        changes = service._query_changes(
            query=f"change:{change_id} status:open",
            limit=5,
            offset=0,
            options=[
                "CURRENT_REVISION",
                "LABELS",
                "DETAILED_LABELS",
                "SUBMIT_REQUIREMENTS",
            ],
        )

        if not changes:
            self.log.warning(
                "No open Gerrit change found for Change-Id %s on %s",
                change_id,
                gerrit_host,
            )
            return None

        # Use the first matching change
        gerrit_change = changes[0]
        self.log.info(
            "Found Gerrit change %s #%d for Change-Id %s",
            gerrit_change.project,
            gerrit_change.number,
            change_id,
        )
        return gerrit_change

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
