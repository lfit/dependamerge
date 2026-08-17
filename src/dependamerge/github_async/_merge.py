# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Merge and auto-merge operations for the async GitHub client.

The merge call, the post-merge verification that distinguishes a real
success from a silently rejected merge, the GitHub error-detail
extraction those messages rely on, and the GraphQL auto-merge enable.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from ._base import _GitHubAsyncBase
from ._errors import PermissionError


class _MergeMixin(_GitHubAsyncBase):
    """Merge and auto-merge operations for ``GitHubAsync``."""

    async def merge_pull_request(
        self, owner: str, repo: str, number: int, merge_method: str = "merge"
    ) -> bool:
        """
        Merge a pull request.

        REST: PUT /repos/{owner}/{repo}/pulls/{pull_number}/merge

        Raises:
            PermissionError: If token lacks required permissions
        """
        # A merge attempt changes the PR's state whether it succeeds or
        # is refused, so any memoised block reason describes the past.
        self.invalidate_block_reason(owner, repo, number)
        try:
            self.log.debug(
                f"Attempting to merge PR #{number} in {owner}/{repo} with method={merge_method}"
            )
            data = await self.put(
                f"/repos/{owner}/{repo}/pulls/{number}/merge",
                json={"merge_method": merge_method},
            )
            # The API returns {"merged": true/false, ...}
            merged = bool(data.get("merged", False))
            if merged:
                self.log.debug(f"Successfully merged PR #{number} in {owner}/{repo}")
            else:
                self.log.warning(
                    f"GitHub API returned merged=false for PR #{number} in {owner}/{repo}: {data}"
                )
            return merged
        except Exception as e:
            # Check for permission errors first (includes workflow scope check)
            perm_error = self._parse_permission_error(e, "merge", owner, repo)
            if perm_error:
                # GitHub returns the "refusing to allow ... workflow" 403
                # only when the *classic* token lacks the ``workflow``
                # scope.  Before repeating that guidance, confirm the scope
                # really is absent: if the token already carries it (or is a
                # fine-grained/app token we cannot introspect and which
                # therefore would not produce this classic-PAT message), the
                # true cause is something else — typically a repository
                # ruleset that restricts workflow-file updates, or an
                # un-authorized SSO session.  Telling the user to add a scope
                # they already hold would be an inaccurate diagnosis.
                if perm_error.operation == "merge_workflow":
                    has_workflow = await self.check_workflow_scope()
                    if has_workflow is True:
                        perm_error = PermissionError(
                            operation="merge_workflow_restricted",
                            message=(
                                f"GitHub refused to merge PR in {owner}/{repo} "
                                "even though the token already has the "
                                "'workflow' scope. The workflow-file update is "
                                "being blocked by something other than token "
                                "scope"
                            ),
                            token_type_guidance={
                                "classic": (
                                    "Check for a repository ruleset that "
                                    "restricts updates to .github/workflows/** "
                                    "and confirm the token is SSO-authorized "
                                    "for this organization"
                                ),
                                "fine_grained": (
                                    "Check for a repository ruleset that "
                                    "restricts updates to .github/workflows/**"
                                ),
                                "fix": (
                                    "Review the repository's rulesets and "
                                    "organization SSO authorization for this "
                                    "token"
                                ),
                            },
                        )
                self.log.debug(
                    f"Permission error merging PR #{number} in {owner}/{repo}: {perm_error}"
                )
                raise perm_error from e

            error_type = type(e).__name__
            error_msg = str(e)
            self.log.debug(
                f"Merge API error for PR #{number} in {owner}/{repo}: {error_type}: {error_msg}"
            )

            github_detail = self._extract_github_error_detail(e)
            if github_detail:
                self.log.debug(
                    f"GitHub merge API response body for #{number}: {github_detail}"
                )

            # Re-check PR state: the merge may have actually succeeded
            # despite the exception (a race where the API call lands
            # but we still see an error from rate-limiting, network, or
            # JSON parsing), and the state adds context to the error we
            # raise.
            return await self._validate_merge_result(
                owner, repo, number, e, github_detail
            )

    @staticmethod
    def _extract_github_error_detail(error: Exception) -> str:
        """Extract GitHub's response-body message from a failed request.

        GitHub puts the *actual* reason here — ruleset violations,
        "Required workflows ... are not satisfied", required-check names,
        etc.  The ``HTTPStatusError`` text only carries the status line
        (e.g. "405 Method Not Allowed"), so without this the real cause is
        silently lost.  Whitespace/newlines are collapsed so the reason
        fits on a single status line.

        Returns an empty string when no detail could be extracted.
        """
        response = getattr(error, "response", None)
        if response is None:
            return ""
        try:
            body = response.json()
            if isinstance(body, dict) and isinstance(body.get("message"), str):
                return " ".join(body["message"].split())
        except Exception:
            # Response body was not JSON (or .json() failed); fall through
            # to the raw-text extraction below rather than failing here.
            pass
        try:
            raw = getattr(response, "text", "") or ""
            return " ".join(raw.split())[:500]
        except Exception:
            return ""

    async def _validate_merge_result(
        self,
        owner: str,
        repo: str,
        number: int,
        error: Exception,
        github_detail: str,
    ) -> bool:
        """Re-check PR state after a merge attempt raised an exception.

        The merge may have actually succeeded despite the exception (a race
        where the API call lands but we still see an error from
        rate-limiting, network, or JSON parsing).  When the PR is confirmed
        merged, return ``True``.  Otherwise raise an enhanced exception that
        preserves the original error text (its HTTP status line is
        string-matched by ``_merge_pr_with_retry`` to classify retryable vs
        terminal failures) and adds GitHub's actionable response body plus
        PR-state context.
        """
        try:
            pr_data_response = await self.get(f"/repos/{owner}/{repo}/pulls/{number}")
            # PR data should always be a dict, not a list
            pr_data = pr_data_response if isinstance(pr_data_response, dict) else {}

            mergeable = pr_data.get("mergeable")
            mergeable_state = pr_data.get("mergeable_state")
            state = pr_data.get("state")
            merged = pr_data.get("merged", False)
            draft = pr_data.get("draft", False)

            # Check if the merge actually succeeded despite the exception.
            # This handles race conditions where the API succeeds but we get
            # an exception due to rate limiting, network issues, JSON
            # parsing, etc.
            if state == "closed" and merged:
                self.log.info(
                    f"PR #{number} in {owner}/{repo} was successfully merged despite exception: {error}"
                )
                return True

            # Enhanced error message.  Always keep the original error text —
            # it carries the HTTP status line (e.g. "405 Method Not
            # Allowed") that ``_merge_pr_with_retry`` string-matches to
            # classify retryable vs terminal failures; dropping it made
            # every blocked/ruleset 405 fall through to the generic retry
            # path (3 attempts + sleeps).  Then *add* GitHub's response body
            # (the actionable reason) when we captured it.
            error_msg = (
                f"Failed to merge PR #{number} in {owner}/{repo}. Error: {str(error)}."
            )
            if github_detail:
                error_msg += f" GitHub: {github_detail}"
            error_msg += (
                f" (PR state: {state}, mergeable: {mergeable}, "
                f"mergeable_state: {mergeable_state})"
            )

            # Note common state-based causes for 405-style errors.
            if mergeable_state == "blocked":
                error_msg += " [blocked by branch protection / required checks]"
            elif mergeable_state == "behind":
                error_msg += " [PR branch is behind base branch]"
            elif mergeable_state == "dirty":
                error_msg += " [PR has merge conflicts]"
            elif draft:
                error_msg += " [cannot merge draft PR]"
            elif state == "closed" and not merged:
                error_msg += " [PR was closed without merging]"
            elif state != "open":
                error_msg += f" [PR is not open, state: {state}]"

            raise Exception(error_msg) from error
        except Exception as inner_e:
            # The enhanced-error path raised successfully (the message
            # starts with "Failed to merge PR") — propagate it unchanged.
            # A bare ``raise`` preserves ``inner_e`` together with its
            # existing ``__cause__`` (set to ``error`` above) and original
            # traceback, whereas ``raise inner_e from error`` would rewrite
            # the chaining.
            if "Failed to merge PR" in str(inner_e):
                raise
            # Otherwise the PR-state re-fetch itself failed.  Still surface
            # GitHub's response body (the actionable reason) when we
            # captured it, rather than dropping back to the bare
            # status-line ``HTTPStatusError``.
            if github_detail:
                raise Exception(
                    f"Failed to merge PR #{number} in {owner}/{repo}. "
                    f"Error: {str(error)}. GitHub: {github_detail}"
                ) from error
            raise error from inner_e

    async def enable_auto_merge(
        self, pull_request_node_id: str, merge_method: str = "MERGE"
    ) -> bool:
        """
        Enable auto-merge on a pull request via GraphQL.

        Auto-merge will automatically merge the PR once all required
        branch protection rules are satisfied.

        Args:
            pull_request_node_id: The GraphQL node ID of the pull request.
            merge_method: Merge method - "MERGE", "SQUASH", or "REBASE".
                Lowercase values ("merge", "squash", "rebase") are
                automatically uppercased.

        Returns:
            True if auto-merge was successfully enabled, False otherwise.
        """
        from ..github_graphql import ENABLE_AUTO_MERGE

        # Normalise to the GraphQL enum (uppercase)
        graphql_method = merge_method.upper()
        if graphql_method not in ("MERGE", "SQUASH", "REBASE"):
            self.log.warning(
                "Invalid merge method for auto-merge: %s; defaulting to MERGE",
                merge_method,
            )
            graphql_method = "MERGE"

        try:
            result = await self.graphql(
                ENABLE_AUTO_MERGE,
                {
                    "pullRequestId": pull_request_node_id,
                    "mergeMethod": graphql_method,
                },
            )
            auto_merge_data = (
                result.get("enablePullRequestAutoMerge", {})
                .get("pullRequest", {})
                .get("autoMergeRequest")
            )
            if auto_merge_data:
                self.log.debug(
                    "Auto-merge enabled for PR %s (method=%s, enabledAt=%s)",
                    pull_request_node_id,
                    auto_merge_data.get("mergeMethod"),
                    auto_merge_data.get("enabledAt"),
                )
                return True
            self.log.debug(
                "Auto-merge response missing autoMergeRequest for PR %s",
                pull_request_node_id,
            )
            return False
        except Exception as e:
            error_msg = str(e)
            # Common reasons auto-merge can't be enabled:
            # - Repository doesn't have auto-merge enabled in settings
            # - PR has conflicts
            # - Required status checks not configured
            self.log.debug(
                "Could not enable auto-merge for PR %s: %s",
                pull_request_node_id,
                error_msg,
            )
            return False
