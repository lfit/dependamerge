# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Enabling GitHub's own auto-merge, and the comment retry it needs.

Auto-merge lets GitHub complete the merge once required checks pass,
which is cheaper than holding a worker open through a long CI run.
"""

from __future__ import annotations

import asyncio

from ..github_async import PermissionError as GitHubPermissionError
from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _AutoMergeEnableMixin(_MergeManagerBase):
    """Turning on GitHub auto-merge for a pull request."""

    async def _post_pr_comment_with_retry(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        html_url: str,
        body: str,
    ) -> bool:
        """Post a PR comment with one retry after a 5s pause.

        Used for the auto-merge audit-trail comment so the PR
        conversation reflects that dependamerge enabled auto-merge.
        Approval comments take a different path: the approval body
        is passed directly to ``approve_pull_request()``, which
        creates a review (not an issue comment), so this helper is
        not used there. If both attempts fail, emit a single
        user-visible warning to the console rather than silently
        failing.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.
            html_url: Full PR URL, used for the warning message.
            body: Markdown body of the comment.

        Returns:
            True if the comment posted successfully (first or
            second attempt), False otherwise.
        """
        if not self._github_client:
            return False

        for attempt in (1, 2):
            try:
                await self._github_client.post_issue_comment(
                    owner, repo, pr_number, body
                )
                return True
            except GitHubPermissionError as exc:
                # Permission errors (typically HTTP 403) are not
                # transient — the token lacks the required scope or
                # the repo's branch protection forbids comments.
                # Skip the retry to avoid a pointless 5s delay per
                # PR and surface the failure right away.
                self.log.debug(
                    "Audit comment post denied (permission) for %s: %s",
                    html_url,
                    exc,
                )
                break
            except Exception as exc:
                # Heuristic: treat 4xx (other than 408/429) as
                # permanent and skip the retry. 5xx, 429 (rate
                # limit), 408 (timeout), and network/transport
                # errors get one retry after a short pause.
                #
                # We check several attribute paths so the
                # heuristic works across the various exception
                # shapes we may see:
                #   * ``exc.status_code`` — some custom wrappers
                #   * ``exc.status`` — ``aiohttp.ClientResponseError``
                #   * ``exc.response.status_code`` — ``httpx`` raises
                #     ``HTTPStatusError`` whose status lives on the
                #     attached ``Response`` object (this is what
                #     ``httpx.Response.raise_for_status()`` produces).
                response = getattr(exc, "response", None)
                status_code = (
                    getattr(exc, "status_code", None)
                    or getattr(exc, "status", None)
                    or getattr(response, "status_code", None)
                    or getattr(response, "status", None)
                )
                permanent = (
                    isinstance(status_code, int)
                    and 400 <= status_code < 500
                    and status_code not in (408, 429)
                )
                self.log.debug(
                    "Audit comment post attempt %d failed for %s: %s"
                    " (status=%r, permanent=%s)",
                    attempt,
                    html_url,
                    exc,
                    status_code,
                    permanent,
                )
                if permanent:
                    break
                if attempt == 1:
                    await asyncio.sleep(5.0)

        # Both attempts failed — surface a single line so the
        # user knows the PR-side audit trail is incomplete.
        self._pr_status(
            f"⚠️ Unable to add pull request comment: {html_url}",
            level="warning",
        )
        return False

    async def _enable_auto_merge_for_pr(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Enable auto-merge on a PR so it merges when checks pass.

        Idempotent and safe to call when auto-merge may already be
        active. Outcomes:

        - GraphQL ``enablePullRequestAutoMerge`` mutation succeeds:
          add the PR to ``_auto_merge_enabled``, post the audit
          comment, and return ``True``.
        - GraphQL mutation reports failure (commonly because
          auto-merge is *already* active on the PR — the response
          omits ``autoMergeRequest`` or the request raises): fall
          back to a REST GET on the PR and inspect ``auto_merge``.
          If non-null, treat as already-enabled — add the PR to
          ``_auto_merge_enabled`` (so the Step 6 skip gate still
          routes it to ``AUTO_MERGE_PENDING`` rather than
          attempting a 405-prone manual merge) and return ``True``,
          but skip the audit comment so re-runs against an
          already-configured PR don't post duplicates.
        - GraphQL mutation reports failure AND the PR has no
          ``auto_merge`` set: auto-merge is genuinely unavailable
          (e.g. the repository setting is off, the PR has
          conflicts, no required-checks are configured). Return
          ``False`` and let the caller fall through to manual
          polling/merge.
        - The PR has no ``node_id`` or there is no GitHub client:
          return ``False`` without making any API calls.

        Args:
            pr_info: Pull request information (must have ``node_id``).
            owner: Repository owner.
            repo: Repository name.

        Returns:
            True if auto-merge is active on the PR after this call
            (whether enabled by this call or already-active before
            it). False if auto-merge is unavailable.
        """
        if not self._github_client:
            return False

        if not pr_info.node_id:
            self.log.debug(
                "Cannot enable auto-merge for %s/%s#%s: missing node_id",
                owner,
                repo,
                pr_info.number,
            )
            return False

        pr_key = f"{owner}/{repo}#{pr_info.number}"

        # Already enabled in this run — skip duplicate API call
        if pr_key in self._auto_merge_enabled:
            return True

        cache_key = f"{owner}/{repo}"
        merge_method = self._pr_merge_methods.get(cache_key, self.default_merge_method)

        enabled = await self._github_client.enable_auto_merge(
            pr_info.node_id, merge_method
        )
        if not enabled:
            # The GraphQL mutation reports failure when auto-merge
            # is already active on the PR (the response omits
            # ``autoMergeRequest`` or the request raises). Check
            # the PR's current auto-merge state via REST so the
            # Step 6 skip gate still routes the PR to
            # ``AUTO_MERGE_PENDING`` instead of falling through to
            # a manual merge attempt that would 405 on pending
            # required checks.
            try:
                pr_payload = await self._github_client.get(
                    f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
                )
            except Exception as exc:
                self.log.debug(
                    "Could not refresh PR %s to check existing auto-merge state: %s",
                    pr_key,
                    exc,
                )
                pr_payload = None

            if (
                isinstance(pr_payload, dict)
                and pr_payload.get("auto_merge") is not None
            ):
                self._auto_merge_enabled.add(pr_key)
                self.log.debug(
                    "Auto-merge already active for %s; treating "
                    "as enabled (no audit comment posted)",
                    pr_key,
                )
                # Skip the audit comment in this branch —
                # someone (a previous run, the author, or the
                # repo's auto-merge bot) already enabled it; we
                # don't want to post a duplicate comment every
                # time dependamerge runs.
                return True
            return False

        self._auto_merge_enabled.add(pr_key)
        self.log.debug(
            "Auto-merge enabled for %s (method=%s)",
            pr_key,
            merge_method,
        )
        # Post a visible audit-trail comment so reviewers can
        # see at a glance that dependamerge enabled auto-merge
        # on the PR.
        audit_comment = (
            "🤖 Dependamerge\nEnabled auto-merge due to pending updates/checks ⏳"
        )
        await self._post_pr_comment_with_retry(
            owner, repo, pr_info.number, pr_info.html_url, audit_comment
        )
        return True

    async def _enable_auto_merge_with_approval(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Approve the current head (if needed) then enable auto-merge.

        Enabling auto-merge is a commitment to let GitHub complete the
        merge as soon as branch protection is satisfied, so the current
        head must already carry our approval — otherwise auto-merge would
        wait forever on a missing review.  This is the *de-facto*
        approve-on-demand trigger for the auto-merge path: when we enable
        auto-merge on a PR whose current version we have not approved, we
        approve it as part of arming auto-merge.

        Approval failures other than typed permission errors are logged
        and swallowed so a transient approval hiccup does not prevent us
        from at least arming auto-merge; the permission error is
        propagated so the caller's dedicated handler can report it.

        Used at the Step 5.5 auto-merge enable point and as the rebase
        module's auto-merge callback (which fires *after* the rebase, so
        we approve the rebased head rather than a soon-to-be-dismissed
        pre-rebase commit).
        """
        if not self.preview_mode:
            try:
                # No propagation delay here: we are arming auto-merge,
                # not dispatching an immediate merge.  GitHub re-checks
                # branch protection when the required checks complete
                # (≫ the propagation window), so sleeping now would
                # only stall the pipeline.
                await self._ensure_pr_approved(
                    pr_info, owner, repo, propagation_delay=False
                )
            except GitHubPermissionError:
                # Surface token permission problems to the caller's
                # dedicated handler rather than masking them here.
                raise
            except Exception as exc:
                self.log.warning(
                    "Could not approve %s/%s#%s before enabling auto-merge: %s",
                    owner,
                    repo,
                    pr_info.number,
                    exc,
                )
        return await self._enable_auto_merge_for_pr(pr_info, owner, repo)
