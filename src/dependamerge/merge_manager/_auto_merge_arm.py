# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Arming GitHub's native auto-merge on a pull request.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio

from ..github_async import PermissionError as GitHubPermissionError
from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _AutoMergeArmMixin(_MergeManagerBase):
    """Arming GitHub's native auto-merge on a pull request."""

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

    async def _ensure_pr_approved(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        *,
        propagation_delay: bool = True,
    ) -> bool:
        """Approve the current PR head on demand and track the approval.

        Thin wrapper around :meth:`_approve_pr` that also records the PR
        in ``_recently_approved`` and applies the post-approval
        propagation delay, exactly as the (now removed) up-front Step 3
        approval used to.  :meth:`_approve_pr` is idempotent — it no-ops
        when the current user already has an active ``APPROVED`` review on
        the current head — so this is safe to call unconditionally at any
        approve-on-demand trigger.

        ``propagation_delay=False`` skips the post-approval sleep.  The
        delay exists to let GitHub propagate the approval into branch-
        protection evaluation before an *immediate* merge dispatch; when
        the caller is arming auto-merge instead (GitHub re-evaluates
        protection when checks complete, typically minutes later) the
        sleep is pure dead time on the critical path.

        Returns:
            True if a *new* approval was submitted, False if the PR was
            already approved (or approval was declined).
        """
        approved = await self._approve_pr(owner, repo, pr_info.number)
        if approved:
            pr_key = f"{owner}/{repo}#{pr_info.number}"
            self._recently_approved.add(pr_key)
            # Give GitHub time to propagate the approval to the branch
            # protection evaluation before a merge is attempted.
            if propagation_delay and self._post_approval_delay > 0:
                self.log.debug(
                    f"Waiting {self._post_approval_delay}s for approval "
                    f"propagation on {pr_key}"
                )
                await asyncio.sleep(self._post_approval_delay)
        return approved

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
