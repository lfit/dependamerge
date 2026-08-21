# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Organisation-level settings and the approval they may mandate.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from typing import Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _OrgSettingsMixin(_MergeManagerBase):
    """Organisation-level settings and the approval they may mandate."""

    async def _get_org_settings(self, owner: str) -> dict[str, Any] | None:
        """
        Get organization-level settings, with caching.

        Organization settings (e.g. web_commit_signoff_required) don't change
        between PRs in the same org, so we cache the result for the lifetime
        of the merge session.

        Args:
            owner: Organization/owner name

        Returns:
            Organization settings dict, or None if the lookup failed
        """
        # Fast path: no lock needed if already cached
        if owner in self._org_settings_cache:
            return self._org_settings_cache[owner]

        # Acquire a per-owner lock so concurrent lookups for the same
        # org are serialised, but lookups for *different* orgs proceed
        # in parallel without blocking each other.
        async with self._org_settings_locks_lock:
            if owner not in self._org_settings_locks:
                self._org_settings_locks[owner] = asyncio.Lock()
            owner_lock = self._org_settings_locks[owner]

        async with owner_lock:
            # Re-check after acquiring the per-owner lock (another
            # task may have populated the cache while we waited).
            if owner in self._org_settings_cache:
                return self._org_settings_cache[owner]

            if not self._github_client:
                return None

            try:
                org_data = await self._github_client.get(f"/orgs/{owner}")
                if isinstance(org_data, dict):
                    self._org_settings_cache[owner] = org_data
                    web_commit_signoff = org_data.get(
                        "web_commit_signoff_required", False
                    )
                    if web_commit_signoff:
                        self.log.debug(f"Organization {owner} requires commit signoff")
                    return org_data
                else:
                    self._org_settings_cache[owner] = None
                    return None
            except Exception as e:
                self.log.debug(
                    f"Could not check organization settings for {owner}: {e}"
                )
                self._org_settings_cache[owner] = None
                return None

    @staticmethod
    def _rules_require_approval(rules: Any) -> bool:
        """Return True if any effective branch rule mandates an approval.

        ``rules`` is the JSON body returned by
        ``GET /repos/{owner}/{repo}/rules/branches/{branch}`` — a flat
        list of the rules that *actually apply* to the branch, with all
        ruleset conditions (repository include/exclude, ref matching)
        already evaluated server-side and org- and repo-level rulesets
        already merged.  We treat a branch as requiring an approval when a
        ``pull_request`` rule asks for at least one approving review.

        This is intentionally org-agnostic: it keys off the rule *type*
        and its ``required_approving_review_count`` parameter, never the
        ruleset's name, so it works for any organization's naming.
        """
        if not isinstance(rules, list):
            return False
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if rule.get("type") != "pull_request":
                continue
            params = rule.get("parameters")
            if not isinstance(params, dict):
                # A pull_request rule with no readable parameters still
                # signals that reviews are governed here; treat the
                # presence of the rule as requiring approval rather than
                # risk a doomed merge-first attempt.
                return True
            count = params.get("required_approving_review_count")
            if isinstance(count, int) and count >= 1:
                return True
        return False

    async def _approve_if_review_mandated(
        self, pr_info: PullRequestInfo, owner: str, repo: str, pr_key: str
    ) -> None:
        """Approve up-front when the base branch mandates a review to merge.

        No-ops in preview mode, when the PR was already approved this run,
        or when the base branch carries no required-approval rule.  This
        is the proactive counterpart to
        :meth:`_approve_and_retry_if_review_required`: detecting the
        requirement (org-agnostically, from the branch's effective rules)
        before dispatch avoids a guaranteed-to-fail merge attempt for
        organizations that gate every merge on an approving review.
        """
        if self.preview_mode or pr_key in self._recently_approved:
            return
        if await self._branch_requires_approval(owner, repo, pr_info.base_branch):
            await self._ensure_pr_approved(pr_info, owner, repo)
