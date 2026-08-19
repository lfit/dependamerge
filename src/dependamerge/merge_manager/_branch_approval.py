# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Whether a branch requires an approving review before merge.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from urllib.parse import quote

from ._base import _MergeManagerBase


class _BranchApprovalMixin(_MergeManagerBase):
    """Whether a branch requires an approving review before merge."""

    async def _branch_requires_approval(
        self, owner: str, repo: str, branch: str
    ) -> bool:
        """Whether ``owner/repo@branch`` mandates an approving review to merge.

        Some organizations enforce a repository ruleset that requires at
        least one approving review before *any* merge is permitted (the
        ``lfreleng-actions`` "Base Protections" ruleset is one example).
        Under "merge first, approve on demand" every such PR would incur a
        guaranteed-to-fail merge attempt before we approve and retry.
        Detecting the requirement up-front lets us approve proactively and
        skip that doomed round-trip.

        Detection is **org-first**: the org's rulesets are enumerated once
        (see :meth:`_org_approval_rulesets`) and their conditions are
        evaluated locally, so a whole org-wide run needs a single ruleset
        query rather than one effective-rules call per repository.  Only
        when a ruleset uses a condition we cannot evaluate locally, or org
        enumeration was not possible, do we fall back to GitHub's
        authoritative per-repo ``rules/branches`` endpoint.  The resolved
        verdict is cached per repo+branch.
        """
        cache_key = f"{owner}/{repo}@{branch}"
        if cache_key in self._branch_approval_cache:
            return self._branch_approval_cache[cache_key]

        async with self._branch_approval_locks_lock:
            if cache_key not in self._branch_approval_locks:
                self._branch_approval_locks[cache_key] = asyncio.Lock()
            branch_lock = self._branch_approval_locks[cache_key]

        async with branch_lock:
            # Re-check after acquiring the per-branch lock (another task
            # may have populated the cache while we waited).
            if cache_key in self._branch_approval_cache:
                return self._branch_approval_cache[cache_key]

            rulesets = await self._org_approval_rulesets(owner)

            requires = False
            # ``None`` means org enumeration failed; consult the per-repo
            # endpoint.  An empty list means the org mandates no approval
            # (repo-level rulesets, if any, are covered by the reactive
            # approve-on-demand safety net).
            need_authoritative = rulesets is None
            for rs in rulesets or []:
                applies = self._ruleset_condition_applies(
                    rs.get("conditions"), repo, branch
                )
                if applies is True:
                    requires = True
                    break
                if applies is None:
                    need_authoritative = True

            if not requires and need_authoritative:
                requires = await self._effective_branch_requires_approval(
                    owner, repo, branch
                )

            self._branch_approval_cache[cache_key] = requires
            if requires:
                self.log.debug(
                    "Branch %s requires an approving review before merge; "
                    "approving proactively",
                    cache_key,
                )
            return requires

    async def _effective_branch_requires_approval(
        self, owner: str, repo: str, branch: str
    ) -> bool:
        """Authoritative per-repo fallback for the approval requirement.

        Uses ``GET /repos/{owner}/{repo}/rules/branches/{branch}`` which
        returns the *effective* rules for the branch — every applicable
        org- and repo-level ruleset, with all conditions already evaluated
        by GitHub.  Used only when the org-first path cannot decide (an
        unrecognised condition type, or org enumeration was unavailable).
        On any error returns ``False`` so the reactive approve-on-demand
        path remains the safety net rather than blocking the merge.
        """
        if not self._github_client:
            return False
        try:
            # Branch names can contain "/" (e.g. "release/v1"); encode the
            # whole segment so it routes to the right endpoint rather than
            # 404ing and being mistaken for "no rules".
            rules = await self._github_client.get(
                f"/repos/{owner}/{repo}/rules/branches/{quote(branch, safe='')}"
            )
            return self._rules_require_approval(rules)
        except Exception as e:
            self.log.debug(
                f"Could not read effective branch rules for {owner}/{repo}@{branch}: {e}"
            )
            return False
