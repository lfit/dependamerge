# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Organisation rulesets and whether one applies to a branch.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
import fnmatch
from typing import Any

from ..output_utils import log_and_print
from ._base import _MergeManagerBase


class _RulesetsMixin(_MergeManagerBase):
    """Organisation rulesets and whether one applies to a branch."""

    async def _org_approval_rulesets(self, org: str) -> list[dict[str, Any]] | None:
        """Enumerate active org rulesets that mandate an approving review.

        Queried **once per org** and cached for the run — the approval
        requirement originates from a single organization ruleset, so it
        is wasteful to rediscover it per repository.  Returns one entry
        per approval-mandating ruleset (``{"name", "conditions"}``), ``[]``
        when the org mandates none, or ``None`` when enumeration failed
        (e.g. the token cannot read org rulesets) so the caller can fall
        back to the authoritative per-repo endpoint.

        The first time an org is found to gate merges on a review a single
        user-facing line is emitted, so the requirement is visible at the
        point of detection rather than buried in debug logs.
        """
        if org in self._org_approval_cache:
            return self._org_approval_cache[org]

        async with self._org_approval_locks_lock:
            if org not in self._org_approval_locks:
                self._org_approval_locks[org] = asyncio.Lock()
            org_lock = self._org_approval_locks[org]

        async with org_lock:
            # Re-check after acquiring the per-org lock (another task may
            # have populated the cache while we waited).
            if org in self._org_approval_cache:
                return self._org_approval_cache[org]

            if not self._github_client:
                return None

            result: list[dict[str, Any]] = []
            try:
                # The list endpoint is paginated (default page size 30),
                # so an org with many rulesets could otherwise silently
                # drop an approval-mandating one.  Walk every page.
                page = 1
                per_page = 100
                while True:
                    rulesets = await self._github_client.get(
                        f"/orgs/{org}/rulesets?per_page={per_page}&page={page}"
                    )
                    if not isinstance(rulesets, list) or not rulesets:
                        break
                    for rs in rulesets:
                        if not isinstance(rs, dict):
                            continue
                        # Only active branch rulesets gate merges;
                        # "evaluate" and "disabled" rulesets do not block,
                        # and tag rulesets are irrelevant to PR merges.
                        if rs.get("enforcement") != "active":
                            continue
                        if rs.get("target", "branch") != "branch":
                            continue
                        rid = rs.get("id")
                        if rid is None:
                            continue
                        detail = await self._github_client.get(
                            f"/orgs/{org}/rulesets/{rid}"
                        )
                        if not isinstance(detail, dict):
                            continue
                        if self._rules_require_approval(detail.get("rules")):
                            result.append(
                                {
                                    "name": rs.get("name", ""),
                                    "conditions": detail.get("conditions") or {},
                                }
                            )
                    if len(rulesets) < per_page:
                        break
                    page += 1
            except Exception as e:
                # Enumeration failed (often a permission/SSO problem).
                # Cache ``None`` so callers consult the per-repo endpoint
                # rather than silently skipping proactive approval.
                self.log.debug(f"Could not enumerate org rulesets for {org}: {e}")
                self._org_approval_cache[org] = None
                return None

            self._org_approval_cache[org] = result
            if result:
                names = ", ".join(r["name"] for r in result if r.get("name")) or (
                    "unnamed ruleset"
                )
                log_and_print(
                    self.log,
                    self._console,
                    "🔐 Organization requires approving reviews before merging\n"
                    f"Ruleset: {names}",
                    level="info",
                )
            return result

    @staticmethod
    def _ruleset_name_matches(
        name: str, include: list[Any], exclude: list[Any]
    ) -> bool:
        """Evaluate a ruleset ``repository_name`` condition against a repo.

        ``include``/``exclude`` are fnmatch-style globs; the sentinel
        ``~ALL`` matches every repository.  A repo is in scope when it
        matches an include pattern and no exclude pattern.
        """

        def match_any(patterns: list[Any]) -> bool:
            for pat in patterns:
                if pat == "~ALL":
                    return True
                if isinstance(pat, str) and fnmatch.fnmatch(name, pat):
                    return True
            return False

        if exclude and match_any(exclude):
            return False
        if not include:
            return False
        return match_any(include)

    @staticmethod
    def _ruleset_ref_matches(
        branch: str, include: list[Any], exclude: list[Any]
    ) -> bool | None:
        """Evaluate a ruleset ``ref_name`` condition against a branch.

        Returns ``True``/``False`` when it can be decided locally, or
        ``None`` when it cannot (so the caller consults the authoritative
        per-repo endpoint).  ``~ALL`` matches any branch.  ``~DEFAULT_BRANCH``
        is treated as in scope: confirming it would need an extra per-repo
        default-branch lookup, and the automation PRs this gates target
        the default branch — a spurious approval on a non-default-base PR
        (which we were about to merge anyway) is harmless.
        """
        ref = f"refs/heads/{branch}"

        def match_any(patterns: list[Any]) -> bool:
            for pat in patterns:
                if pat in ("~ALL", "~DEFAULT_BRANCH"):
                    return True
                if isinstance(pat, str) and (
                    fnmatch.fnmatch(ref, pat) or fnmatch.fnmatch(branch, pat)
                ):
                    return True
            return False

        if exclude and match_any(exclude):
            return False
        if not include:
            # An empty include is unusual; defer to the authoritative
            # endpoint rather than guess.
            return None
        return match_any(include)

    def _ruleset_condition_applies(
        self, conditions: Any, repo: str, branch: str
    ) -> bool | None:
        """Whether a ruleset's ``conditions`` select ``repo@branch``.

        Returns ``True``/``False`` when the verdict is decidable from the
        ``repository_name`` and ``ref_name`` conditions, or ``None`` when
        the ruleset uses a condition type we do not evaluate locally
        (e.g. ``repository_id`` or ``repository_property``) so the caller
        falls back to GitHub's authoritative per-repo evaluation.
        """
        if not isinstance(conditions, dict):
            return None
        # Any condition type beyond the two we evaluate means we cannot be
        # sure locally — signal the caller to ask GitHub directly.
        if any(key not in ("repository_name", "ref_name") for key in conditions):
            return None

        repo_cond = conditions.get("repository_name")
        if isinstance(repo_cond, dict):
            if not self._ruleset_name_matches(
                repo,
                repo_cond.get("include") or [],
                repo_cond.get("exclude") or [],
            ):
                return False

        ref_cond = conditions.get("ref_name")
        if isinstance(ref_cond, dict):
            ref_applies = self._ruleset_ref_matches(
                branch,
                ref_cond.get("include") or [],
                ref_cond.get("exclude") or [],
            )
            if ref_applies is not True:
                # False or None (undecidable) — propagate so an undecidable
                # ref falls back to the authoritative endpoint.
                return ref_applies

        return True
