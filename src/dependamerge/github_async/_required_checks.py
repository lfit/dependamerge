# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Required status check and branch protection lookups.

Collects the required status check contexts for a branch from both
branch protection and repository rulesets, and exposes the raw branch
protection payload.  Both are cached per repo/branch for the session.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from typing import (
    Any,
)

from ._base import _GitHubAsyncBase


class _RequiredChecksMixin(_GitHubAsyncBase):
    """Required-check and branch-protection lookups for ``GitHubAsync``."""

    @staticmethod
    def _required_status_checks_from_detail(
        detail: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return required-status-check entries declared by a ruleset detail."""
        checks: list[dict[str, Any]] = []
        rules = detail.get("rules")
        if not isinstance(rules, list):
            return checks
        for rule in rules:
            if (
                not isinstance(rule, dict)
                or rule.get("type") != "required_status_checks"
            ):
                continue
            params = rule.get("parameters")
            if not isinstance(params, dict):
                continue
            required = params.get("required_status_checks")
            if not isinstance(required, list):
                continue
            for check in required:
                if (
                    isinstance(check, dict)
                    and isinstance(check.get("context"), str)
                    and check["context"]
                ):
                    checks.append(check)
        return checks

    async def _fetch_ruleset_required_checks(
        self, owner: str, repo: str, branch: str, default_branch: str | None
    ) -> tuple[list[dict[str, Any]], bool]:
        """Collect required checks from repo/org rulesets targeting *branch*.

        Returns ``(checks, reliable)`` where ``reliable`` is False when any
        ruleset request failed (so the caller must not cache the verdict).
        """
        checks: list[dict[str, Any]] = []
        try:
            rulesets = await self.get(f"/repos/{owner}/{repo}/rulesets?per_page=100")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.log.debug(
                f"Could not fetch rulesets for {owner}/{repo}: {e}",
                exc_info=True,
            )
            return checks, False
        if not isinstance(rulesets, list):
            return checks, True

        reliable = True
        for ruleset in rulesets:
            if not isinstance(ruleset, dict):
                continue
            ruleset_id = ruleset.get("id")
            if not ruleset_id:
                continue
            try:
                detail = await self.get(f"/repos/{owner}/{repo}/rulesets/{ruleset_id}")
                if not isinstance(detail, dict):
                    continue
                # Filter: skip rulesets that do not target this branch
                conditions = detail.get("conditions", {})
                if isinstance(conditions, dict) and not self._ruleset_applies_to_branch(
                    conditions, branch, default_branch
                ):
                    self.log.debug(
                        f"Ruleset {ruleset_id} does not apply to branch '{branch}'; skipping"
                    )
                    continue
                checks.extend(self._required_status_checks_from_detail(detail))
            except asyncio.CancelledError:
                raise
            except Exception as detail_err:
                reliable = False
                self.log.debug(
                    f"Could not fetch ruleset {ruleset_id} details: {detail_err}",
                    exc_info=True,
                )
        return checks, reliable

    async def _fetch_branch_protection_required_checks(
        self, owner: str, repo: str, branch: str
    ) -> tuple[list[dict[str, Any]], bool]:
        """Collect required checks from classic branch protection.

        Returns ``(checks, reliable)``.  Branch protection may be absent or
        inaccessible with the current token; a plain 404 is the definitive
        "no protection" answer, while anything else leaves the verdict
        unreliable.
        """
        checks: list[dict[str, Any]] = []
        try:
            data = await self.get(
                f"/repos/{owner}/{repo}/branches/{branch}/protection/required_status_checks"
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return checks, "404" in str(e)
        if isinstance(data, dict):
            for ctx in data.get("contexts", []):
                if isinstance(ctx, str) and ctx:
                    checks.append({"context": ctx})
            for check in data.get("checks", []):
                if (
                    isinstance(check, dict)
                    and isinstance(check.get("context"), str)
                    and check["context"]
                ):
                    checks.append(check)
        return checks, True

    async def get_required_status_checks(
        self, owner: str, repo: str, branch: str
    ) -> list[dict[str, Any]]:
        """Required status checks for *branch*, or an empty list.

        Errors read as "nothing required", which suits the callers that
        only want to know whether a particular context is enforced.  A
        caller that must tell "nothing is required" from "could not ask"
        wants :meth:`get_required_status_checks_reliable` instead.
        """
        checks, _reliable = await self.get_required_status_checks_reliable(
            owner, repo, branch
        )
        return checks

    async def get_required_status_checks_reliable(
        self, owner: str, repo: str, branch: str
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        Get required status checks for a branch by inspecting rulesets.

        Only rulesets whose ``conditions.ref_name`` patterns match *branch*
        are considered.  Falls back to branch protection rules if rulesets
        are not available.
        Returns ``(checks, reliable)``, where each check is a dict with
        'context' and optionally 'integration_id', deduplicated by
        ``context``.

        ``reliable`` is False when any lookup failed, which is what makes
        the empty list interpretable.  A token that cannot read rulesets
        --- ``GET /orgs/{org}/rulesets`` returned 403 throughout the
        503-PR run in ``docs/BULK_RUN_PERFORMANCE_AUDIT.md`` --- produces
        exactly the same empty list as a branch with no requirements, and
        a caller reasoning about *absence* must be able to tell them
        apart.

        Results are cached per ``owner/repo@branch`` for the session:
        required-check configuration is repo/branch-level state that does
        not change while dependamerge runs, and the block-reason analysis
        consults it repeatedly (several times per blocked PR).  The
        uncached path costs 2 + N requests (repo + ruleset list + one
        detail GET per ruleset), so the cache saves a burst of API
        traffic on every repeat.  Results assembled while any of those
        requests failed are *not* cached: the fetch treats errors as
        "no required checks", and pinning that error-derived verdict
        for the whole session could misclassify blocked PRs long after
        a transient outage has passed.
        """
        cache_key = f"{owner}/{repo}@{branch}"
        cached = self._required_checks_cache.get(cache_key)
        if cached is not None:
            # Only reliable results are cached, so a hit is reliable.
            return list(cached), True

        required_checks: list[dict[str, Any]] = []
        seen_contexts: set[str] = set()

        def _add(candidates: list[dict[str, Any]]) -> None:
            for check in candidates:
                ctx = check.get("context")
                if not isinstance(ctx, str) or not ctx:
                    continue
                if ctx not in seen_contexts:
                    seen_contexts.add(ctx)
                    required_checks.append(check)

        # Resolve the repo's actual default branch so that ~DEFAULT_BRANCH
        # ruleset conditions are evaluated correctly (not hardcoded to
        # main/master).  A branch that could not be resolved makes the
        # verdict unreliable rather than merely approximate: the ruleset
        # filter then treats ~DEFAULT_BRANCH conditions as applicable on
        # any branch, so on a non-default base the checks collected may
        # belong to a branch this pull request is not targeting.
        default_branch = await self._resolve_default_branch(owner, repo)

        # Try rulesets first (org-level and repo-level)
        ruleset_checks, reliable = await self._fetch_ruleset_required_checks(
            owner, repo, branch, default_branch
        )
        if default_branch is None:
            reliable = False
        _add(ruleset_checks)

        # Fall back to branch protection if no ruleset checks found
        if not required_checks:
            (
                bp_checks,
                bp_reliable,
            ) = await self._fetch_branch_protection_required_checks(owner, repo, branch)
            if not bp_reliable:
                reliable = False
            _add(bp_checks)

        if reliable:
            self._required_checks_cache[cache_key] = list(required_checks)
        return required_checks, reliable

    async def get_branch_protection(
        self, owner: str, repo: str, branch: str
    ) -> dict[str, Any]:
        """
        Get branch protection rules for a branch.

        REST: GET /repos/{owner}/{repo}/branches/{branch}/protection

        Results (including the empty "no protection" result) are cached
        per ``owner/repo@branch`` for the session: the merge pipeline
        calls this once per PR via ``_check_merge_requirements``, but
        protection config is branch-level state that does not change
        mid-run.  Errors other than 404 are not cached so a transient
        failure can succeed on retry.
        """
        cache_key = f"{owner}/{repo}@{branch}"
        cached = self._branch_protection_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            protection_data = await self.get(
                f"/repos/{owner}/{repo}/branches/{branch}/protection"
            )
            # Branch protection data should always be a dict, not a list
            result = protection_data if isinstance(protection_data, dict) else {}
            self._branch_protection_cache[cache_key] = result
            return result
        except Exception as e:
            # Branch protection might not be enabled, return empty dict
            if "404" in str(e):
                self._branch_protection_cache[cache_key] = {}
                return {}
            raise
