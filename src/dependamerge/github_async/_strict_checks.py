# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The "branch must be up to date" protection rule.

Resolves whether a branch requires strict status checks, from either
branch protection or a repository ruleset, plus the ruleset ref-pattern
matching that decides which rulesets apply to a branch.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import (
    Any,
)

from ._base import _GitHubAsyncBase


async def _strict_checks_from_protection(
    api: _GitHubAsyncBase, owner: str, repo: str, branch: str
) -> tuple[bool, bool]:
    """Read the classic branch-protection strict-checks flag.

    Returns:
        Tuple of ``(requires_strict, reliable)``, matching the contract
        of ``_requires_strict_status_checks_uncached``.
    """
    try:
        protection = await api.get_branch_protection(owner, repo, branch)
        checks = protection.get("required_status_checks")
        if isinstance(checks, dict) and checks.get("strict") is True:
            api.log.debug(
                "Branch %s/%s:%s requires up-to-date heads "
                "(classic protection strict checks)",
                owner,
                repo,
                branch,
            )
            return True, True
    except Exception as e:
        # get_branch_protection already maps 404 to {}; anything
        # surfacing here is a transient failure.
        api.log.debug(
            "Error checking classic strict-checks policy for %s/%s:%s: %s",
            owner,
            repo,
            branch,
            e,
        )
        return False, False
    return False, True


async def _strict_checks_from_rulesets(
    api: _GitHubAsyncBase, owner: str, repo: str, branch: str
) -> tuple[bool, bool]:
    """Look for an active ruleset enforcing strict status checks.

    Returns:
        Tuple of ``(requires_strict, reliable)``, matching the contract
        of ``_requires_strict_status_checks_uncached``.
    """
    reliable = True
    try:
        default_branch = await _strict_ruleset_default_branch(api, owner, repo)
        ruleset_ids = await _list_ruleset_ids(api, owner, repo)

        for ruleset_id in ruleset_ids:
            try:
                detail = await api.get(f"/repos/{owner}/{repo}/rulesets/{ruleset_id}")
                if not isinstance(detail, dict):
                    continue
            except Exception as detail_err:
                # An unreadable ruleset could hide a strict
                # required_status_checks rule — the eventual False
                # verdict is no longer definitive.
                reliable = False
                api.log.debug(
                    "Could not fetch ruleset %s for %s/%s: %s",
                    ruleset_id,
                    owner,
                    repo,
                    detail_err,
                )
                continue

            if _ruleset_requires_strict_checks(api, detail, branch, default_branch):
                api.log.debug(
                    "Branch %s/%s:%s requires up-to-date heads (ruleset: %s)",
                    owner,
                    repo,
                    branch,
                    detail.get("name", "unknown"),
                )
                return True, True
    except Exception as e:
        reliable = False
        api.log.debug(
            "Error checking rulesets for strict-checks policy on %s/%s:%s: %s",
            owner,
            repo,
            branch,
            e,
        )

    return False, reliable


async def _strict_ruleset_default_branch(
    api: _GitHubAsyncBase, owner: str, repo: str
) -> str | None:
    """Best-effort default-branch lookup for ruleset ref matching.

    Deliberately separate from ``_resolve_default_branch``: this lookup
    is uncached and its failure is non-fatal, because an unknown default
    branch only makes ``~DEFAULT_BRANCH`` matching conservative rather
    than wrong.
    """
    default_branch: str | None = None
    try:
        repo_data = await api.get(f"/repos/{owner}/{repo}")
        if isinstance(repo_data, dict):
            default_branch = repo_data.get("default_branch")
    except Exception as e:
        api.log.debug(
            "Could not resolve default branch for %s/%s: %s",
            owner,
            repo,
            e,
        )
    return default_branch


async def _list_ruleset_ids(api: _GitHubAsyncBase, owner: str, repo: str) -> list[int]:
    """Page through a repository's rulesets, collecting their ids."""
    ruleset_ids: list[int] = []
    page = 1
    per_page = 100
    while True:
        page_rulesets = await api.get(
            f"/repos/{owner}/{repo}/rulesets?per_page={per_page}&page={page}"
        )
        if not isinstance(page_rulesets, list) or not page_rulesets:
            break
        for rs in page_rulesets:
            if isinstance(rs, dict):
                rs_id = rs.get("id")
                if rs_id is not None:
                    ruleset_ids.append(int(rs_id))
        if len(page_rulesets) < per_page:
            break
        page += 1
    return ruleset_ids


def _ruleset_requires_strict_checks(
    api: _GitHubAsyncBase,
    detail: dict[str, Any],
    branch: str,
    default_branch: str | None,
) -> bool:
    """Check whether one ruleset enforces strict checks on *branch*."""
    if detail.get("enforcement") != "active":
        return False
    conditions = detail.get("conditions", {})
    if isinstance(conditions, dict) and not api._ruleset_applies_to_branch(
        conditions, branch, default_branch
    ):
        return False
    rules = detail.get("rules", [])
    if not isinstance(rules, list):
        return False
    for rule in rules:
        if isinstance(rule, dict) and rule.get("type") == "required_status_checks":
            params = rule.get("parameters")
            if (
                isinstance(params, dict)
                and params.get("strict_required_status_checks_policy") is True
            ):
                return True
    return False


class _StrictChecksMixin(_GitHubAsyncBase):
    """Strict-status-check requirement lookups for ``GitHubAsync``."""

    async def requires_strict_status_checks(
        self, owner: str, repo: str, branch: str = "main"
    ) -> bool:
        """Check whether a branch requires PR heads to be up to date.

        GitHub only rejects the merge of a ``behind`` PR when the
        branch's protection enforces the *strict* status-check policy
        ("Require branches to be up to date before merging").  Without
        it, a behind-but-green PR merges fine and any proactive rebase
        is wasted work (plus a full CI re-run).  The merge pipeline
        uses this to rebase **only when GitHub would actually demand
        it**.

        Uses two complementary sources:

        1. **Classic branch protection** –
           ``required_status_checks.strict`` on the branch protection
           REST payload (already cached by :meth:`get_branch_protection`).
        2. **Repository rulesets** – any active ruleset targeting the
           branch whose ``required_status_checks`` rule sets
           ``strict_required_status_checks_policy``.

        Returns:
            True if either mechanism requires the branch to be up to
            date before merging.

        Results are cached per ``owner/repo@branch`` for the session;
        verdicts derived from transient API errors are not cached so a
        momentary outage cannot pin a wrong answer for the whole run.
        """
        cache_key = f"{owner}/{repo}@{branch}"
        cached = self._requires_strict_checks_cache.get(cache_key)
        if cached is not None:
            return cached
        result, reliable = await self._requires_strict_status_checks_uncached(
            owner, repo, branch
        )
        if reliable:
            self._requires_strict_checks_cache[cache_key] = result
        return result

    async def _requires_strict_status_checks_uncached(
        self, owner: str, repo: str, branch: str
    ) -> tuple[bool, bool]:
        """Uncached implementation of :meth:`requires_strict_status_checks`.

        Returns:
            Tuple of ``(requires_strict, reliable)``.  ``reliable`` is
            False when a transient API error prevented a definitive
            verdict — a ``True`` verdict is always reliable (positive
            evidence), but an error-derived ``False`` must not be
            cached because the requirement may simply have been
            unreadable at that moment.
        """
        strict, protection_reliable = await _strict_checks_from_protection(
            self, owner, repo, branch
        )
        if strict:
            return True, True

        strict, rulesets_reliable = await _strict_checks_from_rulesets(
            self, owner, repo, branch
        )
        if strict:
            return True, True
        return False, protection_reliable and rulesets_reliable

    @staticmethod
    def _ruleset_applies_to_branch(
        conditions: dict[str, Any],
        branch: str,
        default_branch: str | None = None,
    ) -> bool:
        """Check whether a ruleset's ref_name conditions match *branch*.

        Ruleset conditions use ``conditions.ref_name.include`` /
        ``conditions.ref_name.exclude`` arrays.  Recognised patterns:

        * ``~DEFAULT_BRANCH`` — matches when *branch* equals *default_branch*.
          If *default_branch* is not supplied, the match is treated as
          ``True`` (conservative) to avoid silently filtering out rulesets
          for repos whose default branch is something other than
          ``main``/``master``.
        * ``~ALL``            — matches every branch.
        * ``refs/heads/<name>`` — exact ref match.
        * Bare branch name   — treated as ``refs/heads/<name>``.

        If the conditions dict is empty or missing ``ref_name`` the
        ruleset is assumed to apply (conservative).
        """
        # Resolved late, and from the package namespace, because the
        # concrete class is assembled from this mixin: the reference has
        # always been to ``dependamerge.github_async.GitHubAsync``, so
        # keep it that way rather than binding the mixin directly.
        from . import GitHubAsync

        ref_name = conditions.get("ref_name", {})
        if not isinstance(ref_name, dict):
            return True  # No conditions — assume applies

        include = ref_name.get("include", [])
        exclude = ref_name.get("exclude", [])

        full_ref = f"refs/heads/{branch}"

        # Must match at least one include pattern (if any are specified)
        if include and not any(
            GitHubAsync._ref_pattern_matches(p, branch, full_ref, default_branch)
            for p in include
            if isinstance(p, str)
        ):
            return False

        # Must not match any exclude pattern
        if any(
            GitHubAsync._ref_pattern_matches(p, branch, full_ref, default_branch)
            for p in exclude
            if isinstance(p, str)
        ):
            return False

        return True

    @staticmethod
    def _ref_pattern_matches(
        pattern: str,
        branch: str,
        full_ref: str,
        default_branch: str | None,
    ) -> bool:
        """Check whether a single ruleset ref pattern matches *branch*.

        Defined as a static helper method (rather than a closure inside
        ``_ruleset_applies_to_branch``) so it is not re-created on every
        call and can be reused across the include/exclude comprehensions.
        """
        import fnmatch

        if pattern == "~ALL":
            return True
        if pattern == "~DEFAULT_BRANCH":
            if default_branch is None:
                # Unknown default branch — conservatively assume match
                return True
            return branch == default_branch
        # Normalise bare branch names to full refs
        pat = pattern if pattern.startswith("refs/") else f"refs/heads/{pattern}"
        return fnmatch.fnmatchcase(full_ref, pat)
