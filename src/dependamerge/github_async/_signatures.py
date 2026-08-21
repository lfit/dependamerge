# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Commit signature verification and the signature protection rule.

Reads a pull request's commits and their verification state, and
resolves whether the target branch requires signed commits — via
branch protection or a repository ruleset.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import (
    Any,
)
from urllib.parse import quote

from ._base import _GitHubAsyncBase


async def _signatures_from_protection(
    api: _GitHubAsyncBase, owner: str, repo: str, branch: str
) -> tuple[bool, bool]:
    """Read the classic branch-protection signature requirement.

    Returns:
        Tuple of ``(requires_signatures, reliable)``, matching the
        contract of ``_requires_commit_signatures_uncached``.
    """
    try:
        # The signatures endpoint returns 200 with {"enabled": true/false}
        # or 404 when branch protection / signature requirement is absent.
        encoded_branch = quote(branch, safe="")
        sig_data = await api.get(
            f"/repos/{owner}/{repo}/branches/{encoded_branch}/protection/required_signatures"
        )
        if isinstance(sig_data, dict) and sig_data.get("enabled"):
            api.log.debug(
                "Branch %s/%s:%s requires commit signatures (classic protection)",
                owner,
                repo,
                branch,
            )
            return True, True
    except Exception as e:
        # 404 → not enabled; other errors → continue checking rulesets
        if "404" not in str(e):
            api.log.debug(
                "Error checking classic signature requirement for %s/%s:%s: %s",
                owner,
                repo,
                branch,
                e,
            )
            return False, False
    return False, True


async def _signatures_from_rulesets(
    api: _GitHubAsyncBase, owner: str, repo: str, branch: str
) -> tuple[bool, bool]:
    """Look for an active ruleset requiring signed commits.

    Returns:
        Tuple of ``(requires_signatures, reliable)``, matching the
        contract of ``_requires_commit_signatures_uncached``.
    """
    reliable = True
    try:
        default_branch = await _signature_ruleset_default_branch(api, owner, repo)
        ruleset_ids = await _list_signature_ruleset_ids(api, owner, repo)

        for ruleset_id in ruleset_ids:
            try:
                detail = await api.get(f"/repos/{owner}/{repo}/rulesets/{ruleset_id}")
                if not isinstance(detail, dict):
                    continue
            except Exception as detail_err:
                # An unreadable ruleset could hide a
                # required_signatures rule — the eventual False
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

            if _ruleset_requires_signatures(api, detail, branch, default_branch):
                api.log.debug(
                    "Branch %s/%s:%s requires commit signatures (ruleset: %s)",
                    owner,
                    repo,
                    branch,
                    detail.get("name", "unknown"),
                )
                return True, True
    except Exception as e:
        reliable = False
        api.log.debug(
            "Error checking rulesets for signature requirement on %s/%s:%s: %s",
            owner,
            repo,
            branch,
            e,
        )

    return False, reliable


async def _signature_ruleset_default_branch(
    api: _GitHubAsyncBase, owner: str, repo: str
) -> str | None:
    """Resolve the repo's actual default branch for ruleset matching.

    Best-effort: without the default branch we fall through to
    conservative ``~DEFAULT_BRANCH`` matching.
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


async def _list_signature_ruleset_ids(
    api: _GitHubAsyncBase, owner: str, repo: str
) -> list[int]:
    """Paginate through all rulesets to collect their IDs.

    The list endpoint may not include full rules/conditions, so callers
    fetch each ruleset's detail individually (matching the pattern in
    ``get_required_status_checks``).
    """
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


def _ruleset_requires_signatures(
    api: _GitHubAsyncBase,
    detail: dict[str, Any],
    branch: str,
    default_branch: str | None,
) -> bool:
    """Check whether one ruleset requires signatures on *branch*."""
    # Only consider active rulesets
    if detail.get("enforcement") != "active":
        return False
    # Check if this ruleset applies to our branch
    conditions = detail.get("conditions", {})
    if isinstance(conditions, dict) and not api._ruleset_applies_to_branch(
        conditions, branch, default_branch
    ):
        return False
    rules = detail.get("rules", [])
    if isinstance(rules, list):
        for rule in rules:
            if isinstance(rule, dict) and rule.get("type") == "required_signatures":
                return True
    return False


class _SignaturesMixin(_GitHubAsyncBase):
    """Commit-signature queries and requirements for ``GitHubAsync``."""

    async def get_pull_request_commits(
        self, owner: str, repo: str, number: int
    ) -> list[dict[str, Any]]:
        """All commits on a pull request, across pages."""
        out: list[dict[str, Any]] = []
        async for page in self.get_paginated(
            f"/repos/{owner}/{repo}/pulls/{number}/commits",
            per_page=100,
        ):
            if isinstance(page, list):
                out.extend(c for c in page if isinstance(c, dict))
        return out

    async def check_pr_commit_signatures(
        self, owner: str, repo: str, number: int
    ) -> tuple[bool, list[str]]:
        """Check whether all commits on a pull request have verified signatures.

        REST: GET /repos/{owner}/{repo}/pulls/{pull_number}/commits

        Returns:
            Tuple of ``(all_verified, unverified_shas)``.
            ``all_verified`` is True when every commit carries a
            valid signature according to GitHub.
            ``unverified_shas`` contains the abbreviated SHAs of
            any commits whose verification failed.

        Raises:
            Exception: surfaces the underlying API/network error
            on failure rather than silently returning a default.
            Callers that want fail-open or fail-closed semantics
            should wrap the call in ``try``/``except`` and decide
            for themselves — the previous fail-open default
            (returning ``(True, [])``) collided with the
            signature-preservation gate in ``rebase.py``, which
            documents "verified" as a positive confirmation.
        """
        unverified: list[str] = []
        # Iterate over all pages of commits to ensure we don't miss
        # unverified commits on pull requests with >100 commits.
        async for commits in self.get_paginated(
            f"/repos/{owner}/{repo}/pulls/{number}/commits",
            per_page=100,
        ):
            if not isinstance(commits, list):
                # Unexpected response shape: the API returned 200 OK but
                # not the documented list of commits. We cannot determine
                # signature status from this, so we must not pretend every
                # commit is verified (the old fail-open ``(True, [])``
                # default collided with the signature-preservation gate in
                # ``rebase.py``). Surface the uncertainty to the caller.
                raise RuntimeError(
                    "Unexpected response shape from "
                    f"/repos/{owner}/{repo}/pulls/{number}/commits: "
                    f"expected a list, got {type(commits).__name__}"
                )

            for commit_data in commits:
                if not isinstance(commit_data, dict):
                    continue
                raw_sha = commit_data.get("sha")
                sha = str(raw_sha)[:8] if isinstance(raw_sha, str) else "unknown"
                commit_obj = commit_data.get("commit")
                if not isinstance(commit_obj, dict):
                    unverified.append(sha)
                    continue
                verification = commit_obj.get("verification")
                if not isinstance(verification, dict):
                    unverified.append(sha)
                    continue
                if not verification.get("verified", False):
                    unverified.append(sha)

        all_verified = len(unverified) == 0
        return all_verified, unverified

    async def requires_commit_signatures(
        self, owner: str, repo: str, branch: str = "main"
    ) -> bool:
        """
        Check whether a branch requires signed (verified) commits.

        Uses two complementary sources:

        1. **Classic branch protection** – the ``required_signatures``
           sub-resource of the branch protection REST endpoint.
        2. **Repository rulesets** (newer API) – any active ruleset that
           targets the given branch and contains a ``required_signatures``
           rule.

        Returns:
            True if signed commits are required by *either* mechanism.

        Results are cached per ``owner/repo@branch`` for the session:
        the requirement is branch-protection/ruleset configuration that
        does not change while dependamerge runs, and the uncached path
        costs up to 3 + N requests (classic-protection probe, repo
        metadata, ruleset list, one detail GET per ruleset).  Verdicts
        derived from transient API errors are *not* cached, so a
        momentary outage cannot pin a wrong answer for the whole run.
        """
        cache_key = f"{owner}/{repo}@{branch}"
        cached = self._requires_signatures_cache.get(cache_key)
        if cached is not None:
            return cached
        result, reliable = await self._requires_commit_signatures_uncached(
            owner, repo, branch
        )
        if reliable:
            self._requires_signatures_cache[cache_key] = result
        return result

    async def _requires_commit_signatures_uncached(
        self, owner: str, repo: str, branch: str
    ) -> tuple[bool, bool]:
        """Uncached implementation of :meth:`requires_commit_signatures`.

        Returns:
            Tuple of ``(requires_signatures, reliable)``.  ``reliable``
            is False when a transient (non-404) API error prevented a
            definitive verdict — a ``True`` verdict is always reliable
            (positive evidence), but an error-derived ``False`` must
            not be cached because the requirement may simply have been
            unreadable at that moment.
        """
        requires, protection_reliable = await _signatures_from_protection(
            self, owner, repo, branch
        )
        if requires:
            return True, True

        requires, rulesets_reliable = await _signatures_from_rulesets(
            self, owner, repo, branch
        )
        if requires:
            return True, True
        return False, protection_reliable and rulesets_reliable
