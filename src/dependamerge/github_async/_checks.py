# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Check run, status context and branch state queries.

Reads check runs and workflow run names for a ref, the failing commit
status contexts, how far a branch is behind its base, the repository
default branch, and which kind of protection guards a branch.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import (
    Any,
)
from urllib.parse import quote

from ..check_runs import FAILING_CONCLUSIONS, latest_check_run_per_name
from ._base import _GitHubAsyncBase


class _ChecksMixin(_GitHubAsyncBase):
    """Check-run, status and branch-state queries for ``GitHubAsync``."""

    async def get_check_runs_for_ref(
        self, owner: str, repo: str, ref: str
    ) -> list[dict[str, Any]]:
        """Check runs reported against *ref*.

        Returns the raw runs, including superseded duplicates: deciding
        which run is authoritative for a given name belongs to
        :mod:`dependamerge.check_runs`, not here.

        Paginated.  A busy commit carries more than one page, and every
        caller reasons about *absence* --- "nothing is failing", "only
        semantic checks are failing" --- so a truncated page reads as an
        all-clear for whatever sat on page two.
        """
        runs: list[dict[str, Any]] = []
        async for page in self.get_paginated(
            f"/repos/{owner}/{repo}/commits/{ref}/check-runs", per_page=100
        ):
            if not isinstance(page, dict):
                continue
            batch = page.get("check_runs")
            if not batch:
                break
            runs.extend(run for run in batch if isinstance(run, dict))
        return runs

    async def get_failing_workflow_run_names_for_sha(
        self, owner: str, repo: str, head_sha: str
    ) -> list[str]:
        """Names of Actions workflow runs for *head_sha* that did not pass.

        Reported in the **workflow** namespace, which is the one a
        ruleset rejection quotes.  A check run carries its *job* name
        instead --- ``.github/workflows/codeql.yml`` declares the workflow
        ``CodeQL`` and the job ``Audit Repository`` --- so matching a
        rejection against check-run names compares two different
        vocabularies and misses whenever they differ.

        A workflow can have several runs on one commit: a re-run, or a
        dispatch the concurrency group cancelled.  Runs are collapsed to
        the latest per name before their conclusion is read, so a
        superseded failure beside a newer success does not read as
        failing --- the stale-snapshot mistake this whole reading exists
        to avoid, arriving by another route.  The rule is
        :func:`check_runs.latest_check_run_per_name`, shared rather than
        restated.

        Only completed runs count.  A queued or in-progress run has no
        conclusion and has not failed at anything yet.
        """
        collected: list[dict[str, Any]] = []
        async for page in self.get_paginated(
            f"/repos/{owner}/{repo}/actions/runs",
            params={"head_sha": head_sha},
            per_page=100,
        ):
            if not isinstance(page, dict):
                continue
            runs = page.get("workflow_runs")
            if not runs:
                break
            collected.extend(run for run in runs if isinstance(run, dict))

        latest = latest_check_run_per_name(collected)
        failing: list[str] = []
        for run in collected:
            name = (run.get("name") or "").strip()
            if not name or name in failing:
                continue
            conclusion = (latest[name].get("conclusion") or "").strip().lower()
            if conclusion in FAILING_CONCLUSIONS:
                failing.append(name)
        return failing

    async def get_workflow_run_names_for_sha(
        self, owner: str, repo: str, head_sha: str
    ) -> set[str]:
        """Names of Actions workflow runs that exist for *head_sha*.

        Distinct from check runs.  A required workflow that GitHub never
        dispatched has **no workflow run at all** --- not a queued one, not
        a failed one, nothing --- so its absence here is the signal that
        waiting cannot help.  Check runs cannot express that: the
        workflow simply never appears.

        Paginated: a busy commit can carry more than one page of runs,
        and a required workflow sitting on page two would otherwise read
        as absent.  Callers use absence to *stop waiting*, so a false
        absence turns a live workflow into a reported merge failure.

        Request failures **propagate**: they are not converted into an
        empty set, and callers that use absence to stop waiting must
        handle them (``_absent_workflow_runs`` does).  An empty set
        means the lookup succeeded and found nothing, which is itself
        ambiguous --- a commit whose runs are not yet visible looks
        identical --- so that too must read as "unknown" rather than
        "nothing ran".  Both readings land on the same safe action:
        keep waiting.
        """
        names: set[str] = set()
        async for page in self.get_paginated(
            f"/repos/{owner}/{repo}/actions/runs",
            params={"head_sha": head_sha},
            per_page=100,
        ):
            if not isinstance(page, dict):
                continue
            runs = page.get("workflow_runs")
            if not runs:
                break
            for run in runs:
                if isinstance(run, dict):
                    name = run.get("name")
                    if isinstance(name, str) and name:
                        names.add(name)
        return names

    async def get_failing_status_contexts(
        self, owner: str, repo: str, ref: str
    ) -> list[str]:
        """Contexts whose latest commit *status* is failing.

        Distinct from check runs: pre-commit.ci, DCO and other legacy
        integrations report through the commit status API, so a caller
        reasoning about "what is failing" from check runs alone sees only
        half the picture.

        GitHub's combined-status endpoint already collapses each context
        to its latest state, so no deduplication is needed here.

        Paginated: the endpoint defaults to 30 statuses a page, which a
        repository with a moderate integration set exceeds --- and the
        required context that is actually blocking has no reason to be
        among the first thirty.
        """
        failing: list[str] = []
        async for page in self.get_paginated(
            f"/repos/{owner}/{repo}/commits/{ref}/status", per_page=100
        ):
            if not isinstance(page, dict):
                continue
            statuses = page.get("statuses")
            if not statuses:
                break
            for entry in statuses:
                if not isinstance(entry, dict):
                    continue
                if entry.get("state") in ("failure", "error"):
                    context = entry.get("context")
                    if isinstance(context, str) and context and context not in failing:
                        failing.append(context)
        return failing

    async def get_behind_by(
        self, owner: str, repo: str, base_ref: str, head_sha: str
    ) -> int | None:
        """Return how many commits ``head_sha`` is behind ``base_ref``.

        GitHub's ``mergeable_state`` is a single value, so ``blocked``
        (a failing required check) masks ``behind`` (a stale head).
        This helper answers the staleness question independently via
        the compare API, which works regardless of the reported
        mergeable state and regardless of whether the head lives on a
        fork (the SHA is resolvable in the base repository's network).

        Args:
            owner: Base repository owner
            repo: Base repository name
            base_ref: Base branch name (e.g. ``main``)
            head_sha: Head commit SHA of the pull request

        Returns:
            The ``behind_by`` commit count, or ``None`` when the
            comparison could not be performed (API error, unexpected
            payload).  ``None`` means "unknown": callers must not
            interpret it as ``behind_by == 0`` ("up to date"), and
            staleness-driven write actions (e.g. requesting a rebase)
            should require positive evidence (``behind_by > 0``)
            rather than acting on an unknown — the pattern used by
            ``AsyncMergeManager._blocked_pr_needs_rebase``.
        """
        encoded_base = quote(base_ref, safe="")
        try:
            comparison = await self.get(
                f"/repos/{owner}/{repo}/compare/{encoded_base}...{head_sha}"
            )
        except Exception as exc:
            self.log.debug(
                "Compare %s...%s failed for %s/%s: %s",
                base_ref,
                head_sha,
                owner,
                repo,
                exc,
            )
            return None
        if isinstance(comparison, dict):
            behind = comparison.get("behind_by")
            if isinstance(behind, int):
                return behind
        return None

    async def _resolve_default_branch(self, owner: str, repo: str) -> str | None:
        """Return the repository's actual default branch, or ``None``.

        Many repositories default to ``master`` rather than ``main``, so
        callers must never assume a name.  This reads the authoritative
        ``default_branch`` field from the repository metadata and returns
        ``None`` when it cannot be determined (the repo is unreadable or
        the field is absent), letting callers degrade gracefully instead
        of operating on a wrong branch.

        Successful lookups are cached per ``owner/repo`` for the
        session (a repo's default branch does not change mid-run);
        failures are not cached so a transient error can recover.
        """
        cache_key = f"{owner}/{repo}"
        if cache_key in self._default_branch_cache:
            return self._default_branch_cache[cache_key]
        try:
            repo_data = await self.get(f"/repos/{owner}/{repo}")
        except Exception as e:
            self.log.debug(
                "Could not resolve default branch for %s/%s: %s", owner, repo, e
            )
            return None
        if isinstance(repo_data, dict):
            default_branch = repo_data.get("default_branch")
            if isinstance(default_branch, str) and default_branch:
                self._default_branch_cache[cache_key] = default_branch
                return default_branch
        return None

    async def _detect_branch_protection_kind(
        self, owner: str, repo: str, branch: str
    ) -> str:
        """Best-effort classification of what guards a branch.

        Used by :meth:`analyze_block_reason` to describe an otherwise
        unexplained ``BLOCKED`` state accurately instead of asserting
        "branch protection".

        Returns:
            ``"ruleset"``    — one or more repository rulesets apply to the
            branch (reported in preference to classic protection because
            rulesets are invisible to the GraphQL ``branchProtectionRule``
            field and are what most current repositories use).
            ``"protection"`` — a classic branch protection rule applies.
            ``"none"``       — neither could be found (the branch appears
            unguarded, or the token cannot read the configuration).
        """
        # Repository rulesets (newer API): the effective-rules endpoint
        # returns every rule that applies to the branch from any active
        # ruleset.  A non-empty list means a ruleset guards the branch.
        # Branch names can contain '/' (e.g. ``release/v1``), so they must
        # be URL-encoded before interpolation into the REST path.
        encoded_branch = quote(branch, safe="")
        try:
            rules = await self.get(
                f"/repos/{owner}/{repo}/rules/branches/{encoded_branch}"
            )
            if isinstance(rules, list) and rules:
                return "ruleset"
        except Exception as e:
            self.log.debug(
                "Could not read branch rules for %s/%s:%s: %s",
                owner,
                repo,
                branch,
                e,
            )

        # Classic branch protection: 200 = protected, 404 = no rule.
        try:
            await self.get(
                f"/repos/{owner}/{repo}/branches/{encoded_branch}/protection"
            )
            return "protection"
        except Exception as e:
            if "404" not in str(e):
                self.log.debug(
                    "Could not read branch protection for %s/%s:%s: %s",
                    owner,
                    repo,
                    branch,
                    e,
                )

        return "none"
