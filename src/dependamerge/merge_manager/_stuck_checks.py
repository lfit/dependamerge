# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Detection of required checks that will never report.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase


@dataclass(frozen=True)
class _StuckCheckContext:
    """The timing and eligibility facts both check scans share.

    Gathered once per detection so the check-run scan and the status
    scan judge staleness against exactly the same reference points.
    """

    now: datetime
    pr_updated: datetime
    threshold: float
    required_contexts: set[str]


def _parse_ts(value: Any) -> datetime | None:
    """Parse a GitHub timestamp, returning None when it is unusable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        # GitHub returns RFC 3339 with a trailing ``Z``.
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    # A timestamp without tz info parses to a naive datetime, which
    # raises ``TypeError`` when subtracted from the tz-aware ``now``.
    # Treat it as unparsable (fail closed) so the detector degrades
    # gracefully instead of aborting the merge run.
    if ts.tzinfo is None:
        return None
    return ts


def _is_dco_name(name: str) -> bool:
    """Return True when ``name`` looks like a DCO check.

    Matches the common variants emitted by the GitHub DCO App and
    similar bots: ``DCO``, ``dco/dco``, ``dcobot``, and any name
    containing ``signoff`` / ``sign-off`` / ``signed-off``
    (case-insensitive).
    """
    n = (name or "").strip().lower()
    if not n:
        return False
    if n in {"dco", "dco/dco", "dcobot"} or n.startswith("dco/"):
        return True
    return "signoff" in n or "sign-off" in n or "signed-off" in n


def _is_precommit_name(name: str) -> bool:
    """Return True when ``name`` is a pre-commit.ci check.

    pre-commit.ci reports as ``pre-commit.ci - pr`` (and the ``- ci``
    variant).  It is excluded from this detector because dependabot's
    ``recreate`` macro does not retrigger it;
    :meth:`_trigger_stale_precommit_ci` handles it via the
    ``pre-commit.ci run`` comment instead.
    """
    n = (name or "").strip().lower()
    return "pre-commit.ci" in n or "pre-commit-ci" in n


def _is_eligible(name: str, required_contexts: set[str]) -> bool:
    """Return True when a stuck ``name`` should drive recreate.

    Eligible when the check is required on the base branch or is a
    DCO-shaped check, and is *not* a pre-commit.ci check (handled
    separately).

    Restricting to required checks means a non-blocking check is never
    treated as stuck, since it cannot block the merge.  DCO-shaped
    checks are additionally always treated as eligible as a safety
    net: the GitHub DCO App check is the canonical stuck-check case
    and is effectively always blocking where it is enabled, even when
    the required-checks lookup cannot enumerate it.
    """
    if _is_precommit_name(name):
        return False
    return (name or "").strip().lower() in required_contexts or _is_dco_name(name)


def _pr_reference_time(
    pr_data: Any, now: datetime, threshold: float
) -> datetime | None:
    """Return the PR's ``updated_at`` once it is old enough to judge.

    The age floor on PR ``created_at`` / ``updated_at`` avoids false
    positives on PRs that were touched seconds before we observed them
    — in those cases the check is simply running normally and should
    be allowed to finish.  None means the PR is too fresh, or carries
    no usable timing data, in which case the detector fails closed.
    """
    if not isinstance(pr_data, dict):
        return None

    pr_created = _parse_ts(pr_data.get("created_at"))
    pr_updated = _parse_ts(pr_data.get("updated_at"))
    if pr_created is None or pr_updated is None:
        return None

    pr_age = (now - pr_created).total_seconds()
    pr_idle = (now - pr_updated).total_seconds()
    if pr_age < threshold or pr_idle < threshold:
        return None
    return pr_updated


def _stuck_run_candidate(
    runs: Any, ctx: _StuckCheckContext
) -> tuple[str | None, float]:
    """Find the longest-pending eligible check run, if any."""
    candidate_name: str | None = None
    candidate_age = 0.0
    if not isinstance(runs, dict):
        return candidate_name, candidate_age
    for run in runs.get("check_runs") or []:
        if not isinstance(run, dict):
            continue
        name = run.get("name", "")
        if not _is_eligible(name, ctx.required_contexts):
            continue
        status = run.get("status")
        if status not in ("queued", "in_progress"):
            continue
        started = _parse_ts(run.get("started_at"))
        # Use the *latest* of started_at and PR updated_at as the
        # reference so a stale started_at left over from a prior head
        # SHA does not inflate the age.
        ref = max(started, ctx.pr_updated) if started else ctx.pr_updated
        age = (ctx.now - ref).total_seconds()
        if age >= ctx.threshold and age > candidate_age:
            candidate_name = name
            candidate_age = age
    return candidate_name, candidate_age


def _stuck_status_candidate(
    statuses: Any, ctx: _StuckCheckContext
) -> tuple[str | None, float]:
    """Find the longest-pending eligible status context, if any."""
    candidate_name: str | None = None
    candidate_age = 0.0
    if not isinstance(statuses, dict):
        return candidate_name, candidate_age
    for s in statuses.get("statuses") or []:
        if not isinstance(s, dict):
            continue
        ctx_name = s.get("context", "")
        if not _is_eligible(ctx_name, ctx.required_contexts):
            continue
        if s.get("state") != "pending":
            continue
        updated = _parse_ts(s.get("updated_at")) or ctx.pr_updated
        ref = max(updated, ctx.pr_updated)
        age = (ctx.now - ref).total_seconds()
        if age >= ctx.threshold and age > candidate_age:
            candidate_name = ctx_name
            candidate_age = age
    return candidate_name, candidate_age


class _StuckCheckMixin(_MergeManagerBase):
    """Recognising a required check that has stopped making progress."""

    async def _detect_stuck_required_check(
        self,
        pr_info: PullRequestInfo,
    ) -> tuple[bool, str | None, float]:
        """Detect whether a *required* verification check is stuck.

        Required checks (DCO, lint, build, license scans, etc.)
        normally start reporting within a handful of seconds.  When
        one has been queued / in-progress / pending for longer than
        :data:`STUCK_CHECK_THRESHOLD_SECONDS` on a PR that itself was
        created and last updated more than that long ago, treat it
        as stuck so the caller can decide whether to ask dependabot
        to recreate the PR (the only reliable recovery for a
        dependabot PR with no ``recreate``/``rebase`` macro of its
        own once a required check has stalled indefinitely).

        :func:`_is_eligible` decides which checks count, and
        :func:`_pr_reference_time` applies the PR-level age floor.

        Args:
            pr_info: The pull request being evaluated.

        Returns:
            A 3-tuple ``(is_stuck, check_name, age_seconds)``.
            ``check_name`` is the GitHub check / status name of the
            stuck check (or ``None`` when no stuck check was found).
            ``age_seconds`` is the time the check has been pending
            (or ``0.0`` when no candidate check was found).
        """
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        if not self._github_client:
            return False, None, 0.0

        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)
        threshold = _mm.STUCK_CHECK_THRESHOLD_SECONDS
        now = datetime.now(timezone.utc)

        # 1. PR-level age floor — don't fire on PRs we caught right
        #    after they were opened or force-pushed; checks on those
        #    are simply running normally.
        pr_data = await self._stuck_check_pr_data(repo_owner, repo_name, pr_info)
        pr_updated = _pr_reference_time(pr_data, now, threshold)
        if pr_updated is None:
            return False, None, 0.0

        # 2. Determine which checks are *required* on the base branch
        #    so a non-blocking check is never treated as stuck.
        required_contexts = await self._stuck_required_contexts(
            repo_owner, repo_name, pr_info
        )
        ctx = _StuckCheckContext(now, pr_updated, threshold, required_contexts)

        # 3. Examine check-runs and status contexts on the head SHA.
        runs = await self._stuck_check_runs(repo_owner, repo_name, pr_info)
        candidate_name, candidate_age = _stuck_run_candidate(runs, ctx)

        statuses = await self._stuck_check_statuses(repo_owner, repo_name, pr_info)
        status_name, status_age = _stuck_status_candidate(statuses, ctx)
        if status_age > candidate_age:
            candidate_name, candidate_age = status_name, status_age

        if candidate_name is None:
            return False, None, 0.0
        return True, candidate_name, candidate_age

    async def _stuck_check_pr_data(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> Any:
        """Fetch the PR, returning None when it cannot be read."""
        if not self._github_client:
            return None
        try:
            return await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/pulls/{pr_info.number}"
            )
        except Exception as exc:
            self.log.debug(
                "_detect_stuck_required_check: pr fetch failed for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return None

    async def _stuck_required_contexts(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> set[str]:
        """Return the base branch's required status contexts.

        On any failure this falls back to an empty set, leaving the DCO
        safety net in :func:`_is_eligible` as the only eligible matcher.
        """
        required_contexts: set[str] = set()
        if not self._github_client:
            return required_contexts
        try:
            required = await self._github_client.get_required_status_checks(
                repo_owner, repo_name, pr_info.base_branch or "main"
            )
            if isinstance(required, list):
                required_contexts = {
                    str(c.get("context", "")).strip().lower()
                    for c in required
                    if isinstance(c, dict) and c.get("context")
                }
        except Exception as exc:
            self.log.debug(
                "_detect_stuck_required_check: required-checks fetch failed "
                "for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            required_contexts = set()
        return required_contexts

    async def _stuck_check_runs(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> Any:
        """Fetch the head SHA's check runs, returning None on failure."""
        if not self._github_client:
            return None
        try:
            return await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}/check-runs"
            )
        except Exception as exc:
            self.log.debug(
                "_detect_stuck_required_check: check-runs fetch failed for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return None

    async def _stuck_check_statuses(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> Any:
        """Fetch the head SHA's status contexts, returning None on failure."""
        if not self._github_client:
            return None
        try:
            return await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}/status"
            )
        except Exception as exc:
            self.log.debug(
                "_detect_stuck_required_check: status fetch failed for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return None
