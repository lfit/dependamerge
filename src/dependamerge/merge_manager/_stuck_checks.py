# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Detection of a required check that has stopped reporting.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._constants import (
    STUCK_CHECK_THRESHOLD_SECONDS,
)


class _StuckChecksMixin(_MergeManagerBase):
    """Detection of a required check that has stopped reporting."""

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

        Only checks that are *required* on the PR's base branch are
        considered, since a non-required check cannot block the
        merge.  DCO-shaped checks are additionally always treated as
        eligible as a safety net: the GitHub DCO App check is the
        canonical stuck-check case and is effectively always blocking
        where it is enabled, even when the required-checks lookup
        cannot enumerate it.

        ``pre-commit.ci`` checks are explicitly excluded here even
        when required — they have their own dedicated recovery via
        :meth:`_trigger_stale_precommit_ci`, which posts the
        ``pre-commit.ci run`` comment (dependabot's ``recreate`` macro
        does not retrigger pre-commit.ci).

        The age floor on PR ``created_at`` / ``updated_at`` avoids
        false positives on PRs that were touched seconds before we
        observed them — in those cases the check is simply running
        normally and should be allowed to finish.

        Args:
            pr_info: The pull request being evaluated.

        Returns:
            A 3-tuple ``(is_stuck, check_name, age_seconds)``.
            ``check_name`` is the GitHub check / status name of the
            stuck check (or ``None`` when no stuck check was found).
            ``age_seconds`` is the time the check has been pending
            (or ``0.0`` when no candidate check was found).
        """
        if not self._github_client:
            return False, None, 0.0

        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)
        threshold = STUCK_CHECK_THRESHOLD_SECONDS

        from datetime import datetime, timezone

        def _parse_ts(value: Any) -> datetime | None:
            if not isinstance(value, str) or not value:
                return None
            try:
                # GitHub returns RFC 3339 with a trailing ``Z``.
                ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            # A timestamp without tz info parses to a naive datetime,
            # which raises ``TypeError`` when subtracted from the
            # tz-aware ``now`` below.  Treat it as unparsable (fail
            # closed) so the detector degrades gracefully instead of
            # aborting the merge run.
            if ts.tzinfo is None:
                return None
            return ts

        def _is_dco_name(name: str) -> bool:
            """Return True when ``name`` looks like a DCO check.

            Matches the common variants emitted by the GitHub DCO
            App and similar bots: ``DCO``, ``dco/dco``, ``dcobot``,
            and any name containing ``signoff`` / ``sign-off`` /
            ``signed-off`` (case-insensitive).
            """
            n = (name or "").strip().lower()
            if not n:
                return False
            if n in {"dco", "dco/dco", "dcobot"} or n.startswith("dco/"):
                return True
            return "signoff" in n or "sign-off" in n or "signed-off" in n

        def _is_precommit_name(name: str) -> bool:
            """Return True when ``name`` is a pre-commit.ci check.

            pre-commit.ci reports as ``pre-commit.ci - pr`` (and the
            ``- ci`` variant).  It is excluded from this detector
            because dependabot's ``recreate`` macro does not
            retrigger it; ``_trigger_stale_precommit_ci`` handles it
            via the ``pre-commit.ci run`` comment instead.
            """
            n = (name or "").strip().lower()
            return "pre-commit.ci" in n or "pre-commit-ci" in n

        # 1. PR-level age floor — don't fire on PRs we caught right
        #    after they were opened or force-pushed; checks on those
        #    are simply running normally.
        now = datetime.now(timezone.utc)
        try:
            pr_data = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/pulls/{pr_info.number}"
            )
        except Exception as exc:
            self.log.debug(
                "_detect_stuck_required_check: pr fetch failed for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return False, None, 0.0

        if not isinstance(pr_data, dict):
            return False, None, 0.0

        pr_created = _parse_ts(pr_data.get("created_at"))
        pr_updated = _parse_ts(pr_data.get("updated_at"))
        if pr_created is None or pr_updated is None:
            # Without timing data we cannot safely judge stuckness;
            # fail closed.
            return False, None, 0.0

        pr_age = (now - pr_created).total_seconds()
        pr_idle = (now - pr_updated).total_seconds()
        if pr_age < threshold or pr_idle < threshold:
            return False, None, 0.0

        # 2. Determine which checks are *required* on the base branch
        #    so a non-blocking check is never treated as stuck.  On
        #    any failure we fall back to an empty set, leaving the
        #    DCO safety net (below) as the only eligible matcher.
        required_contexts: set[str] = set()
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

        def _is_eligible(name: str) -> bool:
            """Return True when a stuck ``name`` should drive recreate.

            Eligible when the check is required on the base branch or
            is a DCO-shaped check (safety net), and is *not* a
            pre-commit.ci check (handled separately).
            """
            if _is_precommit_name(name):
                return False
            return (name or "").strip().lower() in required_contexts or _is_dco_name(
                name
            )

        # 3. Examine check-runs and status contexts on the head SHA.
        candidate_name: str | None = None
        candidate_age: float = 0.0

        try:
            runs = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}/check-runs"
            )
        except Exception as exc:
            self.log.debug(
                "_detect_stuck_required_check: check-runs fetch failed for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            runs = None

        if isinstance(runs, dict):
            for run in runs.get("check_runs") or []:
                if not isinstance(run, dict):
                    continue
                name = run.get("name", "")
                if not _is_eligible(name):
                    continue
                status = run.get("status")
                if status not in ("queued", "in_progress"):
                    continue
                started = _parse_ts(run.get("started_at"))
                # Use the *latest* of started_at and PR updated_at
                # as the reference so a stale started_at left over
                # from a prior head SHA does not inflate the age.
                ref = max(started, pr_updated) if started else pr_updated
                age = (now - ref).total_seconds()
                if age >= threshold and age > candidate_age:
                    candidate_name = name
                    candidate_age = age

        try:
            statuses = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}/status"
            )
        except Exception as exc:
            self.log.debug(
                "_detect_stuck_required_check: status fetch failed for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            statuses = None

        if isinstance(statuses, dict):
            for s in statuses.get("statuses") or []:
                if not isinstance(s, dict):
                    continue
                ctx = s.get("context", "")
                if not _is_eligible(ctx):
                    continue
                if s.get("state") != "pending":
                    continue
                updated = _parse_ts(s.get("updated_at")) or pr_updated
                ref = max(updated, pr_updated)
                age = (now - ref).total_seconds()
                if age >= threshold and age > candidate_age:
                    candidate_name = ctx
                    candidate_age = age

        if candidate_name is None:
            return False, None, 0.0
        return True, candidate_name, candidate_age
