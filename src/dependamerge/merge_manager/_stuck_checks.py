# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Detection of a required check that has stopped reporting.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._constants import (
    STUCK_CHECK_THRESHOLD_SECONDS,
)

if TYPE_CHECKING:
    from datetime import datetime

    from ..github_async import GitHubAsync


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
        client = self._github_client
        if not client:
            return False, None, 0.0

        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        pr_updated = await self._stuck_check_pr_reference(
            client, pr_info, repo_owner, repo_name, now
        )
        if pr_updated is None:
            return False, None, 0.0

        required_contexts = await self._required_stuck_check_contexts(
            client, pr_info, repo_owner, repo_name
        )

        runs = await self._fetch_stuck_check_runs(
            client, pr_info, repo_owner, repo_name
        )
        candidate_name, candidate_age = self._stuck_check_run_candidate(
            runs, required_contexts, now, pr_updated
        )

        statuses = await self._fetch_stuck_commit_statuses(
            client, pr_info, repo_owner, repo_name
        )
        status_name, status_age = self._stuck_status_context_candidate(
            statuses, required_contexts, now, pr_updated
        )
        if status_age > candidate_age:
            candidate_name, candidate_age = status_name, status_age

        if candidate_name is None:
            return False, None, 0.0
        return True, candidate_name, candidate_age

    async def _stuck_check_pr_reference(
        self,
        client: GitHubAsync,
        pr_info: PullRequestInfo,
        repo_owner: str,
        repo_name: str,
        now: datetime,
    ) -> datetime | None:
        """Return the PR's ``updated_at`` once it clears the age floor.

        A PR opened or force-pushed less than
        :data:`STUCK_CHECK_THRESHOLD_SECONDS` before ``now`` has checks
        that are simply running normally, so this answers ``None`` and
        none of its checks are examined.  ``None`` also covers a PR
        fetch that failed and timestamps that would not parse: without
        timing data stuckness cannot be judged, so the detector fails
        closed.

        Held apart from the check scans because the value it returns is
        both the gate and the reference every per-check age is later
        measured from.  ``now`` is sampled by the caller before this
        fetch, so the fetch's own latency counts towards the ages.
        """
        threshold = STUCK_CHECK_THRESHOLD_SECONDS
        try:
            pr_data = await client.get(
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

        if not isinstance(pr_data, dict):
            return None

        pr_created = self._parse_check_timestamp(pr_data.get("created_at"))
        pr_updated = self._parse_check_timestamp(pr_data.get("updated_at"))
        if pr_created is None or pr_updated is None:
            return None

        pr_age = (now - pr_created).total_seconds()
        pr_idle = (now - pr_updated).total_seconds()
        if pr_age < threshold or pr_idle < threshold:
            return None
        return pr_updated

    @staticmethod
    def _parse_check_timestamp(value: Any) -> datetime | None:
        """Parse a GitHub RFC 3339 timestamp, or answer ``None``.

        ``None`` means "no usable timestamp", which every caller reads
        as too little evidence to judge stuckness.  A value that will
        not parse and one that parses to a naive datetime are equally
        unusable — the latter raises ``TypeError`` when subtracted from
        the tz-aware ``now`` — so both fail closed here, in one place,
        letting the detector degrade gracefully instead of aborting the
        merge run.
        """
        from datetime import datetime

        if not isinstance(value, str) or not value:
            return None
        try:
            # GitHub returns RFC 3339 with a trailing ``Z``.
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if ts.tzinfo is None:
            return None
        return ts

    async def _required_stuck_check_contexts(
        self,
        client: GitHubAsync,
        pr_info: PullRequestInfo,
        repo_owner: str,
        repo_name: str,
    ) -> set[str]:
        """Return the lower-cased contexts required on the base branch.

        A check that cannot block the merge is never worth calling
        stuck, so this set decides what the scans may consider.  A
        branch protection lookup that fails degrades to an empty set
        rather than an error, leaving the DCO safety net in
        :meth:`_is_eligible_stuck_check` as the only eligible matcher —
        the conservative outcome, and the reason the failure is
        swallowed here rather than surfaced.
        """
        required_contexts: set[str] = set()
        try:
            required = await client.get_required_status_checks(
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

    def _is_eligible_stuck_check(self, name: str, required_contexts: set[str]) -> bool:
        """Return True when a stuck ``name`` should drive recreate.

        Eligible when the check is required on the base branch or is a
        DCO-shaped check (safety net), and is *not* a pre-commit.ci
        check (handled separately).  Both scans ask the same question,
        so the rule lives in one place.
        """
        if self._is_precommit_check_name(name):
            return False
        normalised = (name or "").strip().lower()
        return normalised in required_contexts or self._is_dco_check_name(name)

    @staticmethod
    def _is_dco_check_name(name: str) -> bool:
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

    @staticmethod
    def _is_precommit_check_name(name: str) -> bool:
        """Return True when ``name`` is a pre-commit.ci check.

        pre-commit.ci reports as ``pre-commit.ci - pr`` (and the
        ``- ci`` variant).  It is excluded from this detector because
        dependabot's ``recreate`` macro does not retrigger it;
        ``_trigger_stale_precommit_ci`` handles it via the
        ``pre-commit.ci run`` comment instead.
        """
        n = (name or "").strip().lower()
        return "pre-commit.ci" in n or "pre-commit-ci" in n

    async def _fetch_stuck_check_runs(
        self,
        client: GitHubAsync,
        pr_info: PullRequestInfo,
        repo_owner: str,
        repo_name: str,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Read the head SHA's check-runs, or ``None`` on failure.

        The commit status contexts are still worth scanning when this
        half of the picture cannot be read, so a failed fetch is logged
        and answered with ``None`` rather than ending the detection.
        """
        try:
            return await client.get(
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

    def _stuck_check_run_candidate(
        self,
        runs: dict[str, Any] | list[dict[str, Any]] | None,
        required_contexts: set[str],
        now: datetime,
        pr_updated: datetime,
    ) -> tuple[str | None, float]:
        """Return the longest-stuck eligible check-run and its age.

        Answers ``(None, 0.0)`` when nothing eligible has been queued or
        in progress for longer than
        :data:`STUCK_CHECK_THRESHOLD_SECONDS`.  Separate from the status
        context scan because the two APIs describe pendency
        differently, and collapsing them would hide which timestamp is
        compared against what.
        """
        candidate_name: str | None = None
        candidate_age: float = 0.0
        if not isinstance(runs, dict):
            return candidate_name, candidate_age

        for run in runs.get("check_runs") or []:
            if not isinstance(run, dict):
                continue
            name = run.get("name", "")
            if not self._is_eligible_stuck_check(name, required_contexts):
                continue
            status = run.get("status")
            if status not in ("queued", "in_progress"):
                continue
            started = self._parse_check_timestamp(run.get("started_at"))
            # Use the *latest* of started_at and PR updated_at
            # as the reference so a stale started_at left over
            # from a prior head SHA does not inflate the age.
            ref = max(started, pr_updated) if started else pr_updated
            age = (now - ref).total_seconds()
            if age >= STUCK_CHECK_THRESHOLD_SECONDS and age > candidate_age:
                candidate_name = name
                candidate_age = age
        return candidate_name, candidate_age

    async def _fetch_stuck_commit_statuses(
        self,
        client: GitHubAsync,
        pr_info: PullRequestInfo,
        repo_owner: str,
        repo_name: str,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Read the head SHA's status contexts, or ``None`` on failure.

        Any check-run candidate already found still stands when this
        fetch fails, so the failure is logged and answered with ``None``
        rather than discarding the detection so far.
        """
        try:
            return await client.get(
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

    def _stuck_status_context_candidate(
        self,
        statuses: dict[str, Any] | list[dict[str, Any]] | None,
        required_contexts: set[str],
        now: datetime,
        pr_updated: datetime,
    ) -> tuple[str | None, float]:
        """Return the longest-stuck eligible status context and its age.

        The older commit status API reports ``pending`` rather than
        ``queued`` / ``in_progress`` and carries ``updated_at`` rather
        than ``started_at``, so its scan stays distinct from the
        check-run one.  The caller keeps whichever of the two candidates
        is older, so ``(None, 0.0)`` here leaves any check-run candidate
        untouched.
        """
        candidate_name: str | None = None
        candidate_age: float = 0.0
        if not isinstance(statuses, dict):
            return candidate_name, candidate_age

        for s in statuses.get("statuses") or []:
            if not isinstance(s, dict):
                continue
            ctx = s.get("context", "")
            if not self._is_eligible_stuck_check(ctx, required_contexts):
                continue
            if s.get("state") != "pending":
                continue
            updated = self._parse_check_timestamp(s.get("updated_at")) or pr_updated
            ref = max(updated, pr_updated)
            age = (now - ref).total_seconds()
            if age >= STUCK_CHECK_THRESHOLD_SECONDS and age > candidate_age:
                candidate_name = ctx
                candidate_age = age
        return candidate_name, candidate_age
