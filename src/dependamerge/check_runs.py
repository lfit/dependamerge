# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Deduplication of check runs reported against a single commit.

GitHub can attach several check runs carrying the *same name* to one
commit.  The common cause is a superseded run: two workflow events fire
for the same head SHA (for example a ``pull_request`` event delivered
twice with different attributed actors), the workflow's ``concurrency``
group cancels the first, and the cancelled run stays attached to the
commit alongside the successful one.

Treating every reported run as authoritative makes a commit look broken
when it is not: a cancelled, superseded run sits next to a successful
run of the same check.  GitHub's own merge protection evaluates the
latest run per check name, and this module does the same.

The helpers here are deliberately pure and transport-agnostic so the
REST and GraphQL code paths can share one definition of "did this check
pass".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

__all__ = [
    "FAILING_CONCLUSIONS",
    "failing_check_names",
    "latest_check_run_per_name",
]

# Conclusions that mark a check run as not having succeeded.  "cancelled"
# is included because a genuinely cancelled *latest* run should block;
# it is only harmless when superseded, which the deduplication below
# resolves before this set is consulted.
FAILING_CONCLUSIONS = frozenset({"failure", "cancelled", "timed_out"})


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp, tolerating ``Z`` and ``None``."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _run_timestamp(run: Mapping[str, Any]) -> datetime | None:
    """Best available completion time for *run*.

    Prefers completion over start: two runs may start in the same second
    while their completion times still order them correctly.  Accepts
    both REST (``completed_at``) and GraphQL (``completedAt``) spellings.
    """
    for key in ("completed_at", "completedAt", "started_at", "startedAt"):
        parsed = _parse_timestamp(run.get(key))
        if parsed is not None:
            return parsed
    return None


def _is_failing(run: Mapping[str, Any]) -> bool:
    return (run.get("conclusion") or "").strip().lower() in FAILING_CONCLUSIONS


def _supersedes(candidate: Mapping[str, Any], incumbent: Mapping[str, Any]) -> bool:
    """Whether *candidate* should replace *incumbent* as the latest run.

    Ordering is by timestamp.  When timestamps are absent or identical --
    which happens when the API does not expose them, or when duplicate
    runs complete within the same second -- a successful run wins over a
    failing one.  That tie-break is what resolves the superseded-cancel
    race: the surviving run is the one that actually ran to completion.
    """
    candidate_ts = _run_timestamp(candidate)
    incumbent_ts = _run_timestamp(incumbent)

    if candidate_ts is not None and incumbent_ts is not None:
        if candidate_ts != incumbent_ts:
            return candidate_ts > incumbent_ts
    elif candidate_ts is not None or incumbent_ts is not None:
        # Only one side is timestamped.  Prefer the timestamped run: it
        # carries strictly more information than one that reports none.
        return candidate_ts is not None

    # Indistinguishable by time; prefer a run that did not fail.
    return _is_failing(incumbent) and not _is_failing(candidate)


def latest_check_run_per_name(
    runs: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Collapse *runs* to the latest run for each check name.

    Runs without a name are ignored: they cannot be matched against a
    required-check rule, so they carry no signal here.
    """
    latest: dict[str, Mapping[str, Any]] = {}
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        name = (run.get("name") or "").strip()
        if not name:
            continue
        incumbent = latest.get(name)
        if incumbent is None or _supersedes(run, incumbent):
            latest[name] = run
    return latest


def failing_check_names(runs: Iterable[Mapping[str, Any]]) -> list[str]:
    """Names whose *latest* run did not succeed, in first-seen order.

    A name is reported at most once even when several runs carry it.
    """
    materialised = [run for run in runs if isinstance(run, Mapping)]
    latest = latest_check_run_per_name(materialised)

    ordered: list[str] = []
    seen: set[str] = set()
    for run in materialised:
        name = (run.get("name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)

    return [name for name in ordered if _is_failing(latest[name])]
