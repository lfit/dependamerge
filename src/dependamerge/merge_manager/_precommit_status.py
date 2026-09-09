# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Reading what pre-commit.ci has said about a commit.

pre-commit.ci reports through the commit *status* API rather than as a
check run, so answering "what happened" means picking one context out of
a combined-status payload and interpreting its state and timestamps.

Separated from the retrigger logic that consumes it: this module answers
"what does the payload say", while :mod:`_precommit_ci` answers "and
what should we do about it".  Everything here is pure and directly
testable, with no client and no side effects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

PRECOMMIT_CONTEXT = "pre-commit.ci - pr"


def find_precommit_status(status_data: Any) -> dict[str, Any] | None:
    """Pick the pre-commit.ci entry out of a combined status response."""
    if not isinstance(status_data, dict):
        return None
    for s in status_data.get("statuses", []):
        if isinstance(s, dict) and s.get("context") == PRECOMMIT_CONTEXT:
            return s
    return None


def pending_age(precommit_status: dict[str, Any], now: datetime) -> float | None:
    """Seconds the status has been pending, or None when unknowable.

    Uses ``updated_at`` (when pre-commit.ci set the pending status),
    falling back to ``created_at``.
    """
    raw_ts = precommit_status.get("updated_at") or precommit_status.get("created_at")
    if not isinstance(raw_ts, str) or not raw_ts:
        return None
    try:
        ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        return (now - ts).total_seconds()
    except (ValueError, TypeError):
        # ``ValueError``: unparsable timestamp.
        # ``TypeError``: a timestamp lacking tz info parses to a naive
        # datetime, which cannot be subtracted from the tz-aware
        # ``now``.  Either way, degrade to ``None`` (fail closed)
        # rather than abort the run.
        return None


def precommit_outcome(status_data: Any, since: datetime | None = None) -> bool | None:
    """Report a settled pre-commit.ci result, or None while pending.

    Scans past pending entries rather than stopping at the first match,
    so a later context with a final state is still honoured.

    *since* is the moment of the status a retrigger was posted in
    response to.  A terminal state no newer than that is the **same
    reading** the nudge was meant to replace: pre-commit.ci has not
    reacted yet, and reporting it would end the wait on the very error
    being retried.  Such a status reads as pending, so the poll keeps
    going until a genuinely newer result appears.
    """
    if not isinstance(status_data, dict):
        return None
    for s in status_data.get("statuses", []):
        if not isinstance(s, dict):
            continue
        if s.get("context") != PRECOMMIT_CONTEXT:
            continue
        state = s.get("state")
        if state not in ("success", "failure", "error"):
            # state == "pending" --- keep polling
            continue
        if since is not None:
            reported = parse_timestamp(s.get("updated_at") or s.get("created_at"))
            if reported is not None and reported <= since:
                # The pre-trigger reading, unchanged.
                continue
        return bool(state == "success")
    return None


def has_precommit_trigger_comment(comments: Any, since: datetime | None = None) -> bool:
    """Report whether a ``pre-commit.ci run`` comment already exists.

    When *since* is given, only comments posted after it count.  A nudge
    suppresses a second nudge for the *same* incident, not for the life
    of the pull request: pre-commit.ci can error again on a later head,
    after an intervening run succeeded, and a comment from the previous
    episode is no reason to leave the new one stalled.

    A comment whose timestamp cannot be read counts, which keeps an
    unparsable payload on the side of not commenting twice.
    """
    if not isinstance(comments, list):
        return False
    for c in comments:
        if not isinstance(c, dict):
            continue
        body = c.get("body")
        if not (isinstance(body, str) and body.strip() == "pre-commit.ci run"):
            continue
        if since is None:
            return True
        posted = parse_timestamp(c.get("created_at"))
        if posted is None or posted >= since:
            return True
    return False


def parse_timestamp(raw: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp, tolerating ``Z`` and absence."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
