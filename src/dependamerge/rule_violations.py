# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Parsing GitHub's repository rule-violation messages.

When a merge is refused by a ruleset, GitHub returns a single prose
string naming the offending conditions::

    Repository rule violations found
    Required workflows 'AI Slop Scan 🧹, Zizmor Scan 🌈' are not satisfied

Two very different consumers need the names out of that string: the CLI,
to render one bullet per condition, and the merge pipeline, to decide
whether waiting can possibly help.  Parsing it in both places invites
them to drift apart, so the extraction lives here --- pure, shared and
directly testable.

The message is also the *more dependable* source for required-workflow
names than enumerating org rulesets: during the 503-PR run analysed in
``docs/BULK_RUN_PERFORMANCE_AUDIT.md``, ``GET /orgs/{org}/rulesets``
returned 403 while this string was always present on the rejection.
"""

from __future__ import annotations

import re

__all__ = [
    "RULE_VIOLATION_MARKER",
    "is_rule_violation",
    "required_workflow_names",
    "required_status_check_names",
    "violation_verb",
]

RULE_VIOLATION_MARKER = "Repository rule violations found"

_WORKFLOW_MARKER = "Required workflows "
_STATUS_CHECK_MARKER = "Required status check"


def is_rule_violation(reason: str) -> bool:
    """Whether *reason* is a ruleset rejection."""
    return RULE_VIOLATION_MARKER in (reason or "")


def violation_verb(reason: str) -> str:
    """``\"failed\"`` when a condition ran and failed, else ``\"not satisfied\"``.

    The distinction matters: *failed* means a workflow ran and reported
    failure, which retrying cannot fix, whereas *not satisfied* can mean
    it is still running --- or has never started at all.
    """
    return "failed" if "fail" in (reason or "").lower() else "not satisfied"


def required_workflow_names(reason: str) -> list[str]:
    """Workflow names quoted in a ``Required workflows '…'`` clause.

    GitHub can repeat a name within one violation string, so duplicates
    are collapsed while first-seen order is preserved --- callers render
    these as a bullet list.
    """
    if not reason or _WORKFLOW_MARKER not in reason:
        return []
    after_marker = reason.split(_WORKFLOW_MARKER, 1)[1]
    if "'" not in after_marker:
        return []
    _, _, after_first = after_marker.partition("'")
    quoted, _, _rest = after_first.partition("'")
    names = [name.strip() for name in quoted.split(",") if name.strip()]
    return list(dict.fromkeys(names))


def required_status_check_names(reason: str) -> list[str]:
    """Context names quoted in a ``Required status check \"…\"`` clause."""
    if not reason or _STATUS_CHECK_MARKER not in reason:
        return []
    names = [c.strip() for c in re.findall(r'"([^"]+)"', reason) if c.strip()]
    return list(dict.fromkeys(names))
