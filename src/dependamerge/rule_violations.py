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
    "status_check_violation_verb",
    "violation_verb",
    "workflow_name_fragments",
]

RULE_VIOLATION_MARKER = "Repository rule violations found"

_WORKFLOW_MARKER = "Required workflows "
_STATUS_CHECK_MARKER = "Required status check"

# GitHub states the outcome immediately after the closing quote, as in
# ``' are not satisfied`` or ``' failed``.  Anchoring on that wording is
# what lets the closing delimiter be found without assuming the names
# contain no apostrophe of their own.
_OUTCOME_RE = re.compile(
    r"\s*(?:are|is|was|were|have|has)?\s*(?:not satisfied|failed|fail)",
    re.IGNORECASE,
)


def _split_workflow_names(reason: str) -> tuple[str, str] | None:
    """Split ``Required workflows 'A, B' are not satisfied`` in two.

    Returns ``(names, outcome)`` --- the raw comma-separated list and the
    clause saying what became of it --- or ``None`` when *reason* is not
    that shape.

    A workflow name is an arbitrary Actions ``name:`` value and GitHub
    does not escape the quote it wraps the list in, so a workflow called
    ``Don't Fail`` puts an apostrophe *inside* it.  Treating the first
    apostrophe as the closing delimiter would then yield the name ``Don``
    and the outcome ``t Fail' are not satisfied`` --- read as "failed",
    which would skip the recovery path for a workflow that had merely
    not started yet.

    The closing delimiter is therefore the *last* apostrophe an
    *outcome* follows.  Last rather than first because a name may
    contain a quoted phrase of its own: in ``'CI 'Fail Fast'' are not
    satisfied`` the inner quote is followed by ``Fail Fast``, which the
    outcome pattern matches, so taking the first would cut the name
    short at ``CI`` and read the rest as a failure.  Trailing prose is
    safe from the later choice: an apostrophe in ``GitHub's ruleset``
    has no outcome after it and never qualifies.  When no candidate
    qualifies the first apostrophe is used, preserving the previous
    behaviour for message shapes not seen here.
    """
    if not reason or _WORKFLOW_MARKER not in reason:
        return None
    after_marker = reason.split(_WORKFLOW_MARKER, 1)[1]
    if "'" not in after_marker:
        return None
    _, _, body = after_marker.partition("'")
    closing = -1
    for index, char in enumerate(body):
        if char == "'" and _OUTCOME_RE.match(body, index + 1):
            closing = index
    if closing == -1:
        closing = body.find("'")
    if closing == -1:
        return body, ""
    return body[:closing], body[closing + 1 :]


def is_rule_violation(reason: str) -> bool:
    """Whether *reason* is a ruleset rejection."""
    return RULE_VIOLATION_MARKER in (reason or "")


def violation_verb(reason: str) -> str:
    """``"failed"`` when a condition ran and failed, else ``"not satisfied"``.

    The distinction matters: *failed* means a workflow ran and reported
    failure, which retrying cannot fix, whereas *not satisfied* can mean
    it is still running --- or has never started at all.

    Only the text **after** the quoted condition names is inspected.
    The enclosing exception always begins ``Failed to merge PR …``, so
    scanning the whole string would classify every rejection as
    ``failed``; a condition *name* containing "fail" would do the same.
    """
    clause = _verb_clause(reason)
    return "failed" if "fail" in clause.lower() else "not satisfied"


def _verb_clause(reason: str) -> str:
    """The portion of *reason* that states the outcome.

    For a workflow violation that is whatever follows the closing quote
    of the name list (``' are not satisfied``), located by
    :func:`_split_workflow_names` so an apostrophe inside a name cannot
    be mistaken for it.  For a status-check violation the names are
    individually quoted, so the text after the final quote is used.
    Falls back to the whole string when neither shape is recognised.

    One rejection can name *both* kinds, in which case the workflow
    clause runs on into the status-check one.  The tail is therefore cut
    at the status-check marker, so a status context that has already
    failed cannot report the workflows as failed when they are merely
    unfinished --- the exact pair that arises while required workflows
    are still queued.
    """
    if not reason:
        return ""
    workflow = _split_workflow_names(reason)
    if workflow is not None:
        return workflow[1].split(_STATUS_CHECK_MARKER, 1)[0]
    if _STATUS_CHECK_MARKER in reason:
        # ``Required status check "X" is failing.`` --- take everything
        # after the last quoted name.
        idx = reason.rfind('"')
        if idx != -1:
            return reason[idx + 1 :]
        return reason[reason.find(_STATUS_CHECK_MARKER) :]
    return reason


def workflow_name_fragments(reason: str) -> list[str]:
    """The comma-separated pieces of the quoted list, in order.

    Unlike :func:`required_workflow_names` this keeps duplicates.  A
    workflow name may itself contain a comma, so the pieces are only a
    *guess* at the names --- and reconciling that guess against the runs
    that actually dispatched needs the sequence exactly as GitHub wrote
    it.  Collapsing ``'Build, Build'`` to a single piece makes that name
    impossible to rejoin, and a name that cannot be rejoined reads as
    never dispatched, which stops the wait on a workflow that ran.
    """
    workflow = _split_workflow_names(reason)
    if workflow is None:
        return []
    return [name.strip() for name in workflow[0].split(",") if name.strip()]


def required_workflow_names(reason: str) -> list[str]:
    """Workflow names quoted in a ``Required workflows '…'`` clause.

    GitHub can repeat a name within one violation string, so duplicates
    are collapsed while first-seen order is preserved --- callers render
    these as a bullet list.  Callers that need to *reconcile* the pieces
    against observed runs want :func:`workflow_name_fragments` instead.
    """
    return list(dict.fromkeys(workflow_name_fragments(reason)))


def required_status_check_names(reason: str) -> list[str]:
    """Context names quoted in a ``Required status check \"…\"`` clause."""
    if not reason or _STATUS_CHECK_MARKER not in reason:
        return []
    names = [c.strip() for c in re.findall(r'"([^"]+)"', reason) if c.strip()]
    return list(dict.fromkeys(names))


def status_check_violation_verb(reason: str) -> str:
    """The outcome stated for the ``Required status check`` clause.

    :func:`violation_verb` answers for the rejection as a whole and
    resolves a *workflow* clause in preference, so on a rejection naming
    both kinds it reports the workflows' outcome.  Applying that to the
    status checks is wrong whenever the two differ, which is the usual
    case: workflows that have not finished sit alongside a status
    context that has already failed.

    Status-check names are wrapped in double quotes and workflow names
    in single ones, so the last double quote closes the final context
    name.  The clause is cut at the workflow marker for the same reason
    :func:`_verb_clause` cuts at the status-check one --- either clause
    may come first, and neither may borrow the other's outcome.
    """
    if not reason or _STATUS_CHECK_MARKER not in reason:
        return "not satisfied"
    closing = reason.rfind('"')
    clause = (
        reason[closing + 1 :]
        if closing != -1
        else reason[reason.find(_STATUS_CHECK_MARKER) :]
    )
    clause = clause.split(_WORKFLOW_MARKER, 1)[0]
    return "failed" if "fail" in clause.lower() else "not satisfied"
