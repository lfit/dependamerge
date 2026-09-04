# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The failure reason a stored merge exception can be read off.

When a merge attempt raises, the exception text usually carries a more
actionable explanation than anything that can be inferred afterwards
from the PR's state, so it is consulted first.
"""

from __future__ import annotations

import re

from ..models import PullRequestInfo
from ._base import _MergeManagerBase

#: The HTTP status in an httpx error message, e.g. ``Client error '405
#: Method Not Allowed' for url '…'``.
_HTTP_STATUS_RE = re.compile(r"'(\d{3}) [^']*' for url")

#: Statuses on which GitHub reached a verdict about *the pull request*.
#: The merge endpoint returns 405 for "not mergeable" --- which is how a
#: ruleset violation arrives, body and all --- and 409 when the head has
#: moved under it.
#:
#: 422 is deliberately absent.  GitHub documents it as "Validation
#: Failed", which covers a malformed *request* --- a ``merge_method``
#: the repository does not allow, for instance --- and that is a
#: persistent configuration error, not a state that a later reading can
#: clear.  Treating it as a verdict would tell an operator to re-run
#: something that will fail identically every time.
#:
#: An allowlist rather than a list of exclusions.  Naming the statuses
#: that describe the attempt instead means every status nobody thought
#: of --- a 404 on the merge endpoint, a 410, whatever GitHub adds next
#: --- defaults to being treated as a verdict, and so becomes
#: withdrawable on a later clean reading.  The safe default is the other
#: way round: an unrecognised status keeps its message.
_STATE_VERDICT_STATUSES = frozenset({"405", "409"})


def _describes_the_attempt(error_msg: str) -> bool:
    """Whether the error is about the request rather than the PR.

    GitHub attaches a response body to transport failures too --- a 502
    arrives as ``Client error '502 Bad Gateway' … GitHub: Server
    Error``, a missing repository as ``GitHub: Not Found`` --- so the
    presence of a body says nothing about what the body is *about*.  The
    status does, and reading it here keeps a server or access error from
    being classified as a mergeability verdict merely because GitHub was
    polite enough to explain itself.

    A message carrying no recognisable status is left to the caller's
    other branches, which is why this returns ``False`` rather than
    guessing.
    """
    match = _HTTP_STATUS_RE.search(error_msg)
    if match is None:
        return False
    return match.group(1) not in _STATE_VERDICT_STATUSES


def _is_state_verdict(error_msg: str) -> bool:
    """Whether the status *positively* says GitHub judged the PR.

    The complement of :func:`_describes_the_attempt` only for messages
    that carry a status at all.  A message with none --- a retry-exhausted
    network timeout, say --- is neither: nothing about it says the pull
    request was judged, so a failure explained by it must not be
    withdrawn on a later clean reading.
    """
    match = _HTTP_STATUS_RE.search(error_msg)
    return match is not None and match.group(1) in _STATE_VERDICT_STATUSES


class _FailureSummaryFromExceptionMixin(_MergeManagerBase):
    """Reading a merge failure reason out of the exception that caused it."""

    def _failure_summary_from_exception(
        self,
        pr_key: str,
        last_exception: Exception,
        pr_info: PullRequestInfo,
    ) -> tuple[str, bool] | None:
        """
        Derive a failure reason from a stored merge exception.

        Args:
            pr_key: The ``owner/repo#number`` key the exception is stored under
            last_exception: The exception the last merge attempt raised
            pr_info: Pull request information

        Returns:
            A ``(reason, refused)`` pair, or None when the exception says
            nothing conclusive and the caller should fall back to
            state-based analysis.

            ``refused`` marks the reason as GitHub declining the merge
            **on the pull request's state** --- a ruleset violation, an
            unmet required check.  Only such a reason is a snapshot that
            can expire, so only such a reason may be withdrawn when a
            later reading finds the PR mergeable.  A missing token scope,
            a 403 or a 502 is a fact about the run or the transport: the
            PR looking clean afterwards says nothing about it, and
            withdrawing it would hide the one message that explains what
            to fix.
        """
        error_msg = str(last_exception)
        self.log.debug(f"Last exception for {pr_key}: {error_msg[:200]}")
        # The merge layer (github_async.merge_pull_request) embeds
        # GitHub's own explanation after a "GitHub: " marker — the
        # ruleset violation, "Required workflows ... are not
        # satisfied", required-check names, etc.  This is the
        # actionable cause, so surface it ahead of any generic
        # state-based inference.  We trim the PR-state context we
        # appended after it so the reason stays concise.
        marker = "GitHub: "
        if marker in error_msg:
            detail = error_msg.split(marker, 1)[1]
            detail = detail.split(" (PR state:", 1)[0].strip()
            if detail:
                # GitHub's own words, which are the best text available
                # either way --- but only a verdict on the pull request
                # when the status positively says one was reached.  A
                # body with no recognisable status (a wrapped or
                # transport exception that still quoted GitHub) says
                # nothing about mergeability, and must not be
                # withdrawable on a later clean reading.
                return detail[:300], _is_state_verdict(error_msg)
        # Workflow-scope failures surface in several phrasings: the
        # PermissionError messages we raise ("Missing 'workflow' scope",
        # "Missing workflow permissions") and GitHub's own response body
        # ("refusing to allow ... without `workflow` scope").  Match all
        # of them, but require the word "workflow" so unrelated 403s do
        # not get mislabelled as a scope problem.
        error_lower = error_msg.lower()
        if "workflow" in error_lower and (
            "missing 'workflow' scope" in error_lower
            or "missing workflow permissions" in error_lower
            or "refusing to allow" in error_lower
        ):
            return "missing 'workflow' token scope", False
        # The token already had the 'workflow' scope but GitHub still
        # refused the workflow-file update — a ruleset or SSO problem,
        # not a scope problem.  Report it as such rather than telling the
        # user to add a scope they already hold.
        elif "blocked by something other than token scope" in error_lower:
            return (
                "workflow update blocked by repository ruleset or SSO "
                "(token already has 'workflow' scope)"
            ), False
        elif "403" in error_msg and "forbidden" in error_lower:
            return "insufficient permissions", False
        # Surface transient HTTP errors (502, 405 etc.) accurately instead
        # of falling through to infer a reason from mergeable_state, which
        # may be stale or misleading (e.g. "clean" → "branch protection").
        elif "405" in error_msg and "Method Not Allowed" in error_msg:
            if pr_info.mergeable_state in ("clean", "unstable"):
                return (
                    "GitHub API returned transient 405 error "
                    "(PR appears mergeable — GitHub may be experiencing issues, "
                    "see https://www.githubstatus.com)"
                ), False
            # For non-clean states, fall through to state-based analysis below
        elif "502" in error_msg:
            return (
                "GitHub API returned 502 Bad Gateway "
                "(GitHub may be experiencing issues, "
                "see https://www.githubstatus.com)"
            ), False

        return None
