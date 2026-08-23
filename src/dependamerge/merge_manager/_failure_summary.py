# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The failure reason a stored merge exception can be read off.

When a merge attempt raises, the exception text usually carries a more
actionable explanation than anything that can be inferred afterwards
from the PR's state, so it is consulted first.
"""

from __future__ import annotations

from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _FailureSummaryFromExceptionMixin(_MergeManagerBase):
    """Reading a merge failure reason out of the exception that caused it."""

    def _failure_summary_from_exception(
        self,
        pr_key: str,
        last_exception: Exception,
        pr_info: PullRequestInfo,
    ) -> str | None:
        """
        Derive a failure reason from a stored merge exception.

        Args:
            pr_key: The ``owner/repo#number`` key the exception is stored under
            last_exception: The exception the last merge attempt raised
            pr_info: Pull request information

        Returns:
            The reason, or None when the exception says nothing conclusive
            and the caller should fall back to state-based analysis.
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
                return detail[:300]
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
            return "missing 'workflow' token scope"
        # The token already had the 'workflow' scope but GitHub still
        # refused the workflow-file update — a ruleset or SSO problem,
        # not a scope problem.  Report it as such rather than telling the
        # user to add a scope they already hold.
        elif "blocked by something other than token scope" in error_lower:
            return (
                "workflow update blocked by repository ruleset or SSO "
                "(token already has 'workflow' scope)"
            )
        elif "403" in error_msg and "forbidden" in error_lower:
            return "insufficient permissions"
        # Surface transient HTTP errors (502, 405 etc.) accurately instead
        # of falling through to infer a reason from mergeable_state, which
        # may be stale or misleading (e.g. "clean" → "branch protection").
        elif "405" in error_msg and "Method Not Allowed" in error_msg:
            if pr_info.mergeable_state in ("clean", "unstable"):
                return (
                    "GitHub API returned transient 405 error "
                    "(PR appears mergeable — GitHub may be experiencing issues, "
                    "see https://www.githubstatus.com)"
                )
            # For non-clean states, fall through to state-based analysis below
        elif "502" in error_msg:
            return (
                "GitHub API returned 502 Bad Gateway "
                "(GitHub may be experiencing issues, "
                "see https://www.githubstatus.com)"
            )

        return None
