# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Evidence gathering and phrasing for block-reason analysis.

The probes that read a pull request's reviews, comments, checks and
required-check configuration, and the ordered set of phrasings that
turn that evidence into one human-readable reason.  Split out of
``dependamerge.github_async._block_reason`` purely to keep each module
reviewable.
"""

from __future__ import annotations

from typing import NamedTuple

from ..bot_identity import is_copilot
from ..check_runs import failing_check_names
from ._base import _GitHubAsyncBase


class _ReviewSignals(NamedTuple):
    """Review verdicts observed on a pull request."""

    approved: bool
    human_changes_requested: bool
    unresolved_copilot_reviews: int


class _CheckSignals(NamedTuple):
    """Check names observed on a head commit, grouped by status."""

    failing: list[str]
    completed: set[str]
    reported: set[str]
    pending: set[str]


async def _collect_review_signals(
    api: _GitHubAsyncBase, owner: str, repo: str, number: int
) -> _ReviewSignals:
    """Read the PR's reviews for approval and change-request verdicts."""
    approved = False
    human_changes_requested = False
    unresolved_copilot_reviews = 0

    try:
        reviews = await api.get(f"/repos/{owner}/{repo}/pulls/{number}/reviews")
        if isinstance(reviews, list):
            for review in reviews:
                if not isinstance(review, dict):
                    continue
                state = review.get("state")
                author = (review.get("user") or {}).get("login", "")

                if state == "APPROVED":
                    approved = True
                elif state == "CHANGES_REQUESTED":
                    if is_copilot(author):
                        unresolved_copilot_reviews += 1
                    else:
                        human_changes_requested = True
    except Exception:
        # Review data is best-effort; on API error leave the
        # approval/changes flags at their safe defaults.
        pass

    return _ReviewSignals(approved, human_changes_requested, unresolved_copilot_reviews)


async def _count_unresolved_copilot_comments(
    api: _GitHubAsyncBase, owner: str, repo: str, number: int
) -> int:
    """Count Copilot review comments that read as unresolved."""
    unresolved_copilot_comments = 0
    try:
        comments = await api.get(f"/repos/{owner}/{repo}/pulls/{number}/comments")
        if isinstance(comments, list):
            for comment in comments:
                if not isinstance(comment, dict):
                    continue
                author = (comment.get("user") or {}).get("login", "")
                # Count unresolved Copilot comments (those without replies dismissing them)
                if is_copilot(author):
                    # Simple heuristic: if comment doesn't have "DISMISSED" or similar resolution text
                    body = comment.get("body", "").lower()
                    if "dismissed" not in body and "resolved" not in body:
                        unresolved_copilot_comments += 1
    except Exception:
        # Review comments are best-effort; ignore fetch errors and
        # leave the Copilot comment count unchanged.
        pass
    return unresolved_copilot_comments


async def _collect_check_signals(
    api: _GitHubAsyncBase, owner: str, repo: str, head_sha: str
) -> _CheckSignals:
    """Read check runs and status contexts for a head commit."""
    # Check runs and status contexts - look for failing (check this first as it's most specific)
    failing_checks: list[str] = []
    completed_check_names: set[str] = set()
    # Track all reported check names regardless of status so that
    # queued/in_progress checks are not misclassified as "missing".
    reported_check_names: set[str] = set()
    pending_check_names: set[str] = set()
    try:
        # Check runs (newer GitHub Apps API)
        runs = await api.get(f"/repos/{owner}/{repo}/commits/{head_sha}/check-runs")
        if isinstance(runs, dict):
            raw_runs = [
                run for run in (runs.get("check_runs") or []) if isinstance(run, dict)
            ]
            # Status classification deliberately considers *every*
            # reported run, not just the latest: a name carrying both
            # a completed run and a fresh in_progress re-run is still
            # pending, and must not be collapsed away here.
            for run in raw_runs:
                name = (run.get("name") or "").strip()
                if not name:
                    # An unnamed run cannot be matched against a
                    # required-check rule.  Recording it produces
                    # only misleading output such as "Blocked by
                    # failing check: unknown", so drop it here just
                    # as the deduplication helper does.
                    continue
                status = run.get("status")
                reported_check_names.add(name)
                if status == "completed":
                    completed_check_names.add(name)
                elif status in ("queued", "in_progress"):
                    pending_check_names.add(name)
            # Failure, by contrast, is decided by the latest run per
            # name.  A commit can carry several runs under one name
            # when a duplicate workflow event causes ``concurrency``
            # to cancel a superseded run; that cancelled run must not
            # mask the successful one that replaced it.
            failing_checks.extend(failing_check_names(raw_runs))
    except Exception:
        # Check-runs API may be unavailable; proceed with whatever
        # checks were collected so far.
        pass

    try:
        statuses = await api.get(f"/repos/{owner}/{repo}/commits/{head_sha}/status")
        if isinstance(statuses, dict):
            for s in statuses.get("statuses") or []:
                if not isinstance(s, dict):
                    continue
                context = s.get("context", "unknown")
                state = s.get("state")
                reported_check_names.add(context)
                if state in ["success", "neutral"]:
                    completed_check_names.add(context)
                elif state == "pending":
                    pending_check_names.add(context)
                if state in ["failure", "error"]:
                    # Avoid duplicates if both check-run and status exist for same service
                    if context not in failing_checks:
                        failing_checks.append(context)
    except Exception:
        # Status API may be unavailable; proceed with whatever
        # status contexts were collected so far.
        pass

    return _CheckSignals(
        failing_checks,
        completed_check_names,
        reported_check_names,
        pending_check_names,
    )


async def _resolve_block_reason_base_branch(
    api: _GitHubAsyncBase,
    owner: str,
    repo: str,
    number: int,
    base_branch: str | None,
) -> str | None:
    """Resolve the PR's actual base branch.

    It drives both the required status-check lookup and the final
    guard-kind classification, so a wrong value (e.g. assuming "main" on
    a repo that defaults to "master") produces a misleading block
    reason.  Prefer the caller-supplied value, then the PR's own base
    ref; if neither is available, fall back to the repository's real
    default branch rather than a hardcoded name, and only give up
    (returning ``None``) when nothing can be determined.
    """
    if base_branch is None:
        try:
            pr_data = await api.get(f"/repos/{owner}/{repo}/pulls/{number}")
            if isinstance(pr_data, dict):
                ref = (pr_data.get("base") or {}).get("ref")
                if isinstance(ref, str) and ref:
                    base_branch = ref
        except Exception as pr_err:
            api.log.debug(
                f"Could not read base branch for {owner}/{repo}#{number}: {pr_err}"
            )

    if base_branch is None:
        base_branch = await api._resolve_default_branch(owner, repo)

    return base_branch


async def _collect_required_check_gaps(
    api: _GitHubAsyncBase,
    owner: str,
    repo: str,
    number: int,
    base_branch: str | None,
    checks: _CheckSignals,
) -> tuple[list[str], list[str]]:
    """Detect missing/pending required status checks (e.g. stale pre-commit.ci)."""
    missing_required_checks: list[str] = []
    pending_required_checks: list[str] = []

    # Only inspect required status checks when we know which branch to
    # query; an assumed branch would yield checks for the wrong ref.
    if base_branch is not None:
        try:
            required_checks = await api.get_required_status_checks(
                owner, repo, base_branch
            )
            for check in required_checks:
                ctx = check.get("context", "")
                if not ctx:
                    continue
                if ctx in checks.reported:
                    if ctx not in checks.completed and ctx in checks.pending:
                        pending_required_checks.append(ctx)
                else:
                    # Never reported via either API — truly missing
                    missing_required_checks.append(ctx)
        except Exception as req_err:
            api.log.debug(
                f"Could not check required status checks for "
                f"{owner}/{repo}#{number}: {req_err}"
            )

    return missing_required_checks, pending_required_checks


def _check_block_reason(
    failing_checks: list[str],
    missing_required_checks: list[str],
    pending_required_checks: list[str],
) -> str | None:
    """Phrase a check-derived blocker, most specific first."""
    if failing_checks:
        if len(failing_checks) == 1:
            return f"Blocked by failing check: {failing_checks[0]}"
        else:
            return f"Blocked by {len(failing_checks)} failing checks"

    if missing_required_checks:
        if len(missing_required_checks) == 1:
            return f"Blocked by missing required status: {missing_required_checks[0]}"
        else:
            names = ", ".join(missing_required_checks)
            return f"Blocked by {len(missing_required_checks)} missing required statuses: {names}"

    if pending_required_checks:
        if len(pending_required_checks) == 1:
            return f"Blocked by pending required check: {pending_required_checks[0]}"
        else:
            names = ", ".join(pending_required_checks)
            return f"Blocked by {len(pending_required_checks)} pending required checks: {names}"

    return None


def _review_block_reason(
    human_changes_requested: bool,
    unresolved_copilot_reviews: int,
    unresolved_copilot_comments: int,
) -> str | None:
    """Phrase a review-derived blocker."""
    if human_changes_requested:
        return "Human reviewer requested changes"

    if unresolved_copilot_reviews > 0:
        if unresolved_copilot_comments > 0:
            return f"Blocked by {unresolved_copilot_reviews} Copilot reviews, {unresolved_copilot_comments} comments"
        else:
            return f"Blocked by {unresolved_copilot_reviews} unresolved Copilot reviews"

    if unresolved_copilot_comments > 0:
        return f"Blocked by {unresolved_copilot_comments} unresolved Copilot comments"

    return None


def _pending_block_reason(pending_check_names: set[str]) -> str | None:
    """Phrase a still-running check as a temporary blocker.

    No *required* check is failing, missing, or pending, and no
    human/Copilot review is blocking — but if any check on the head
    commit is still queued or in progress, the PR is only *temporarily*
    blocked. This matters for checks enforced through a repository
    ruleset's "required workflows": those never appear in the classic
    required-status-checks list, so the pending-required-checks phrasing
    cannot see them. Surface them here, *before* the "requires approval"
    fallback, so the merge pipeline waits for them (and arms auto-merge)
    instead of failing the PR outright while its workflows are still
    running.
    """
    # Any name in ``pending_check_names`` has a queued/in-progress run
    # and is therefore still running. We must NOT subtract
    # ``completed_check_names``: GitHub can report two runs with the
    # same name (a re-run leaves one ``completed`` entry and a fresh
    # ``in_progress`` one), and the set difference would cancel the
    # name out and hide a check that is genuinely still running.
    #
    # Defensively filter to non-empty strings: a malformed API
    # payload can report ``name``/``context`` as ``null``, and mixing
    # ``None`` with strings would make ``sorted``/``join`` raise. This
    # branch is best-effort, so drop anything that is not a usable name.
    pending_only = sorted(
        name for name in pending_check_names if isinstance(name, str) and name
    )
    if pending_only:
        if len(pending_only) == 1:
            return f"Blocked by pending check: {pending_only[0]}"
        names = ", ".join(pending_only)
        return f"Blocked by {len(pending_only)} pending checks: {names}"
    return None


async def _guarded_block_reason(
    api: _GitHubAsyncBase, owner: str, repo: str, base_branch: str | None
) -> str:
    """Phrase a blocker for which no failing condition was found.

    Checks pass, the PR is approved, and no changes are requested — yet
    GitHub still reports the PR as blocked.  Rather than *asserting*
    "branch protection" (which is invisible to this code path when the
    repository uses rulesets), determine what kind of rule actually
    guards the branch and keep the wording non-committal: we know the
    branch is guarded, not that a specific condition is failing.
    """
    if base_branch is None:
        # The base branch could not be resolved, so no branch-specific
        # inspection ran.  Say exactly that rather than implying we
        # looked for protection rules and found none.
        return (
            "Blocked for an undetermined reason "
            "(GitHub reports the PR as blocked, but the PR's base "
            "branch could not be determined, so its protection rules "
            "and required checks could not be inspected)"
        )
    kind = await api._detect_branch_protection_kind(owner, repo, base_branch)
    if kind == "ruleset":
        return "Blocked by repository ruleset (no specific failing condition detected)"
    if kind == "protection":
        return "Blocked by branch protection (no specific failing condition detected)"
    return (
        "Blocked for an undetermined reason "
        "(GitHub reports the PR as blocked but no failing checks, "
        "required reviews, or visible protection rules were found; "
        "the repository may use rulesets this token cannot read)"
    )
