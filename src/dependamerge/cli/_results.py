# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Merge-result reporting, failure formatting, and merge ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..merge_manager import (
    MergeResult,
)
from ..models import ComparisonResult, PullRequestInfo
from ..rule_violations import (
    RULE_VIOLATION_MARKER,
    is_rule_violation,
    required_status_check_names,
    required_workflow_names,
    violation_verb,
)

if TYPE_CHECKING:
    pass

from ._app import console


@dataclass(frozen=True)
class _MergePreview:
    """Outcome of a preview merge pass, ready for confirmation."""

    merge_results: list[MergeResult]
    all_prs_to_merge: list[tuple[PullRequestInfo, ComparisonResult | None]]
    merged_count: int
    total_to_merge: int


def _print_final_merge_summary(real_results: list[MergeResult]) -> None:
    """Print the post-run 🚀 Final Results line and per-outcome recap.

    Shared by the org / repo / similar-PR confirmed-merge paths so
    every outcome category (including closed-without-merge) renders
    identically regardless of scope.
    """
    final_merged = sum(1 for r in real_results if r.status.value == "merged")
    final_failed = sum(1 for r in real_results if r.status.value == "failed")
    final_skipped = sum(1 for r in real_results if r.status.value == "skipped")
    final_blocked = sum(1 for r in real_results if r.status.value == "blocked")
    final_closed = sum(1 for r in real_results if r.status.value == "closed")
    final_auto_merge = sum(
        1 for r in real_results if r.status.value == "auto_merge_pending"
    )
    parts = [f"{final_merged} merged"]
    if final_auto_merge > 0:
        parts.append(f"{final_auto_merge} auto-merge pending")
    parts.append(f"{final_failed} failed")
    if final_skipped > 0:
        parts.append(f"{final_skipped} skipped")
    if final_blocked > 0:
        parts.append(f"{final_blocked} blocked")
    if final_closed > 0:
        parts.append(f"{final_closed} closed")
    console.print(f"\n🚀 Final Results: {', '.join(parts)}")
    if final_skipped > 0:
        console.print(f"⏭️ Skipped {final_skipped} PRs")
    if final_blocked > 0:
        console.print(f"🛑 Blocked {final_blocked} PRs")
    if final_closed > 0:
        console.print(f"🚪 Closed without merging: {final_closed} PRs")
    if final_auto_merge > 0:
        console.print(f"⏳ Auto-merge pending for {final_auto_merge} PRs")

    _print_failed_pr_details(real_results)


def _format_failure_reason(reason: str) -> list[str]:
    """Expand a failure reason into consistent display lines.

    Failures are rendered in a consistent shape so the final summary is
    actionable at a glance:

      1. the failed PR URL (prepended by the caller),
      2. a single **failure-type** line whose parts are joined with
         `` / `` and carry no trailing colon, e.g.
         ``Repository rule violations found / Required workflows failed``,
      3. one bullet (``• ``) per individual failing condition.

    Repository-ruleset violations arrive from GitHub as one long string
    that crams the offending workflow / status-check names into a
    quoted, comma-separated clause.  We split that into the type line
    plus a bullet per name for both the ``Required workflows`` and
    ``Required status check(s)`` variants.  Reasons we do not recognise
    are returned unchanged as a single line.
    """
    if is_rule_violation(reason):
        ruleset = RULE_VIOLATION_MARKER
        verb = violation_verb(reason)
        workflows = required_workflow_names(reason)
        if workflows:
            return [
                f"{ruleset} / Required workflows {verb}",
                *(f"• {name}" for name in workflows),
            ]
        checks = required_status_check_names(reason)
        if checks:
            return [
                f"{ruleset} / Required status checks {verb}",
                *(f"• {name}" for name in checks),
            ]
    return [reason]


def _print_failed_pr_details(
    merge_results: list[MergeResult],
) -> None:
    """Print URL and reason for every non-merged PR in the result list.

    A bare ``Failed: 1`` line in the summary forces the user to
    scroll back through the merge output to find which PR failed
    and why.  During a real merge run the per-PR status lines are
    no longer printed to the console at all (progress is conveyed
    by the live tracker counters), so this end-of-run report is
    the *only* place reasons appear.  It therefore covers every
    non-merged terminal outcome — failed, blocked, skipped, closed
    and auto-merge pending — one section per outcome.
    """
    sections: list[tuple[str, str]] = [
        ("failed", "\n❌ Failed PRs:"),
        ("blocked", "\n🛑 Blocked PRs:"),
        ("skipped", "\n⏭️ Skipped PRs:"),
        ("closed", "\n🚪 Closed PRs:"),
        ("auto_merge_pending", "\n🤖 Auto-merge pending PRs:"),
    ]
    for status_value, heading in sections:
        matching = [r for r in merge_results if r.status.value == status_value]
        if not matching:
            continue
        console.print(heading)
        for r in matching:
            url = getattr(r.pr_info, "html_url", "<unknown>")
            reason = r.error or "no reason reported"
            body = "\n".join(f"     {line}" for line in _format_failure_reason(reason))
            # markup=False so bracketed reasons are not eaten by Rich.
            console.print(f"   • {url}\n{body}", markup=False)


def _display_merge_results(
    merge_results: list[MergeResult],
    no_confirm: bool,
) -> None:
    """Print the final summary of merge results."""
    merged_count = sum(1 for r in merge_results if r.status.value == "merged")
    failed_count = sum(1 for r in merge_results if r.status.value == "failed")
    skipped_count = sum(1 for r in merge_results if r.status.value == "skipped")
    blocked_count = sum(1 for r in merge_results if r.status.value == "blocked")
    closed_count = sum(1 for r in merge_results if r.status.value == "closed")
    auto_merge_count = sum(
        1 for r in merge_results if r.status.value == "auto_merge_pending"
    )

    if failed_count > 0:
        if not no_confirm:
            console.print(f"❌ Would fail to merge {failed_count} PRs")
        else:
            console.print(f"❌ Failed {failed_count} PRs")
    if skipped_count > 0:
        console.print(f"⏭️ Skipped {skipped_count} PRs")
    if blocked_count > 0:
        console.print(f"🛑 Blocked {blocked_count} PRs")
    if closed_count > 0:
        console.print(f"🚪 Closed without merging: {closed_count} PRs")
    if auto_merge_count > 0:
        console.print(f"⏳ Auto-merge pending for {auto_merge_count} PRs")

    if no_confirm:
        parts = [f"{merged_count} merged"]
        if auto_merge_count > 0:
            parts.append(f"{auto_merge_count} auto-merge pending")
        parts.append(f"{failed_count} failed")
        if skipped_count > 0:
            parts.append(f"{skipped_count} skipped")
        if blocked_count > 0:
            parts.append(f"{blocked_count} blocked")
        if closed_count > 0:
            parts.append(f"{closed_count} closed")
        console.print(f"📈 Final Results: {', '.join(parts)}")

    _print_failed_pr_details(merge_results)


def _repo_merge_order(
    prs: list[PullRequestInfo],
) -> list[PullRequestInfo]:
    """Order a single repository's PRs for oldest-first merging.

    Sorts ascending by PR number, i.e. in the order the automation raised
    them (oldest first).  Merging the oldest PR first minimises the rebase
    churn imposed on the newer siblings: each merge advances the base
    branch, so a newer sibling merged ahead of an older one would leave
    the older PR ``behind`` and trigger an avoidable rebase + CI wait.

    This is the single-repository analogue of the within-repository key
    used owner-wide by :func:`_owner_merge_order`, keeping both schemes'
    intra-repository sequencing identical.
    """
    return sorted(prs, key=lambda p: p.number)


def _owner_merge_order(
    prs: list[PullRequestInfo],
) -> list[PullRequestInfo]:
    """Order owner-wide PRs for striped merging.

    Sorts so that:

    - Repositories with the most in-scope PRs come first.  These take the
      longest to drain (each merge can make the next sibling ``behind`` /
      ``dirty`` and trigger a rebase + CI wait), so starting them earliest
      gives them the most wall-clock head start under the striped
      scheduler's concurrent per-repository workers.
    - Within a repository, PRs ascend by number, i.e. in the order the
      automation raised them (oldest first).  Merging the oldest first
      minimises the rebase churn imposed on the newer siblings.

    Ties between equally-sized repositories break on repository name so
    the order (and the grouped listing derived from it) is deterministic.
    """
    counts: dict[str, int] = {}
    for pr in prs:
        counts[pr.repository_full_name] = counts.get(pr.repository_full_name, 0) + 1
    return sorted(
        prs,
        key=lambda p: (
            -counts[p.repository_full_name],
            p.repository_full_name,
            p.number,
        ),
    )


def _print_prs_grouped_by_repo(
    prs: list[PullRequestInfo],
) -> None:
    """Print a PR list grouped by repository for owner-wide readability.

    Emits a header per repository followed by its PRs indented beneath,
    so a large owner-wide list stays scannable.  Repositories are listed
    in the order they first appear in ``prs`` (the caller passes them in
    merge order via :func:`_owner_merge_order`, so the listing mirrors the
    sequence in which they will be merged); PRs within a repository are
    shown in the supplied order.
    """
    by_repo: dict[str, list[PullRequestInfo]] = {}
    for pr in prs:
        by_repo.setdefault(pr.repository_full_name, []).append(pr)

    for repo, repo_prs in by_repo.items():
        console.print(f"\n📁 {repo} ({len(repo_prs)} PR(s))")
        for pr in repo_prs:
            console.print(f"  #{pr.number} {pr.title} (by {pr.author})")
