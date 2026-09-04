# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Result reporting for a completed merge pass.

Turns the per-PR results into the closing summary, expands the raw
failure reasons into readable guidance, and lists the PRs that could not
be merged.
"""

from ..merge_manager import (
    MergeResult,
)
from ..rule_violations import (
    RULE_VIOLATION_MARKER,
    is_rule_violation,
    required_status_check_names,
    required_workflow_names,
    status_check_violation_verb,
    violation_verb,
)
from ._app import console


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
    final_unsettled = sum(1 for r in real_results if r.status.value == "unsettled")
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
    if final_unsettled > 0:
        parts.append(f"{final_unsettled} unsettled")
    if final_closed > 0:
        parts.append(f"{final_closed} closed")
    console.print(f"\n🚀 Final Results: {', '.join(parts)}")
    if final_skipped > 0:
        console.print(f"⏭️ Skipped {final_skipped} PRs")
    if final_blocked > 0:
        console.print(f"🛑 Blocked {final_blocked} PRs")
    if final_unsettled > 0:
        console.print(
            f"⏱️ Unsettled {final_unsettled} PRs "
            "(judged before their checks finished; re-run to merge)"
        )
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

    One rejection can name **both** kinds, and each is reported: a
    failing status context such as ``pre-commit.ci - pr`` used to be
    dropped whenever a workflow was named alongside it, leaving the
    summary listing conditions that pass and omitting the one that
    blocks.  Each kind also carries its own verb, because workflows that
    have not finished routinely accompany a context that has failed.
    """
    if not is_rule_violation(reason):
        return [reason]

    workflows = required_workflow_names(reason)
    checks = required_status_check_names(reason)
    if not workflows and not checks:
        return [reason]

    headings = [RULE_VIOLATION_MARKER]
    if workflows:
        headings.append(f"Required workflows {violation_verb(reason)}")
    if checks:
        headings.append(f"Required status checks {status_check_violation_verb(reason)}")

    # Bullets are labelled only when both kinds are present.  With one
    # kind the heading already says which, so labelling would add noise
    # to every existing report; with both, an unlabelled name does not
    # say which heading it belongs under.
    both = bool(workflows and checks)
    bullets = [
        f"• Required workflow: {name}" if both else f"• {name}" for name in workflows
    ] + [f"• Required status check: {name}" if both else f"• {name}" for name in checks]
    return [" / ".join(headings), *bullets]


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
    non-merged terminal outcome — failed, blocked, unsettled, skipped,
    closed and auto-merge pending — one section per outcome.
    """
    sections: list[tuple[str, str]] = [
        ("failed", "\n❌ Failed PRs:"),
        ("blocked", "\n🛑 Blocked PRs:"),
        ("unsettled", "\n⏱️ Unsettled PRs (re-run to merge):"),
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
    unsettled_count = sum(1 for r in merge_results if r.status.value == "unsettled")
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
    if unsettled_count > 0:
        console.print(
            f"⏱️ Unsettled {unsettled_count} PRs "
            "(judged before their checks finished; re-run to merge)"
        )
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
        if unsettled_count > 0:
            parts.append(f"{unsettled_count} unsettled")
        if closed_count > 0:
            parts.append(f"{closed_count} closed")
        console.print(f"📈 Final Results: {', '.join(parts)}")

    _print_failed_pr_details(merge_results)
