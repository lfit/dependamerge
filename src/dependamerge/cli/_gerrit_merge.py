# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Gerrit submission preview, execution, and the Gerrit merge flow.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from ..gerrit import (
    GerritAuthError,
    GerritChangeInfo,
    GerritComparisonResult,
    GerritRestError,
    GerritSubmitResult,
)
from ..netrc import (
    GerritCredentials,
)
from ..progress_tracker import MergeProgressTracker
from ..url_parser import (
    ParsedGerritTopicUrl,
    ParsedUrl,
)

if TYPE_CHECKING:
    pass

from . import _deps
from ._gerrit_resolve import (
    _find_and_print_similar_changes,
    _maybe_rebase_gerrit_change,
    _resolve_gerrit_candidates,
    _resolve_gerrit_credentials_or_exit,
    _resolve_gerrit_only_automation,
    _resolve_gerrit_source_change,
)
from ._sha import (
    _generate_gerrit_continue_sha,
)


def _confirm_gerrit_submission(
    source_change: GerritChangeInfo,
    console: Console,
) -> bool:
    """Prompt for the continue SHA before a real Gerrit submission.

    Mirrors the interactive preview-then-confirm flow of the GitHub
    path (_handle_preview_confirmation): the user must type a SHA
    derived from the source change to proceed.

    Returns:
        True when the user confirmed and the submission should proceed.
    """
    continue_sha = _generate_gerrit_continue_sha(source_change)
    console.print()
    console.print(f"To proceed with merging enter: {continue_sha}")

    try:
        if "pytest" in sys.modules or os.getenv("TESTING"):
            console.print("⚠️ Test mode detected - skipping interactive prompt")
            return False

        user_input = input(
            "Enter the string above to continue (or press Enter to cancel): "
        ).strip()

        if user_input == continue_sha:
            return True
        if user_input == "":
            console.print("❌ Merge cancelled by user.")
        else:
            console.print("❌ Invalid input. Merge cancelled.")
    except KeyboardInterrupt:
        console.print("\n❌ Merge cancelled by user.")
    except EOFError:
        console.print("\n❌ Merge cancelled.")
    return False


def _print_gerrit_final_summary(
    results: list[GerritSubmitResult],
    all_changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
    console: Console,
) -> None:
    """Print the post-run 🚀 Final Results line and failure recap.

    Mirrors the GitHub ``_print_final_merge_summary`` /
    ``_print_failed_pr_details`` pair: per-change status lines are
    not printed while the submission runs (progress is conveyed by
    the live tracker counters), so this end-of-run report is the
    only place failure reasons appear.

    Args:
        results: Submit results, one per attempted change.
        all_changes: The (change, comparison) tuples that were
            submitted; used to recover each change's web URL for the
            failure recap (``GerritSubmitResult`` carries only the
            project and number).
        console: Rich console to print to.
    """
    submitted = sum(1 for r in results if r.submitted)
    failed = [r for r in results if not r.success]
    reviewed_only = sum(1 for r in results if r.reviewed and not r.submitted)

    parts = [f"{submitted} submitted", f"{len(failed)} failed"]
    if reviewed_only > 0:
        parts.append(f"{reviewed_only} reviewed (not submitted)")
    console.print(f"\n🚀 Final Results: {', '.join(parts)}")

    if not failed:
        return
    url_by_key = {
        (change.project, change.number): change.url for change, _ in all_changes
    }
    console.print("\n❌ Failed changes:")
    for result in failed:
        url = url_by_key.get((result.project, result.change_number)) or (
            f"{result.project} #{result.change_number}"
        )
        reason = result.error or "no reason reported"
        # markup=False so bracketed reasons are not eaten by Rich.
        console.print(f"   • {url}\n     {reason}", markup=False)


def _preview_gerrit_submission(
    source_change: GerritChangeInfo,
    all_changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
    no_confirm: bool,
    dry_run: bool,
    console: Console,
) -> bool:
    """Warn about permissions and preview the run, then decide to proceed.

    Permissions are per-project in Gerrit, so this checks the source
    change and warns if the user may lack sufficient permissions. It
    then either stops (dry run), prompts for confirmation
    (interactive), or proceeds straight to submission (--no-confirm).

    Returns:
        True when submission should proceed, False when the caller
        should stop (dry run, or the user declined confirmation).
    """
    permission_warnings = source_change.get_permission_warnings()
    if permission_warnings:
        console.print("\n⚠️ Permission warnings:")
        for warning in permission_warnings:
            console.print(f"   • {warning}")
        console.print(
            "\n   Note: Permissions vary by project. The operation may still "
            "succeed on some changes."
        )

    if not no_confirm or dry_run:
        label = "Dry run" if dry_run else "Preview"
        console.print(
            f"\n📊 {label}: {len(all_changes)} changes would be reviewed and submitted"
        )
        if source_change.has_required_permissions():
            console.print(
                "   ✅ You appear to have required permissions (+2 Code-Review, submit)"
            )
        else:
            console.print(
                "   ⚠️ You may not have all required permissions (see warnings above)"
            )
        if dry_run:
            console.print("\n🧪 Dry run: no changes were reviewed or submitted.")
            return False
        if not _confirm_gerrit_submission(source_change, console):
            return False

    return True


def _run_gerrit_submission(
    parsed_url: ParsedUrl | ParsedGerritTopicUrl,
    credentials: GerritCredentials,
    all_changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
    show_progress: bool,
    console: Console,
) -> None:
    """Submit all changes in parallel and print the final summary.

    Live in-place progress mirrors the GitHub merge path: the submit
    manager records each change's transitory and terminal states
    against the tracker while the parallel submission runs, and
    failures are recapped afterwards by _print_gerrit_final_summary
    instead of interleaved lines. The tracker is created (unstarted)
    before the submit manager so it can be handed over, but only
    started inside the try/finally so it is always stopped, even when
    submission setup or the run itself raises.
    """
    console.print(f"\n🚀 Submitting {len(all_changes)} changes...")

    progress_tracker: MergeProgressTracker | None = None
    if show_progress:
        progress_tracker = MergeProgressTracker(
            parsed_url.host,
            operation_label="Submitting changes",
            operation_icon="▶️",
            unit_label="changes",
        )
        progress_tracker.set_total_prs(len(all_changes))

    submit_manager = _deps.create_submit_manager(
        host=parsed_url.host,
        base_path=parsed_url.base_path,
        username=credentials.username,
        password=credentials.password,
        progress_tracker=progress_tracker,
    )

    try:
        if progress_tracker is not None:
            progress_tracker.start()
        results = submit_manager.submit_changes_parallel(all_changes)
    finally:
        if progress_tracker is not None:
            progress_tracker.stop()

    _print_gerrit_final_summary(results, all_changes, console)


def _handle_gerrit_merge(
    parsed_url: ParsedUrl | ParsedGerritTopicUrl,
    no_confirm: bool,
    similarity_threshold: float,
    verbose: bool,
    console: Console,
    no_netrc: bool = False,
    netrc_file: Path | None = None,
    netrc_optional: bool = True,
    dry_run: bool = False,
    override: str | None = None,
    topic: str | None = None,
    show_progress: bool = True,
) -> None:
    """
    Handle merge operation for a Gerrit change or topic search URL.

    Args:
        parsed_url: Parsed Gerrit change URL (host, project, change
            number) or topic search URL (host, topic).
        no_confirm: If True, skip confirmation prompt.
        similarity_threshold: Threshold for matching similar changes.
        verbose: Enable verbose output.
        console: Rich console for output.
        no_netrc: If True, skip .netrc credential lookup.
        netrc_file: Explicit path to a .netrc file.
        netrc_optional: If True, don't fail if netrc not found.
        dry_run: If True, preview only and never review or submit any
            change, even when ``no_confirm`` is also set.
        override: SHA hash to override non-automation change restriction.
        topic: Explicit topic to scope the similar-change search to.
            When omitted, the topic is taken from the search URL (if
            given) or from the source change itself.
        show_progress: If True, drive a live Rich progress tracker
            during submission (in-place counters, matching the GitHub
            merge path) instead of printing per-change status lines.
    """
    credentials = _resolve_gerrit_credentials_or_exit(
        parsed_url, no_netrc, netrc_file, verbose, console
    )

    console.print(f"🔍 Examining Gerrit change on {parsed_url.host}...")

    try:
        service = _deps.create_gerrit_service(
            host=parsed_url.host,
            base_path=parsed_url.base_path,
            username=credentials.username,
            password=credentials.password,
        )

        if not service.is_authenticated:
            console.print("⚠️ Warning: Service created but may not be authenticated")

        source_change, topic_changes = _resolve_gerrit_source_change(
            service, parsed_url, topic, credentials, console
        )

        comparator = _deps.create_gerrit_comparator(
            similarity_threshold=similarity_threshold
        )
        only_automation = _resolve_gerrit_only_automation(
            source_change, comparator, override, console
        )

        source_change = _maybe_rebase_gerrit_change(
            service, source_change, credentials, console
        )

        candidates = _resolve_gerrit_candidates(
            service, source_change, parsed_url, topic, topic_changes, console
        )
        similar_changes = _find_and_print_similar_changes(
            service, comparator, source_change, candidates, only_automation, console
        )

        # Prepare list of changes to submit (similar + source)
        source_entry: tuple[GerritChangeInfo, GerritComparisonResult | None] = (
            source_change,
            None,
        )
        all_changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]] = [
            *similar_changes,
            source_entry,
        ]

        if not _preview_gerrit_submission(
            source_change, all_changes, no_confirm, dry_run, console
        ):
            return

        _run_gerrit_submission(
            parsed_url, credentials, all_changes, show_progress, console
        )

    except typer.Exit:
        # Re-raise typer.Exit without treating it as an error
        raise
    except GerritAuthError as e:
        console.print(f"❌ Gerrit authentication failed: {e}")
        console.print("   Check your GERRIT_USERNAME and GERRIT_PASSWORD")
        raise typer.Exit(1) from None
    except GerritRestError as e:
        console.print(f"❌ Gerrit API error: {e}")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"❌ Error during Gerrit merge operation: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1) from None
