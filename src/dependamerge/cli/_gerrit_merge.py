# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Reviewing and submitting a batch of Gerrit changes.

Prints the changes that match the source, previews what submission would
do, and --- once confirmed --- votes and submits them.
"""

from pathlib import Path

import typer
from rich.console import Console

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
import dependamerge.cli as _pkg

from ..gerrit import (
    GerritAuthError,
    GerritChangeComparator,
    GerritChangeInfo,
    GerritComparisonResult,
    GerritRestError,
    GerritService,
    GerritServiceError,
)
from ..netrc import (
    GerritCredentials,
)
from ..progress_tracker import MergeProgressTracker
from ..url_parser import (
    ParsedGerritTopicUrl,
    ParsedUrl,
)
from ._gerrit_setup import (
    _maybe_rebase_gerrit_change,
    _print_gerrit_final_summary,
    _resolve_gerrit_candidates,
    _resolve_gerrit_credentials_or_exit,
    _resolve_gerrit_only_automation,
    _resolve_gerrit_source_change,
)
from ._similarity import _format_gerrit_similarity


def _find_and_print_similar_changes(
    service: GerritService,
    comparator: GerritChangeComparator,
    source_change: GerritChangeInfo,
    candidates: list[GerritChangeInfo] | None,
    only_automation: bool,
    console: Console,
) -> list[tuple[GerritChangeInfo, GerritComparisonResult]]:
    """Score candidates against the source change and print the matches."""
    similar_changes = service.find_similar_changes(
        source_change,
        comparator,
        only_automation=only_automation,
        candidates=candidates,
    )

    console.print(f"Found {len(similar_changes)} similar changes:")
    for change, comparison in similar_changes:
        console.print(f"  • {change.project} #{change.number}: {change.subject}")
        console.print(f"    {_format_gerrit_similarity(comparison)}")

    return similar_changes


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
        if not _pkg._confirm_gerrit_submission(source_change, console):
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

    submit_manager = _pkg.create_submit_manager(
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
        service = _pkg.create_gerrit_service(
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

        comparator = _pkg.create_gerrit_comparator(
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
    except GerritServiceError as e:
        # Not a transport failure: the request succeeded but the
        # response could not be used.  Distinct from the generic
        # handler below, which would label a diagnosis we made
        # deliberately as unexpected.
        console.print(f"❌ Gerrit service error: {e}")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"❌ Error during Gerrit merge operation: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1) from None
