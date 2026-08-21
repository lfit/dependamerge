# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Gerrit credential, source-change, and candidate resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from ..error_codes import (
    ExitCode,
    exit_with_error,
)
from ..gerrit import (
    GerritChangeComparator,
    GerritChangeInfo,
    GerritComparisonResult,
    GerritService,
)
from ..netrc import (
    GerritCredentials,
    NetrcParseError,
)
from ..url_parser import (
    ParsedGerritTopicUrl,
    ParsedUrl,
)

if TYPE_CHECKING:
    pass

from . import _deps
from ._display import (
    _display_change_info,
    _format_gerrit_similarity,
)
from ._sha import (
    _generate_gerrit_override_sha,
)


def _resolve_gerrit_credentials_or_exit(
    parsed_url: ParsedUrl | ParsedGerritTopicUrl,
    no_netrc: bool,
    netrc_file: Path | None,
    verbose: bool,
    console: Console,
) -> GerritCredentials:
    """Resolve Gerrit credentials, or print guidance and exit.

    Args:
        parsed_url: Parsed Gerrit change or topic URL (for the host).
        no_netrc: If True, skip .netrc credential lookup.
        netrc_file: Explicit path to a .netrc file.
        verbose: When True, report which auth method was used.
        console: Rich console for output.

    Returns:
        Valid Gerrit credentials.

    Raises:
        typer.Exit: When no valid credentials can be resolved.
    """
    try:
        credentials = _deps.resolve_gerrit_credentials(
            host=parsed_url.host,
            use_netrc=not no_netrc,
            netrc_file=netrc_file,
        )
    except NetrcParseError as e:
        console.print(f"⚠️ Error parsing .netrc file: {e}")
        credentials = None

    if credentials is None or not credentials.is_valid:
        console.print("❌ Gerrit credentials not found.")
        console.print("   Options:")
        console.print("   1. Create a ~/.netrc file with Gerrit credentials")
        console.print(
            "   2. Set GERRIT_USERNAME and GERRIT_PASSWORD environment variables"
        )
        console.print(
            "   Tip: Source your .secrets.gerrit file and run use_lf or use_onap"
        )
        raise typer.Exit(1)

    if verbose:
        console.print(f"🔑 Using credentials from {credentials.auth_method_display()}")

    return credentials


def _resolve_gerrit_source_change(
    service: GerritService,
    parsed_url: ParsedUrl | ParsedGerritTopicUrl,
    topic: str | None,
    credentials: GerritCredentials,
    console: Console,
) -> tuple[GerritChangeInfo, list[GerritChangeInfo] | None]:
    """Fetch and display the source change, validating its state.

    For a topic search URL there is no explicit change number: the
    first open change in the topic anchors the batch and the rest
    become candidates.

    Returns:
        The source change and, when a topic URL was given, the list of
        open changes sharing that topic (else None).

    Raises:
        typer.Exit: When no change is found, or the source change is
            already merged or abandoned.
    """
    topic_changes: list[GerritChangeInfo] | None = None
    if isinstance(parsed_url, ParsedGerritTopicUrl):
        search_topic = topic or parsed_url.topic
        console.print(f"📋 Fetching changes with topic '{search_topic}'...")
        topic_changes = [
            c for c in service.get_changes_by_topic(search_topic) if c.is_open
        ]
        if not topic_changes:
            console.print(f"❌ No open changes found with topic '{search_topic}'")
            raise typer.Exit(1)
        # Refetch the anchor change: list queries omit the label,
        # permission, and action detail the checks below rely on.
        source_change = service.get_change_info(topic_changes[0].number)
    else:
        console.print(f"📋 Fetching change {parsed_url.change_number}...")
        source_change = service.get_change_info(parsed_url.change_number)

    if source_change is None:
        console.print("❌ Change not found")
        raise typer.Exit(1)

    # Display source change info using Rich table (same style as GitHub)
    _display_change_info(
        source_change,
        console=console,
        auth_method=credentials.auth_method_display(),
    )

    if source_change.status == "MERGED":
        console.print("\n✅ Change is already merged.")
        raise typer.Exit(0)

    if source_change.status == "ABANDONED":
        console.print("\n❌ Change has been abandoned.")
        raise typer.Exit(1)

    return source_change, topic_changes


def _resolve_gerrit_only_automation(
    source_change: GerritChangeInfo,
    comparator: GerritChangeComparator,
    override: str | None,
    console: Console,
) -> bool:
    """Decide whether the batch is automation-only, honouring --override.

    Automation source changes match only other automation changes. A
    non-automation source requires a matching override SHA to proceed
    and then widens matching beyond automation.

    Returns:
        True when only automation changes should be matched.

    Raises:
        SystemExit: When the source is a non-automation change without a
            valid override SHA.
    """
    if comparator.is_automation_change(source_change):
        return True

    expected_sha = _generate_gerrit_override_sha(source_change)
    if not override:
        owner = source_change.owner.strip()
        subject = source_change.subject.strip()
        subject_preview = subject if len(subject) <= 50 else f"{subject[:50]}..."
        console.print("Source change is not from a recognized automation tool.")
        console.print(
            "To submit this and similar changes, run again with: "
            f"--override {expected_sha}"
        )
        console.print(
            f"This SHA is based on the owner '{owner}' and subject '{subject_preview}'",
            style="dim",
        )
        raise typer.Exit(0)

    if override.strip().lower() != expected_sha:
        exit_with_error(
            ExitCode.VALIDATION_ERROR,
            message="❌ Invalid override SHA provided",
            details=(
                f"Expected SHA for this change and owner: --override {expected_sha}"
            ),
        )

    console.print(
        "Override SHA validated. Proceeding with non-automation change merge."
    )
    return False


def _maybe_rebase_gerrit_change(
    service: GerritService,
    source_change: GerritChangeInfo,
    credentials: GerritCredentials,
    console: Console,
) -> GerritChangeInfo:
    """Rebase the source change when it has merge conflicts.

    Returns:
        The (possibly refreshed) source change. Unchanged when the
        change is already mergeable.

    Raises:
        typer.Exit: When the rebase fails, whether due to conflicts
            needing manual resolution or another error.
    """
    if source_change.mergeable is not False:
        return source_change

    console.print("\n⚠️ Change has merge conflicts. Attempting to rebase...")
    rebase_result = service.rebase_change(source_change.number)

    if rebase_result["success"]:
        console.print("✅ Rebase successful! Refreshing change info...")
        source_change = service.get_change_info(source_change.number)
        _display_change_info(
            source_change,
            console=console,
            auth_method=credentials.auth_method_display(),
        )
        return source_change

    if rebase_result["conflict"]:
        console.print("\n❌ Rebase failed due to merge conflicts:")
        if rebase_result["conflicting_files"]:
            console.print("\n   Conflicting files:")
            for file_path in rebase_result["conflicting_files"]:
                console.print(f"   • {file_path}")
        console.print(
            "\n💡 To resolve: manually rebase the change locally and push a new patchset."
        )
        console.print(f"   git review -d {source_change.number}")
        console.print(f"   git rebase origin/{source_change.branch}")
        console.print("   # resolve conflicts, then:")
        console.print("   git review")
        raise typer.Exit(1)

    console.print(f"\n❌ Rebase failed: {rebase_result['error']}")
    raise typer.Exit(1)


def _resolve_gerrit_candidates(
    service: GerritService,
    source_change: GerritChangeInfo,
    parsed_url: ParsedUrl | ParsedGerritTopicUrl,
    topic: str | None,
    topic_changes: list[GerritChangeInfo] | None,
    console: Console,
) -> list[GerritChangeInfo] | None:
    """Resolve the candidate changes to compare against, by topic.

    An explicit --topic wins, then the topic from a search URL, then
    the source change's own topic. A server-side topic query is far
    cheaper and more reliable than scanning every open change; when no
    topic is available, None lets the caller fall back to a full scan.
    """
    effective_topic = topic or source_change.topic
    if isinstance(parsed_url, ParsedGerritTopicUrl):
        effective_topic = topic or parsed_url.topic

    if not effective_topic:
        console.print(f"\n🔍 Searching for similar changes on {parsed_url.host}...")
        return None

    console.print(
        f"\n🔍 Searching for changes with topic "
        f"'{effective_topic}' on {parsed_url.host}..."
    )
    # topic_changes (when set) was fetched with this same topic
    if topic_changes is not None:
        return topic_changes
    return [c for c in service.get_changes_by_topic(effective_topic) if c.is_open]


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
