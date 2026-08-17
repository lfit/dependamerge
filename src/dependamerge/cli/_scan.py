# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Automation-author authorization and the similar-PR org scan.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..error_codes import (
    ExitCode,
    exit_with_error,
)

if TYPE_CHECKING:
    pass

from ._app import console
from ._context import (
    _MergeContext,
)
from ._display import (
    _format_condensed_similarity,
)
from ._sha import (
    _generate_override_sha,
    _validate_override_sha,
)


def _validate_automation_author(ctx: _MergeContext) -> None:
    """Gate a human-authored source PR behind an explicit opt-in.

    Automation-authored sources pass straight through.  A human-authored
    source needs either ``--include-human-prs`` (the documented opt-in,
    which also governs which similar PRs are acted on) or a matching
    ``--override`` SHA, retained so existing invocations keep working.

    With neither, fail fast.  Previously this printed override guidance
    and exited *successfully*, which is indistinguishable from "nothing
    to merge" when scripted, and meant ``--include-human-prs`` appeared
    to do nothing when pointed at a human-authored PR.

    Raises:
        SystemExit: When the source is human-authored and unauthorized,
            or when a supplied override SHA does not match.
    """
    assert ctx.github_client is not None
    assert ctx.source_pr is not None

    if ctx.github_client.is_automation_author(ctx.source_pr.author):
        return

    human_source_notice = (
        f"👤 Source PR is human-authored (by {ctx.source_pr.author}); "
        "proceeding because --include-human-prs was supplied."
    )

    # Deriving the override SHA costs an extra API call per pull request.
    # Skip it when --include-human-prs alone already authorizes the run
    # and there is no supplied override to check it against; at
    # organisation scale that latency and rate-limit pressure adds up.
    if ctx.include_human_prs and not ctx.override:
        console.print(human_source_notice)
        return

    commit_messages = ctx.github_client.get_pull_request_commits(
        ctx.owner, ctx.repo_name, ctx.pr_number
    )
    first_commit_line = commit_messages[0].split("\n")[0] if commit_messages else ""
    expected_sha = _generate_override_sha(ctx.source_pr, first_commit_line)

    # A supplied override must match whichever gate ultimately authorizes
    # the run.  A wrong SHA means the operator is looking at a different
    # PR than they think, and that is worth stopping for even when
    # --include-human-prs would otherwise have been sufficient.
    if ctx.override and not _validate_override_sha(
        ctx.override, ctx.source_pr, first_commit_line
    ):
        exit_with_error(
            ExitCode.VALIDATION_ERROR,
            message="❌ Invalid override SHA provided",
            details=(f"Expected SHA for this PR and author: --override {expected_sha}"),
        )

    if ctx.include_human_prs:
        console.print(human_source_notice)
        return

    if ctx.override:
        console.print(
            "Override SHA validated. Proceeding with non-automation PR merge."
        )
        console.print(
            "ℹ️ --include-human-prs is the documented way to authorize "
            "human-authored PRs; --override remains supported.",
            style="dim",
        )
        return

    exit_with_error(
        ExitCode.VALIDATION_ERROR,
        message=(
            f"❌ Source PR is human-authored (by {ctx.source_pr.author}), "
            "not from a recognized automation tool"
        ),
        details=(
            "dependamerge acts on automation PRs by default.\n"
            "To include human-authored PRs, run again with: --include-human-prs\n"
            f"To authorize only this PR instead: --override {expected_sha}\n"
            f"That SHA derives from the author '{ctx.source_pr.author}' and "
            f"commit message '{first_commit_line[:50]}...'"
        ),
    )


def _scan_and_find_similar(ctx: _MergeContext) -> None:
    """Scan org repositories and populate *ctx.all_similar_prs*."""
    assert ctx.github_client is not None
    assert ctx.source_pr is not None
    assert ctx.comparator is not None

    console.print(f"Checking owner: {ctx.owner}")

    # Start the progress tracker now — this is where the
    # long-running owner-wide scan begins.
    if ctx.progress_tracker:
        ctx.progress_tracker.start()

    # Repository enumeration and counting is handled internally
    # by GitHubService via a single-pass GraphQL query that
    # extracts totalCount on the first page and feeds it to the
    # progress tracker automatically.

    from ..github_service import GitHubService

    async def _find_similar():
        svc = GitHubService(
            token=ctx.token,
            progress_tracker=ctx.progress_tracker,
            debug_matching=ctx.debug_matching,
        )
        try:
            assert ctx.github_client is not None
            assert ctx.source_pr is not None
            assert ctx.comparator is not None
            only_automation = ctx.github_client.is_automation_author(
                ctx.source_pr.author
            )
            return await svc.find_similar_prs(
                ctx.owner,
                ctx.source_pr,
                ctx.comparator,
                only_automation=only_automation,
            )
        finally:
            await svc.close()

    ctx.all_similar_prs = asyncio.run(_find_similar())

    if ctx.progress_tracker:
        ctx.progress_tracker.stop()
        summary = ctx.progress_tracker.get_summary()
        elapsed_time = summary.get("elapsed_time")
        total_prs_analyzed = summary.get("total_prs_analyzed")
        completed_repositories = summary.get("completed_repositories")
        similar_prs_found = summary.get("similar_prs_found")
        errors_count = summary.get("errors_count", 0)
        console.print(f"\n✅ Analysis completed in {elapsed_time}")
        console.print(
            f"📊 Analyzed {total_prs_analyzed} PRs across "
            f"{completed_repositories} repositories"
        )
        console.print(f"🔍 Found {similar_prs_found} similar PRs")
        if errors_count > 0:
            console.print(f"⚠️ {errors_count} errors encountered during analysis")
        # The trailing blank line delineates the similar-PR list that
        # follows.  When no similar PRs were found there is no list to
        # separate, so the blank line is suppressed (see below).
        if ctx.all_similar_prs:
            console.print()
    else:
        console.print(f"\n🔍 Found {len(ctx.all_similar_prs)} similar PRs")

    if not ctx.all_similar_prs:
        # Not a failure: the supplied PR is simply the only one to
        # merge.  Use a neutral "skip ahead" glyph rather than ❌ so
        # the output does not read like an error.
        console.print("⏩ No similar PRs found for this owner")

    for target_pr, comparison in ctx.all_similar_prs:
        console.print(f"  • {target_pr.repository_full_name} #{target_pr.number}")
        console.print(f"    {_format_condensed_similarity(comparison)}")
