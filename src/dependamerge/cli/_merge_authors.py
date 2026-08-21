# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Authorship gate for the source pull request.

Only automation-authored PRs merge without ceremony.  A human-authored
source PR needs either ``--include-human-prs`` or a matching
``--override`` token before the run may proceed.
"""

from ..error_codes import (
    ExitCode,
    exit_with_error,
)
from ._app import console
from ._context import _MergeContext
from ._sha import _generate_override_sha, _validate_override_sha


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
