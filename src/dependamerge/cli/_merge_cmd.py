# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The ``merge`` command and its options.

The body is deliberately thin: it validates the options, resolves the URL
shape, builds the shared context, and hands off to the matching handler
in :mod:`._merge_dispatch`.
"""

from pathlib import Path

import typer

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
from ..merge_manager import (
    DEFAULT_MERGE_TIMEOUT,
)
from ._app import DEFAULT_MAX_WAIT, app
from ._context import _MergeContext
from ._merge_dispatch import (
    _dispatch_gerrit,
    _normalise_topic,
    _parse_merge_target,
    _resolve_gerrit_target,
    _run_guarded,
    _run_single_pr_merge,
    _validate_max_wait,
)
from ._merge_inputs import _validate_merge_inputs
from ._org_merge import _handle_org_merge
from ._repo_merge import _handle_repo_merge


@app.command()
def merge(
    pr_url: str = typer.Argument(
        ...,
        help="GitHub PR URL, repository URL, owner/org URL, or Gerrit change URL",
    ),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help="Skip confirmation prompt and merge immediately",
    ),
    similarity_threshold: float = typer.Option(
        0.8, "--threshold", help="Similarity threshold for matching PRs (0.0-1.0)"
    ),
    merge_method: str = typer.Option(
        "merge", "--merge-method", help="Merge method: merge, squash, or rebase"
    ),
    token: str | None = typer.Option(
        None, "--token", help="GitHub token (or set GITHUB_TOKEN env var)"
    ),
    override: str | None = typer.Option(
        None,
        "--override",
        help=(
            "SHA hash authorizing a single non-automation PR/change. "
            "Prefer --include-human-prs; this remains supported"
        ),
    ),
    no_fix: bool = typer.Option(
        False,
        "--no-fix",
        help="Do not attempt to automatically fix out-of-date branches",
    ),
    fix_semantic_title: bool = typer.Option(
        True,
        "--fix-semantic-title/--no-fix-semantic-title",
        help=(
            "Repair an automation PR whose title differs from its single "
            "commit's subject, which permanently fails a semantic pull "
            "request check. The title is set to the commit subject, and "
            "GitHub's edited event re-runs the check. Only applied when "
            "that check is the sole failure and the difference is an "
            "elided version fragment. Default: enabled."
        ),
    ),
    rebase_local: bool = typer.Option(
        True,
        "--rebase-local/--no-rebase-local",
        help=(
            "When rebasing a behind PR, prefer a local ``git`` clone + "
            "rebase + force-push-with-lease over the GitHub REST "
            "``update-branch`` endpoint when the base branch requires "
            "verified signatures or the PR is from pre-commit-ci[bot]. "
            "The local path inherits ``~/.gitconfig`` so commits stay "
            "signed; the REST path is faster but produces unsigned "
            "commits that break verification. Default: enabled."
        ),
    ),
    merge_timeout: float = typer.Option(
        DEFAULT_MERGE_TIMEOUT,
        "--merge-timeout",
        help=(
            "Timeout in seconds for async merge operations (rebase, "
            "pre-commit.ci, recreate). Default: "
            f"{DEFAULT_MERGE_TIMEOUT:.0f}"
        ),
    ),
    max_wait: float = typer.Option(
        DEFAULT_MAX_WAIT,
        "--max-wait",
        help=(
            "Owner/user-wide runs only: global wall-clock ceiling in "
            "seconds for the whole batch. The run returns once every PR "
            "has merged or this elapses (anything still in flight keeps "
            "auto-merge armed and is reported as pending). 0 = "
            "fire-and-forget (arm auto-merge and return immediately, "
            f"never block). Default: {DEFAULT_MAX_WAIT:.0f}"
        ),
    ),
    show_progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show real-time progress updates"
    ),
    debug_matching: bool = typer.Option(
        False,
        "--debug-matching",
        help="Show detailed scoring information for PR matching",
    ),
    dismiss_copilot: bool = typer.Option(
        False,
        "--dismiss-copilot",
        help="Automatically dismiss unresolved GitHub Copilot review comments",
    ),
    force: str = typer.Option(
        "code-owners",
        "--force",
        help="Override level: 'none', 'code-owners', 'protection-rules', 'all' (default: code-owners)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose debug logging",
    ),
    no_netrc: bool = typer.Option(
        False,
        "--no-netrc",
        help="Disable .netrc credential lookup for Gerrit authentication",
    ),
    netrc_file: Path | None = typer.Option(
        None,
        "--netrc-file",
        help="Explicit path to .netrc file for Gerrit credentials",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    netrc_optional: bool = typer.Option(
        True,
        "--netrc-optional/--netrc-required",
        help="Whether to fail if .netrc file is not found (default: optional)",
    ),
    submit_gerrit_changes: bool = typer.Option(
        False,
        "--submit-gerrit-changes",
        help="Explicitly request Gerrit submission for GitHub2Gerrit PRs (already the default when neither --skip-gerrit-changes nor --ignore-github2gerrit is given)",
    ),
    skip_gerrit_changes: bool = typer.Option(
        False,
        "--skip-gerrit-changes",
        help="Skip PRs that have GitHub2Gerrit comments instead of merging them",
    ),
    ignore_github2gerrit: bool = typer.Option(
        False,
        "--ignore-github2gerrit",
        help="Ignore GitHub2Gerrit comments and merge PRs in GitHub as normal",
    ),
    include_human_prs: bool = typer.Option(
        False,
        "--include-human-prs",
        help=(
            "Authorize human-authored PRs. Required when the source PR is "
            "human-authored, and includes human-authored PRs in the similar-PR "
            "set (prompting for confirmation unless --no-confirm)"
        ),
    ),
    topic: str | None = typer.Option(
        None,
        "--topic",
        help=(
            "Gerrit only: scope the similar-change search to this topic. "
            "When omitted, the topic is extracted automatically from a "
            "Gerrit topic search URL or from the source change itself."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Analyze and preview only: never approve, merge, rebase, or "
            "close anything. Skips the write-permission pre-flight so it "
            "runs under a read-only token (e.g. in CI). Implies "
            "preview-only and suppresses confirmation prompts."
        ),
    ),
):
    """
    Bulk approve/merge pull requests or Gerrit changes.

    Supports GitHub PRs, GitHub repository URLs, and Gerrit Code Review changes.

    By default, runs in interactive mode showing what changes will apply,
    then prompts to proceed with merge. Use --no-confirm to merge immediately.

    For GitHub PRs (single PR URL), this command will:

    1. Analyze the provided PR

    2. Find similar PRs across the owner (organization or user account)

    3. Approve and merge matching PRs

    4. Automatically fix out-of-date branches (use --no-fix to disable)

    For GitHub repository URLs, this command will:

    1. Fetch all open PRs in the specified repository

    2. Filter to automation PRs only (unless --include-human-prs is given)

    3. Approve and merge matching PRs in bulk

    Repository URL formats accepted:
      https://github.com/owner/repo
      https://github.com/owner/repo/
      https://github.com/owner/repo/pulls

    For GitHub owner (organization or user) URLs, this command will:

    1. Enumerate every non-archived, non-fork repository owned by the
       organization or user (forks are excluded)

    2. Fetch their open automation PRs (unless --include-human-prs)

    3. Bulk merge them with a striped scheduler that processes at most
       one PR per repository at a time, spreading work across
       repositories to avoid racing GitHub's mergeability propagation

    Owner URL formats accepted:
      https://github.com/owner
      https://github.com/owner/
      https://github.com/orgs/owner
      https://github.com/orgs/owner/repositories

    For Gerrit changes, this command will:

    1. Analyze the provided change

    2. Find similar open changes on the server (scoped to the change's
       topic when it has one, or the --topic flag)

    3. Review (+2 Code-Review) and submit matching changes

    Gerrit topic search URLs are also accepted; all open changes
    sharing the topic form the batch:
      https://gerrit.example.org/q/topic:some-topic
      https://gerrit.onap.org/r/q/topic:some-topic

    Merges similar PRs/changes from the same automation tool.

    For user generated bulk PRs, use the --override flag with SHA hash.

    GitHub2Gerrit handling:
    By default, PRs with GitHub2Gerrit mapping comments are detected and
    the corresponding Gerrit changes are submitted (+2 Code-Review + submit).
    Use --skip-gerrit-changes to skip these PRs, or --ignore-github2gerrit
    to merge them in GitHub as normal (which may leave orphaned Gerrit changes).

    GitHub Force levels:
    - none: Respect all protections
    - code-owners: Bypass code owner review requirements (default)
    - protection-rules: Bypass branch protection checks (requires permissions)
    - all: Attempt merge despite most warnings (not recommended)

    Authentication (Gerrit):
    Credentials are loaded in this order:
    1. .netrc file (if not disabled with --no-netrc)
    2. Environment variables: GERRIT_USERNAME and GERRIT_PASSWORD

    .netrc search order: ./netrc, ~/.netrc, ~/_netrc (Windows)
    Use --netrc-file to specify an explicit path.
    """
    github2gerrit_mode = _validate_merge_inputs(
        submit_gerrit_changes,
        skip_gerrit_changes,
        ignore_github2gerrit,
        force,
        verbose,
    )
    _validate_max_wait(max_wait)
    topic = _normalise_topic(topic)

    target = _parse_merge_target(pr_url)

    ctx = _MergeContext(
        pr_url=pr_url,
        max_wait=max_wait,
        no_confirm=no_confirm,
        similarity_threshold=similarity_threshold,
        merge_method=merge_method,
        token=token,
        override=override,
        no_fix=no_fix,
        fix_semantic_title=fix_semantic_title,
        merge_timeout=merge_timeout,
        show_progress=show_progress,
        debug_matching=debug_matching,
        dismiss_copilot=dismiss_copilot,
        force=force,
        verbose=verbose,
        no_netrc=no_netrc,
        netrc_file=netrc_file,
        netrc_optional=netrc_optional,
        github2gerrit_mode=github2gerrit_mode,
        include_human_prs=include_human_prs,
        rebase_local=rebase_local,
        dry_run=dry_run,
    )

    gerrit_target = _resolve_gerrit_target(target, topic)
    if gerrit_target is not None:
        _dispatch_gerrit(gerrit_target, ctx, topic)
        return

    if target.org is not None:
        org = target.org
        ctx.pr_url = org.original_url
        _run_guarded(
            ctx,
            "\u274c Error during owner-wide merge operation",
            lambda: _handle_org_merge(org, ctx),
        )
        return

    if target.repo is not None:
        repo = target.repo
        ctx.pr_url = repo.original_url
        _run_guarded(
            ctx,
            "\u274c Error during repository merge operation",
            lambda: _handle_repo_merge(repo, ctx),
        )
        return

    assert target.url is not None
    ctx.pr_url = target.url.original_url
    _run_guarded(
        ctx,
        "\u274c Error during merge operation",
        lambda: _run_single_pr_merge(ctx),
    )
