# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Target resolution, error containment, and the single-PR merge flow.

Holds the parts of the ``merge`` command that are independent of its
Typer signature, so that module stays close to the user-facing
interface it declares.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import NoReturn
from urllib.parse import urlparse

import typer

from ..error_codes import (
    DependamergeError,
    ExitCode,
    convert_git_error,
    convert_github_api_error,
    convert_network_error,
    exit_for_github_api_error,
    exit_with_error,
    is_github_api_permission_error,
    is_network_error,
)
from ..git_ops import GitError
from ..github_async import (
    GraphQLError,
    RateLimitError,
    SecondaryRateLimitError,
)
from ..models import ComparisonResult, PullRequestInfo
from ..progress_tracker import MergeProgressTracker, ProgressTracker
from ..url_parser import (
    ParsedGerritTopicUrl,
    ParsedOrgUrl,
    ParsedRepoUrl,
    ParsedUrl,
    UrlParseError,
    parse_change_url,
    parse_gerrit_topic_url,
    parse_org_url,
    parse_repo_url,
)
from . import _parallel
from ._app import console
from ._context import (
    _fetch_and_validate_source_pr,
    _init_github_merge,
    _MergeContext,
    _print_debug_matching,
)
from ._gerrit_merge import _handle_gerrit_merge
from ._org_merge import _handle_org_merge
from ._parallel import (
    _handle_preview_confirmation,
    _restart_merge_progress_tracker,
)
from ._permissions import _maybe_check_merge_permissions
from ._repo_merge import _handle_repo_merge
from ._results import _display_merge_results
from ._scan import _scan_and_find_similar, _validate_automation_author


@dataclass(frozen=True)
class _MergeTarget:
    """What a ``merge`` argument resolved to, at most one being set."""

    parsed_url: ParsedUrl | None = None
    parsed_topic: ParsedGerritTopicUrl | None = None
    parsed_repo: ParsedRepoUrl | None = None
    parsed_org: ParsedOrgUrl | None = None

    @property
    def gerrit(self) -> ParsedUrl | ParsedGerritTopicUrl | None:
        """The Gerrit target, when the argument named one."""
        if self.parsed_topic is not None:
            return self.parsed_topic
        if self.parsed_url is not None and self.parsed_url.is_gerrit:
            return self.parsed_url
        return None


def _report_url_parse_failure(
    pr_url: str,
    change_err: UrlParseError,
    org_err: UrlParseError,
    repo_err: UrlParseError,
) -> NoReturn:
    """Print whichever parse error gives the most actionable guidance."""
    # Show the most relevant error: if the URL targets a
    # non-github.com host the original parse_change_url error
    # gives host-appropriate guidance (e.g. Gerrit tips),
    # whereas parse_repo_url only talks about github.com.
    from ..url_parser import _host_matches

    # Prepend scheme if missing so urlparse can extract the
    # hostname.  Without a scheme, schemeless URLs like
    # "gerrit.example.org/..." are parsed as a path with no
    # hostname, causing the wrong error to be shown.
    _norm = pr_url
    if not _norm.startswith(("http://", "https://")):
        _norm = "https://" + _norm
    try:
        host = urlparse(_norm).hostname or ""
    except Exception:
        host = ""
    if host and not _host_matches(host.lower(), "github.com"):
        # Non-github host.  An owner-shaped path (``/orgs/owner``
        # or a single bare segment) most likely means the user
        # aimed an owner-wide URL at a non-github host (e.g.
        # GHE), so surface parse_org_url's actionable rejection
        # ("Owner-wide URL parsing is only supported for
        # github.com … use a direct PR URL") instead of the
        # generic parse_change_url "cannot determine platform"
        # message.  Any other shape (including Gerrit-style
        # URLs) keeps the platform-agnostic guidance.
        segs = [s for s in urlparse(_norm).path.split("/") if s]
        if segs and (segs[0] == "orgs" or len(segs) == 1):
            console.print(f"❌ Invalid URL: {org_err}")
        else:
            console.print(f"❌ Invalid URL: {change_err}")
    else:
        console.print(f"❌ Invalid URL: {repo_err}")
    raise typer.Exit(1) from None


def _parse_merge_target(pr_url: str) -> _MergeTarget:
    """Resolve the ``merge`` argument to a PR, topic, owner, or repository."""
    # Try as a specific PR/change URL first, then a Gerrit topic search
    # URL, then an owner-wide URL (bare owner / orgs/owner), then a
    # single repository URL.
    parsed_url: ParsedUrl | None = None
    parsed_topic: ParsedGerritTopicUrl | None = None
    change_err: UrlParseError | None = None
    try:
        parsed_url = parse_change_url(pr_url)
    except UrlParseError as e:
        change_err = e
        # Not a PR/change URL — try a Gerrit topic search URL next, so
        # pasted dashboard URLs like /q/topic:some-topic work directly.
        try:
            parsed_topic = parse_gerrit_topic_url(pr_url)
        except UrlParseError:
            parsed_topic = None
    if parsed_url is not None or parsed_topic is not None or change_err is None:
        return _MergeTarget(parsed_url=parsed_url, parsed_topic=parsed_topic)

    # Not a PR URL — try owner-wide before repository.  parse_org_url
    # is strict (only a bare owner or the canonical orgs/owner forms),
    # so a two-segment owner/repo URL falls through to parse_repo_url.
    # Trying owner-wide first is required so /orgs/owner is not
    # mis-parsed by parse_repo_url as owner="orgs", repo="owner".
    try:
        return _MergeTarget(parsed_org=parse_org_url(pr_url))
    except UrlParseError as org_err:
        # Not an owner URL — try as a repository URL
        try:
            return _MergeTarget(parsed_repo=parse_repo_url(pr_url))
        except UrlParseError as repo_err:
            _report_url_parse_failure(pr_url, change_err, org_err, repo_err)


@contextlib.contextmanager
def _merge_error_guard(
    tracker: Callable[[], MergeProgressTracker | ProgressTracker | None],
    message: str,
) -> Iterator[None]:
    """Stop the live tracker and map escaping errors onto exit codes."""

    def _stop() -> None:
        current = tracker()
        if current:
            current.stop()

    try:
        yield
    except DependamergeError as exc:
        _stop()
        exc.display_and_exit()
    except (KeyboardInterrupt, SystemExit):
        _stop()
        raise
    except typer.Exit:
        _stop()
        raise
    except (
        GitError,
        RateLimitError,
        SecondaryRateLimitError,
        GraphQLError,
    ) as exc:
        _stop()
        if isinstance(exc, GitError):
            converted_error = convert_git_error(exc)
        else:
            converted_error = convert_github_api_error(exc)
        converted_error.display_and_exit()
    except Exception as e:
        _stop()
        if is_github_api_permission_error(e):
            exit_for_github_api_error(exception=e)
        elif is_network_error(e):
            converted_error = convert_network_error(e)
            converted_error.display_and_exit()
        else:
            exit_with_error(
                ExitCode.GENERAL_ERROR,
                message=message,
                details=str(e),
                exception=e,
            )


def _merge_source_and_similar(ctx: _MergeContext) -> None:
    """Merge the source PR together with the similar PRs found for it."""
    if not ctx.no_confirm or ctx.dry_run:
        console.print("\n🔍 Dependamerge Evaluation\n")

    assert ctx.source_pr is not None
    source_entry: tuple[PullRequestInfo, ComparisonResult | None] = (
        ctx.source_pr,
        None,
    )
    all_prs_to_merge: list[tuple[PullRequestInfo, ComparisonResult | None]] = [
        *ctx.all_similar_prs,
        source_entry,
    ]
    # For the real merge (``--no-confirm``) the scan has already
    # stopped the progress tracker; stand up a fresh one so the
    # background wait-status ticker has a live display to update
    # while PRs sit in the Step 5.5 auto-merge wait.  Preview and
    # dry runs keep the stopped tracker (one line per PR, no wait
    # loop) because they never execute a real merge.
    if ctx.no_confirm and not ctx.dry_run:
        _restart_merge_progress_tracker(ctx, len(all_prs_to_merge))
    try:
        merge_results = _parallel._run_parallel_merge(
            ctx,
            all_prs_to_merge,
            preview=ctx.dry_run or not ctx.no_confirm,
            # No similar-PR list was printed when none were found, so
            # skip the blank line before the merge banner.
            leading_blank=bool(ctx.all_similar_prs),
        )
    finally:
        if (
            ctx.no_confirm
            and not ctx.dry_run
            and ctx.show_progress
            and ctx.progress_tracker
        ):
            ctx.progress_tracker.stop()

    if not merge_results:
        console.print("❌ No PRs were processed")
        return

    merged_count = sum(1 for r in merge_results if r.status.value == "merged")

    # Dry run: report what *would* happen and stop before any prompt
    # or real merge.  ``no_confirm=False`` selects the "Would …"
    # preview phrasing in the results summary.
    if ctx.dry_run:
        _display_merge_results(merge_results, no_confirm=False)
        return

    if not ctx.no_confirm:
        _handle_preview_confirmation(
            ctx,
            merge_results,
            all_prs_to_merge,
            merged_count,
            len(merge_results),
        )
        return

    _display_merge_results(merge_results, ctx.no_confirm)


def _run_single_pr_merge(ctx: _MergeContext) -> None:
    """Analyse the source PR, find its siblings, and merge the batch."""
    _init_github_merge(ctx)
    _fetch_and_validate_source_pr(ctx)
    _maybe_check_merge_permissions(ctx)

    # Debug matching info for source PR
    if ctx.debug_matching:
        _print_debug_matching(ctx)

    _validate_automation_author(ctx)

    # Scan org and find similar PRs
    _scan_and_find_similar(ctx)

    _merge_source_and_similar(ctx)


def _dispatch_merge(
    ctx: _MergeContext,
    target: _MergeTarget,
    topic: str | None,
) -> None:
    """Route a resolved target to the Gerrit, owner, repo, or PR flow."""
    if topic and target.gerrit is None:
        console.print("\u274c --topic is only supported for Gerrit URLs")
        raise typer.Exit(1)

    gerrit_target = target.gerrit
    if gerrit_target is not None:
        _handle_gerrit_merge(
            parsed_url=gerrit_target,
            no_confirm=ctx.no_confirm,
            similarity_threshold=ctx.similarity_threshold,
            verbose=ctx.verbose,
            console=console,
            no_netrc=ctx.no_netrc,
            netrc_file=ctx.netrc_file,
            netrc_optional=ctx.netrc_optional,
            dry_run=ctx.dry_run,
            override=ctx.override,
            topic=topic,
            show_progress=ctx.show_progress,
        )
        return

    if target.parsed_org is not None:
        ctx.pr_url = target.parsed_org.original_url
        with _merge_error_guard(
            lambda: ctx.progress_tracker,
            "\u274c Error during owner-wide merge operation",
        ):
            _handle_org_merge(target.parsed_org, ctx)
        return

    if target.parsed_repo is not None:
        ctx.pr_url = target.parsed_repo.original_url
        with _merge_error_guard(
            lambda: ctx.progress_tracker,
            "\u274c Error during repository merge operation",
        ):
            _handle_repo_merge(target.parsed_repo, ctx)
        return

    assert target.parsed_url is not None
    ctx.pr_url = target.parsed_url.original_url
    with _merge_error_guard(
        lambda: ctx.progress_tracker, "\u274c Error during merge operation"
    ):
        _run_single_pr_merge(ctx)
