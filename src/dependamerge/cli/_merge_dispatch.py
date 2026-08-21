# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
URL resolution and dispatch for the ``merge`` command.

``merge`` accepts four URL shapes --- a pull request or Gerrit change, a
Gerrit topic search, a GitHub owner, and a single repository --- each of
which routes to a different handler.  Resolving which shape was given,
guarding the options that only apply to some of them, and mapping the
handlers' exceptions onto exit codes all live here, so the command
itself stays close to a declaration of its options.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn
from urllib.parse import urlparse

import typer

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
import dependamerge.cli as _pkg

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
from ._app import console
from ._context import _MergeContext
from ._gerrit_merge import _handle_gerrit_merge
from ._merge_authors import _validate_automation_author
from ._merge_inputs import (
    _fetch_and_validate_source_pr,
    _init_github_merge,
)
from ._merge_permissions import _maybe_check_merge_permissions
from ._merge_report import _display_merge_results
from ._merge_scan import (
    _handle_preview_confirmation,
    _restart_merge_progress_tracker,
    _scan_and_find_similar,
)
from ._pr_display import _print_debug_matching


@dataclass
class _MergeTarget:
    """The URL shape ``merge`` was given; at most one field is set."""

    url: ParsedUrl | None = None
    topic: ParsedGerritTopicUrl | None = None
    repo: ParsedRepoUrl | None = None
    org: ParsedOrgUrl | None = None


def _validate_max_wait(max_wait: float) -> None:
    """Reject a negative ``--max-wait``.

    The documented contract is 0 = fire-and-forget and > 0 = wall-clock
    ceiling; a negative value has no defined meaning and would otherwise
    be silently coerced into a surprising "instant no-wait" run
    (max_wait <= 0).  Fail fast so the flag's behaviour stays aligned
    with its help text.  The isinstance guard tolerates direct Python
    calls to ``merge`` (e.g. in tests), where Typer's ``OptionInfo``
    default object is passed unresolved.

    Args:
        max_wait: The wall-clock ceiling supplied on the command line.

    Raises:
        typer.Exit: The value is negative.
    """
    if isinstance(max_wait, int | float) and max_wait < 0:
        console.print(
            "❌ Invalid --max-wait: must be 0 (fire-and-forget) or a "
            "positive number of seconds"
        )
        raise typer.Exit(1)


def _normalise_topic(topic: str | None) -> str | None:
    """Reduce ``--topic`` to a non-empty string or ``None``.

    Tolerates direct Python calls to ``merge`` (e.g. in tests), where
    Typer's ``OptionInfo`` default object is passed unresolved and would
    otherwise be truthy.

    Args:
        topic: The raw ``--topic`` value.

    Returns:
        The stripped topic, or ``None`` when none was supplied.
    """
    if not isinstance(topic, str):
        return None
    if not topic.strip():
        return None
    return topic.strip()


def _report_unparsable_url(
    pr_url: str,
    change_err: UrlParseError,
    org_err: UrlParseError,
    repo_err: UrlParseError,
) -> NoReturn:
    """Report the most relevant parse failure and exit.

    If the URL targets a non-github.com host the original
    ``parse_change_url`` error gives host-appropriate guidance (e.g.
    Gerrit tips), whereas ``parse_repo_url`` only talks about github.com.

    Args:
        pr_url: The URL as the operator typed it.
        change_err: Failure from ``parse_change_url``.
        org_err: Failure from ``parse_org_url``.
        repo_err: Failure from ``parse_repo_url``.

    Raises:
        typer.Exit: Always.
    """
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
    """Resolve which of the accepted URL shapes ``pr_url`` is.

    Tries a specific PR/change URL first, then a Gerrit topic search URL,
    then an owner-wide URL (bare owner / orgs/owner), then a single
    repository URL.

    Args:
        pr_url: The URL as the operator typed it.

    Returns:
        The target, with exactly one field set.

    Raises:
        typer.Exit: The URL matches none of the accepted shapes.
    """
    target = _MergeTarget()
    change_err: UrlParseError | None = None
    try:
        target.url = parse_change_url(pr_url)
    except UrlParseError as e:
        change_err = e
        # Not a PR/change URL — try a Gerrit topic search URL next, so
        # pasted dashboard URLs like /q/topic:some-topic work directly.
        try:
            target.topic = parse_gerrit_topic_url(pr_url)
        except UrlParseError:
            target.topic = None
    if target.url is None and target.topic is None and change_err is not None:
        # Not a PR URL — try owner-wide before repository.  parse_org_url
        # is strict (only a bare owner or the canonical orgs/owner forms),
        # so a two-segment owner/repo URL falls through to parse_repo_url.
        # Trying owner-wide first is required so /orgs/owner is not
        # mis-parsed by parse_repo_url as owner="orgs", repo="owner".
        try:
            target.org = parse_org_url(pr_url)
        except UrlParseError as org_err:
            # Not an owner URL — try as a repository URL
            try:
                target.repo = parse_repo_url(pr_url)
            except UrlParseError as repo_err:
                _report_unparsable_url(pr_url, change_err, org_err, repo_err)
    return target


def _resolve_gerrit_target(
    target: _MergeTarget,
    topic: str | None,
) -> ParsedUrl | ParsedGerritTopicUrl | None:
    """Return the Gerrit target to submit, if the URL named one.

    Args:
        target: The resolved URL shape.
        topic: The normalised ``--topic`` value.

    Returns:
        The change or topic URL to submit, or ``None`` when the URL is a
        GitHub one.

    Raises:
        typer.Exit: ``--topic`` was given for a non-Gerrit URL.
    """
    is_gerrit = target.topic is not None or (
        target.url is not None and target.url.is_gerrit
    )
    if topic and not is_gerrit:
        console.print("❌ --topic is only supported for Gerrit URLs")
        raise typer.Exit(1)

    if not is_gerrit:
        return None
    if target.topic is not None:
        return target.topic
    assert target.url is not None
    return target.url


def _dispatch_gerrit(
    gerrit_target: ParsedUrl | ParsedGerritTopicUrl,
    ctx: _MergeContext,
    topic: str | None,
) -> None:
    """Submit a Gerrit change or topic using the collected options.

    Args:
        gerrit_target: The change or topic URL to submit.
        ctx: Shared merge context populated with CLI parameters.
        topic: The normalised ``--topic`` value.
    """
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


def _run_guarded(
    ctx: _MergeContext,
    message: str,
    action: Callable[[], None],
) -> None:
    """Run a merge handler, mapping its failures onto exit codes.

    Every merge path shares this ladder: structured errors display and
    exit themselves, interrupts and ``typer.Exit`` propagate untouched,
    known git and GitHub API errors are converted, and anything else is
    categorised before exiting.  The progress display is stopped first in
    every case so a failure never leaves a live render behind.

    Args:
        ctx: Shared merge context, holding the progress tracker.
        message: Operation-specific text for the fallback error.
        action: The handler to run.
    """
    try:
        action()
    except DependamergeError as exc:
        if ctx.progress_tracker:
            ctx.progress_tracker.stop()
        exc.display_and_exit()
    except (KeyboardInterrupt, SystemExit):
        if ctx.progress_tracker:
            ctx.progress_tracker.stop()
        raise
    except typer.Exit:
        if ctx.progress_tracker:
            ctx.progress_tracker.stop()
        raise
    except (
        GitError,
        RateLimitError,
        SecondaryRateLimitError,
        GraphQLError,
    ) as exc:
        if ctx.progress_tracker:
            ctx.progress_tracker.stop()
        if isinstance(exc, GitError):
            converted_error = convert_git_error(exc)
        else:
            converted_error = convert_github_api_error(exc)
        converted_error.display_and_exit()
    except Exception as e:
        if ctx.progress_tracker:
            ctx.progress_tracker.stop()
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


def _run_single_pr_merge(ctx: _MergeContext) -> None:
    """Merge one PR and every similar PR found across its owner.

    Args:
        ctx: Shared merge context populated with CLI parameters.
    """
    _init_github_merge(ctx)
    _fetch_and_validate_source_pr(ctx)
    _maybe_check_merge_permissions(ctx)

    # Debug matching info for source PR
    if ctx.debug_matching:
        _print_debug_matching(ctx)

    _validate_automation_author(ctx)

    # Scan org and find similar PRs
    _scan_and_find_similar(ctx)

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
        merge_results = _pkg._run_parallel_merge(
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
