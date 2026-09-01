# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Working out what the ``merge`` command was pointed at.

A target may be a single change, a Gerrit topic search, a repository or
an owner, and the four shapes are told apart by trying their parsers in
an order that matters (see :func:`_parse_merge_target`).  This module
answers *what was asked for*; :mod:`._merge_dispatch` acts on the
answer.

Separated so neither outgrows a reviewable size, and so the parse-order
reasoning sits next to the failure reporting that depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn
from urllib.parse import urlparse

import typer

from ..url_parser import (
    ParsedGerritTopicUrl,
    ParsedOrgUrl,
    ParsedRepoUrl,
    ParsedUrl,
    UrlParseError,
    _host_matches,
    parse_change_url,
    parse_gerrit_topic_url,
    parse_org_url,
    parse_repo_url,
)
from ._app import console


@dataclass
class _MergeTarget:
    """The URL shape ``merge`` was given; at most one field is set."""

    url: ParsedUrl | None = None
    topic: ParsedGerritTopicUrl | None = None
    repo: ParsedRepoUrl | None = None
    org: ParsedOrgUrl | None = None


def _target_host(target: _MergeTarget) -> str:
    """Return the host of whichever shape was parsed.

    Every parsed shape carries its own host, but the caller holds a
    union of four.  Reading it in one place keeps the GitHub Enterprise
    base URLs derived from the host the operator actually named, rather
    than from a default that happens to be right on github.com and
    silently wrong everywhere else.

    Args:
        target: The parsed target.

    Returns:
        The hostname, or an empty string when nothing was parsed.
    """
    for shape in (target.url, target.topic, target.repo, target.org):
        if shape is not None:
            return shape.host
    return ""


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
