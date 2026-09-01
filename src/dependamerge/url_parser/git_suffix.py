# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Deciding when a trailing ``.git`` is a clone-URL artefact.

Clone URLs carry the suffix and web URLs do not, so a remote copied
from ``git remote -v`` would otherwise name a repository literally
called ``dependamerge.git``.  Removing it unconditionally is worse,
because on some paths the trailing text is a *value* rather than a
repository name, and on others removing it repairs a malformed URL
into a valid reference to something the operator never named.

This lives apart from :mod:`shorthand` because the question is its own:
it is about what a path *means*, not about expanding an abbreviation.
"""

from __future__ import annotations

from .models import UrlParseError

#: The Gerrit path segment that separates a project from its change
#: number, and the prefix its REST API uses.
_GERRIT_CHANGE_SEP = "+"
_GERRIT_CHANGE_VIEW = "c"
_GERRIT_REST_PREFIX = "changes"


def _is_change_number(segment: str) -> bool:
    """Report whether a path segment is a change or pull request number.

    The trailing ``.git`` is discounted before the test, because that
    suffix is exactly what the caller is deciding whether to remove.

    Args:
        segment: A single path segment.

    Returns:
        True when the segment is a run of digits.
    """
    return segment.removesuffix(".git").isdigit()


def _is_github_pull_request(segments: list[str]) -> bool:
    """Match ``/owner/repo/pull/N``, the GitHub pull request shape.

    Position matters.  A Gerrit project may nest a segment called
    ``pull`` at any depth, so only the third segment counts.
    """
    return (
        len(segments) >= 4
        and segments[2].lower() == "pull"
        and _is_change_number(segments[3])
    )


def _is_gerrit_change(segments: list[str]) -> bool:
    """Match ``[/base]/c/<project>/+/N``, the Gerrit change shape.

    The ``/c/`` view segment leads, optionally behind a single base
    path segment, and the number follows the ``+`` separator.
    """
    lowered = [s.lower() for s in segments]
    if _GERRIT_CHANGE_VIEW not in lowered[:2]:
        return False
    view = lowered.index(_GERRIT_CHANGE_VIEW)
    if _GERRIT_CHANGE_SEP not in segments[view:]:
        return False
    sep = segments.index(_GERRIT_CHANGE_SEP, view)
    return sep + 1 < len(segments) and _is_change_number(segments[sep + 1])


def _is_gerrit_rest_change(segments: list[str]) -> bool:
    """Match ``/changes/N``, the Gerrit REST shape.

    Anchored at the root, so a project nesting a ``changes`` segment is
    left alone.
    """
    return (
        len(segments) >= 2
        and segments[0].lower() == _GERRIT_REST_PREFIX
        and _is_change_number(segments[1])
    )


def _is_github_host(host: str) -> bool:
    """Report whether this tool treats ``host`` as GitHub.

    The import is deferred because :mod:`hosts` imports :mod:`shorthand`,
    which imports this module; a module-level import would close that
    cycle.

    Args:
        host: The hostname from the URL being normalised.

    Returns:
        True for github.com, its subdomains, and any declared
        Enterprise host.
    """
    from .hosts import is_supported_github_host

    try:
        return is_supported_github_host(host)
    except UrlParseError:
        # Unusable host configuration says nothing about this path, and
        # normalisation must not fail over settings it does not need.
        # An unreadable declaration is treated as no declaration.
        return False


def _names_a_change(segments: list[str], host: str) -> bool:
    """Report whether a path addresses one change rather than a project.

    Each shape mirrors the corresponding regex in the change parser,
    **including where the marker sits**, so that this agrees with the
    parser it protects rather than being a second, divergent test.

    Which shapes apply depends on the host, because the markers are
    ordinary names on the other platform.  ``changes`` and ``c`` are
    valid GitHub logins, so ``github.com/changes/123.git`` is a clone
    URL for the repository ``123``, not Gerrit change 123.  The change
    parser resolves this the same way, by asking about the host before
    trying the Gerrit shapes.

    A ``/pull/N`` path stays a GitHub shape on every host, matching
    ``_is_github_url``.  On an undeclared host that is also the safe
    reading: keeping the suffix refuses the URL, whereas removing it
    would repair a malformed target into a live pull request.

    Args:
        segments: The non-empty path segments, in order.
        host: The hostname the URL names.

    Returns:
        True when the path names an individual change.
    """
    if _is_github_host(host):
        return _is_github_pull_request(segments)
    return (
        _is_github_pull_request(segments)
        or _is_gerrit_change(segments)
        or _is_gerrit_rest_change(segments)
    )


def strips_git_suffix(path: str, host: str) -> bool:
    """Report whether ``.git`` on this path is a clone-URL artefact.

    It is, on a repository path.  A clone URL's path *is* the project,
    so anything naming something else within a project is not one, and
    two shapes have to be excluded.

    A Gerrit search carries a *value* in its trailing text:
    ``/q/topic:release.git`` names a topic that genuinely ends in
    ``.git``, and trimming it silently searches for the wrong thing.
    The test is the final segment carrying a query operator's colon,
    encoded or otherwise --- not the presence of a ``q`` segment, which
    an owner may legitimately be called.  ``github.com/q/widget.git`` is
    a clone URL belonging to the owner ``q``.

    A change URL such as ``/acme/widget/pull/7.git`` is malformed: no
    such clone URL exists.  Trimming the suffix repairs it into a
    perfectly valid reference to pull request 7, so the tool acts on a
    change the operator never named.  Such a URL stays invalid.

    Args:
        path: The URL path, without query or fragment.
        host: The hostname the URL names.  Required, because the same
            path means different things on the two platforms.

    Returns:
        True when a trailing ``.git`` should come off.
    """
    segments = [s for s in path.split("/") if s]
    if not segments:
        return False
    last = segments[-1].lower()
    if ":" in last or "%3a" in last:
        return False
    return not _names_a_change(segments, host)


def strip_git_suffix(path: str) -> str:
    """Remove a trailing ``.git`` from a URL path.

    Clone URLs carry it and web URLs do not, so without this a remote
    copied from ``git remote -v`` yields a repository literally named
    ``dependamerge.git``.

    A path that does not end in ``.git`` is returned untouched ---
    including its trailing slashes, which callers record verbatim as
    the original URL.
    """
    trimmed = path.rstrip("/")
    # Guard on the final *segment*, not the whole path: "/.git" has no
    # repository name in front of the suffix, so there is nothing to
    # strip down to.
    last = trimmed.rsplit("/", 1)[-1]
    if last.endswith(".git") and len(last) > len(".git"):
        return trimmed[: -len(".git")]
    return path
