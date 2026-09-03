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

    Which shapes apply depends on the host, but only where the two
    platforms genuinely collide.  ``changes`` and ``c`` are valid
    GitHub logins, so ``github.com/changes/123.git`` is a clone URL for
    the repository ``123`` rather than Gerrit change 123, and the root
    ``/changes/N`` form is therefore read as Gerrit off GitHub only.
    The change parser resolves the same ambiguity the same way, by
    asking about the host before trying that shape.

    The full ``/c/<project>/+/N`` shape stays protected everywhere.  A
    ``+`` segment cannot appear in a GitHub owner or repository name,
    so nothing is given up by honouring it, and dropping it let a
    malformed ``/c/project/+/123.git`` be repaired into a live change
    reference on a declared Enterprise host.

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
        return _is_github_pull_request(segments) or _is_gerrit_change(segments)
    return (
        _is_github_pull_request(segments)
        or _is_gerrit_change(segments)
        or _is_gerrit_rest_change(segments)
    )


def _names_a_page_route(segments: list[str], host: str) -> bool:
    """Report whether a path addresses a GitHub web page, not a project.

    ``/orgs/acme/repositories`` and ``/acme/widget/pulls`` are pages the
    site serves, so no clone URL ends in them.  Removing a ``.git``
    tail repairs such a URL into a valid target --- and for the ``orgs``
    routes, a *broader* one: ``/orgs/acme.git`` would become an
    owner-wide merge of everything ``acme`` owns.

    GitHub only.  These are route names there and ordinary project path
    segments on Gerrit, which is the same distinction
    :func:`_names_a_change` draws for ``changes``.

    Args:
        segments: The non-empty path segments, in order.
        host: The hostname the URL names.

    Returns:
        True when the path names a page rather than a project.
    """
    if not _is_github_host(host):
        return False
    if segments[0].lower() == "orgs":
        return True
    if len(segments) < 2:
        # A GitHub clone URL always names an owner *and* a repository,
        # so a single segment is not one.  Trimming the suffix turns
        # ``github.com/acme.git`` into the owner URL for ``acme`` and
        # the dispatcher then merges every repository they own ---
        # scope invented out of a malformed URL.  Gerrit projects do
        # sit at the root, which is why this is gated on the host.
        return True
    # The suffix is still attached at this point, so it is discounted
    # before the comparison --- the last segment reads ``pulls.git``.
    return len(segments) >= 3 and segments[-1].lower().removesuffix(".git") == "pulls"


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
    if _names_a_page_route(segments, host):
        return False
    return not _names_a_change(segments, host)


def has_stray_git_suffix(path: str) -> bool:
    """Report whether a path's final segment still ends in ``.git``.

    Normalisation removes the suffix where it is a clone-URL artefact
    and *preserves* it everywhere else, as a marker that the target is
    not a clone URL.  A parser that reaches a preserved suffix is
    therefore looking at a malformed target.

    Honouring the marker here is what turns "left unchanged" into
    "refused".  Leaving it to normalisation alone was not enough:
    ``/acme.git`` merely became the owner ``acme.git``, and
    ``/pull/7/files.git`` still matched the pull request shape, because
    both parsers accept trailing segments.

    Not for every parser.  A repository *may* be called ``widget.git``,
    reached through the clone URL ``widget.git.git``, and a Gerrit topic
    may end in ``.git`` too --- in both the suffix is part of a name
    rather than a leftover.

    Args:
        path: The URL path, without query or fragment.

    Returns:
        True when the last segment carries a leftover ``.git``.
    """
    segments = [s for s in path.split("/") if s]
    if not segments:
        return False
    last = segments[-1].lower()
    # Any case-insensitive ``.git`` ending counts, including a segment
    # that is *exactly* ``.git``.  This helper is consulted only by the
    # owner and change parsers, where a repository name can never be
    # the last segment, so there is nothing to protect: requiring a
    # name in front let ``/pull/7/.git`` through, and matching case
    # let ``/pull/7/files.GIT`` through, both acting on pull request 7.
    return last.endswith(".git")


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
