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

#: Path segments that introduce an individual change, taken from the
#: change parser: ``/owner/repo/pull/N`` on GitHub, and ``/c/.../+/N``
#: or ``/changes/N`` on Gerrit.  A path carrying one names a change
#: rather than a project, so it is not a clone URL.
_CHANGE_MARKERS = frozenset({"pull", "+", "changes"})


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


def _names_a_change(segments: list[str]) -> bool:
    """Report whether a path addresses one change rather than a project.

    The markers are the ones this package's own change parser uses ---
    ``/pull/`` for GitHub, ``/+/`` and ``/changes/`` for Gerrit --- so
    that this agrees with the parser it protects instead of being a
    second, divergent shape test.

    A marker counts only when a number follows it, which is what the
    change parsers require.  Gerrit projects nest, so a project may
    genuinely contain a ``pull`` segment: ``/org/pull/widget.git`` is a
    clone URL and must keep its stripping.

    Args:
        segments: The non-empty path segments, in order.

    Returns:
        True when the path names an individual change.
    """
    return any(
        marker.lower() in _CHANGE_MARKERS and _is_change_number(following)
        for marker, following in zip(segments, segments[1:], strict=False)
    )


def strips_git_suffix(path: str) -> bool:
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

    Returns:
        True when a trailing ``.git`` should come off.
    """
    segments = [s for s in path.split("/") if s]
    if not segments:
        return False
    last = segments[-1].lower()
    if ":" in last or "%3a" in last:
        return False
    return not _names_a_change(segments)


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
