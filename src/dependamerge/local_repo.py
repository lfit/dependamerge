# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Inferring the target from the repository the operator is standing in.

Running ``dependamerge merge`` with no URL should mean "this
repository".  Working that out is a matter of asking git where it is
and what it points at, then deciding whether the answer describes a
GitHub repository or a Gerrit project.

Gerrit needs identifying separately rather than being treated as an
odd-looking GitHub remote.  Its changes are not addressable as
``/owner/repo``, so a Gerrit checkout that silently fell through to the
GitHub path would fail somewhere far less informative than here.

The Gerrit heuristics in this module rank evidence rather than trusting
a name.  None of them is a trust decision --- they choose which parser
to use on the operator's own checkout, never whether to send a
credential somewhere.  Host *authorisation* stays in
``url_parser.hosts``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .git_ops import GitError, run_git
from .gitreview import GitReviewInfo, parse_gitreview
from .url_parser import ChangeSource, normalize_target

log = logging.getLogger("dependamerge.local_repo")

#: Gerrit's default SSH port.  A remote using it is Gerrit; nothing
#: else conventionally listens there.
_GERRIT_SSH_PORT = "29418"

#: Bound on every git call.  ``run_git`` has no default timeout, and
#: these run before the operator has seen any output, so a hung git
#: would look like the tool itself hanging on startup.
_GIT_TIMEOUT = 10.0

#: Remotes consulted, in order.  ``origin`` is the conventional
#: upstream; ``upstream`` is the fork convention, where ``origin`` is
#: the operator's own fork and not what they mean to merge.
_REMOTE_PREFERENCE = ("upstream", "origin")


@dataclass(frozen=True)
class LocalTarget:
    """What the current checkout points at.

    Attributes:
        source: Which platform the checkout belongs to.
        url: A URL the existing parsers understand.  Empty for Gerrit,
            whose changes are not addressable from the checkout alone.
        remote: The git remote the answer came from, for reporting.
        root: The repository's working tree root.
        gitreview: Gerrit's ``.gitreview``, when the checkout has one.
    """

    source: ChangeSource
    url: str
    remote: str
    root: Path
    gitreview: GitReviewInfo | None = None

    @property
    def is_gerrit(self) -> bool:
        """Check whether the checkout belongs to a Gerrit server."""
        return self.source == ChangeSource.GERRIT


def _git(args: list[str], cwd: Path | None) -> str | None:
    """Run a read-only git command, returning stripped stdout or None.

    Every failure mode here --- not a repository, no such remote, git
    missing entirely --- is an ordinary "cannot infer" answer rather
    than an error, because the caller always has the option of asking
    the operator for a URL.
    """
    try:
        result = run_git(args, cwd=cwd, check=True, timeout=_GIT_TIMEOUT)
    except GitError as exc:
        log.debug("git %s failed: %s", " ".join(args[1:]), exc)
        return None
    output = result.stdout.strip()
    return output or None


def repository_root(cwd: Path | None = None) -> Path | None:
    """Return the working tree root, or None outside a repository."""
    root = _git(["git", "rev-parse", "--show-toplevel"], cwd)
    return Path(root) if root else None


def remote_url(
    root: Path, *, preference: tuple[str, ...] = _REMOTE_PREFERENCE
) -> tuple[str, str] | None:
    """Return the first configured remote from ``preference``.

    Args:
        root: The repository root.
        preference: Remote names to try, most preferred first.

    Returns:
        A ``(remote_name, url)`` pair, or None when the repository has
        none of them configured.
    """
    for name in preference:
        url = _git(["git", "remote", "get-url", name], root)
        if url:
            return (name, url)
    return None


def host_suggests_gerrit(host: str) -> bool:
    """Report whether a hostname reads like a Gerrit server.

    A weak, last-resort hint used only after ``.gitreview`` and the SSH
    port have had their say.  Matching is on whole dot-separated labels
    so that ``gerrit.example.org`` and ``review.gerrit.example.org``
    qualify while ``notgerrit.example.org`` does not.

    NOT a security check.  It decides which parser to try on the
    operator's own checkout; it never authorises sending anything
    anywhere.  Host authorisation lives in
    :func:`~dependamerge.url_parser.hosts.is_supported_github_host`.

    Args:
        host: The hostname from a git remote.

    Returns:
        True when the name suggests Gerrit.
    """
    labels = (host or "").strip().lower().split(".")
    return any(label == "gerrit" or label.startswith("gerrit-") for label in labels)


def _read_gitreview(root: Path) -> GitReviewInfo | None:
    """Parse ``.gitreview`` from the working tree, if it has one.

    The existing parser is pure, but the only fetch path was the GitHub
    contents API; a checkout has the file on disk.
    """
    path = root / ".gitreview"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log.debug("could not read %s: %s", path, exc)
        return None
    return parse_gitreview(text)


def _looks_like_gerrit_remote(url: str) -> bool:
    """Report whether a remote URL is a Gerrit one."""
    raw = url.strip()
    # The port is definitive: Gerrit's SSH daemon owns 29418.  It has to
    # come from the *authority*, though.  An scp-style remote puts the
    # path after the colon, so a substring test would read
    # ``git@github.com:29418/widget.git`` --- an owner named 29418 ---
    # as a Gerrit server.
    scheme_match = re.match(r"\A[A-Za-z][A-Za-z0-9+.-]*://([^/]+)", raw)
    if scheme_match:
        authority = scheme_match.group(1).rsplit("@", 1)[-1]
        _, _, port = authority.rpartition(":")
        if port == _GERRIT_SSH_PORT:
            return True

    normalized = normalize_target(raw)
    host = normalized.split("://", 1)[-1].split("/", 1)[0]
    return host_suggests_gerrit(host)


def detect_local_target(cwd: Path | None = None) -> LocalTarget | None:
    """Work out what the current checkout points at.

    Evidence is ranked, strongest first: a ``.gitreview`` file is
    Gerrit's own declaration and settles it; then the remote's SSH
    port; then the shape of its hostname.  Anything else is treated as
    GitHub, which is what the URL parsers already assume.

    Args:
        cwd: Directory to inspect.  Defaults to the process's own.

    Returns:
        The inferred target, or None when the directory is not a git
        repository or has no usable remote.
    """
    root = repository_root(cwd)
    if root is None:
        return None

    found = remote_url(root)
    remote_name, url = found if found else ("", "")

    gitreview = _read_gitreview(root)
    if gitreview is not None and gitreview.is_valid:
        # Gerrit's own declaration of where the project lives, and the
        # reason .gitreview exists at all.  Trusted over the remote,
        # which may point at a replica or a personal mirror.
        return LocalTarget(
            source=ChangeSource.GERRIT,
            url="",
            remote=remote_name,
            root=root,
            gitreview=gitreview,
        )

    if not url:
        return None

    if _looks_like_gerrit_remote(url):
        return LocalTarget(
            source=ChangeSource.GERRIT,
            url="",
            remote=remote_name,
            root=root,
        )

    return LocalTarget(
        source=ChangeSource.GITHUB,
        url=normalize_target(url),
        remote=remote_name,
        root=root,
    )
