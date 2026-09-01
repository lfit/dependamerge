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
from .url_parser import (
    ChangeSource,
    UrlParseError,
    is_supported_github_host,
    normalize_target,
)

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

#: Credentials in a URL, for any scheme.
_USERINFO_RE = re.compile(r"\A([A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@")

#: Any ``scheme://`` prefix.
_SCHEME_RE = re.compile(r"\A[A-Za-z][A-Za-z0-9+.-]*://")

#: scp-style remote: ``[user@]host:path``.
_SCP_REMOTE_RE = re.compile(r"\A(?:(?P<user>[^@/]+)@)?(?P<host>[^@/:]+):(?!//)[^\s]+\Z")


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
        host: The server the checkout belongs to, however it was
            determined.  Reported back to the operator, so it is
            populated for a Gerrit checkout recognised by its remote
            as well as one carrying a ``.gitreview``.
        project: The Gerrit project, when it is known.  A remote gives
            this away in its path even without a ``.gitreview``.
    """

    source: ChangeSource
    url: str
    remote: str
    root: Path
    gitreview: GitReviewInfo | None = None
    host: str = ""
    project: str = ""

    @property
    def is_gerrit(self) -> bool:
        """Check whether the checkout belongs to a Gerrit server."""
        return self.source == ChangeSource.GERRIT


def _git(args: list[str], cwd: Path | None) -> str | None:
    """Run a read-only git command, returning stripped stdout or None.

    Every failure mode here --- not a repository, no such remote, git
    missing entirely, an unreadable working directory --- is an
    ordinary "cannot infer" answer rather than an error, because the
    caller always has the option of asking the operator for a URL.

    ``OSError`` is caught alongside ``GitError`` deliberately.
    ``run_git`` converts timeouts and non-zero exits, but process
    creation failures --- a missing ``git``, a working directory that
    does not exist --- surface as ``FileNotFoundError`` and would
    otherwise crash the command with a traceback instead of the
    guidance.
    """
    try:
        result = run_git(args, cwd=cwd, check=True, timeout=_GIT_TIMEOUT)
    except (GitError, OSError) as exc:
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

    normalized = _remote_web_url(raw)
    if normalized is None:
        return False
    host = normalized.split("://", 1)[-1].split("/", 1)[0]
    # An explicit declaration outranks a guess about the name.
    # Enterprise hostnames are arbitrary, so an operator may well have
    # declared one carrying a ``gerrit`` label, and treating it as
    # Gerrit anyway would make that declaration unusable.  The stronger
    # Gerrit evidence still wins: the SSH port above, and ``.gitreview``
    # which the caller consults first.
    try:
        if is_supported_github_host(host):
            return False
    except UrlParseError as exc:
        # A malformed *GitHub* host setting says nothing about whether
        # this remote is Gerrit.  Raising here would abort inference
        # for a Gerrit checkout over configuration it never consults,
        # and this runs before the merge command's error guard, so it
        # would surface as a traceback.  A GitHub target still reports
        # the same setting through the parsers.
        log.debug("ignoring unusable GitHub host configuration: %s", exc)
    return host_suggests_gerrit(host)


def _names_a_server(url: str) -> bool:
    """Report whether a git remote addresses a server at all.

    A remote is a URL, an scp-style address, or a filesystem path.  It
    is never *shorthand*: that is a convenience for what a human types,
    and applying it here is actively dangerous --- a relative remote
    like ``mirror/widget.git`` would expand to a real, unrelated GitHub
    repository and an omitted-target merge would act on it.

    Args:
        url: The remote URL as git reports it.

    Returns:
        True when the remote names a host rather than a local path.
    """
    raw = url.strip()
    if _SCHEME_RE.match(raw):
        return True
    scp = _SCP_REMOTE_RE.match(raw)
    if scp is None:
        return False
    # Userinfo settles it.  ``git@ghe:acme/widget.git`` addresses a
    # server whose DNS name has a single label, which an internal
    # Enterprise installation may well have, and no filesystem path
    # carries a ``user@`` prefix.
    if scp.group("user"):
        return True
    # ``C:/repos/widget.git`` is a Windows drive, not a host, and the
    # scp pattern cannot tell them apart on its own.  Without userinfo
    # a real remote host is dotted, or is localhost.
    authority = scp.group("host").lower()
    return "." in authority or authority == "localhost"


def _safe_for_log(url: str) -> str:
    """Redact any credentials from a remote before it reaches a log.

    The module contract is that no credential survives into output, and
    a log is output.  ``git_ops.redact_text`` only recognises http(s)
    URLs, but a remote this module *declines* may use any scheme ---
    ``ftp://user:password@host/repo.git`` among them --- so the
    userinfo is removed regardless of scheme.

    A token hides in three places, not one.  The query and fragment go
    too, because this runs on remotes that normalisation has *not*
    accepted, so it cannot rely on ``_remote_web_url`` having dropped
    them, and a git remote needs neither.

    Args:
        url: The remote URL as git reports it.

    Returns:
        The URL with any credentials removed from every position.
    """
    redacted = _USERINFO_RE.sub(r"\1***@", url)
    return redacted.split("?", 1)[0].split("#", 1)[0]


def _remote_web_url(url: str) -> str | None:
    """Normalise a git remote into a URL safe to show and to parse.

    Credentials reach a remote in two ways.  ``normalize_target``
    removes URL userinfo, but a query string can carry one too ---
    a remote ending ``/owner/repo.git?token=SECRET`` --- and this URL
    is printed back to the operator when a target is inferred.  A git
    remote never needs a query or a fragment, so both are dropped.

    Args:
        url: The remote URL as git reports it.

    Returns:
        A web URL with no credentials in any position, or None when
        the remote is not one a target can be derived from.  A local
        ``file://`` mirror is a perfectly valid remote that names no
        server to merge against, so it is an ordinary "cannot infer"
        answer rather than an error.
    """
    raw = url.strip()
    if not _names_a_server(raw):
        # A filesystem path, relative or absolute.  Nothing to target,
        # and emphatically not something to run through shorthand
        # expansion.
        log.debug("remote %s is a local path, not a server", _safe_for_log(raw))
        return None
    try:
        normalized = normalize_target(raw)
    except UrlParseError as exc:
        log.debug("remote %s is not a usable target: %s", _safe_for_log(raw), exc)
        return None
    stripped = normalized.split("?", 1)[0].split("#", 1)[0]
    if not stripped.startswith(("http://", "https://")):
        log.debug("remote %s does not name a server", _safe_for_log(raw))
        return None
    return stripped


def _gerrit_identity_from_remote(url: str) -> tuple[str, str]:
    """Extract ``(host, project)`` from a Gerrit remote URL.

    Gerrit remotes name the project in their path, so a checkout with
    no ``.gitreview`` still identifies itself.

    Args:
        url: The remote URL.

    Returns:
        The host and project, either of which may be empty.
    """
    normalized = _remote_web_url(url)
    if normalized is None:
        return ("", "")
    remainder = normalized.split("://", 1)[-1]
    host, _, path = remainder.partition("/")
    return (host, path.strip("/"))


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
            host=gitreview.host,
            project=gitreview.project,
        )

    if not url:
        return None

    if _looks_like_gerrit_remote(url):
        # Recognised from the remote alone.  Its host and path are the
        # only identity available, and reporting them is the difference
        # between actionable guidance and a bare refusal.
        host, project = _gerrit_identity_from_remote(url)
        return LocalTarget(
            source=ChangeSource.GERRIT,
            url="",
            remote=remote_name,
            root=root,
            host=host,
            project=project,
        )

    normalized = _remote_web_url(url)
    if normalized is None:
        # A remote git can use but this tool cannot target, such as a
        # local mirror.  Nothing to infer, and the caller can still ask
        # the operator for a URL.
        return None
    return LocalTarget(
        source=ChangeSource.GITHUB,
        url=normalized,
        remote=remote_name,
        root=root,
        host=normalized.split("://", 1)[-1].split("/", 1)[0],
    )
