# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Normalisation of abbreviated and git-remote target forms.

Every parser in this package used to begin by pasting ``https://`` onto
anything that lacked a scheme.  That works for ``github.com/owner/repo``
and fails silently for everything else: ``lfreleng-actions/dependamerge``
became the host ``lfreleng-actions`` with the path ``/dependamerge``, and
``git@github.com:owner/repo.git`` became a hostname of ``git@github.com``.

:func:`normalize_target` replaces that step for all of them, so the
shorthand forms are understood once rather than four times, and the
parsers downstream keep receiving ordinary absolute URLs.

The one genuinely ambiguous case is a two-segment input: is
``a/b`` an owner and a repository, or a host and an owner?  GitHub logins
are restricted to alphanumerics and hyphens --- they cannot contain a dot
--- so a dotted first segment is a host and an undotted one is a login.
Ports and ``localhost`` are covered for completeness.  Repository names
*may* contain dots, but a repository name is never the first segment of
a shorthand, so the rule is not lossy.
"""

from __future__ import annotations

import os
import re

# aislop-ignore-file ai-slop/hardcoded-url -- This module parses and builds
# GitHub/Gerrit URLs, so URL literals here are the subject matter, not
# stray configuration.  Enterprise hosts are always derived from the
# caller's input or from an explicit environment override.

#: The host assumed when a shorthand names no host of its own.
DEFAULT_GITHUB_HOST = "github.com"

#: Environment variables consulted, in order, for the default host.
#: ``GH_HOST`` is the GitHub CLI's own variable, so an operator who has
#: already pointed ``gh`` at their Enterprise instance gets the same
#: shorthand behaviour here without configuring anything twice.
_HOST_ENV_VARS = ("DEPENDAMERGE_GITHUB_HOST", "GH_HOST")

# scp-style remote: [user@]host:path, with no scheme and no leading
# slash on the path.  Whether a bare ``host:something`` is scp or
# ``host:port`` is decided in code --- see :func:`_is_scp_remote`.
_SCP_REMOTE_RE = re.compile(
    r"\A(?:(?P<user>[^@/]+)@)?(?P<host>[^@/:]+):(?P<path>[^\s]+)\Z"
)

# A path that is purely a number, i.e. indistinguishable from a port.
_NUMERIC_PATH_RE = re.compile(r"\A\d+(?:/|\Z)")

# A path segment that names a host rather than an owner.
_PORT_SUFFIX_RE = re.compile(r":\d+\Z")

# GitHub login grammar: alphanumerics and hyphens, no leading or
# trailing hyphen, at most 39 characters.  Used to decide whether a
# bare token is plausibly an owner before treating it as one, so that
# genuine rubbish still fails fast with "Invalid URL" instead of being
# expanded into a request for a repository that cannot exist.
_OWNER_RE = re.compile(r"\A[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")

#: Host named by ``--github-host`` on the command line, if any.
#:
#: Process-wide because it is process-wide configuration: the flag is a
#: higher-priority source of the same setting the environment variables
#: provide, and the parsers that consult it are pure functions reached
#: long before any per-run context object exists.  Set once at command
#: entry via :func:`set_github_host`.
_HOST_OVERRIDE: str | None = None


def set_github_host(host: str | None) -> None:
    """Record the host named by ``--github-host``.

    Takes priority over ``DEPENDAMERGE_GITHUB_HOST`` and ``GH_HOST``
    for both purposes those serve: the default a bare shorthand
    resolves against, and the set of hosts permitted at all.  Naming a
    host on the command line is at least as deliberate as exporting it.

    Anything that is not a string is treated as absent.  This tolerates
    direct Python calls to the commands (as the tests make), where
    Typer's ``OptionInfo`` default object arrives unresolved --- the
    same allowance ``_normalise_topic`` and ``_validate_max_wait``
    make.

    Args:
        host: The hostname, or None to clear a previous value.
    """
    global _HOST_OVERRIDE
    if not isinstance(host, str):
        _HOST_OVERRIDE = None
        return
    cleaned = _clean_host(host)
    _HOST_OVERRIDE = cleaned or None


def github_host_override() -> str | None:
    """Return the host named by ``--github-host``, if any."""
    return _HOST_OVERRIDE


def enterprise_hosts() -> tuple[str, ...]:
    """Return the GitHub Enterprise hosts the operator has declared.

    Enterprise installs use arbitrary hostnames, so there is no way to
    recognise one from the name alone.  Trusting whatever host appears
    in a URL would mean sending the caller's token wherever a pasted or
    mistyped link points, so a host has to be declared before it is
    used.

    Declared by ``--github-host`` on the command line, by
    ``DEPENDAMERGE_GITHUB_HOSTS`` (comma-separated), and by the
    single-host ``DEPENDAMERGE_GITHUB_HOST`` / ``GH_HOST`` variables
    that also set the default for shorthand --- naming a host as your
    default is a clear enough statement that you trust it.

    Returns:
        A tuple of lowercased hostnames, without duplicates.
    """
    seen: dict[str, None] = {}
    if _HOST_OVERRIDE:
        seen[_HOST_OVERRIDE] = None
    raw = os.environ.get("DEPENDAMERGE_GITHUB_HOSTS") or ""
    for candidate in raw.split(","):
        host = _clean_host(candidate)
        if host:
            seen[host] = None
    for name in _HOST_ENV_VARS:
        host = _clean_host(os.environ.get(name) or "")
        if host:
            seen[host] = None
    return tuple(seen)


def _clean_host(value: str) -> str:
    """Reduce a configured value to a bare lowercase hostname.

    Raises rather than trimming a port.  Ports are unsupported end to
    end --- ``urlparse`` drops them before the API base URLs are built
    --- so a configured ``host:8443`` would otherwise expand shorthand
    into a URL its own parser then rejects, or quietly address port
    443.

    Raises:
        UrlParseError: If the configured value names a port.
    """
    from .models import UrlParseError

    host = _strip_scheme(value.strip()).strip("/").split("/", 1)[0].lower()
    if not host:
        return ""
    name, _, port = host.rpartition(":")
    if name and port:
        raise UrlParseError(
            f"Configured GitHub host {host!r} names a port, which is not "
            "supported: the port cannot be carried through to the API "
            f"base URL, so requests would go to {name} on the default "
            "port instead. Configure the host without a port."
        )
    return host


def default_github_host() -> str:
    """Return the host a bare shorthand should resolve against.

    Resolution order, highest priority first:

    1. ``--github-host`` on the command line
    2. ``DEPENDAMERGE_GITHUB_HOST``
    3. ``GH_HOST`` --- the GitHub CLI's own variable, so an operator
       who has already pointed ``gh`` at an Enterprise instance does
       not configure the same thing twice
    4. github.com

    Returns:
        A bare lowercase hostname.
    """
    if _HOST_OVERRIDE:
        return _HOST_OVERRIDE
    for name in _HOST_ENV_VARS:
        value = _clean_host(os.environ.get(name) or "")
        if value:
            return value
    return DEFAULT_GITHUB_HOST


def _strip_scheme(value: str) -> str:
    """Remove any ``scheme://`` prefix from ``value``."""
    return re.sub(r"\A[A-Za-z][A-Za-z0-9+.-]*://", "", value)


def looks_like_host(segment: str) -> bool:
    """Report whether a leading path segment names a host.

    GitHub owner logins are alphanumerics and hyphens only, so a dot
    settles it, and so does a port.

    Bare ``localhost`` deliberately does *not* qualify.  It satisfies
    the login grammar and names a real GitHub account, so treating it
    as a host would contradict the documented rule and make
    ``localhost/widget`` unreachable.  A local server still works when
    named with a port or an explicit scheme.

    Args:
        segment: The first segment of a schemeless target.

    Returns:
        True when the segment should be read as a hostname.
    """
    segment = segment.strip().lower()
    if not segment:
        return False
    if _PORT_SUFFIX_RE.search(segment):
        return True
    return "." in segment


def looks_like_owner(segment: str) -> bool:
    """Report whether a segment is a plausible GitHub owner login.

    Logins are alphanumerics and hyphens, no leading or trailing
    hyphen, at most 39 characters.  Checking the shape keeps the
    shorthand from swallowing arbitrary text: ``lfreleng-actions`` is an
    owner, ``not a url`` is not.

    Args:
        segment: The first segment of a schemeless target.

    Returns:
        True when the segment could name an owner.
    """
    return bool(_OWNER_RE.match(segment.strip()))


def _strips_git_suffix(path: str) -> bool:
    """Report whether ``.git`` on this path is a clone-URL artefact.

    It is, on a repository path.  It is not inside a Gerrit search,
    where the trailing text is a *value*: ``/q/topic:release.git`` names
    a topic that genuinely ends in ``.git``, and trimming it silently
    searches for the wrong thing.

    The test is the final segment carrying a query operator's colon,
    encoded or otherwise --- not the presence of a ``q`` segment, which
    an owner may legitimately be called.  ``github.com/q/widget.git`` is
    a clone URL belonging to the owner ``q``.

    Args:
        path: The URL path, without query or fragment.

    Returns:
        True when a trailing ``.git`` should come off.
    """
    segments = [s for s in path.split("/") if s]
    if not segments:
        return False
    last = segments[-1].lower()
    return ":" not in last and "%3a" not in last


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


def _is_scp_remote(match: re.Match[str]) -> bool:
    """Decide whether an ambiguous ``host:tail`` is an scp remote.

    ``git@github.com:29418/widget.git`` is a clone URL for an owner
    named ``29418``; ``ghe.example.com:8443/acme`` is a host and a
    port.  Userinfo settles it --- a port never follows one --- and
    without userinfo a purely numeric leading segment is read as a
    port, which is the commoner intent.
    """
    if match.group("user"):
        return True
    return not _NUMERIC_PATH_RE.match(match.group("path"))


def normalize_target(value: str, *, default_host: str | None = None) -> str:
    """Expand an abbreviated or git-remote target into an absolute URL.

    Understood forms, in addition to ordinary ``http(s)://`` URLs which
    pass through with only a ``.git`` suffix removed:

    ==============================  =====================================
    Input                           Result
    ==============================  =====================================
    ``owner``                       ``https://github.com/owner``
    ``owner/repo``                  ``https://github.com/owner/repo``
    ``github.com/owner``            ``https://github.com/owner``
    ``ghe.example.com/owner/repo``  ``https://ghe.example.com/owner/repo``
    ``git@github.com:owner/repo``   ``https://github.com/owner/repo``
    ``ssh://git@host:29418/proj``   ``https://host/proj``
    ==============================  =====================================

    Args:
        value: The raw target as typed, pasted, or read from a remote.
        default_host: Host to assume for a shorthand that names none.
            Defaults to :func:`default_github_host`.

    Returns:
        An absolute ``http(s)://`` URL.  Empty input is returned
        unchanged so the existing "URL cannot be empty" errors keep
        their wording and their callers.
    """
    value = (value or "").strip()
    if not value:
        return value

    # A rooted path names no host and is not a shorthand --- nobody
    # types "/owner/repo" to mean an abbreviation.  Left alone so the
    # parsers still reject it with "URL must include a hostname" rather
    # than silently resolving it against the default host.
    if value.startswith("/"):
        return value

    host = (default_host or default_github_host()).strip().lower()

    # ssh://, git://, and friends: keep the host and path, drop the
    # scheme, any credentials, and any port.  A Gerrit SSH remote such
    # as ssh://user@gerrit.example.org:29418/releng/tool differs from
    # its web URL only in those three things.
    scheme_match = re.match(r"\A([A-Za-z][A-Za-z0-9+.-]*)://(.*)\Z", value, re.DOTALL)
    if scheme_match:
        scheme = scheme_match.group(1).lower()
        if scheme in ("http", "https"):
            # Credentials are stripped even here, where the scheme and
            # port are kept.  A clone remote may embed a token, and the
            # normalised URL is printed back to the operator when a
            # target is inferred from a checkout --- which would put
            # that token in the terminal and in any captured log.
            return _rebuild(
                f"{scheme}://"
                + _strip_userinfo(scheme_match.group(2), strip_port=False)
            )
        remainder = scheme_match.group(2)
        # A non-web scheme carries a transport port --- Gerrit's 29418,
        # for instance --- which has no bearing on the web URL, so it
        # goes along with any credentials.
        return _rebuild("https://" + _strip_userinfo(remainder, strip_port=True))

    # scp-style remote, which has no scheme at all.
    scp = _SCP_REMOTE_RE.match(value)
    if scp and _is_scp_remote(scp):
        return _rebuild(f"https://{scp.group('host')}/{scp.group('path').lstrip('/')}")

    # Schemeless.  Whether the first segment is a host or an owner is
    # what decides between a URL and a shorthand.
    first = value.split("/", 1)[0]
    if looks_like_host(first):
        # Schemeless but host-shaped, so this is a web URL missing only
        # its scheme.  Any port is part of that URL and is kept.
        return _rebuild("https://" + _strip_userinfo(value, strip_port=False))

    if not looks_like_owner(first):
        # Neither a host nor a possible login.  Returned unchanged so
        # the parsers reject it on their own terms, rather than being
        # expanded into a plausible-looking URL that cannot resolve.
        return value

    return _rebuild(f"https://{host}/{value.lstrip('/')}")


def _strip_userinfo(authority_and_path: str, *, strip_port: bool) -> str:
    """Drop ``user@`` --- and optionally ``:port`` --- from ``host/path``."""
    parts = authority_and_path.split("/", 1)
    authority = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    if "@" in authority:
        authority = authority.rsplit("@", 1)[1]
    if strip_port:
        authority = _PORT_SUFFIX_RE.sub("", authority)
    return f"{authority}/{rest}" if rest else authority


def _rebuild(url: str, *, strip_git: bool = True) -> str:
    """Reassemble ``url``, optionally dropping a ``.git`` path suffix."""
    if not strip_git:
        return url
    match = re.match(r"\A(https?://[^/]+)(/.*)?\Z", url, re.DOTALL)
    if not match:
        return url
    authority = match.group(1)
    path = match.group(2) or ""
    if not path:
        return authority
    # Preserve any query or fragment; only the path carries ``.git``.
    split = re.match(r"\A([^?#]*)(.*)\Z", path, re.DOTALL)
    assert split is not None
    path_part = split.group(1)
    if not _strips_git_suffix(path_part):
        return url
    return f"{authority}{strip_git_suffix(path_part)}{split.group(2)}"
