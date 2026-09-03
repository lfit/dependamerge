# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Which GitHub hosts the operator has declared, and which is the default.

Enterprise installs use arbitrary hostnames, so there is no way to
recognise one from the name alone.  Trusting whatever host appears in a
URL would mean sending the caller's token wherever a pasted or mistyped
link points, so a host has to be *declared* before it is used.

Separated from the shorthand expansion that consumes it: this module
answers "which hosts may we address, and which do we assume", while
:mod:`shorthand` answers "what did the operator type".
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator

from .models import UrlParseError

# aislop-ignore-file ai-slop/hardcoded-url -- The dotcom hostname is the
# subject matter here, not stray configuration.

#: The host assumed when a shorthand names no host of its own.
DEFAULT_GITHUB_HOST = "github.com"

#: Environment variables consulted, in order, for the default host.
#: ``GH_HOST`` is the GitHub CLI's own variable, so an operator who has
#: already pointed ``gh`` at their Enterprise instance gets the same
#: shorthand behaviour here without configuring anything twice.
_HOST_ENV_VARS = ("DEPENDAMERGE_GITHUB_HOST", "GH_HOST")

#: A bare hostname: dot-separated labels of letters, digits and
#: hyphens.  Anything else in a configured value --- userinfo above
#: all --- makes the string address a different server than it looks
#: like, so it is refused rather than trimmed.
_HOSTNAME_RE = re.compile(
    r"\A[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*\Z"
)

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
    Typer's ``OptionInfo`` default object arrives unresolved.

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


def iter_enterprise_hosts() -> Iterator[str]:
    """Yield declared Enterprise hosts lazily, in precedence order.

    Lazily, so a caller that only needs to know whether *some* host
    matches can stop at the first one.  Building the whole tuple
    validates every source, so a malformed lower-priority value --- a
    stale ``GH_HOST`` naming a port, say --- rejected a host that a
    higher-priority ``--github-host`` had already declared, which made
    the documented precedence ineffective.  A bad value is still
    reported, but only once nothing above it has answered the question.

    Yields:
        Lowercased hostnames, without duplicates.

    Raises:
        UrlParseError: A value reached during iteration names a port.
    """
    seen: set[str] = set()

    def _fresh(host: str) -> bool:
        if not host or host in seen:
            return False
        seen.add(host)
        return True

    if _HOST_OVERRIDE and _fresh(_HOST_OVERRIDE):
        yield _HOST_OVERRIDE
    # Single-host settings first, matching the order
    # :func:`default_github_host` resolves them in.  Reading the
    # declaration list first meant a malformed entry there raised
    # before a valid, higher-priority default could match --- the same
    # ineffective precedence this laziness was added to fix, just one
    # source further along.
    for name in _HOST_ENV_VARS:
        host = _clean_host(os.environ.get(name) or "")
        if _fresh(host):
            yield host
    raw = os.environ.get("DEPENDAMERGE_GITHUB_HOSTS") or ""
    for candidate in raw.split(","):
        host = _clean_host(candidate)
        if _fresh(host):
            yield host


def enterprise_hosts() -> tuple[str, ...]:
    """Return the GitHub Enterprise hosts the operator has declared.

    Declared by ``--github-host`` on the command line, by
    ``DEPENDAMERGE_GITHUB_HOSTS`` (comma-separated), and by the
    single-host ``DEPENDAMERGE_GITHUB_HOST`` / ``GH_HOST`` variables
    that also set the default for shorthand --- naming a host as your
    default is a clear enough statement that you trust it.

    Reads every source, so any malformed value is reported.  A caller
    asking only whether one host is declared should use
    :func:`iter_enterprise_hosts`, which stops at the first match.

    Returns:
        A tuple of lowercased hostnames, without duplicates.
    """
    return tuple(iter_enterprise_hosts())


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


def _safe_to_show(host: str) -> str:
    """Redact a rejected configuration value before it reaches output.

    These messages describe a value the operator got *wrong*, and a
    plausible way to get it wrong is to paste a whole URL --- which may
    carry a token in its userinfo or query string.  An accidentally
    pasted ``https://user:TOKEN@host`` reaches the port branch, because
    the text after the colon looks like a port, so echoing the value
    verbatim printed the token to the terminal and into any captured
    log.

    The shape is what makes the message actionable, not the secret, so
    enough is kept to recognise the mistake.

    Args:
        host: The reduced configuration value.

    Returns:
        The value with any credentials removed.
    """
    without_query = host.split("?", 1)[0].split("#", 1)[0]
    if "@" in without_query:
        return f"***@{without_query.rsplit('@', 1)[-1]}"
    return without_query


def _clean_host(value: str) -> str:
    """Reduce a configured value to a bare lowercase hostname.

    Raises rather than trimming a port.  Ports are unsupported end to
    end --- ``urlparse`` drops them before the API base URLs are built
    --- so a configured ``host:8443`` would otherwise expand shorthand
    into a URL its own parser then rejects, or quietly address port
    443.

    Raises anything that is not a bare hostname, for a sharper reason.
    A configured value is *trusted*: it decides where the token goes.
    Stripping only the scheme and path left userinfo in place, so
    ``https://github.com@evil.example`` reduced to the "host"
    ``github.com@evil.example`` --- which passed the declaration check
    by looking like github.com, and then addressed ``evil.example``,
    because that is the authority a URL of that shape actually names.

    Raises:
        UrlParseError: The value names a port, or is not a hostname.
    """
    host = _strip_scheme(value.strip()).strip("/").split("/", 1)[0].lower()
    if not host:
        return ""
    name, _, port = host.rpartition(":")
    if name and port:
        shown = _safe_to_show(host)
        raise UrlParseError(
            f"Configured GitHub host {shown!r} names a port, which is not "
            "supported: the port cannot be carried through to the API "
            f"base URL, so requests would go to {_safe_to_show(name)} on "
            "the default port instead. Configure the host without a port."
        )
    if not _HOSTNAME_RE.match(host):
        raise UrlParseError(
            f"Configured GitHub host {_safe_to_show(host)!r} is not a bare "
            "hostname. Credentials, paths and query strings are not "
            "accepted here, because a value of that shape addresses a "
            "different server than it appears to. Configure the hostname "
            "on its own."
        )
    return host


def _strip_scheme(value: str) -> str:
    """Remove any ``scheme://`` prefix from ``value``."""
    return re.sub(r"\A[A-Za-z][A-Za-z0-9+.-]*://", "", value)
