# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Hostname matching and API base-URL derivation.

Both concerns are host-level policy shared by every URL parser, and both
are security-sensitive: :func:`_host_matches` is the single approved way
to compare hostnames in this codebase, and :func:`derive_api_urls` is the
single place encoding the dotcom-versus-GHE base-URL rule.
"""

from __future__ import annotations

from .shorthand import enterprise_hosts

# aislop-ignore-file ai-slop/hardcoded-url -- This module parses and builds
# GitHub/Gerrit URLs, so URL literals here are the subject matter, not
# stray configuration: example URLs in error/usage messages and
# docstrings, plus the canonical https://api.github.com endpoints for
# GitHub.  Enterprise hosts are always derived from the caller's input.


def _host_matches(
    hostname: str,
    target: str,
    *,
    allow_subdomains: bool = True,
) -> bool:
    """Check if hostname matches target using secure comparison.

    Uses exact equality or subdomain matching with a leading dot
    to prevent substring bypass attacks.

    SECURITY: This function is the approved way to check hostnames
    in this codebase. Do NOT use Python's ``in`` operator on hostname
    strings — see CodeQL rule py/incomplete-url-substring-sanitization.

    Args:
        hostname: The parsed hostname to check (lowercase).
        target: The target hostname to match against.
        allow_subdomains: If True, also matches \\*.target.

    Returns:
        True if hostname matches target or is a subdomain of target.
    """
    if not hostname or not target:
        return False
    hostname = hostname.lower()
    target = target.lower()
    if hostname == target:
        return True
    if allow_subdomains and hostname.endswith(f".{target}"):
        return True
    return False


def is_supported_github_host(host: str) -> bool:
    """Report whether ``host`` may be treated as a GitHub API endpoint.

    github.com and its subdomains always qualify.  A GitHub Enterprise
    Server install qualifies once the operator has declared it --- see
    :func:`~dependamerge.url_parser.shorthand.enterprise_hosts`.

    SECURITY: the declaration requirement is the point.  Enterprise
    hostnames are arbitrary, so accepting any host that merely appears
    in a URL would send the caller's token to whatever a pasted or
    mistyped link names.  Comparison goes through :func:`_host_matches`
    rather than substring tests.

    Args:
        host: The hostname to check.

    Returns:
        True when requests may be directed at this host.
    """
    host = (host or "").strip().lower()
    if not host:
        return False
    if _host_matches(host, "github.com"):
        return True
    return any(
        _host_matches(host, declared, allow_subdomains=False)
        for declared in enterprise_hosts()
    )


def reject_port_bearing_host(netloc: str, scope: str) -> None:
    """Refuse a target that names a port, valid or otherwise.

    ``urlparse`` reports ``hostname`` without the port, and ``host`` is
    what every parsed model carries and what :func:`derive_api_urls`
    builds from.  A port therefore survives normalisation and is then
    dropped, so ``ghe.example.com:8443`` would quietly be addressed on
    443.

    Refusing is the honest answer while the port has nowhere to live.
    Silently discarding it would send requests to a server the operator
    did not name.

    A *malformed* port is refused too.  ``https://github.com:notaport/o/r``
    yields a hostname of ``github.com``, so treating it as portless
    would route a plainly broken URL to dotcom as though nothing were
    wrong.

    Args:
        netloc: The authority as parsed, possibly ``host:port``.
        scope: What was being parsed, e.g. ``"Repository"``.

    Raises:
        UrlParseError: When ``netloc`` carries a port of any kind.
    """
    from .models import UrlParseError

    # An IPv6 literal is bracketed and full of colons; only what follows
    # the closing bracket can be a port.
    remainder = netloc
    prefix = ""
    if netloc.startswith("["):
        closing = netloc.find("]")
        if closing == -1:
            return
        prefix = netloc[: closing + 1]
        remainder = netloc[closing + 1 :]

    if ":" not in remainder:
        return

    host, _, port = remainder.rpartition(":")
    host = f"{prefix}{host}" if prefix else host

    if not port:
        # A bare trailing colon names no port; harmless, and urlparse
        # already ignores it.
        return

    if not port.isdigit():
        raise UrlParseError(
            f"{scope} URL has a malformed port ({netloc}). Ports must be numeric."
        )

    raise UrlParseError(
        f"{scope} URL parsing does not support a port ({netloc}). "
        f"The port cannot be carried through to the API base URL, so "
        f"requests would go to {host} on the default port instead. "
        "Use a host without a port, or a direct pull request URL."
    )


def unsupported_host_message(host: str, scope: str) -> str:
    """Build the rejection shown for an undeclared host.

    Args:
        host: The offending hostname.
        scope: What was being parsed, e.g. ``"Repository"``.

    Returns:
        A message naming the host and how to permit it.
    """
    return (
        f"{scope} URL parsing is not enabled for host: {host}. "
        "github.com works out of the box; for a GitHub Enterprise "
        "Server install, declare it first with "
        "DEPENDAMERGE_GITHUB_HOSTS=host1,host2 (or GH_HOST=host). "
        "Hosts are declared rather than inferred so a mistyped URL "
        "cannot send your token somewhere unintended."
    )


def derive_api_urls(host: str) -> tuple[str, str]:
    """Derive the (REST, GraphQL) API base URLs for a GitHub host.

    This is the single place that encodes the dotcom-vs-GHE base-URL
    rule.  github.com (and its subdomains) use the dedicated
    ``api.github.com`` host, while GitHub Enterprise Server installs
    serve the API from ``https://HOST/api/v3`` (REST) and
    ``https://HOST/api/graphql`` (GraphQL).

    GHE hosts must be declared by the operator before they are used;
    callers gate on :func:`is_supported_github_host` first.  This
    function derives URLs for whatever host it is given and performs no
    trust check of its own.

    Args:
        host: The hostname (e.g. ``github.com`` or ``ghe.example.com``).

    Returns:
        A ``(api_url, graphql_url)`` tuple.

    Raises:
        ValueError: If ``host`` is empty or whitespace-only, which would
            otherwise yield a subtly broken base URL such as
            ``https:///api/v3``.
    """
    host = (host or "").strip().lower()
    if not host:
        raise ValueError("derive_api_urls requires a non-empty host")
    if _host_matches(host, "github.com"):
        return ("https://api.github.com", "https://api.github.com/graphql")
    # GitHub Enterprise Server base URLs.
    return (f"https://{host}/api/v3", f"https://{host}/api/graphql")
