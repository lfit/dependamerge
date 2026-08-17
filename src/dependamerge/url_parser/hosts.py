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


def derive_api_urls(host: str) -> tuple[str, str]:
    """Derive the (REST, GraphQL) API base URLs for a GitHub host.

    This is the single place that encodes the dotcom-vs-GHE base-URL
    rule.  github.com (and its subdomains) use the dedicated
    ``api.github.com`` host, while GitHub Enterprise Server installs
    serve the API from ``https://HOST/api/v3`` (REST) and
    ``https://HOST/api/graphql`` (GraphQL).

    GHE is not yet wired through the service/client constructors (the
    URL parsers still reject non-github.com hosts), but centralising
    the derivation here means enabling GHE later is a matter of relaxing
    that single guard and threading the returned URLs through — see the
    GHE tracking issue.

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
