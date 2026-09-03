# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Parsing for repository-wide and owner-wide GitHub URLs.

These are the bulk-operation entry points: a repository URL scopes work
to one repository, an owner URL to every repository beneath an
organization or user account.
"""

from __future__ import annotations

from urllib.parse import urlparse

from .git_suffix import has_stray_git_suffix
from .hosts import (
    is_supported_github_host,
    reject_port_bearing_host,
    unsupported_host_message,
)
from .models import ChangeSource, ParsedOrgUrl, ParsedRepoUrl, UrlParseError
from .shorthand import default_github_host, looks_like_owner, normalize_target

# aislop-ignore-file ai-slop/hardcoded-url -- This module parses and builds
# GitHub/Gerrit URLs, so URL literals here are the subject matter, not
# stray configuration: example URLs in error/usage messages and
# docstrings, plus the canonical https://api.github.com endpoints for
# GitHub.  Enterprise hosts are always derived from the caller's input.


def parse_repo_url(url: str) -> ParsedRepoUrl:
    """
    Parse a GitHub repository URL (not a specific PR).

    Supports formats:
        https://github.com/owner/repo
        https://github.com/owner/repo/
        https://github.com/owner/repo/pulls

    Args:
        url: The URL to parse.

    Returns:
        A ParsedRepoUrl instance with the extracted components.

    Raises:
        UrlParseError: If the URL format is not recognized as a valid repository URL.
    """
    url = url.strip()
    if not url:
        raise UrlParseError("URL cannot be empty")

    # Expand shorthand ("owner", "owner/repo"), git remote forms, and a
    # missing scheme into an absolute URL.  Centralised so every parser
    # understands the same set of abbreviations.
    url = normalize_target(url)

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise UrlParseError(f"Invalid URL format: {exc}") from exc

    if not parsed.hostname:
        raise UrlParseError("URL must include a hostname")

    host = parsed.hostname.lower()
    path = parsed.path.rstrip("/")

    # github.com is always available; a GitHub Enterprise Server host is
    # available once the operator has declared it.  Enterprise hostnames
    # are arbitrary, so requiring a declaration is what stops a mistyped
    # URL directing a token at an unintended host.  This is the single
    # choke point for repository URLs --- do NOT scatter additional host
    # checks elsewhere.
    reject_port_bearing_host(parsed.netloc.lower(), "Repository")
    if not is_supported_github_host(host):
        raise UrlParseError(unsupported_host_message(host, "Repository"))

    # Try to extract owner/repo from the path
    # Expected: /owner/repo or /owner/repo/pulls
    # Strip the path, remove "pulls" suffix if present
    parts = [p for p in path.split("/") if p]

    if parts and parts[0].lower() == "orgs":
        # ``/orgs/<login>`` is GitHub's owner route, so its first
        # segment is never a repository owner --- the name is reserved,
        # and no account holds it.  Without this, an owner URL that the
        # owner parser refuses falls through to here and is read as the
        # repository ``acme.git`` under the owner ``orgs``, which simply
        # moves a malformed target from one mode to another.
        raise UrlParseError(
            f"Not a repository URL: {url}. '/orgs/' introduces an owner, "
            "so this names no repository. Drop the '/orgs/' prefix for a "
            "repository, or give the owner URL to merge across it."
        )

    # Remove a "pulls" *page* suffix, which is what /owner/repo/pulls
    # is.  Only when an owner and a repository remain: "pulls" is a
    # legal repository name, and stripping it unconditionally left
    # /owner/pulls as a single segment, so every repository actually
    # called "pulls" was rejected as a malformed URL.
    if len(parts) > 2 and parts[-1] == "pulls":
        parts = parts[:-1]

    if len(parts) < 2:
        raise UrlParseError(
            f"Invalid GitHub repository URL format. Expected: https://{host}/owner/repo"
        )

    # After stripping "pulls", require exactly 2 parts (owner/repo)
    if len(parts) != 2:
        # Check if this is a PR URL (owner/repo/pull/…) before giving a generic error.
        # Match any path starting with /owner/repo/pull/ regardless of whether
        # the PR segment is numeric — /pull/abc is still clearly a PR-shaped URL
        # and deserves the more specific guidance.
        if len(parts) >= 3 and parts[2] == "pull":
            raise UrlParseError(
                "This looks like a pull request URL, not a repository URL. "
                "Pass the full PR URL (…/pull/<number>) directly to merge "
                "a single PR, or use the repository URL (…/owner/repo) for "
                "bulk operations."
            )
        raise UrlParseError(
            f"Invalid GitHub repository URL format. Expected: https://{host}/owner/repo"
        )

    owner = parts[0]
    repo = parts[1]

    return ParsedRepoUrl(
        source=ChangeSource.GITHUB,
        host=host,
        owner=owner,
        repo=repo,
        project=f"{owner}/{repo}",
        original_url=url,
    )


def parse_org_url(url: str) -> ParsedOrgUrl:
    """
    Parse a GitHub organization/owner URL (not a specific repo or PR).

    Supports the following owner-wide forms (trailing slashes are
    cosmetic and ignored):
        https://github.com/owner
        https://github.com/owner/
        https://github.com/orgs/owner
        https://github.com/orgs/owner/repositories

    The owner may be an organization or a personal user account; the two
    are indistinguishable here and are disambiguated at runtime when the
    repositories are enumerated.

    Args:
        url: The URL to parse.

    Returns:
        A ParsedOrgUrl instance with the extracted owner.

    Raises:
        UrlParseError: If the URL is not recognised as an owner-wide URL.
    """
    url = url.strip()
    if not url:
        raise UrlParseError("URL cannot be empty")

    # Expand shorthand ("owner", "owner/repo"), git remote forms, and a
    # missing scheme into an absolute URL.  Centralised so every parser
    # understands the same set of abbreviations.
    url = normalize_target(url)

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise UrlParseError(f"Invalid URL format: {exc}") from exc

    if not parsed.hostname:
        raise UrlParseError("URL must include a hostname")

    host = parsed.hostname.lower()
    path = parsed.path.rstrip("/")

    # SECURITY: github.com is always available; a GitHub Enterprise
    # Server host is available once the operator has declared it.
    # Enterprise hostnames are arbitrary, so requiring a declaration is
    # what stops a mistyped URL directing a token at an unintended
    # host.  This is the single choke point for owner-wide URLs --- do
    # NOT scatter additional host checks elsewhere.
    reject_port_bearing_host(parsed.netloc.lower(), "Owner-wide")
    if not is_supported_github_host(host):
        raise UrlParseError(unsupported_host_message(host, "Owner-wide"))

    if has_stray_git_suffix(path):
        # Normalisation preserves the suffix on anything that is not a
        # clone URL, and an owner path is never one.  Without this the
        # marker achieved nothing: ``/acme.git`` simply became the owner
        # ``acme.git`` and reached owner-wide dispatch anyway.
        raise UrlParseError(
            f"Not an owner URL: {url}. The trailing '.git' makes this a "
            "clone URL for a repository, not an owner. Give the owner "
            "on its own, or a repository URL."
        )

    parts = [p for p in path.split("/") if p]

    # Normalise the canonical GitHub org forms:
    #   /orgs/owner               -> owner
    #   /orgs/owner/repositories  -> owner
    if parts and parts[0] == "orgs":
        rest = parts[1:]
        if rest and rest[-1] == "repositories":
            rest = rest[:-1]
        if len(rest) != 1:
            raise UrlParseError(
                f"Invalid GitHub organization URL format. Expected: "
                f"https://{host}/orgs/owner"
            )
        owner = rest[0]
        return ParsedOrgUrl(
            source=ChangeSource.GITHUB,
            host=host,
            owner=owner,
            original_url=url,
        )

    # Bare owner form: exactly one path segment.
    if len(parts) != 1:
        raise UrlParseError(
            f"Invalid GitHub owner URL format. Expected: "
            f"https://{host}/owner (an organization or user login)"
        )

    owner = parts[0]
    return ParsedOrgUrl(
        source=ChangeSource.GITHUB,
        host=host,
        owner=owner,
        original_url=url,
    )


def parse_owner_target(value: str) -> tuple[str, str]:
    """Extract an owner login *and its host* from a CLI argument.

    The companion to :func:`parse_owner_arg`, which returns the login
    alone.  Dropping the host is safe only while github.com is the sole
    possibility; with GitHub Enterprise Server hosts available, a
    command that keeps the login and forgets the host accepts
    ``https://ghe.example.com/acme`` and then scans ``acme`` on
    github.com --- the wrong server, silently.

    Args:
        value: The raw CLI argument.

    Returns:
        An ``(owner, host)`` pair.  A bare login resolves against the
        default host.

    Raises:
        UrlParseError: If ``value`` is empty or is a URL that is not a
            recognised owner URL on a permitted host.
    """
    value = (value or "").strip()
    if not value:
        raise UrlParseError("Owner name or URL cannot be empty")

    bare = value.rstrip("/")
    if not bare:
        raise UrlParseError("Owner name or URL cannot be empty")
    if "/" not in bare and "://" not in bare:
        if not looks_like_owner(bare):
            # The same boundary the shorthand expansion enforces: text
            # that cannot be a login is rejected rather than sent to
            # the API as an owner that cannot exist.
            raise UrlParseError(
                f"Not a valid GitHub owner name: {bare!r}. Logins are "
                "alphanumerics and hyphens, at most 39 characters."
            )
        return (bare, default_github_host())

    parsed = parse_org_url(value)
    return (parsed.owner, parsed.host)


def parse_owner_arg(value: str) -> str:
    """Extract an owner login from a CLI argument.

    The owner-wide *report* commands (``status`` and ``blocked``) accept
    either a bare login or any of the GitHub owner URL forms that
    :func:`parse_org_url` understands.  This single helper normalises all
    of them to a plain login so the commands no longer rely on a naive
    ``split("/")[-1]`` that silently mis-parses the canonical
    ``/orgs/owner/repositories`` form (it would return ``repositories``).

    Accepted inputs:
        owner
        owner/
        https://github.com/owner
        https://github.com/owner/
        github.com/owner
        https://github.com/orgs/owner
        https://github.com/orgs/owner/repositories

    A bare token — optionally with one or more trailing slashes but no
    other path separator and no scheme — is treated as a login and
    returned verbatim (minus the trailing slashes).  This preserves the
    long-standing ability to pass just an organization/user name, and to
    pass ``owner/`` (the ``status``/``blocked`` commands historically
    accepted a trailing slash via ``rstrip("/")``).  Anything that still
    looks like a URL (an embedded ``/`` or a scheme) is delegated to
    :func:`parse_org_url`, which enforces the github.com-only guard and
    the canonical forms.

    Args:
        value: The raw CLI argument.

    Returns:
        The extracted owner login.

    Raises:
        UrlParseError: If ``value`` is empty or is a URL that is not a
            recognised github.com owner URL.
    """
    value = (value or "").strip()
    if not value:
        raise UrlParseError("Owner name or URL cannot be empty")

    # A bare login has no scheme and no embedded path separator once any
    # trailing slashes are removed; accept it as-is so plain names like
    # "lfreleng-actions" and the historical "lfreleng-actions/" form keep
    # working.
    bare = value.rstrip("/")
    if not bare:
        # The input was only slashes (e.g. "////"); there is no login to
        # extract, so treat it the same as an empty value.
        raise UrlParseError("Owner name or URL cannot be empty")
    if "/" not in bare and "://" not in bare:
        if not looks_like_owner(bare):
            raise UrlParseError(
                f"Not a valid GitHub owner name: {bare!r}. Logins are "
                "alphanumerics and hyphens, at most 39 characters."
            )
        return bare

    return parse_org_url(value).owner
