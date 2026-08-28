# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Parsing for individual change URLs: GitHub pull requests and Gerrit changes.

Supported URL formats:

GitHub:
    https://github.com/owner/repo/pull/123
    https://github.enterprise.com/owner/repo/pull/456

Gerrit:
    https://gerrit.linuxfoundation.org/infra/c/project/name/+/12345
    https://gerrit.example.org/c/project/+/67890
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from .hosts import _host_matches
from .models import ChangeSource, ParsedUrl, UrlParseError
from .shorthand import normalize_target

# aislop-ignore-file ai-slop/hardcoded-url -- This module parses and builds
# GitHub/Gerrit URLs, so URL literals here are the subject matter, not
# stray configuration: example URLs in error/usage messages and
# docstrings, plus the canonical https://api.github.com endpoints for
# GitHub.  Enterprise hosts are always derived from the caller's input.


def parse_change_url(url: str) -> ParsedUrl:
    """
    Parse a GitHub PR URL or Gerrit change URL.

    Args:
        url: The URL to parse.

    Returns:
        A ParsedUrl instance with the extracted components.

    Raises:
        UrlParseError: If the URL format is not recognized or invalid.
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

    # Detect platform based on URL characteristics
    if _is_github_url(host, path):
        return _parse_github_url(host, path, url)
    elif _is_gerrit_url(host, path):
        return _parse_gerrit_url(host, path, url)
    else:
        raise UrlParseError(
            f"Cannot determine platform for URL: {url}. "
            "Expected GitHub PR URL (containing /pull/) or "
            "Gerrit change URL (containing /c/.../+/)."
        )


def _is_github_url(host: str, path: str) -> bool:
    """Check if the URL is a GitHub URL using secure host comparison.

    SECURITY: Uses exact hostname matching via _host_matches(), not
    substring checks, to prevent bypass attacks via crafted hostnames.
    See CodeQL rule py/incomplete-url-substring-sanitization.

    Detection heuristics:
    - Host matches 'github.com' (exact or subdomain)
    - Path contains '/pull/' (for GitHub Enterprise with unknown hosts)
    """
    # SECURITY: Use _host_matches() — never use `"github.com" in host`
    if _host_matches(host, "github.com"):
        return True

    # Path-based detection for GitHub Enterprise with unknown hosts
    if "/pull/" in path:
        return True

    return False


def _is_gerrit_url(host: str, path: str) -> bool:
    """Check if the URL is a Gerrit URL using structural validation.

    SECURITY: Uses Gerrit's distinctive URL path structure rather than
    hostname substring matching. See CodeQL rule
    py/incomplete-url-substring-sanitization.

    Detection heuristics:
    - Path contains '/c/' and '/+/' (Gerrit change URL pattern)
    - Path starts with '/changes/' (Gerrit REST API pattern)
    """
    # Primary: Gerrit change URL structure is definitive
    if "/c/" in path and "/+/" in path:
        return True

    # Secondary: Gerrit REST API pattern
    if path.startswith("/changes/"):
        return True

    return False


def _parse_github_url(host: str, path: str, original_url: str) -> ParsedUrl:
    """
    Parse a GitHub pull request URL.

    Expected format: https://github.com/owner/repo/pull/123
    """
    # Pattern: /owner/repo/pull/number
    match = re.match(r"^/([^/]+)/([^/]+)/pull/(\d+)(?:/.*)?$", path)
    if not match:
        raise UrlParseError(
            f"Invalid GitHub PR URL format. Expected: "
            f"https://{host}/owner/repo/pull/123"
        )

    owner = match.group(1)
    repo = match.group(2)
    pr_number = int(match.group(3))

    return ParsedUrl(
        source=ChangeSource.GITHUB,
        host=host,
        base_path=None,
        project=f"{owner}/{repo}",
        change_number=pr_number,
        original_url=original_url,
    )


def _parse_gerrit_url(host: str, path: str, original_url: str) -> ParsedUrl:
    """
    Parse a Gerrit change URL.

    Expected formats:
        https://gerrit.example.org/c/project/+/12345
        https://gerrit.example.org/infra/c/project/name/+/12345

    The base_path (e.g., "infra") is optional and appears before /c/.
    """
    # Pattern: optional_base_path/c/project_path/+/number
    # The project path can contain multiple segments (e.g., releng/tool)
    match = re.match(r"^(?:/([^/]+))?/c/(.+)/\+/(\d+)(?:/.*)?$", path)

    if not match:
        # Try alternative pattern without base path
        match = re.match(r"^/c/(.+)/\+/(\d+)(?:/.*)?$", path)
        if match:
            base_path = None
            project = match.group(1)
            change_number = int(match.group(2))
        else:
            raise UrlParseError(
                f"Invalid Gerrit change URL format. Expected: "
                f"https://{host}/c/project/+/12345 or "
                f"https://{host}/base/c/project/+/12345"
            )
    else:
        base_path = match.group(1)  # May be None
        project = match.group(2)
        change_number = int(match.group(3))

    if not project:
        raise UrlParseError("Gerrit URL must include a project name")

    if change_number <= 0:
        raise UrlParseError("Gerrit change number must be positive")

    return ParsedUrl(
        source=ChangeSource.GERRIT,
        host=host,
        base_path=base_path,
        project=project,
        change_number=change_number,
        original_url=original_url,
    )


def detect_source(url: str) -> ChangeSource:
    """
    Detect the source platform from a URL without full parsing.

    This is a convenience function for quick platform detection.

    Args:
        url: The URL to analyze.

    Returns:
        The detected ChangeSource.

    Raises:
        UrlParseError: If the platform cannot be determined.
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

    host = parsed.hostname.lower() if parsed.hostname else ""
    path = parsed.path.rstrip("/")

    if _is_github_url(host, path):
        return ChangeSource.GITHUB
    elif _is_gerrit_url(host, path):
        return ChangeSource.GERRIT
    else:
        raise UrlParseError(f"Cannot determine platform for URL: {url}")
