# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
URL detection and parsing for GitHub PRs and Gerrit changes.

This module provides unified URL parsing that distinguishes between GitHub
pull request URLs and Gerrit change URLs, extracting the necessary components
for each platform.

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
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse


class ChangeSource(Enum):
    """Enumeration of supported code review platforms."""

    GITHUB = "github"
    GERRIT = "gerrit"


class UrlParseError(ValueError):
    """Raised when a URL cannot be parsed as a valid change URL."""


@dataclass(frozen=True)
class ParsedUrl:
    """
    Parsed change URL with platform-specific components.

    Attributes:
        source: The code review platform (GitHub or Gerrit).
        host: The hostname of the server.
        base_path: The base path for Gerrit servers (e.g., "infra").
                   None for GitHub or Gerrit without a base path.
        project: The project identifier. For GitHub this is "owner/repo",
                 for Gerrit this is the project path (e.g., "releng/tool").
        change_number: The PR number (GitHub) or change number (Gerrit).
        original_url: The original URL that was parsed.
    """

    source: ChangeSource
    host: str
    base_path: str | None
    project: str
    change_number: int
    original_url: str

    @property
    def is_github(self) -> bool:
        """Check if this URL is from GitHub."""
        return self.source == ChangeSource.GITHUB

    @property
    def is_gerrit(self) -> bool:
        """Check if this URL is from Gerrit."""
        return self.source == ChangeSource.GERRIT


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

    # Ensure URL has a scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise UrlParseError(f"Invalid URL format: {exc}") from exc

    if not parsed.netloc:
        raise UrlParseError("URL must include a hostname")

    host = parsed.netloc.lower()
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
    """
    Check if the URL appears to be a GitHub URL.

    Detection heuristics:
    - Host contains 'github.com' or 'github'
    - Path contains '/pull/'
    """
    # Check for GitHub domain patterns
    if "github.com" in host or "github" in host:
        return True

    # Check for PR path pattern (most reliable indicator)
    if "/pull/" in path:
        return True

    return False


def _is_gerrit_url(host: str, path: str) -> bool:
    """
    Check if the URL appears to be a Gerrit URL.

    Detection heuristics:
    - Path contains '/c/' and '/+/' (Gerrit change URL pattern)
    - Host contains 'gerrit'
    """
    # Gerrit change URL pattern: /c/project/+/number
    if "/c/" in path and "/+/" in path:
        return True

    # Check for Gerrit domain pattern
    if "gerrit" in host:
        # Could be a Gerrit dashboard or other page
        # Only return True if we also have a recognizable path pattern
        if "/c/" in path or path.startswith("/changes/"):
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

    # Validate extracted components
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

    # Ensure URL has a scheme for parsing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise UrlParseError(f"Invalid URL format: {exc}") from exc

    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")

    if _is_github_url(host, path):
        return ChangeSource.GITHUB
    elif _is_gerrit_url(host, path):
        return ChangeSource.GERRIT
    else:
        raise UrlParseError(f"Cannot determine platform for URL: {url}")


__all__ = [
    "ChangeSource",
    "ParsedUrl",
    "UrlParseError",
    "detect_source",
    "parse_change_url",
]
