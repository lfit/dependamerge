# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Dataclasses and enums describing parsed change, repository and owner URLs.

The parsing functions that build these live alongside them in
:mod:`dependamerge.url_parser`; this module holds only the result types
so that every parser can share them without importing one another.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# aislop-ignore-file ai-slop/hardcoded-url -- This module parses and builds
# GitHub/Gerrit URLs, so URL literals here are the subject matter, not
# stray configuration: example URLs in error/usage messages and
# docstrings, plus the canonical https://api.github.com endpoints for
# GitHub.  Enterprise hosts are always derived from the caller's input.


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


@dataclass(frozen=True)
class ParsedGerritTopicUrl:
    """
    Parsed Gerrit topic search URL.

    Represents a Gerrit query URL that scopes work to a topic, e.g.
    ``https://gerrit.onap.org/r/q/topic:update-settings``.  Only
    ``topic:`` queries are supported; other search operators in the
    query are ignored for parsing purposes.

    Attributes:
        source: The code review platform (always Gerrit).
        host: The hostname of the Gerrit server.
        base_path: The base path for the Gerrit server (e.g., "r"),
                   or None when the server is mounted at the root.
        topic: The Gerrit topic name extracted from the query.
        original_url: The original URL that was parsed.
    """

    source: ChangeSource
    host: str
    base_path: str | None
    topic: str
    original_url: str

    @property
    def is_gerrit(self) -> bool:
        """Check if this URL is from Gerrit."""
        return self.source == ChangeSource.GERRIT


@dataclass(frozen=True)
class ParsedOrgUrl:
    """
    Parsed organization/owner URL (not a specific repo or PR).

    Represents an owner-wide scope, e.g. ``https://github.com/owner``.
    The owner may be either a GitHub organization or a personal user
    account; the two are indistinguishable from the URL alone and are
    disambiguated at runtime when enumerating repositories.

    Attributes:
        source: The code review platform (GitHub only for now).
        host: The hostname of the server.
        owner: The organization or user login.
        original_url: The original URL that was parsed.
    """

    source: ChangeSource
    host: str
    owner: str
    original_url: str

    @property
    def is_github(self) -> bool:
        """Check if this URL is from GitHub."""
        return self.source == ChangeSource.GITHUB


@dataclass(frozen=True)
class ParsedRepoUrl:
    """
    Parsed repository URL (not a specific PR/change).

    Attributes:
        source: The code review platform (GitHub only for now).
        host: The hostname of the server.
        owner: The repository owner/organization.
        repo: The repository name.
        project: The full "owner/repo" string.
        original_url: The original URL that was parsed.
    """

    source: ChangeSource
    host: str
    owner: str
    repo: str
    project: str
    original_url: str

    @property
    def is_github(self) -> bool:
        """Check if this URL is from GitHub."""
        return self.source == ChangeSource.GITHUB
