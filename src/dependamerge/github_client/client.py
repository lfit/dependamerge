# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The ``GitHubClient`` entry point.

This module holds the client's construction and the one operation that
needs no network access at all, :meth:`GitHubClient.parse_pr_url`.

The API-backed operations live in :mod:`.queries` (reads),
:mod:`.actions` (writes) and :mod:`.status` (mergeability
interpretation), and are mixed in here, so the client exposes exactly the
method surface it always did.  The logger name is unchanged so records
keep reporting as ``dependamerge.github_client``.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from ..url_parser import _host_matches
from .actions import _GitHubActionMixin
from .queries import _GitHubQueryMixin
from .status import _GitHubStatusMixin

logger = logging.getLogger("dependamerge.github_client")


class GitHubClient(_GitHubQueryMixin, _GitHubActionMixin, _GitHubStatusMixin):
    """GitHub API client for managing pull requests."""

    def __init__(self, token: str | None = None):
        """Initialize GitHub client with token."""
        resolved = token or os.getenv("GITHUB_TOKEN")
        if not resolved:
            raise ValueError(
                "GitHub token is required. Set GITHUB_TOKEN environment variable."
            )
        self.token: str = resolved

    def __repr__(self) -> str:
        """Safe repr that never exposes the token value."""
        return "GitHubClient(token=***)"

    def parse_pr_url(self, url: str) -> tuple[str, str, int]:
        """Parse GitHub PR URL to extract owner, repo, and PR number."""
        # SECURITY: Use urlparse for host extraction, not substring checks.
        # See CodeQL rule py/incomplete-url-substring-sanitization.
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not _host_matches(host, "github.com"):
            raise ValueError(f"Invalid GitHub PR URL: {url}")

        # Use parsed.path to ignore query strings and fragments
        # when splitting.
        parts = parsed.path.strip("/").split("/")
        if "pull" not in parts:
            raise ValueError(f"Invalid GitHub PR URL: {url}")

        # Find the 'pull' segment and get the PR number
        try:
            pull_index = parts.index("pull")
            if pull_index + 1 >= len(parts):
                raise ValueError("PR number not found after 'pull'")

            owner = parts[pull_index - 2]
            repo = parts[pull_index - 1]
            pr_number = int(parts[pull_index + 1])

            return owner, repo, pr_number
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid GitHub PR URL: {url}") from e
