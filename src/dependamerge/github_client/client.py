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
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from ..url_parser import (
    UrlParseError,
    _host_matches,
    default_github_host,
    derive_api_urls,
    is_supported_github_host,
    normalize_target,
    reject_port_bearing_host,
    unsupported_host_message,
)
from .actions import _GitHubActionMixin
from .queries import _GitHubQueryMixin
from .status import _GitHubStatusMixin

logger = logging.getLogger("dependamerge.github_client")

# The path shape a pull request URL must have before declaring its host
# could possibly help.
_PR_PATH_RE = re.compile(
    r"\A/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)(?:/.*)?\Z"
)

if TYPE_CHECKING:
    from ..github_async import GitHubAsync


class GitHubClient(_GitHubQueryMixin, _GitHubActionMixin, _GitHubStatusMixin):
    """GitHub API client for managing pull requests."""

    def __init__(self, token: str | None = None, *, host: str | None = None):
        """Initialize GitHub client with token.

        Args:
            token: GitHub token; falls back to ``GITHUB_TOKEN``.
            host: The GitHub host to address.  Defaults to github.com,
                or to whatever ``DEPENDAMERGE_GITHUB_HOST``/``GH_HOST``
                names.  A GitHub Enterprise Server host changes the API
                base URLs, which is why it has to be carried here rather
                than assumed at the transport layer.
        """
        resolved = token or os.getenv("GITHUB_TOKEN")
        if not resolved:
            raise ValueError(
                "GitHub token is required. Set GITHUB_TOKEN environment variable."
            )
        self.token: str = resolved
        self.host: str = (host or default_github_host()).strip().lower()
        self.api_url, self.graphql_url = derive_api_urls(self.host)

    def _new_async(self, **kwargs: Any) -> GitHubAsync:
        """Build a transport client aimed at this instance's host.

        Every operation opens its own client, so without a single
        factory each new call site is another chance to silently fall
        back to github.com on an Enterprise run.

        ``GitHubAsync`` is resolved here rather than bound at import
        time, so that tests patching
        ``dependamerge.github_async.GitHubAsync`` still intercept it ---
        the same convention the query and action mixins have always
        used, and the reason their imports are function-local.
        """
        from ..github_async import GitHubAsync as _GitHubAsync

        return _GitHubAsync(
            token=self.token,
            api_url=self.api_url,
            graphql_url=self.graphql_url,
            **kwargs,
        )

    def __repr__(self) -> str:
        """Safe repr that never exposes the token value."""
        return "GitHubClient(token=***)"

    def parse_pr_url(self, url: str) -> tuple[str, str, int]:
        """Parse GitHub PR URL to extract owner, repo, and PR number.

        Accepts the same shorthand the URL parsers do, so
        ``acme/widget/pull/7`` and a scheme-less host both work here as
        well as in ``merge``.
        """
        # SECURITY: Use urlparse for host extraction, not substring checks.
        # See CodeQL rule py/incomplete-url-substring-sanitization.
        parsed = urlparse(normalize_target(url, default_host=self.host))
        # Same reasoning as the repository and owner parsers: a port
        # cannot survive into the API base URL, so accepting one would
        # send the token to the default port of a server the operator
        # did not name.  A malformed port is refused for the same
        # reason it is there --- it resolves to a bare hostname.
        try:
            reject_port_bearing_host(parsed.netloc.lower(), "Pull request")
        except UrlParseError as exc:
            raise ValueError(str(exc)) from exc
        host = (parsed.hostname or "").lower()
        if not is_supported_github_host(host):
            # Declaration guidance only helps when the URL is otherwise
            # a pull request.  Offering it for something that is not
            # one sends the operator to configure a host, only to meet
            # the real error afterwards.
            if host and _PR_PATH_RE.match(parsed.path):
                raise ValueError(unsupported_host_message(host, "Pull request"))
            raise ValueError(f"Invalid GitHub PR URL: {url}")

        # This client's API base URLs were fixed at construction, so a
        # URL naming a *different* permitted host would be acted on
        # against the wrong server --- and an owner/repository pair can
        # exist on both.  Declaring two hosts must not let one stand in
        # for the other.
        if not _host_matches(host, self.host, allow_subdomains=False) and not (
            _host_matches(host, "github.com") and _host_matches(self.host, "github.com")
        ):
            raise ValueError(
                f"Pull request URL names host {host}, but this client is "
                f"configured for {self.host}. Acting on it here would "
                "address the wrong server."
            )

        # The same strict shape gates every host.  Splitting the path
        # and indexing backwards from ``pull`` let a short path wrap
        # around: ``/pull/7`` put ``pull`` at index 0, so the negative
        # indexes returned owner ``pull`` and repository ``7``, and
        # ``/a/pull/7`` returned the pair transposed.  Reading the
        # groups the pattern already captures removes the arithmetic
        # that made that possible.
        match = _PR_PATH_RE.match(parsed.path)
        if match is None:
            raise ValueError(f"Invalid GitHub PR URL: {url}")
        return (
            match.group("owner"),
            match.group("repo"),
            int(match.group("number")),
        )
