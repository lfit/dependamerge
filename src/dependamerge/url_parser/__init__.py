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

Gerrit topic search (see parse_gerrit_topic_url):
    https://gerrit.example.org/q/topic:some-topic
    https://gerrit.onap.org/r/q/topic:some-topic

Shorthand (see normalize_target):
    lfreleng-actions                    -> owner-wide
    acme/widget                         -> repository-wide
    acme/widget/pull/7                  -> a single pull request
    git@github.com:acme/widget.git      -> repository-wide
"""

from __future__ import annotations

from .change import (
    _is_gerrit_url,
    _is_github_url,
    _parse_gerrit_url,
    _parse_github_url,
    detect_source,
    parse_change_url,
)
from .hosts import _host_matches, derive_api_urls
from .models import (
    ChangeSource,
    ParsedGerritTopicUrl,
    ParsedOrgUrl,
    ParsedRepoUrl,
    ParsedUrl,
    UrlParseError,
)
from .repos import parse_org_url, parse_owner_arg, parse_repo_url
from .shorthand import (
    DEFAULT_GITHUB_HOST,
    default_github_host,
    looks_like_host,
    looks_like_owner,
    normalize_target,
    strip_git_suffix,
)
from .topic import parse_gerrit_topic_url

__all__ = [
    "DEFAULT_GITHUB_HOST",
    "ChangeSource",
    "ParsedGerritTopicUrl",
    "ParsedOrgUrl",
    "ParsedRepoUrl",
    "ParsedUrl",
    "UrlParseError",
    "_host_matches",
    "default_github_host",
    "derive_api_urls",
    "detect_source",
    "looks_like_host",
    "looks_like_owner",
    "normalize_target",
    "parse_change_url",
    "parse_gerrit_topic_url",
    "parse_org_url",
    "parse_owner_arg",
    "parse_repo_url",
    "strip_git_suffix",
]
