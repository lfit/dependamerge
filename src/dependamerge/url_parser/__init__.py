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
from .git_suffix import has_stray_git_suffix
from .hosts import (
    _host_matches,
    canonical_web_host,
    clone_url_for,
    derive_api_urls,
    is_supported_github_host,
    pull_request_url_for,
    reject_port_bearing_host,
    unsupported_host_message,
)
from .models import (
    ChangeSource,
    ParsedGerritTopicUrl,
    ParsedOrgUrl,
    ParsedRepoUrl,
    ParsedUrl,
    UrlParseError,
)
from .repos import (
    parse_org_url,
    parse_owner_arg,
    parse_owner_target,
    parse_repo_url,
)
from .shorthand import (
    DEFAULT_GITHUB_HOST,
    default_github_host,
    enterprise_hosts,
    github_host_override,
    looks_like_host,
    looks_like_owner,
    normalize_target,
    set_github_host,
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
    "canonical_web_host",
    "clone_url_for",
    "default_github_host",
    "derive_api_urls",
    "detect_source",
    "enterprise_hosts",
    "github_host_override",
    "is_supported_github_host",
    "looks_like_host",
    "looks_like_owner",
    "normalize_target",
    "parse_change_url",
    "parse_gerrit_topic_url",
    "parse_org_url",
    "parse_owner_arg",
    "parse_owner_target",
    "parse_repo_url",
    "pull_request_url_for",
    "reject_port_bearing_host",
    "set_github_host",
    "has_stray_git_suffix",
    "strip_git_suffix",
    "unsupported_host_message",
]
