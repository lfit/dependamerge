# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Repository listing queries.

These are the lightweight "just the repositories" queries: no pull
request nodes, so a single page reveals the owner-wide repository total
and the archived/fork flags needed to decide what to scan.  The
``Organization`` and ``User`` variants are separated only by their root
field; see :mod:`dependamerge.github_graphql.pull_requests` for the
heavier queries that carry PR payloads.
"""

from __future__ import annotations

# Lightweight query to list repositories without PR nodes for accurate counting.
# totalCount is provided by the GitHub GraphQL API for free on connection
# objects, so the first page immediately reveals the org-wide repo total
# without requiring a separate counting pass.
#
# ``isFork`` is included so owner-wide bulk operations can exclude fork
# repositories without a second round-trip; existing consumers that only
# read ``nameWithOwner`` / ``isArchived`` simply ignore the extra field.
ORG_REPOS_ONLY = """
query($org: String!, $reposCursor: String) {
  organization(login: $org) {
    repositories(first: 100, after: $reposCursor, orderBy: { field: NAME, direction: ASC }) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        nameWithOwner
        isArchived
        isFork
      }
    }
  }
}
"""

# User-account counterpart of ORG_REPOS_ONLY.  The ``repositories``
# connection exists on both ``Organization`` and ``User``, so this query
# is structurally identical apart from the ``user(login:)`` root.  It is
# used as a runtime fallback when an owner login is not an organization
# (the ``organization`` field returns null).  The ``$org`` variable name
# is retained for call-site uniformity even though it carries a user
# login here.
USER_REPOS_ONLY = """
query($org: String!, $reposCursor: String) {
  user(login: $org) {
    repositories(first: 100, after: $reposCursor, orderBy: { field: NAME, direction: ASC }) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        nameWithOwner
        isArchived
        isFork
      }
    }
  }
}
"""
