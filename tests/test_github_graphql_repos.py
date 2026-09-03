# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for repository-listing GraphQL documents."""

from __future__ import annotations

from dependamerge.github_graphql import ORG_REPOS_ONLY, USER_REPOS_ONLY


def test_user_repos_query_limits_owner_affiliations_to_owner() -> None:
    assert "user(login:" in USER_REPOS_ONLY
    assert "ownerAffiliations: OWNER" in USER_REPOS_ONLY


def test_org_repos_query_does_not_set_owner_affiliations() -> None:
    assert "organization(login:" in ORG_REPOS_ONLY
    assert "ownerAffiliations" not in ORG_REPOS_ONLY
