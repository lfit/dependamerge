# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Live GitHub Enterprise Server integration tests for the CLI.

Issue #343 asks for GHE support to be validated against a real
instance --- authentication, GraphQL parity, and the base-URL split ---
which no amount of synthetic hosts can stand in for.  The unit suite
proves the *plumbing* carries a host; only a real server proves the
host is the right one and that the API behind it behaves.

Every test here runs the real CLI in dry-run mode, so a read-only
token suffices and nothing is mutated.  All of them skip cleanly when
the Enterprise configuration is absent, which is the normal case: this
file is inert until somebody points it at an instance.

Configuration:
    DEPENDAMERGE_IT_GHE_HOST    Required.  The Enterprise hostname,
                                e.g. ``ghe.corp.example.com``.
    DEPENDAMERGE_IT_GHE_TOKEN   Required.  A token for *that* instance;
                                a dotcom token will not authenticate.
    DEPENDAMERGE_IT_GHE_ORG     Owner login to enumerate.
    DEPENDAMERGE_IT_GHE_REPO    Optional ``owner/repo`` to scope the
                                repository-wide check to.

Run with ``--run-integration`` (or ``DEPENDAMERGE_RUN_INTEGRATION=1``).
"""

from __future__ import annotations

import os

import pytest

from dependamerge.cli import app
from dependamerge.url_parser import derive_api_urls

from .conftest import combined_output

pytestmark = pytest.mark.integration


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


@pytest.fixture(scope="session")
def ghe_host() -> str:
    """The Enterprise host, or skip the whole suite."""
    host = _env("DEPENDAMERGE_IT_GHE_HOST")
    if not host:
        pytest.skip(
            "DEPENDAMERGE_IT_GHE_HOST not set; skipping live GHE integration tests"
        )
    return host


@pytest.fixture(scope="session")
def ghe_token() -> str:
    """A token for the Enterprise instance, or skip."""
    token = _env("DEPENDAMERGE_IT_GHE_TOKEN")
    if not token:
        pytest.skip("DEPENDAMERGE_IT_GHE_TOKEN not set; skipping live GHE tests")
    return token


@pytest.fixture(scope="session")
def ghe_org() -> str:
    """An owner login on the Enterprise instance, or skip."""
    owner = _env("DEPENDAMERGE_IT_GHE_ORG")
    if not owner:
        pytest.skip("DEPENDAMERGE_IT_GHE_ORG not set; skipping live GHE tests")
    return owner


@pytest.fixture(autouse=True)
def _declare_ghe_host(monkeypatch, ghe_host):
    """Declare the instance, as an operator would.

    Exercises the declaration mechanism itself rather than bypassing
    it: if this were unnecessary, the host guard would not be doing its
    job.
    """
    monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", ghe_host)


class TestLiveEnterpriseAuthentication:
    """The token reaches the Enterprise API, not dotcom."""

    def test_owner_wide_status_authenticates(
        self, runner, ghe_host, ghe_token, ghe_org
    ):
        # ``status`` is read-only and exercises the GraphQL path, which
        # is where a dotcom-shaped base URL fails most visibly.
        result = runner.invoke(
            app,
            [
                "status",
                f"https://{ghe_host}/{ghe_org}",
                "--token",
                ghe_token,
                "--no-progress",
            ],
        )
        output = combined_output(result)
        assert "api.github.com" not in output, (
            "the run addressed dotcom despite an Enterprise target"
        )
        assert "401" not in output and "Unauthorized" not in output, output

    def test_shorthand_resolves_against_the_instance(
        self, runner, monkeypatch, ghe_host, ghe_token, ghe_org
    ):
        # With the host as the default, a bare login must reach the
        # Enterprise instance rather than looking the owner up on
        # github.com --- where it may not exist, or may be someone else.
        monkeypatch.setenv("GH_HOST", ghe_host)
        result = runner.invoke(
            app, ["status", ghe_org, "--token", ghe_token, "--no-progress"]
        )
        output = combined_output(result)
        assert "api.github.com" not in output, output


class TestLiveEnterpriseMergePreview:
    """The dry-run merge path reaches the instance."""

    def test_owner_wide_dry_run(self, runner, ghe_host, ghe_token, ghe_org):
        result = runner.invoke(
            app,
            [
                "merge",
                f"https://{ghe_host}/{ghe_org}",
                "--token",
                ghe_token,
                "--dry-run",
                "--no-progress",
            ],
        )
        output = combined_output(result)
        # The pre-existing github.com-only guard produced exactly this,
        # and its removal is what makes Enterprise merges reachable.
        assert "only supported" not in output, output
        assert "not enabled for host" not in output, output

    def test_repository_wide_dry_run(self, runner, ghe_host, ghe_token):
        project = _env("DEPENDAMERGE_IT_GHE_REPO")
        if not project:
            pytest.skip("DEPENDAMERGE_IT_GHE_REPO not set")
        result = runner.invoke(
            app,
            [
                "merge",
                f"https://{ghe_host}/{project}",
                "--token",
                ghe_token,
                "--dry-run",
                "--no-progress",
            ],
        )
        output = combined_output(result)
        assert "only supported" not in output, output


class TestLiveEnterpriseBaseUrls:
    """The derived base URLs match what the instance actually serves."""

    def test_rest_and_graphql_endpoints_respond(self, ghe_host, ghe_token):
        # Asserts the dotcom-versus-Enterprise split is right for this
        # deployment, which is the one thing a synthetic host cannot
        # tell us: GHE serves /api/v3 and /api/graphql from the same
        # host, and a proxy or a subpath install would not.
        import httpx

        api_url, graphql_url = derive_api_urls(ghe_host)
        headers = {"Authorization": f"Bearer {ghe_token}"}

        with httpx.Client(timeout=20.0) as client:
            rest = client.get(f"{api_url}/user", headers=headers)
            assert rest.status_code == 200, (
                f"{api_url}/user returned {rest.status_code}; "
                "the derived REST base URL is wrong for this instance"
            )

            gql = client.post(
                graphql_url,
                headers=headers,
                json={"query": "query { viewer { login } }"},
            )
            assert gql.status_code == 200, (
                f"{graphql_url} returned {gql.status_code}; "
                "the derived GraphQL base URL is wrong for this instance"
            )
            assert "errors" not in gql.json(), gql.text


class TestLiveEnterpriseDeclarationIsEnforced:
    """An undeclared instance is refused even when reachable."""

    def test_undeclared_host_is_refused(
        self, runner, monkeypatch, ghe_host, ghe_token, ghe_org
    ):
        # The declaration requirement is a security control, so it is
        # worth proving against a host that genuinely exists rather
        # than only against a fictional one.
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOSTS", raising=False)
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.delenv("GH_HOST", raising=False)

        result = runner.invoke(
            app,
            [
                "merge",
                f"https://{ghe_host}/{ghe_org}",
                "--token",
                ghe_token,
                "--dry-run",
            ],
        )
        assert result.exit_code == 1
        assert "not enabled for host" in combined_output(result)
