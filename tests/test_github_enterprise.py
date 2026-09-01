# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Tests for GitHub Enterprise Server host support.

Enterprise hostnames are arbitrary, so the host cannot be recognised
from its name.  These tests cover both halves of the resulting design:
a declared host is usable everywhere github.com is, and an undeclared
one is refused before any request carries a token to it.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dependamerge.cli import app
from dependamerge.cli._merge_permissions import _check_merge_permissions
from dependamerge.close_manager import AsyncCloseManager
from dependamerge.github_client import GitHubClient
from dependamerge.github_service import GitHubService
from dependamerge.merge_manager import AsyncMergeManager
from dependamerge.url_parser import (
    UrlParseError,
    derive_api_urls,
    enterprise_hosts,
    is_supported_github_host,
    parse_change_url,
    parse_org_url,
    parse_owner_target,
    parse_repo_url,
)

GHE = "ghe.corp.example.com"


@pytest.fixture
def declared_ghe(monkeypatch):
    """Declare a GHE host for the duration of a test."""
    monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
    monkeypatch.delenv("GH_HOST", raising=False)
    monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", GHE)
    return GHE


@pytest.fixture
def no_declared_hosts(monkeypatch):
    """Ensure no host is declared, whatever the ambient environment."""
    for name in ("DEPENDAMERGE_GITHUB_HOST", "DEPENDAMERGE_GITHUB_HOSTS", "GH_HOST"):
        monkeypatch.delenv(name, raising=False)


class TestDeriveApiUrls:
    """The dotcom-versus-Enterprise base URL rule."""

    def test_dotcom_uses_the_dedicated_api_host(self):
        assert derive_api_urls("github.com") == (
            "https://api.github.com",
            "https://api.github.com/graphql",
        )

    def test_subdomain_of_dotcom_is_still_dotcom(self):
        api, gql = derive_api_urls("foo.github.com")
        assert api == "https://api.github.com"
        assert gql == "https://api.github.com/graphql"

    def test_enterprise_serves_from_its_own_host(self):
        # GHE mounts REST under /api/v3 and GraphQL under /api/graphql,
        # rather than splitting them onto a separate hostname.
        assert derive_api_urls(GHE) == (
            f"https://{GHE}/api/v3",
            f"https://{GHE}/api/graphql",
        )

    def test_empty_host_is_refused(self):
        # Would otherwise yield the subtly broken "https:///api/v3".
        with pytest.raises(ValueError, match="non-empty host"):
            derive_api_urls("")


class TestHostDeclaration:
    """A host must be declared before it is addressed."""

    def test_dotcom_needs_no_declaration(self, no_declared_hosts):
        assert is_supported_github_host("github.com") is True
        assert is_supported_github_host("foo.github.com") is True

    def test_undeclared_enterprise_host_is_refused(self, no_declared_hosts):
        assert is_supported_github_host(GHE) is False

    def test_declared_enterprise_host_is_accepted(self, declared_ghe):
        assert is_supported_github_host(GHE) is True

    def test_declaration_list_is_split_on_commas(self, monkeypatch):
        monkeypatch.setenv(
            "DEPENDAMERGE_GITHUB_HOSTS", " first.example.com , second.example.com "
        )
        assert is_supported_github_host("first.example.com") is True
        assert is_supported_github_host("second.example.com") is True
        assert is_supported_github_host("third.example.com") is False

    def test_default_host_variable_also_declares(self, monkeypatch):
        # Naming a host as your default is a clear enough statement
        # that you trust it; requiring it twice would be busywork.
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOSTS", raising=False)
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.setenv("GH_HOST", GHE)
        assert is_supported_github_host(GHE) is True

    def test_declaration_does_not_extend_to_subdomains(self, declared_ghe):
        # Declaring one host must not hand over a whole DNS subtree;
        # an attacker controlling a subdomain would otherwise inherit
        # the trust.
        assert is_supported_github_host(f"evil.{GHE}") is False

    def test_declaration_is_not_a_substring_match(self, declared_ghe):
        # The classic bypass: a host merely *containing* the declared
        # one. See CodeQL py/incomplete-url-substring-sanitization.
        assert is_supported_github_host(f"{GHE}.evil.example") is False

    def test_scheme_and_path_are_stripped_from_a_declaration(self, monkeypatch):
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.delenv("GH_HOST", raising=False)
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", f"https://{GHE}/")
        assert enterprise_hosts() == (GHE,)
        assert is_supported_github_host(GHE) is True


class TestEnterpriseUrlParsing:
    """Declared hosts parse in every shape github.com does."""

    def test_repo_url(self, declared_ghe):
        parsed = parse_repo_url(f"https://{GHE}/acme/widget")
        assert parsed.host == GHE
        assert parsed.project == "acme/widget"

    def test_org_url(self, declared_ghe):
        parsed = parse_org_url(f"https://{GHE}/acme")
        assert (parsed.host, parsed.owner) == (GHE, "acme")

    def test_shorthand_resolves_against_the_declared_default(self, monkeypatch):
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOSTS", raising=False)
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.setenv("GH_HOST", GHE)
        parsed = parse_repo_url("acme/widget")
        assert parsed.host == GHE

    def test_pull_request_url_needs_no_declaration(self, no_declared_hosts):
        # A /pull/ path identifies a GitHub PR structurally, so single
        # change URLs have always been host-agnostic.  Left that way.
        parsed = parse_change_url(f"https://{GHE}/acme/widget/pull/7")
        assert parsed.host == GHE
        assert parsed.change_number == 7

    @pytest.mark.parametrize(
        ("parser", "url"),
        [
            pytest.param(parse_repo_url, f"https://{GHE}/acme/widget", id="repo"),
            pytest.param(parse_org_url, f"https://{GHE}/acme", id="org"),
        ],
    )
    def test_undeclared_host_is_rejected_with_guidance(
        self, no_declared_hosts, parser, url
    ):
        with pytest.raises(UrlParseError) as excinfo:
            parser(url)
        message = str(excinfo.value)
        assert "not enabled for host" in message
        assert GHE in message
        # The message has to say what to do about it, or the operator
        # is left guessing that GHE is unsupported outright.
        assert "DEPENDAMERGE_GITHUB_HOSTS" in message

    def test_unrelated_host_is_still_rejected(self, declared_ghe):
        # Declaring one enterprise host must not open the door to
        # every other host.
        with pytest.raises(UrlParseError, match="not enabled for host"):
            parse_repo_url("https://gitlab.com/acme/widget")


class TestPortBearingTargets:
    """Ports are refused rather than silently discarded.

    ``urlparse`` reports ``hostname`` without the port, and ``host`` is
    what the parsed models carry and what the API base URLs derive
    from.  A port therefore survives normalisation and is then dropped,
    so accepting one would address a server the operator did not name.
    """

    @pytest.mark.parametrize(
        "parser",
        [parse_repo_url, parse_org_url],
    )
    def test_port_is_refused_with_an_explanation(self, monkeypatch, parser):
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", "ghe.example.com:8443")
        with pytest.raises(UrlParseError) as excinfo:
            parser("https://ghe.example.com:8443/acme/widget")
        assert "does not support a port" in str(excinfo.value)

    def test_ordinary_hosts_are_unaffected(self):
        assert parse_repo_url("https://github.com/acme/widget").host == "github.com"


class TestEveryClientReachesTheDeclaredHost:
    """Regression: the host has to reach *all* the transports.

    Enterprise support fails in the least obvious way when only some
    call sites carry the host --- discovery succeeds against the
    enterprise server while the writes go to github.com.
    """

    def test_permission_preflight_uses_the_enterprise_base(self, declared_ghe, mocker):
        # Runs before anything else on every non-dry merge, so getting
        # this wrong aborts the run before the host-aware clients exist.
        # Let asyncio.run actually run: patching it would leave the
        # coroutine unawaited, which the suite now treats as a failure.
        api = mocker.AsyncMock()
        api.check_token_permissions = mocker.AsyncMock(
            return_value={"merge": {"has_permission": True}}
        )
        built = mocker.patch("dependamerge.cli._merge_permissions.GitHubAsync")
        built.return_value.__aenter__ = mocker.AsyncMock(return_value=api)
        built.return_value.__aexit__ = mocker.AsyncMock(return_value=None)

        ctx = mocker.MagicMock()
        ctx.host = GHE
        ctx.token = "t"
        ctx.no_fix = True
        mocker.patch(
            "dependamerge.cli._merge_permissions._source_pr_modifies_workflows",
            return_value=False,
        )

        _check_merge_permissions(ctx)

        assert built.call_args.kwargs["api_url"] == f"https://{GHE}/api/v3"
        assert built.call_args.kwargs["graphql_url"] == f"https://{GHE}/api/graphql"

    def test_close_manager_uses_the_enterprise_base(self, declared_ghe):
        manager = AsyncCloseManager(token="t", host=GHE)
        assert manager.host == GHE

    def test_merge_manager_accepts_a_host(self, declared_ghe):
        manager = AsyncMergeManager(token="t", host=GHE)
        assert manager.host == GHE

    def test_service_uses_the_enterprise_base(self, declared_ghe):
        # Asserted on the constructed transport rather than on a patched
        # constructor: ``GitHubAsync`` is bound in more than one module
        # of this package, so patching one of them would prove less
        # than it appears to.
        service = GitHubService(token="t", host=GHE)
        assert service._api.api_url == f"https://{GHE}/api/v3"
        assert service._api.graphql_url == f"https://{GHE}/api/graphql"

    def test_service_defaults_to_dotcom(self, no_declared_hosts):
        service = GitHubService(token="t")
        assert service._api.api_url == "https://api.github.com"


class TestOwnerArgumentKeepsItsHost:
    """``status`` and ``blocked`` must not lose the host.

    Returning the login alone was safe only while github.com was the
    sole possibility.  With Enterprise hosts available, an accepted
    owner URL whose host was dropped scans the wrong server in silence.
    """

    def test_bare_login_uses_the_default_host(self, no_declared_hosts):
        assert parse_owner_target("acme") == ("acme", "github.com")

    def test_enterprise_owner_url_keeps_its_host(self, declared_ghe):
        assert parse_owner_target(f"https://{GHE}/acme") == ("acme", GHE)

    def test_canonical_orgs_form_keeps_its_host(self, declared_ghe):
        assert parse_owner_target(f"https://{GHE}/orgs/acme/repositories") == (
            "acme",
            GHE,
        )

    def test_undeclared_host_is_still_refused(self, no_declared_hosts):
        with pytest.raises(UrlParseError, match="not enabled for host"):
            parse_owner_target("https://evil.example.com/acme")


class TestShorthandReachesTheCloseCommand:
    """``parse_pr_url`` understands the same shorthand as ``merge``."""

    def test_shorthand_pull_request(self, no_declared_hosts):
        client = GitHubClient("t")
        assert client.parse_pr_url("acme/widget/pull/7") == ("acme", "widget", 7)

    def test_scheme_less_host(self, no_declared_hosts):
        client = GitHubClient("t")
        assert client.parse_pr_url("github.com/acme/widget/pull/7") == (
            "acme",
            "widget",
            7,
        )

    def test_shorthand_resolves_against_the_clients_host(self, declared_ghe):
        # The client's own host is the default for a shorthand, so a
        # close run started from an enterprise URL stays on that server.
        client = GitHubClient("t", host=GHE)
        assert client.parse_pr_url("acme/widget/pull/7") == ("acme", "widget", 7)


class TestEnterpriseRepositoryMergeRuns:
    """The documented Enterprise flow reaches the merge path.

    Regression: relaxing the parser guard was not enough on its own.
    A second, github.com-only guard in the repository handler still
    rejected every Enterprise host, so the flow this PR advertises
    could not run at all while every unit test passed.
    """

    runner = CliRunner()

    def test_declared_host_reaches_repository_mode(self, declared_ghe):
        result = self.runner.invoke(
            app,
            [
                "merge",
                f"https://{GHE}/acme/widget",
                "--token",
                "t",
                "--dry-run",
            ],
        )
        out = result.stdout
        assert "Repository mode" in out
        assert "only supported" not in out

    def test_undeclared_host_is_still_refused(self, no_declared_hosts):
        result = self.runner.invoke(
            app,
            ["merge", f"https://{GHE}/acme/widget", "--token", "t", "--dry-run"],
        )
        assert result.exit_code == 1
        assert "Repository mode" not in result.stdout


class TestClientCarriesTheHost:
    """The resolved host reaches the transport layer."""

    def test_dotcom_client_uses_the_api_host(self):
        client = GitHubClient("t", host="github.com")
        assert client.api_url == "https://api.github.com"
        assert client.graphql_url == "https://api.github.com/graphql"

    def test_enterprise_client_uses_its_own_base(self, declared_ghe):
        client = GitHubClient("t", host=GHE)
        assert client.api_url == f"https://{GHE}/api/v3"
        assert client.graphql_url == f"https://{GHE}/api/graphql"

    def test_transport_is_built_with_the_enterprise_base(self, declared_ghe, mocker):
        # The regression that matters: every operation opens its own
        # transport, so a missed call site addresses github.com while
        # everything else addresses the enterprise host.
        built = mocker.patch("dependamerge.github_async.GitHubAsync")
        client = GitHubClient("t", host=GHE)
        client._new_async()
        assert built.call_args.kwargs["api_url"] == f"https://{GHE}/api/v3"
        assert built.call_args.kwargs["graphql_url"] == f"https://{GHE}/api/graphql"

    def test_pr_url_on_a_declared_host_is_accepted(self, declared_ghe):
        client = GitHubClient("t", host=GHE)
        owner, repo, number = client.parse_pr_url(f"https://{GHE}/acme/widget/pull/7")
        assert (owner, repo, number) == ("acme", "widget", 7)

    def test_pr_url_on_an_undeclared_host_is_refused(self, no_declared_hosts):
        client = GitHubClient("t")
        with pytest.raises(ValueError, match="Invalid GitHub PR URL"):
            client.parse_pr_url("https://evil.example.com/acme/widget/pull/7")
