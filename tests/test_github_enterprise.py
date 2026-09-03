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

import asyncio
import logging

import pytest
import typer
from typer.testing import CliRunner

from dependamerge.cli import app
from dependamerge.cli._merge_permissions import _check_merge_permissions
from dependamerge.close_manager import AsyncCloseManager
from dependamerge.error_codes import is_github_api_permission_error
from dependamerge.github_async import GitHubAsync
from dependamerge.github_async._permissions import (
    _unauthorized_permission_error,
    web_host_for,
)
from dependamerge.github_client import GitHubClient
from dependamerge.github_service import GitHubService
from dependamerge.merge_manager import AsyncMergeManager
from dependamerge.models import PullRequestInfo
from dependamerge.resolve_conflicts import FixOrchestrator
from dependamerge.url_parser import (
    UrlParseError,
    clone_url_for,
    default_github_host,
    derive_api_urls,
    enterprise_hosts,
    is_supported_github_host,
    parse_change_url,
    parse_org_url,
    parse_owner_target,
    parse_repo_url,
    pull_request_url_for,
    set_github_host,
)

GHE = "ghe.corp.example.com"


def _option_names(command: str) -> set[str]:
    """Return the option strings a command accepts.

    Asks the command object rather than parsing ``--help``, which is a
    rendering and varies with colour support and terminal width.
    """
    cli = typer.main.get_command(app)
    subcommand = cli.commands[command]  # type: ignore[attr-defined]
    names: set[str] = set()
    for param in subcommand.params:
        names.update(param.opts)
    return names


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

    @pytest.mark.parametrize(
        "parser",
        [parse_repo_url, parse_org_url],
    )
    def test_malformed_port_is_refused(self, parser):
        # urlparse reports a hostname of github.com for this, so
        # treating a non-numeric port as "no port" routed a plainly
        # broken URL to dotcom as though nothing were wrong.
        with pytest.raises(UrlParseError, match="malformed port"):
            parser("https://github.com:notaport/acme/widget")

    def test_pull_request_url_refuses_a_port(self, no_declared_hosts):
        client = GitHubClient("t")
        with pytest.raises(ValueError, match="does not support a port"):
            client.parse_pr_url("https://github.com:8443/acme/widget/pull/7")

    def test_pull_request_url_refuses_a_malformed_port(self, no_declared_hosts):
        client = GitHubClient("t")
        with pytest.raises(ValueError, match="malformed port"):
            client.parse_pr_url("https://github.com:notaport/acme/widget/pull/7")


class TestPermissionGuidanceFollowsTheHost:
    """Token guidance must name the server the token belongs to.

    Settings pages and ``gh auth refresh`` are per-installation, so an
    Enterprise operator sent to github.com is being pointed at a site
    that knows nothing about their credentials.
    """

    def test_dotcom_api_maps_to_the_web_host(self):
        assert web_host_for("https://api.github.com") == "github.com"

    def test_enterprise_api_maps_to_its_own_host(self):
        assert web_host_for(f"https://{GHE}/api/v3") == GHE

    def test_missing_api_url_falls_back_to_dotcom(self):
        assert web_host_for("") == "github.com"

    @pytest.mark.parametrize(
        "api_url",
        [
            f"https://user:ghp_SECRETTOKEN@{GHE}/api/v3",
            f"https://ghp_SECRETTOKEN@{GHE}/api/v3",
        ],
    )
    def test_credentials_never_reach_the_guidance(self, api_url):
        # The authority includes userinfo, so reading ``netloc`` put a
        # caller-supplied credential into the settings URL and the
        # ``gh -h`` argument --- both printed to the terminal.
        assert web_host_for(api_url) == GHE

    def test_the_rendered_guidance_is_clean(self):
        # Asserting on the helper alone would pass if the credential
        # reached the message by another route, so this drives the
        # error the operator actually sees.
        api = GitHubAsync(
            token="t",
            api_url=f"https://user:ghp_SECRETTOKEN@{GHE}/api/v3",
            graphql_url=f"https://{GHE}/api/graphql",
        )
        error = api._parse_permission_error(
            Exception("403 Forbidden"), "merge", "acme", "widget"
        )

        # A 403 must be recognised as a permission problem; if it were
        # not, the assertions below would vacuously pass on ``None``.
        assert error is not None
        guidance = " ".join(error.token_type_guidance.values())
        assert "SECRETTOKEN" not in guidance.upper()
        # And the host is still named: dropping it entirely would also
        # satisfy the assertion above while making the guidance useless.
        assert GHE in guidance

    def test_the_host_is_lowercased(self):
        assert web_host_for("https://GHE.Example.COM/api/v3") == "ghe.example.com"

    def test_unauthorised_guidance_names_the_enterprise_host(self):
        error = _unauthorized_permission_error("merge", GHE)
        guidance = " ".join(error.token_type_guidance.values())
        assert GHE in guidance
        assert "github.com" not in guidance

    def test_unauthorised_guidance_still_defaults_to_dotcom(self):
        error = _unauthorized_permission_error("merge")
        assert "github.com" in " ".join(error.token_type_guidance.values())


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
        # Asserting the stored host alone would pass even if the
        # derive_api_urls wiring were removed, so enter the manager and
        # check the transport it actually built.
        manager = AsyncCloseManager(token="t", host=GHE)
        assert manager.host == GHE

        async def _check() -> tuple[str, str]:
            async with manager:
                client = manager._github_client
                assert client is not None
                return (client.api_url, client.graphql_url)

        api_url, graphql_url = asyncio.run(_check())
        assert api_url == f"https://{GHE}/api/v3"
        assert graphql_url == f"https://{GHE}/api/graphql"

    def test_merge_manager_uses_the_enterprise_base(self, declared_ghe, mocker):
        manager = AsyncMergeManager(token="t", host=GHE)
        assert manager.host == GHE

        built = mocker.patch("dependamerge.merge_manager.GitHubAsync")
        built.return_value.__aenter__ = mocker.AsyncMock()
        mocker.patch("dependamerge.merge_manager._lifecycle.GitHubService")
        mocker.patch("dependamerge.merge_manager._lifecycle.PullRequestStatePoller")

        asyncio.run(manager.__aenter__())

        assert built.call_args.kwargs["api_url"] == f"https://{GHE}/api/v3"
        assert built.call_args.kwargs["graphql_url"] == f"https://{GHE}/api/graphql"

    def test_fix_orchestrator_uses_the_enterprise_base(self, declared_ghe):
        # The orchestrator builds its transport deep inside an async
        # fetch, so assert the derivation its host feeds instead of
        # only that the host was stored.
        orchestrator = FixOrchestrator("t", host=GHE)
        assert orchestrator._host == GHE
        assert derive_api_urls(orchestrator._host) == (
            f"https://{GHE}/api/v3",
            f"https://{GHE}/api/graphql",
        )

    def test_fix_orchestrator_defaults_to_dotcom(self, no_declared_hosts):
        assert FixOrchestrator("t")._host == "github.com"

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

    def test_shared_client_ignores_a_broken_default(self, monkeypatch):
        # A caller passing its own client has fixed endpoints already,
        # so resolving the default host for it made an unrelated
        # misconfiguration fail a construction that never uses it.
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.setenv("GH_HOST", "broken:8443")

        shared = GitHubAsync(token="t")
        service = GitHubService(token="t", client=shared)

        assert service._api is shared


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

    @pytest.mark.parametrize(
        "value",
        ["not a url", "has$dollar", "-leading", "x" * 40],
    )
    def test_bare_value_must_look_like_a_login(self, no_declared_hosts, value):
        # The bare-token shortcut used to return anything verbatim, so
        # ``status "not a url"`` started an API scan for an owner that
        # cannot exist.  Same boundary the shorthand expansion enforces.
        #
        # ``trailing-`` is deliberately absent: ``johan--`` is a real
        # account, and this shortcut accepted it before the boundary
        # existed, so refusing it here would be a regression rather
        # than a tightening.
        with pytest.raises(UrlParseError, match="Not a valid GitHub owner name"):
            parse_owner_target(value)

    def test_a_trailing_hyphen_is_accepted(self, no_declared_hosts):
        assert parse_owner_target("johan--") == ("johan--", "github.com")

    @pytest.mark.parametrize("value", ["lfreleng-actions", "acme", "acme/"])
    def test_valid_logins_still_pass(self, no_declared_hosts, value):
        owner, _ = parse_owner_target(value)
        assert owner == value.rstrip("/")


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

    def test_declared_host_reaches_repository_mode(self, declared_ghe, mocker):
        # ``_init_repo_merge_client`` is left real --- it is the code
        # the removed guard lived in --- while the fetch beyond it is
        # stubbed, so this stays offline and cannot pass merely because
        # a network failure printed different words.
        fetch = mocker.patch(
            "dependamerge.cli._repo_merge._fetch_repo_prs", return_value=[]
        )

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
        assert result.exit_code == 0, out
        assert "Repository mode" in out
        assert "only supported" not in out
        fetch.assert_called_once()

    def test_undeclared_host_is_still_refused(self, no_declared_hosts):
        result = self.runner.invoke(
            app,
            ["merge", f"https://{GHE}/acme/widget", "--token", "t", "--dry-run"],
        )
        assert result.exit_code == 1
        assert "Repository mode" not in result.stdout


class TestGithubHostFlag:
    """``--github-host`` and its precedence over the environment.

    The flag is a higher-priority source of the same setting the
    environment variables provide, and serves both their purposes: the
    host a shorthand resolves against, and the set of hosts permitted
    at all.  Naming a host on the command line is at least as
    deliberate as exporting it.
    """

    runner = CliRunner()

    def test_flag_sets_the_default_host(self, no_declared_hosts):
        set_github_host(GHE)
        assert default_github_host() == GHE

    def test_flag_declares_the_host(self, no_declared_hosts):
        assert is_supported_github_host(GHE) is False
        set_github_host(GHE)
        assert is_supported_github_host(GHE) is True

    def test_flag_resolves_shorthand(self, no_declared_hosts):
        set_github_host(GHE)
        assert parse_repo_url("acme/widget").host == GHE

    def test_flag_beats_dependamerge_variable(self, monkeypatch):
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOST", "env.example.com")
        set_github_host(GHE)
        assert default_github_host() == GHE

    def test_flag_beats_gh_host(self, monkeypatch):
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.setenv("GH_HOST", "env.example.com")
        set_github_host(GHE)
        assert default_github_host() == GHE

    def test_environment_order_without_the_flag(self, monkeypatch):
        # DEPENDAMERGE_GITHUB_HOST outranks GH_HOST, so a project
        # setting is not overridden by whatever ``gh`` happens to use.
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOST", "first.example.com")
        monkeypatch.setenv("GH_HOST", "second.example.com")
        assert default_github_host() == "first.example.com"

    def test_clearing_the_flag_falls_back_to_the_environment(self, monkeypatch):
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.setenv("GH_HOST", "env.example.com")
        set_github_host(GHE)
        set_github_host(None)
        assert default_github_host() == "env.example.com"

    def test_scheme_and_path_are_stripped_from_the_flag(self, no_declared_hosts):
        set_github_host(f"https://{GHE}/")
        assert default_github_host() == GHE
        assert is_supported_github_host(GHE) is True

    def test_empty_flag_is_treated_as_absent(self, no_declared_hosts):
        # Typer passes None when the flag is omitted; an empty string
        # from a shell expansion must not become a hostname.
        set_github_host("")
        assert default_github_host() == "github.com"

    def test_unresolved_typer_default_is_treated_as_absent(self, no_declared_hosts):
        # Calling a command directly as a Python function --- which the
        # tests do --- passes Typer's OptionInfo through unresolved.
        # The same allowance _normalise_topic and _validate_max_wait
        # make, and without it every such call raises AttributeError.
        set_github_host(typer.Option(None, "--github-host"))  # type: ignore[arg-type]
        assert default_github_host() == "github.com"

    @pytest.mark.parametrize(
        "command",
        ["merge", "close", "status", "blocked"],
    )
    def test_flag_is_offered_by_every_target_taking_command(self, command):
        # Reads the command's parameters rather than its rendered help.
        #
        # Scraping ``--help`` passed locally and failed on CI, because
        # Rich colours the flag when it detects a terminal and the
        # escape sequences land *between* the characters --- so the
        # literal "--github-host" is absent from output that displays
        # it perfectly.  Width-based wrapping truncates the table for
        # the same reason.  The option's existence is the claim; how it
        # is drawn is not.
        assert "--github-host" in _option_names(command)

    def test_flag_reaches_the_merge_path(self, no_declared_hosts, mocker):
        # End to end: without the flag this host is undeclared and the
        # run is refused, so reaching repository mode proves the flag
        # travelled all the way through parsing.  The fetch is stubbed
        # so the assertion cannot ride on a network error.
        fetch = mocker.patch(
            "dependamerge.cli._repo_merge._fetch_repo_prs", return_value=[]
        )

        result = self.runner.invoke(
            app,
            [
                "merge",
                "acme/widget",
                "--github-host",
                GHE,
                "--token",
                "t",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "Repository mode" in result.stdout
        fetch.assert_called_once()

    def test_without_the_flag_the_same_run_is_refused(self, no_declared_hosts):
        result = self.runner.invoke(
            app,
            ["merge", f"https://{GHE}/acme/widget", "--token", "t", "--dry-run"],
        )
        assert "Repository mode" not in result.stdout


class TestClientRefusesAnotherHost:
    """A client bound to one host must not act on another's URL.

    Both hosts may be declared and both may hold an ``acme/widget``.
    Parsing a URL from one while the client addresses the other would
    act on the wrong instance under a name that looks right.
    """

    @pytest.fixture
    def two_declared_hosts(self, monkeypatch):
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.delenv("GH_HOST", raising=False)
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", f"{GHE},other.example.com")

    def test_url_from_a_different_declared_host_is_refused(self, two_declared_hosts):
        client = GitHubClient("t", host=GHE)
        with pytest.raises(ValueError, match="address the wrong server"):
            client.parse_pr_url("https://other.example.com/acme/widget/pull/7")

    def test_dotcom_url_is_refused_by_an_enterprise_client(self, declared_ghe):
        client = GitHubClient("t", host=GHE)
        with pytest.raises(ValueError, match="address the wrong server"):
            client.parse_pr_url("https://github.com/acme/widget/pull/7")

    def test_matching_host_is_accepted(self, declared_ghe):
        client = GitHubClient("t", host=GHE)
        assert client.parse_pr_url(f"https://{GHE}/acme/widget/pull/7") == (
            "acme",
            "widget",
            7,
        )

    def test_dotcom_subdomains_are_the_same_instance(self, no_declared_hosts):
        # foo.github.com and github.com are one service, so this must
        # not become a false positive on the ordinary path.
        client = GitHubClient("t", host="github.com")
        assert client.parse_pr_url("https://foo.github.com/acme/widget/pull/7") == (
            "acme",
            "widget",
            7,
        )


class TestUndeclaredHostErrorIsActionable:
    """The rejection shown must be the one that says what to do.

    An undeclared Enterprise repository URL has two path segments, and
    the reporter used to answer those with ``parse_change_url``'s
    "cannot determine platform" --- true, but it leaves the operator
    with no way forward.
    """

    runner = CliRunner()

    def test_repository_shape_reports_the_declaration_instructions(
        self, no_declared_hosts
    ):
        result = self.runner.invoke(
            app, ["merge", f"https://{GHE}/acme/widget", "--token", "t"]
        )
        out = result.stdout
        assert "DEPENDAMERGE_GITHUB_HOSTS" in out
        assert "Cannot determine platform" not in out

    def test_gerrit_shape_keeps_the_platform_guidance(self, no_declared_hosts):
        # A structurally Gerrit-looking path is a different mistake and
        # keeps the platform-agnostic message.
        result = self.runner.invoke(
            app,
            ["merge", "https://review.example.org/c/proj/+/abc", "--token", "t"],
        )
        assert "Invalid URL" in result.stdout

    def test_bare_host_does_not_advise_declaring(self, no_declared_hosts):
        # No path at all names no target on any host, so declaring it
        # cannot change the outcome.
        result = self.runner.invoke(
            app, ["merge", "https://invalid-url.com", "--token", "t"]
        )
        assert "Invalid URL" in result.stdout
        assert "DEPENDAMERGE_GITHUB_HOSTS" not in result.stdout


class TestCloneUrlFallbacksFollowTheHost:
    """Synthesised clone URLs must name the right server.

    A clone URL is missing from some API payloads and gets built from
    the repository's full name.  Hard-coding github.com there sends a
    clone --- and a force-push, carrying the token --- to dotcom for a
    repository that lives on an Enterprise server.
    """

    def test_enterprise_host(self, declared_ghe):
        assert clone_url_for(GHE, "acme/widget") == f"https://{GHE}/acme/widget.git"

    def test_dotcom(self):
        assert (
            clone_url_for("github.com", "acme/widget")
            == "https://github.com/acme/widget.git"
        )

    def test_empty_host_falls_back_to_dotcom(self):
        assert clone_url_for("", "acme/widget") == "https://github.com/acme/widget.git"

    @pytest.mark.parametrize("host", ["api.github.com", "foo.github.com"])
    def test_dotcom_subdomains_canonicalise(self, host):
        # The parsers accept github.com *and its subdomains* as one
        # service, so api.github.com reaches URL construction --- and
        # building a clone URL under it names an address that serves
        # no repositories.
        assert clone_url_for(host, "acme/widget") == (
            "https://github.com/acme/widget.git"
        )
        assert pull_request_url_for(host, "acme/widget", 7) == (
            "https://github.com/acme/widget/pull/7"
        )

    def test_enterprise_host_is_not_canonicalised(self, declared_ghe):
        assert pull_request_url_for(GHE, "acme/widget", 7) == (
            f"https://{GHE}/acme/widget/pull/7"
        )

    def test_rebase_plan_uses_the_host_for_missing_urls(self, declared_ghe):
        # The local rebase path clones and force-pushes, so a wrong
        # host here is the most consequential of the fallbacks.
        from dependamerge.rebase.local_plan import _build_rebase_plan

        pr = PullRequestInfo(
            number=1,
            title="t",
            body=None,
            author="dependabot[bot]",
            head_sha="abc",
            base_branch="main",
            head_branch="dep/x",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[],
            repository_full_name="acme/widget",
            html_url=f"https://{GHE}/acme/widget/pull/1",
            head_repo_full_name="acme/widget",
        )

        plan = _build_rebase_plan(
            pr_info=pr,
            owner="acme",
            repo="widget",
            log=logging.getLogger("test"),
            host=GHE,
        )

        assert plan is not None
        # ``origin_url`` is where the rebase clones from and force-pushes
        # to, so this is the field that matters most.
        assert GHE in plan.origin_url
        assert "github.com" not in plan.origin_url
        assert "github.com" not in plan.upstream_url


class TestUndeclaredPullRequestUrlExplainsItself:
    """A structurally valid PR URL on an undeclared host says why.

    The repository and owner parsers name the environment variable;
    a direct-PR user was told only "Invalid GitHub PR URL", which
    reads as though the URL were malformed.
    """

    def test_message_names_the_remedy(self, no_declared_hosts):
        client = GitHubClient("t")
        with pytest.raises(ValueError) as excinfo:
            client.parse_pr_url(f"https://{GHE}/acme/widget/pull/7")
        message = str(excinfo.value)
        assert "DEPENDAMERGE_GITHUB_HOSTS" in message
        assert GHE in message

    def test_genuinely_malformed_url_keeps_its_own_message(self, no_declared_hosts):
        client = GitHubClient("t")
        with pytest.raises(ValueError, match="Invalid GitHub PR URL"):
            client.parse_pr_url("https://github.com/acme/widget")

    def test_the_error_is_not_reported_as_a_credentials_fault(self, no_declared_hosts):
        # ``is_github_api_permission_error`` matches *substrings of the
        # message*, and "token" is one of them.  The remedy text ends
        # "...cannot send your token somewhere unintended", so the
        # undeclared-host error was reported as "provide a GITHUB_TOKEN
        # with the required permissions" --- advice for a fault that
        # was not there, which also hid the message saying what to do.
        client = GitHubClient("t")
        with pytest.raises(ValueError) as excinfo:
            client.parse_pr_url(f"https://{GHE}/acme/widget/pull/7")

        assert "token" in str(excinfo.value)  # the substring that misled it
        assert is_github_api_permission_error(excinfo.value) is False

    def test_a_url_error_is_typed_so_callers_can_tell_it_apart(self, no_declared_hosts):
        # UrlParseError subclasses ValueError, so callers catching the
        # latter keep working.  Raising bare ValueError discarded the
        # only signal that separates a URL problem from an API one.
        client = GitHubClient("t")
        with pytest.raises(UrlParseError):
            client.parse_pr_url(f"https://{GHE}/acme/widget/pull/7")
        with pytest.raises(ValueError):
            client.parse_pr_url(f"https://{GHE}/acme/widget/pull/7")

    @pytest.mark.parametrize(
        ("exception", "expected"),
        [
            (Exception("Bad credentials"), True),
            (Exception("Resource not accessible by integration"), True),
            (Exception("HTTP 403 Forbidden"), True),
            (ValueError("invalid token supplied"), True),
        ],
    )
    def test_real_permission_errors_still_classify(self, exception, expected):
        # The complement: excluding URL errors must not stop a genuine
        # credentials failure being reported as one.
        assert is_github_api_permission_error(exception) is expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://invalid-url.com",
            "https://invalid-url.com/acme",
            "https://invalid-url.com/acme/widget",
        ],
    )
    def test_non_pull_request_paths_do_not_advise_declaring(
        self, no_declared_hosts, url
    ):
        # Declaring the host cannot turn these into pull requests, so
        # the guidance would send the operator to configure something
        # and then meet the real error anyway.
        client = GitHubClient("t")
        with pytest.raises(ValueError) as excinfo:
            client.parse_pr_url(url)
        assert "Invalid GitHub PR URL" in str(excinfo.value)
        assert "DEPENDAMERGE_GITHUB_HOSTS" not in str(excinfo.value)


class TestConfigurationErrorsAreReported:
    """A bad host configuration is a message, not a traceback.

    The refusal is raised by the configuration readers, which run
    before any command's error handling. Left alone it escaped as an
    uncaught exception on an ordinary mistake.
    """

    runner = CliRunner()

    def test_flag_with_a_port_is_reported(self, no_declared_hosts):
        result = self.runner.invoke(
            app,
            [
                "merge",
                "acme/widget",
                "--github-host",
                "ghe.example.com:8443",
                "--token",
                "t",
            ],
        )
        assert result.exit_code == 1
        assert "names a port" in result.stdout
        assert not isinstance(result.exception, UrlParseError)

    def test_environment_with_a_port_is_reported(self, monkeypatch):
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.setenv("GH_HOST", "ghe.example.com:8443")
        result = self.runner.invoke(app, ["merge", "acme/widget", "--token", "t"])
        assert result.exit_code == 1
        assert "names a port" in result.stdout
        assert not isinstance(result.exception, UrlParseError)

    def test_broken_environment_does_not_block_an_explicit_gerrit_url(
        self, monkeypatch
    ):
        # A Gerrit target never consults a GitHub default, so validating
        # the environment up front blocked a run for a reason that has
        # nothing to do with it.  Only the flag is validated eagerly.
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.setenv("GH_HOST", "broken:8443")
        result = self.runner.invoke(
            app,
            [
                "merge",
                "https://gerrit.example.org/c/proj/+/123",
                "--token",
                "t",
                "--dry-run",
            ],
        )
        assert "names a port" not in result.stdout


class TestOwnerCommandsExplainAnUndeclaredHost:
    """``status`` and ``blocked`` surface the parser's remedy.

    Swallowing the parse error left an Enterprise operator with only
    "invalid owner", which names neither the host nor the fix.
    """

    runner = CliRunner()

    @pytest.mark.parametrize("command", ["status", "blocked"])
    def test_message_names_the_remedy(self, no_declared_hosts, command):
        result = self.runner.invoke(
            app, [command, f"https://{GHE}/acme", "--token", "t"]
        )
        assert result.exit_code == 1
        assert "DEPENDAMERGE_GITHUB_HOSTS" in result.stdout
        assert GHE in result.stdout


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
        # The message changed to name the remedy; the refusal itself is
        # what this test guards.  See
        # TestUndeclaredPullRequestUrlExplainsItself for the wording.
        with pytest.raises(ValueError, match="not enabled for host"):
            client.parse_pr_url("https://evil.example.com/acme/widget/pull/7")
