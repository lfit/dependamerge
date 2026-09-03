# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Tests for shorthand and git-remote target normalisation.

``normalize_target`` runs ahead of every URL parser, so these tests
cover both halves of its job: expanding the abbreviations a human types
and the remote forms git prints, and *declining* to expand anything
else so that the parsers still reject rubbish on their own terms.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dependamerge.cli import app
from dependamerge.url_parser import (
    is_supported_github_host,
    parse_change_url,
    parse_gerrit_topic_url,
    parse_org_url,
    parse_repo_url,
)
from dependamerge.url_parser.models import UrlParseError
from dependamerge.url_parser.shorthand import (
    DEFAULT_GITHUB_HOST,
    default_github_host,
    enterprise_hosts,
    looks_like_host,
    looks_like_owner,
    normalize_target,
    set_github_host,
    strip_git_suffix,
)


class TestLooksLikeHost:
    """The rule that separates ``owner/repo`` from ``host/owner``."""

    @pytest.mark.parametrize(
        "segment",
        [
            "github.com",
            "ghe.corp.example.com",
            "gerrit.linuxfoundation.org",
            "localhost:3000",
            "example.com:8443",
        ],
    )
    def test_host_shaped_segments(self, segment):
        assert looks_like_host(segment) is True

    @pytest.mark.parametrize(
        "segment",
        ["lfreleng-actions", "acme", "o", "some-org-name", "", "localhost"],
    )
    def test_owner_shaped_segments(self, segment):
        # A GitHub login cannot contain a dot, which is what makes the
        # two-segment case decidable at all.  Bare ``localhost``
        # satisfies the login grammar and names a real account, so it
        # is an owner here; a local server is named with a port or an
        # explicit scheme.
        assert looks_like_host(segment) is False

    def test_bare_localhost_is_an_owner_shorthand(self):
        assert normalize_target("localhost/widget") == (
            "https://github.com/localhost/widget"
        )

    def test_localhost_with_a_port_is_a_host(self):
        assert normalize_target("localhost:3000/acme/widget") == (
            "https://localhost:3000/acme/widget"
        )


class TestLooksLikeOwner:
    """Shorthand expansion is gated on the GitHub login grammar."""

    @pytest.mark.parametrize(
        "segment",
        ["lfreleng-actions", "acme", "a", "A1", "not-a-url", "x" * 39],
    )
    def test_valid_logins(self, segment):
        assert looks_like_owner(segment) is True

    @pytest.mark.parametrize("segment", ["a--b--t", "acme--tools"])
    def test_consecutive_hyphens_are_accepted(self, segment):
        # Deliberate, and checked against the API rather than against
        # the signup form's wording.  GitHub's *user* signup rejects
        # consecutive hyphens, but organisation names did not always,
        # and ``a--b--t`` is a real organisation (id 7857740, created
        # 2014).  Owner-wide merging is mostly an organisation
        # operation, so excluding "--" here would make a resolvable
        # target unreachable by shorthand.
        #
        # The gate exists to stop obvious rubbish, not to reimplement
        # GitHub's account policy: too permissive costs a clear 404,
        # too strict reports "Invalid URL" for an owner that exists.
        # Enterprise adds a second reason --- accounts provisioned
        # over LDAP or SAML need not follow the dotcom signup grammar.
        assert looks_like_owner(segment) is True

    @pytest.mark.parametrize(
        ("segment", "why"),
        [
            ("not a url", "spaces"),
            ("-leading", "leading hyphen"),
            ("trailing-", "trailing hyphen"),
            ("has_underscore", "underscore"),
            ("has$dollar", "punctuation"),
            ("x" * 40, "too long"),
            ("", "empty"),
        ],
    )
    def test_invalid_logins(self, segment, why):
        assert looks_like_owner(segment) is False, why


class TestStripGitSuffix:
    """``.git`` comes off; everything else is left exactly as given."""

    def test_removes_git_suffix(self):
        assert strip_git_suffix("/owner/repo.git") == "/owner/repo"

    def test_removes_git_suffix_with_trailing_slash(self):
        assert strip_git_suffix("/owner/repo.git/") == "/owner/repo"

    def test_preserves_trailing_slash_when_no_git_suffix(self):
        # The parsers record the input as ``original_url``, so a path
        # without ``.git`` must survive byte for byte.
        assert strip_git_suffix("/owner/") == "/owner/"

    def test_leaves_dotted_repo_names_alone(self):
        assert strip_git_suffix("/owner/repo.js") == "/owner/repo.js"

    def test_bare_dot_git_is_not_a_suffix(self):
        assert strip_git_suffix("/.git") == "/.git"

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            (
                "https://github.com/acme/tool.git.git",
                "https://github.com/acme/tool.git",
            ),
            (
                "git@github.com:acme/tool.git.git",
                "https://github.com/acme/tool.git",
            ),
        ],
    )
    def test_a_repository_named_dot_git_is_reachable(self, url, expected):
        # Exactly one suffix comes off, never a greedy sweep.  A
        # repository genuinely called ``tool.git`` has the clone URL
        # ``tool.git.git``, so this is the form git itself reports and
        # it round-trips to the right project.  Stripping repeatedly
        # would make such a repository unreachable.
        assert normalize_target(url) == expected

    def test_the_project_survives_the_round_trip(self):
        assert (
            parse_repo_url("https://github.com/acme/tool.git.git").project
            == "acme/tool.git"
        )


class TestCredentialsAreNotCarried:
    """Embedded credentials never survive normalisation.

    A clone remote may carry a token, and a target inferred from a
    checkout is printed back to the operator --- so anything left in
    the URL lands in the terminal and in any captured log.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            "https://someuser:ghp_SECRETTOKEN@github.com/acme/widget.git",
            "http://someuser:ghp_SECRETTOKEN@github.com/acme/widget",
            "ssh://someuser:ghp_SECRETTOKEN@github.com/acme/widget.git",
            "git@github.com:acme/widget.git",
        ],
    )
    def test_no_userinfo_survives(self, raw):
        result = normalize_target(raw)
        assert "@" not in result
        assert "ghp_SECRETTOKEN" not in result

    def test_the_rest_of_the_url_is_untouched(self):
        assert (
            normalize_target("https://u:p@github.com/acme/widget.git")
            == "https://github.com/acme/widget"
        )

    def test_scheme_and_port_survive_credential_stripping(self):
        # Only the credentials go: a web URL keeps its scheme and port.
        assert (
            normalize_target("http://u:p@ghe.example.com:8443/acme/widget")
            == "http://ghe.example.com:8443/acme/widget"
        )


class TestConfiguredHostWithAPort:
    """A configured host naming a port is a configuration error.

    Ports are unsupported end to end, so silently trimming one would
    address port 443 on a server the operator did not name, and keeping
    it made shorthand expand into a URL its own parser rejects.
    """

    @pytest.mark.parametrize(
        "variable",
        ["DEPENDAMERGE_GITHUB_HOST", "GH_HOST"],
    )
    def test_ported_default_is_refused(self, monkeypatch, variable):
        for name in (
            "DEPENDAMERGE_GITHUB_HOST",
            "GH_HOST",
            "DEPENDAMERGE_GITHUB_HOSTS",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(variable, "ghe.example.com:8443")
        with pytest.raises(UrlParseError, match="names a port"):
            default_github_host()

    def test_ported_declaration_is_refused(self, monkeypatch):
        # DEPENDAMERGE_GITHUB_HOSTS only declares; it never sets the
        # default, so it is enterprise_hosts() that must reject it.
        for name in (
            "DEPENDAMERGE_GITHUB_HOST",
            "GH_HOST",
            "DEPENDAMERGE_GITHUB_HOSTS",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", "ghe.example.com:8443")
        with pytest.raises(UrlParseError, match="names a port"):
            enterprise_hosts()

    def test_portless_configuration_is_fine(self, monkeypatch):
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.setenv("GH_HOST", "ghe.example.com")
        assert default_github_host() == "ghe.example.com"


class TestUnsupportedSchemes:
    """Only web URLs and git transports are accepted.

    Rewriting an arbitrary scheme to https turned a target that should
    be refused into a real GitHub operation --- the parsers read only
    the netloc and path, so ``javascript://github.com/acme/widget``
    parsed as an ordinary repository.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            "javascript://github.com/acme/widget",
            "file:///etc/passwd",
            "data://github.com/acme/widget",
            "ftp://github.com/acme/widget",
        ],
    )
    def test_unsupported_scheme_is_refused(self, raw):
        with pytest.raises(UrlParseError, match="Unsupported URL scheme"):
            normalize_target(raw)

    @pytest.mark.parametrize(
        "raw",
        [
            "javascript://github.com/acme/widget",
            "file:///etc/passwd",
        ],
    )
    def test_parsers_refuse_them_too(self, raw):
        # The check has to bite at the parser, not merely in
        # normalisation, or the netloc still reaches a real request.
        with pytest.raises(UrlParseError):
            parse_repo_url(raw)

    @pytest.mark.parametrize(
        "raw",
        [
            "ssh://git@github.com/acme/widget.git",
            "git://github.com/acme/widget.git",
            "https://github.com/acme/widget",
            "http://github.com/acme/widget",
        ],
    )
    def test_supported_schemes_still_work(self, raw):
        assert normalize_target(raw).startswith(("http://", "https://"))


class TestDefaultHostIsResolvedLazily:
    """An explicit URL does not depend on the GitHub default.

    The default was resolved before the function knew whether it was
    needed, so a misconfigured ``GH_HOST`` broke even a Gerrit URL that
    never consults it.
    """

    @pytest.fixture(autouse=True)
    def _broken_default(self, monkeypatch):
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.setenv("GH_HOST", "broken:8443")

    def test_explicit_gerrit_topic_is_unaffected(self):
        parsed = parse_gerrit_topic_url("https://gerrit.example.org/q/topic:x")
        assert parsed.topic == "x"

    def test_explicit_github_url_is_unaffected(self):
        assert parse_repo_url("https://github.com/acme/widget").project == (
            "acme/widget"
        )

    def test_shorthand_still_reports_the_misconfiguration(self):
        # The shorthand branch is the one that needs the default, so it
        # is the one that must still complain.
        with pytest.raises(UrlParseError, match="names a port"):
            normalize_target("acme/widget")


class TestAConfiguredHostMustBeBare:
    """A configured host decides where the token goes, so it is trusted.

    Stripping only the scheme and path left userinfo in place, and a
    URL of that shape addresses the authority *after* the ``@``. So
    ``https://github.com@evil.example`` reduced to a "host" that passed
    the declaration check by looking like github.com, then sent the
    token to ``evil.example``.
    """

    @pytest.mark.parametrize(
        "configured",
        [
            "https://github.com@evil.example",
            "github.com@evil.example",
            "user:pw@evil.example",
            "github.com?x=1",
            "github.com#frag",
        ],
    )
    def test_a_value_that_is_not_a_hostname_is_refused(self, configured, monkeypatch):
        monkeypatch.setenv("GH_HOST", configured)

        with pytest.raises(UrlParseError):
            default_github_host()

    def test_the_declaration_check_cannot_be_fooled(self, monkeypatch):
        # The sharpest part: the crafted value did not merely slip
        # through, it *passed* ``is_supported_github_host`` by looking
        # like github.com, so the guard reported the token as safe to
        # send to a host the operator never named.
        monkeypatch.setenv("GH_HOST", "https://github.com@evil.example")

        with pytest.raises(UrlParseError):
            is_supported_github_host("github.com@evil.example")

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [
            ("github.com", "github.com"),
            ("ghe.corp.example.com", "ghe.corp.example.com"),
            ("localhost", "localhost"),
            # A scheme and a trailing slash are still tolerated: those
            # are shapes an operator plausibly pastes, and neither
            # changes which server is addressed.
            ("https://ghe.example.com/", "ghe.example.com"),
        ],
    )
    def test_ordinary_hosts_are_unaffected(self, configured, expected, monkeypatch):
        monkeypatch.setenv("GH_HOST", configured)

        assert default_github_host() == expected

    @pytest.mark.parametrize(
        "configured",
        [
            # Reaches the *port* branch, because the text after the
            # colon looks like one --- which is how a pasted URL leaks
            # its token through an error message.
            "https://user:ghp_SECRETTOKEN@host",
            "ghe.example.com?token=ghp_SECRETTOKEN",
            "ghe.example.com#ghp_SECRETTOKEN",
        ],
    )
    def test_a_rejected_value_is_not_echoed_verbatim(self, configured, monkeypatch):
        # These messages describe a value the operator got wrong, and
        # pasting a whole URL is a plausible way to get it wrong.
        #
        # Credentials live in the userinfo, the query and the fragment,
        # so those go.  The hostname itself stays: it is not a secret,
        # and it is the part that tells the operator which setting is
        # at fault.
        monkeypatch.setenv("GH_HOST", configured)

        with pytest.raises(UrlParseError) as excinfo:
            default_github_host()

        assert "SECRETTOKEN" not in str(excinfo.value).upper()

    def test_the_message_still_identifies_the_mistake(self, monkeypatch):
        # Redaction must not reduce this to an unactionable complaint:
        # asserting only the secret's absence would pass if the value
        # were dropped altogether.
        monkeypatch.setenv("GH_HOST", "https://user:ghp_SECRETTOKEN@ghe.example.com")

        with pytest.raises(UrlParseError, match=r"\*\*\*@ghe\.example\.com"):
            default_github_host()

    def test_a_value_with_no_secret_is_shown_in_full(self, monkeypatch):
        monkeypatch.setenv("GH_HOST", "broken:8443")

        with pytest.raises(UrlParseError, match="broken:8443"):
            default_github_host()


class TestMalformedTargetsAreRefusedEndToEnd:
    """Preserving ``.git`` only helps if a parser then refuses it.

    Normalisation leaving these unchanged was asserted as "stays
    invalid", which was not true: ``/acme.git`` became the owner
    ``acme.git`` and reached owner-wide dispatch, and the change shapes
    accept trailing segments so ``/pull/7/files.git`` matched pull
    request 7. The suffix is a marker; honouring it is the parsers' job.

    Driven through the command, because asserting normalisation is what
    let the false claim look verified.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/acme.git",
            "https://github.com/orgs/acme.git",
            "https://github.com/acme/widget/pull/7.git",
            "https://github.com/acme/widget/pull/7/files.git",
            "https://gerrit.example.org/c/p/+/123/files.git",
        ],
    )
    def test_the_command_refuses_them(self, url):
        result = CliRunner().invoke(app, ["merge", url, "--token", "t", "--dry-run"])

        assert result.exit_code != 0, result.stdout
        assert "Invalid URL" in result.stdout
        # Neither dispatch mode may be reached: the earlier fix moved
        # ``/orgs/acme.git`` from owner-wide into *repository* mode
        # rather than refusing it, which the parser test did not catch.
        assert "Owner mode" not in result.stdout
        assert "Repository mode" not in result.stdout

    @pytest.mark.parametrize(
        ("url", "parser", "attribute", "expected"),
        [
            # A repository may genuinely be called ``widget.git``; its
            # clone URL then ends ``.git.git``.
            (
                "https://github.com/acme/widget.git.git",
                parse_repo_url,
                "project",
                "acme/widget.git",
            ),
            # A Gerrit topic may end in ``.git`` too.
            (
                "https://gerrit.example.org/q/topic:release.git",
                parse_gerrit_topic_url,
                "topic",
                "release.git",
            ),
            # And the ordinary shapes are untouched.
            ("https://github.com/orgs/acme", parse_org_url, "owner", "acme"),
            (
                "https://github.com/acme/widget/pull/7/files",
                parse_change_url,
                "change_number",
                7,
            ),
        ],
    )
    def test_legitimate_targets_are_untouched(self, url, parser, attribute, expected):
        assert getattr(parser(url), attribute) == expected


class TestPageRoutesKeepTheirGitSuffix:
    """No clone URL ends in a GitHub page route.

    Removing a ``.git`` tail from one repairs a malformed URL into a
    valid target, and for the ``orgs`` routes a *broader* one: an
    owner-wide merge of everything the owner has.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/orgs/acme.git",
            "https://github.com/orgs/acme/repositories.git",
            "https://github.com/acme/widget/pulls.git",
            # A GitHub clone URL always names an owner *and* a
            # repository, so one segment is not one.  Trimming turned
            # this into the owner URL for ``acme``, and the dispatcher
            # merged every repository they own.
            "https://github.com/acme.git",
        ],
    )
    def test_the_suffix_stays_so_the_url_stays_invalid(self, url):
        assert normalize_target(url) == url

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            # The route words are ordinary names in any other position,
            # and both of these are real repositories.
            (
                "https://github.com/clerk/orgs.git",
                "https://github.com/clerk/orgs",
            ),
            (
                "https://github.com/csabella/pulls.git",
                "https://github.com/csabella/pulls",
            ),
            # Route names on GitHub, project path segments on Gerrit.
            (
                "https://gerrit.example.org/orgs/acme.git",
                "https://gerrit.example.org/orgs/acme",
            ),
            (
                "https://gerrit.example.org/a/b/pulls.git",
                "https://gerrit.example.org/a/b/pulls",
            ),
            # Gerrit projects do sit at the root, so a single segment is
            # a genuine clone URL there.  This is why the rule above is
            # gated on the host rather than applied everywhere.
            (
                "https://gerrit.example.org/project.git",
                "https://gerrit.example.org/project",
            ),
        ],
    )
    def test_clone_urls_are_unaffected(self, url, expected):
        assert normalize_target(url) == expected


class TestReservedRouteShorthand:
    """A shorthand's first segment is an owner, never a URL route.

    ``orgs`` is a path segment GitHub reserves, so expanding the
    shorthand ``orgs/acme`` produced ``https://github.com/orgs/acme``
    --- the canonical *owner-wide* URL.  A two-segment shorthand names
    one repository, so that silently widened the request into a merge
    of everything ``acme`` owns.  Scope must never broaden by
    accident, so the ambiguity is reported instead of guessed.
    """

    def test_reserved_first_segment_is_refused(self):
        with pytest.raises(UrlParseError) as excinfo:
            normalize_target("orgs/acme")
        message = str(excinfo.value)
        assert "orgs" in message
        assert "acme" in message

    def test_the_explicit_url_still_means_owner_wide(self):
        # Only the *shorthand* is ambiguous.  Typed in full, this is
        # GitHub's own owner URL and keeps its meaning.
        assert (
            normalize_target("https://github.com/orgs/acme")
            == "https://github.com/orgs/acme"
        )

    def test_a_reserved_word_elsewhere_is_an_ordinary_name(self):
        # Only the first segment is a route.  ``clerk/orgs`` is a real
        # repository, and nothing about it is ambiguous.
        assert normalize_target("clerk/orgs") == "https://github.com/clerk/orgs"

    def test_a_bare_reserved_word_is_refused_with_a_reason(self):
        # Building the two-segment message unconditionally raised
        # IndexError, so ``merge orgs`` crashed through normalisation.
        # Letting it expand instead was no better: the parsers rejected
        # ``https://github.com/orgs`` as a malformed *repository* URL,
        # which explains nothing.  GitHub reserves the name, so no such
        # account exists and the reason is known here.
        with pytest.raises(UrlParseError, match="not an owner"):
            normalize_target("orgs")

    @pytest.mark.parametrize("target", ["orgs", "orgs/acme"])
    def test_the_cli_reports_it_rather_than_crashing(self, target):
        # The normalisation assertion above is not enough on its own:
        # an earlier version passed it while ``merge orgs`` still failed
        # downstream with an unrelated message.  This exercises the
        # path the operator actually takes.
        result = CliRunner().invoke(app, ["merge", target, "--token", "t", "--dry-run"])

        assert result.exit_code != 0
        assert "not an owner" in result.stdout
        assert "Invalid GitHub repository URL format" not in result.stdout


class TestGerritTopicNeedsItsOwnHost:
    """Owner shorthand must not manufacture a Gerrit target.

    Shorthand is a GitHub convenience and resolves against the GitHub
    default host, so ``q/topic:x`` expanded to
    ``https://github.com/q/topic:x`` --- whose path this parser then
    accepted, dispatching a Gerrit topic run against github.com.
    """

    def test_owner_shorthand_is_refused(self):
        with pytest.raises(UrlParseError, match="no host"):
            parse_gerrit_topic_url("q/topic:x")

    @pytest.mark.parametrize(
        ("url", "host", "topic"),
        [
            ("gerrit.example.org/q/topic:x", "gerrit.example.org", "x"),
            (
                "https://gerrit.example.org/q/topic:release",
                "gerrit.example.org",
                "release",
            ),
            (
                "https://gerrit.onap.org/r/q/topic:update",
                "gerrit.onap.org",
                "update",
            ),
            (
                "https://gerrit.example.org/#/q/topic:legacy",
                "gerrit.example.org",
                "legacy",
            ),
        ],
    )
    def test_every_host_bearing_form_still_parses(self, url, host, topic):
        # The complement, including the scheme-less and legacy-fragment
        # forms: requiring a host must not cost any shape that names one.
        parsed = parse_gerrit_topic_url(url)

        assert parsed.host == host
        assert parsed.topic == topic


class TestRepositoryNamedPulls:
    """``/owner/repo/pulls`` is a page; ``pulls`` is also a repo name.

    Stripping the suffix unconditionally left ``/owner/pulls`` with a
    single segment, so every repository actually called ``pulls`` was
    rejected as malformed.  There are more than fifty on github.com.
    """

    @pytest.mark.parametrize(
        "raw",
        ["csabella/pulls", "https://github.com/csabella/pulls"],
    )
    def test_a_repository_named_pulls_resolves(self, raw):
        assert parse_repo_url(raw).project == "csabella/pulls"

    def test_the_pulls_page_suffix_still_comes_off(self):
        # The complement: the suffix exists because this form is
        # accepted, and narrowing the rule must not lose it.
        assert (
            parse_repo_url("https://github.com/acme/widget/pulls").project
            == "acme/widget"
        )


class TestDeclarationPrecedenceShortCircuits:
    """A losing declaration must not veto the winning one.

    Authorisation materialised every declared source, so validating a
    lower-priority value that lost the precedence contest could reject
    a host the operator had just named with ``--github-host``.
    """

    def test_a_malformed_lower_priority_value_does_not_block(self, monkeypatch):
        monkeypatch.setenv("GH_HOST", "broken:8443")
        set_github_host("ghe.example.com")

        assert is_supported_github_host("ghe.example.com") is True
        assert is_supported_github_host("github.com") is True

    def test_a_malformed_list_does_not_block_a_single_host(self, monkeypatch):
        # The single-host settings are what ``default_github_host``
        # resolves, so they are yielded first.  Reading the declaration
        # list before them let a malformed entry there raise before a
        # valid default could match --- the same ineffective precedence,
        # one source further along.
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOST", "ghe.example.com")
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", "broken:8443")

        assert is_supported_github_host("ghe.example.com") is True

    def test_the_malformed_value_is_still_reported(self, monkeypatch):
        # It is not swallowed: a caller that genuinely has to read
        # every declaration still meets the configuration error.
        monkeypatch.setenv("GH_HOST", "broken:8443")
        set_github_host("ghe.example.com")

        with pytest.raises(UrlParseError, match="broken:8443"):
            enterprise_hosts()


class TestNormalizeTarget:
    """Expansion of every accepted input form."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Shorthand
            ("lfreleng-actions", "https://github.com/lfreleng-actions"),
            ("acme/widget", "https://github.com/acme/widget"),
            ("acme/widget/pull/7", "https://github.com/acme/widget/pull/7"),
            # ``orgs/acme`` is deliberately absent: see
            # TestReservedRouteShorthand.  It expanded here until that
            # was found to widen a two-segment shorthand, which names
            # one repository, into an owner-wide URL.
            # Scheme-less but host-shaped
            ("github.com/acme", "https://github.com/acme"),
            ("github.com/acme/widget", "https://github.com/acme/widget"),
            (
                "ghe.corp.example.com/acme/widget",
                "https://ghe.corp.example.com/acme/widget",
            ),
            # Already absolute
            (
                "https://github.com/acme/widget/pull/7",
                "https://github.com/acme/widget/pull/7",
            ),
            ("http://github.com/acme/widget", "http://github.com/acme/widget"),
        ],
    )
    def test_expansions(self, raw, expected):
        assert normalize_target(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("git@github.com:acme/widget.git", "https://github.com/acme/widget"),
            ("git@github.com:acme/widget", "https://github.com/acme/widget"),
            (
                "ssh://git@github.com/acme/widget.git",
                "https://github.com/acme/widget",
            ),
            ("https://github.com/acme/widget.git", "https://github.com/acme/widget"),
        ],
    )
    def test_git_remote_forms(self, raw, expected):
        # These are what ``git remote get-url`` prints, so they must
        # round-trip to the web URL the parsers understand.
        assert normalize_target(raw) == expected

    def test_ssh_port_is_dropped(self):
        # A Gerrit SSH remote's 29418 is a transport port with no
        # bearing on the web URL.
        assert (
            normalize_target("ssh://git@gerrit.example.org:29418/releng/tool")
            == "https://gerrit.example.org/releng/tool"
        )

    def test_web_port_is_kept(self):
        # By contrast a port on a scheme-less web host is part of the
        # address and must survive normalisation.  Whether the rest of
        # the stack can *use* it is a separate question --- see
        # TestPortBearingTargets.
        assert (
            normalize_target("ghe.example.com:8443/acme/widget")
            == "https://ghe.example.com:8443/acme/widget"
        )

    def test_gerrit_topic_colon_is_not_a_port(self):
        url = "https://gerrit.onap.org/r/q/topic:update-settings"
        assert normalize_target(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "https://gerrit.example.org/q/topic:release.git",
            "https://gerrit.example.org/r/q/topic:release.git",
        ],
    )
    def test_gerrit_topic_keeps_a_dot_git_value(self, url):
        # Inside a Gerrit search the trailing text is a *value*, not a
        # repository name.  Trimming it searched for the wrong topic.
        assert normalize_target(url) == url

    def test_gerrit_topic_with_a_git_suffix_parses_intact(self):
        parsed = parse_gerrit_topic_url(
            "https://gerrit.example.org/q/topic:release.git"
        )
        assert parsed.topic == "release.git"

    @pytest.mark.parametrize(
        "raw",
        [
            "git@github.com:q/widget.git",
            "https://github.com/q/widget.git",
        ],
    )
    def test_owner_named_q_still_gets_clone_handling(self, raw):
        # ``q`` is a legal GitHub login, so treating any ``q`` segment
        # as a Gerrit search left the suffix on a perfectly ordinary
        # clone URL.  The colon in the final segment is the real signal.
        assert normalize_target(raw) == "https://github.com/q/widget"

    def test_encoded_colon_also_marks_a_query_value(self):
        url = "https://gerrit.example.org/q/topic%3Arelease.git"
        assert normalize_target(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/acme/widget/pull/7.git",
            "https://github.com/acme/widget/pull/7/files.git",
            "https://gerrit.example.org/c/acme/widget/+/123.git",
            "https://gerrit.example.org/r/c/acme/widget/+/123.git",
            "https://gerrit.example.org/changes/123.git",
        ],
    )
    def test_change_urls_keep_a_git_suffix(self, url):
        # No such clone URL exists, so trimming the suffix *repaired* a
        # malformed URL into a valid reference to a change the operator
        # never named.  It has to stay invalid.
        assert normalize_target(url) == url

    @pytest.mark.parametrize(
        "path",
        [
            # A marker in the wrong position is part of the project.
            # Gerrit projects nest, so each of these is a clone URL.
            "org/pull/123",
            "org/changes/123",
            "org/pull/widget",
            "acme/pull",
            "changes",
        ],
    )
    def test_a_marker_off_position_is_part_of_the_project(self, path):
        # Matching "a marker with a number after it" anywhere would
        # classify these as changes and leave the suffix on, so
        # checkout inference would report the wrong project name.
        # Each shape is anchored where the change parser anchors it.
        assert (
            normalize_target(f"https://gerrit.example.org/{path}.git")
            == f"https://gerrit.example.org/{path}"
        )

    @pytest.mark.parametrize(
        "host",
        ["github.com", "ghe.example.com", "gerrit.example.org"],
    )
    def test_the_full_gerrit_change_shape_is_protected_everywhere(
        self, host, monkeypatch
    ):
        # A ``+`` segment cannot appear in a GitHub owner or repository
        # name, so this shape collides with nothing and costs nothing
        # to honour.  Gating it on the host let a malformed
        # ``/c/project/+/123.git`` be repaired into a live change
        # reference on a declared Enterprise host.
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", "ghe.example.com")
        url = f"https://{host}/c/project/+/123.git"

        assert normalize_target(url) == url

    @pytest.mark.parametrize("owner", ["changes", "c"])
    def test_gerrit_markers_are_ordinary_logins_on_github(self, owner):
        # ``changes`` and ``c`` are valid GitHub logins, so these are
        # clone URLs for the repository ``123``.  Applying the Gerrit
        # shapes regardless of host left the suffix on, and the
        # repository came out as ``123.git``.
        assert (
            normalize_target(f"https://github.com/{owner}/123.git")
            == f"https://github.com/{owner}/123"
        )

    def test_a_declared_enterprise_host_is_read_as_github(self, monkeypatch):
        # The same reasoning has to reach Enterprise, or a declared
        # host keeps the Gerrit reading of its own repository names.
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", "ghe.example.com")

        assert (
            normalize_target("https://ghe.example.com/changes/123.git")
            == "https://ghe.example.com/changes/123"
        )

    def test_unusable_host_configuration_does_not_break_normalisation(
        self, monkeypatch
    ):
        # Consulting the declaration must not let unrelated broken
        # configuration make normalisation raise.
        monkeypatch.setenv("GH_HOST", "broken:8443")

        assert (
            normalize_target("https://github.com/acme/widget.git")
            == "https://github.com/acme/widget"
        )

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/acme/widget/pull/7.git",
            "https://gerrit.example.org/c/acme/widget/+/123.git",
        ],
    )
    def test_change_urls_with_a_git_suffix_are_refused(self, url):
        # The unstripped suffix is only the mechanism; what matters is
        # that no operation is reachable through it.  The message moved
        # from the shape mismatch to the stray suffix, which is the more
        # specific reason and now fires first --- the refusal these
        # assert is unchanged.
        with pytest.raises(UrlParseError, match="trailing '.git'"):
            parse_change_url(url)

    def test_a_project_named_pull_still_gets_clone_handling(self):
        # Gerrit projects nest, so ``pull`` can be a real path segment.
        # Only the third segment is a GitHub pull request marker.
        assert (
            normalize_target("https://gerrit.example.org/org/pull/widget.git")
            == "https://gerrit.example.org/org/pull/widget"
        )

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_empty_input_passes_through(self, raw):
        # Left empty so callers keep raising their own "URL cannot be
        # empty" error with its existing wording.
        assert normalize_target(raw) == ""

    def test_rooted_path_is_not_expanded(self):
        # Nobody types "/owner/repo" as an abbreviation; expanding it
        # would turn a hostname error into a silent guess.
        assert normalize_target("/acme/widget") == "/acme/widget"

    @pytest.mark.parametrize("raw", ["not a url", "has$dollar", "x" * 40])
    def test_non_login_text_is_not_expanded(self, raw):
        assert normalize_target(raw) == raw

    def test_explicit_default_host(self):
        assert (
            normalize_target("acme/widget", default_host="ghe.example.com")
            == "https://ghe.example.com/acme/widget"
        )


class TestDefaultGithubHost:
    """The default host honours the usual environment overrides."""

    def test_falls_back_to_dotcom(self, monkeypatch):
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.delenv("GH_HOST", raising=False)
        assert default_github_host() == DEFAULT_GITHUB_HOST

    def test_gh_host_is_honoured(self, monkeypatch):
        # Reusing the GitHub CLI's variable means an operator who has
        # already pointed ``gh`` at their Enterprise instance does not
        # configure the same thing twice.
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.setenv("GH_HOST", "ghe.corp.example.com")
        assert default_github_host() == "ghe.corp.example.com"

    def test_project_variable_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOST", "first.example.com")
        monkeypatch.setenv("GH_HOST", "second.example.com")
        assert default_github_host() == "first.example.com"

    def test_scheme_is_stripped_from_override(self, monkeypatch):
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.setenv("GH_HOST", "https://ghe.corp.example.com/")
        assert default_github_host() == "ghe.corp.example.com"

    def test_shorthand_resolves_against_the_override(self, monkeypatch):
        monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
        monkeypatch.setenv("GH_HOST", "ghe.corp.example.com")
        assert (
            normalize_target("acme/widget")
            == "https://ghe.corp.example.com/acme/widget"
        )


class TestShorthandReachesTheParsers:
    """End to end: the abbreviations resolve to the right parse."""

    def test_bare_owner_parses_as_owner(self):
        parsed = parse_org_url("lfreleng-actions")
        assert parsed.owner == "lfreleng-actions"
        assert parsed.host == "github.com"

    def test_owner_repo_parses_as_repository(self):
        parsed = parse_repo_url("acme/widget")
        assert (parsed.owner, parsed.repo) == ("acme", "widget")
        assert parsed.project == "acme/widget"

    def test_owner_repo_pull_parses_as_change(self):
        parsed = parse_change_url("acme/widget/pull/7")
        assert parsed.project == "acme/widget"
        assert parsed.change_number == 7

    def test_clone_url_parses_as_repository(self):
        # Regression: ``.git`` used to survive into the repo name, so
        # this returned a repository called "widget.git".
        parsed = parse_repo_url("https://github.com/acme/widget.git")
        assert parsed.repo == "widget"
        assert parsed.project == "acme/widget"

    def test_scp_remote_parses_as_repository(self):
        parsed = parse_repo_url("git@github.com:acme/widget.git")
        assert (parsed.owner, parsed.repo) == ("acme", "widget")

    def test_rooted_path_still_rejected(self):
        with pytest.raises(UrlParseError, match="URL must include a hostname"):
            parse_change_url("/acme/widget/pull/7")
