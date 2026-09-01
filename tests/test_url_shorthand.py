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

from dependamerge.url_parser import (
    parse_change_url,
    parse_gerrit_topic_url,
    parse_org_url,
    parse_repo_url,
)
from dependamerge.url_parser.models import UrlParseError
from dependamerge.url_parser.shorthand import (
    DEFAULT_GITHUB_HOST,
    default_github_host,
    looks_like_host,
    looks_like_owner,
    normalize_target,
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
            "localhost",
            "localhost:3000",
            "example.com:8443",
        ],
    )
    def test_host_shaped_segments(self, segment):
        assert looks_like_host(segment) is True

    @pytest.mark.parametrize(
        "segment",
        ["lfreleng-actions", "acme", "o", "some-org-name", ""],
    )
    def test_owner_shaped_segments(self, segment):
        # A GitHub login cannot contain a dot, which is what makes the
        # two-segment case decidable at all.
        assert looks_like_host(segment) is False


class TestLooksLikeOwner:
    """Shorthand expansion is gated on the GitHub login grammar."""

    @pytest.mark.parametrize(
        "segment",
        ["lfreleng-actions", "acme", "a", "A1", "not-a-url", "x" * 39],
    )
    def test_valid_logins(self, segment):
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


class TestNormalizeTarget:
    """Expansion of every accepted input form."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Shorthand
            ("lfreleng-actions", "https://github.com/lfreleng-actions"),
            ("acme/widget", "https://github.com/acme/widget"),
            ("acme/widget/pull/7", "https://github.com/acme/widget/pull/7"),
            ("orgs/acme", "https://github.com/orgs/acme"),
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
