# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Tests for inferring the merge target from the local checkout.

These build real git repositories rather than mocking ``run_git``.
The behaviour under test is a conversation with git --- which remote
wins, what a missing one does, how a detached or bare tree behaves ---
and a mock would only assert that we call git the way we already think
we do.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dependamerge.cli import app
from dependamerge.gitreview import GitReviewInfo
from dependamerge.local_repo import (
    LocalTarget,
    _safe_for_log,
    detect_local_target,
    host_suggests_gerrit,
    remote_url,
    repository_root,
)
from dependamerge.url_parser import ChangeSource, UrlParseError

GITHUB_REMOTE = "https://github.com/acme/widget.git"
GERRIT_SSH_REMOTE = "ssh://jdoe@gerrit.example.org:29418/releng/tool"

GITREVIEW = """[gerrit]
host=gerrit.linuxfoundation.org
port=29418
project=releng/tool.git
"""


@pytest.fixture(autouse=True)
def isolated_git_config(monkeypatch, tmp_path):
    """Detach git from the ambient user and system configuration.

    Without this the tests are not hermetic.  ``git remote get-url``
    applies ``url.<base>.insteadOf`` rewrites, so a developer whose
    global config rewrites https to ssh sees different URLs from CI ---
    which is exactly what happened while writing these.
    """
    empty = tmp_path / "empty-gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty))
    monkeypatch.delenv("GIT_CONFIG_NOSYSTEM", raising=False)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        timeout=30,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """An initialised git repository with no remotes."""
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    return root


class TestRepositoryRoot:
    """Locating the working tree."""

    def test_finds_the_root_from_the_root(self, repo):
        assert repository_root(repo) == repo.resolve()

    def test_finds_the_root_from_a_subdirectory(self, repo):
        nested = repo / "src" / "deep"
        nested.mkdir(parents=True)
        assert repository_root(nested) == repo.resolve()

    def test_returns_none_outside_a_repository(self, tmp_path):
        outside = tmp_path / "plain"
        outside.mkdir()
        assert repository_root(outside) is None


class TestRemoteSelection:
    """Which remote speaks for the checkout."""

    def test_origin_is_used_when_alone(self, repo):
        _git(repo, "remote", "add", "origin", GITHUB_REMOTE)
        assert remote_url(repo) == ("origin", GITHUB_REMOTE)

    def test_upstream_wins_over_origin(self, repo):
        # In the fork workflow ``origin`` is the operator's own fork.
        # Merging there is almost never what "this repository" means.
        _git(repo, "remote", "add", "origin", "https://github.com/me/widget.git")
        _git(repo, "remote", "add", "upstream", GITHUB_REMOTE)
        assert remote_url(repo) == ("upstream", GITHUB_REMOTE)

    def test_returns_none_without_a_known_remote(self, repo):
        _git(repo, "remote", "add", "backup", GITHUB_REMOTE)
        assert remote_url(repo) is None


class TestHostSuggestsGerrit:
    """The weakest of the Gerrit signals."""

    @pytest.mark.parametrize(
        "host",
        [
            "gerrit.example.org",
            "review.gerrit.example.org",
            "gerrit-review.googlesource.com",
            "GERRIT.example.org",
        ],
    )
    def test_gerrit_hosts(self, host):
        assert host_suggests_gerrit(host) is True

    @pytest.mark.parametrize(
        "host",
        [
            "github.com",
            # Matching on whole labels, so a name that merely contains
            # "gerrit" does not qualify.
            "notgerrit.example.org",
            "mygerrit.example.org",
            "",
        ],
    )
    def test_non_gerrit_hosts(self, host):
        assert host_suggests_gerrit(host) is False


class TestDetectLocalTarget:
    """End to end inference from a checkout."""

    def test_github_remote_becomes_a_repository_url(self, repo):
        _git(repo, "remote", "add", "origin", GITHUB_REMOTE)

        target = detect_local_target(repo)

        assert target is not None
        assert target.source == ChangeSource.GITHUB
        # ``.git`` is gone, so this parses as a repository rather than
        # one named "widget.git".
        assert target.url == "https://github.com/acme/widget"
        assert target.remote == "origin"

    def test_scp_style_remote_is_understood(self, repo):
        _git(repo, "remote", "add", "origin", "git@github.com:acme/widget.git")

        target = detect_local_target(repo)

        assert target is not None
        assert target.url == "https://github.com/acme/widget"

    @pytest.mark.parametrize(
        "remote",
        [
            "https://someuser:ghp_SECRET@github.com/acme/widget.git",
            "https://github.com/acme/widget.git?token=ghp_SECRET",
            "https://github.com/acme/widget.git#ghp_SECRET",
        ],
    )
    def test_no_credential_survives_into_the_inferred_url(self, repo, remote):
        # The inferred URL is printed back to the operator, so anything
        # left in it lands in the terminal and in any captured log.
        # Userinfo is only one of the places a token can hide.
        _git(repo, "remote", "add", "origin", remote)

        target = detect_local_target(repo)

        assert target is not None
        assert "ghp_SECRET" not in target.url
        assert target.url == "https://github.com/acme/widget"

    def test_gitreview_marks_the_checkout_as_gerrit(self, repo):
        # Even with a GitHub-looking remote: .gitreview is Gerrit's own
        # declaration, and github2gerrit repositories genuinely have
        # both.
        _git(repo, "remote", "add", "origin", GITHUB_REMOTE)
        (repo / ".gitreview").write_text(GITREVIEW, encoding="utf-8")

        target = detect_local_target(repo)

        assert target is not None
        assert target.is_gerrit
        assert target.gitreview is not None
        assert target.gitreview.host == "gerrit.linuxfoundation.org"
        assert target.gitreview.project == "releng/tool"

    def test_gerrit_ssh_port_is_detected_without_gitreview(self, repo):
        _git(repo, "remote", "add", "origin", GERRIT_SSH_REMOTE)

        target = detect_local_target(repo)

        assert target is not None
        assert target.is_gerrit
        assert target.gitreview is None
        # Identity still comes through: the guidance names the host and
        # project, which is what makes it actionable.
        assert target.host == "gerrit.example.org"
        assert target.project == "releng/tool"

    def test_gerrit_hostname_detection_also_reports_identity(self, repo):
        _git(repo, "remote", "add", "origin", "https://gerrit.example.org/releng/tool")

        target = detect_local_target(repo)

        assert target is not None
        assert target.is_gerrit
        assert target.host == "gerrit.example.org"
        assert target.project == "releng/tool"

    def test_scp_path_beginning_with_the_gerrit_port_is_not_gerrit(self, repo):
        # An scp remote puts the *path* after the colon, so an owner
        # named 29418 looked like a Gerrit server to a substring test.
        _git(repo, "remote", "add", "origin", "git@github.com:29418/widget.git")

        target = detect_local_target(repo)

        assert target is not None
        assert target.source == ChangeSource.GITHUB
        assert target.url == "https://github.com/29418/widget"

    def test_gerrit_hostname_is_detected_without_port(self, repo):
        _git(repo, "remote", "add", "origin", "https://gerrit.example.org/releng/tool")

        target = detect_local_target(repo)

        assert target is not None
        assert target.is_gerrit

    def test_a_declared_host_outranks_the_hostname_guess(self, repo, monkeypatch):
        # Enterprise hostnames are arbitrary, so an operator may declare
        # one carrying a ``gerrit`` label.  Letting the name-shape guess
        # win would classify it as Gerrit and make the declaration
        # unusable: bare ``dependamerge merge`` would refuse the
        # checkout it was configured for.
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", "gerrit.example.org")
        _git(
            repo,
            "remote",
            "add",
            "origin",
            "https://gerrit.example.org/acme/widget.git",
        )

        target = detect_local_target(repo)

        assert target is not None
        assert not target.is_gerrit
        assert target.url == "https://gerrit.example.org/acme/widget"

    def test_the_gerrit_ssh_port_outranks_a_declaration(self, repo, monkeypatch):
        # The complement: a declaration must not switch off the stronger
        # evidence.  Port 29418 belongs to Gerrit's SSH daemon.
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", "gerrit.example.org")
        _git(
            repo,
            "remote",
            "add",
            "origin",
            "ssh://git@gerrit.example.org:29418/acme/widget",
        )

        target = detect_local_target(repo)

        assert target is not None
        assert target.is_gerrit

    def test_none_outside_a_repository(self, tmp_path):
        outside = tmp_path / "plain"
        outside.mkdir()
        assert detect_local_target(outside) is None

    def test_missing_working_directory_does_not_crash(self, tmp_path):
        # ``run_git`` converts timeouts and non-zero exits to GitError,
        # but process creation failures stay OSError.  Catching only
        # GitError crashed the command with a traceback instead of
        # showing the no-usable-remote guidance.
        assert detect_local_target(tmp_path / "does-not-exist") is None

    def test_none_without_a_remote(self, repo):
        assert detect_local_target(repo) is None

    @pytest.mark.parametrize(
        "remote",
        [
            "file:///srv/widget.git",
            "/srv/widget.git",
            # Relative paths are the dangerous ones: run through
            # shorthand expansion, ``mirror/widget.git`` becomes a real
            # and unrelated GitHub repository, and an omitted-target
            # merge would act on it.
            "mirror/widget.git",
            "../sibling/widget.git",
            # A Windows drive letter is not a host, though it looks
            # like one to an scp-style pattern.
            "C:/repos/widget.git",
        ],
    )
    def test_local_mirror_remote_is_not_a_target(self, repo, remote):
        # A perfectly valid remote that names no server to merge
        # against.  "Cannot infer" rather than an error --- and
        # certainly not a traceback, since this runs before the merge
        # command's error handling.
        _git(repo, "remote", "add", "origin", remote)

        assert detect_local_target(repo) is None

    @pytest.mark.parametrize(
        ("remote", "expected"),
        [
            ("git@github.com:acme/widget.git", "https://github.com/acme/widget"),
            (
                "https://github.com/acme/widget.git",
                "https://github.com/acme/widget",
            ),
            (
                "ssh://git@github.com/acme/widget.git",
                "https://github.com/acme/widget",
            ),
            # An internal Enterprise installation may answer to a
            # single-label DNS name.  Userinfo is what separates this
            # from the Windows drive above: no filesystem path carries
            # a ``user@`` prefix.
            ("git@ghe:acme/widget.git", "https://ghe/acme/widget"),
        ],
    )
    def test_remotes_that_do_name_a_server_still_work(self, repo, remote, expected):
        # The complement: tightening the check must not reject the
        # forms git actually prints for a hosted remote.
        _git(repo, "remote", "add", "origin", remote)

        target = detect_local_target(repo)

        assert target is not None
        assert target.url == expected

    def test_malformed_gitreview_falls_back_to_the_remote(self, repo):
        # A .gitreview without a host tells us nothing, so the remote
        # still gets its say rather than the checkout being declared
        # Gerrit on the strength of an empty file.
        _git(repo, "remote", "add", "origin", GITHUB_REMOTE)
        (repo / ".gitreview").write_text("[gerrit]\nport=29418\n", encoding="utf-8")

        target = detect_local_target(repo)

        assert target is not None
        assert target.source == ChangeSource.GITHUB


class TestNoUrlUsesTheLocalCheckout:
    """Omitting the argument means "this repository"."""

    runner = CliRunner()

    def test_github_checkout_is_used_as_the_target(self, mocker):
        target = LocalTarget(
            source=ChangeSource.GITHUB,
            url="https://github.com/acme/widget",
            remote="origin",
            root=Path("/tmp/widget"),
        )
        mocker.patch(
            "dependamerge.cli._merge_target.detect_local_target", return_value=target
        )
        parsed = mocker.patch(
            "dependamerge.cli._merge_target.parse_repo_url",
            side_effect=UrlParseError("stop here"),
        )

        self.runner.invoke(app, ["merge", "--token", "test_token"])

        # The inferred URL reaches the parsers, which is the whole
        # point: inference adds a source of URLs, not a second path.
        assert parsed.call_args[0][0] == "https://github.com/acme/widget"

    def test_inference_is_announced(self, mocker):
        target = LocalTarget(
            source=ChangeSource.GITHUB,
            url="https://github.com/acme/widget",
            remote="upstream",
            root=Path("/tmp/widget"),
        )
        mocker.patch(
            "dependamerge.cli._merge_target.detect_local_target", return_value=target
        )
        # Stop at dispatch.  Without this the command runs the real
        # repository merge, so the unit suite makes network requests
        # with a fake token and these assertions pass on whatever was
        # printed before the failure.
        handler = mocker.patch("dependamerge.cli._merge_cmd._handle_repo_merge")

        result = self.runner.invoke(app, ["merge", "--token", "test_token"])

        # Acting on a repository the operator did not name has to be
        # visible, and it has to say *which* remote it chose.
        assert result.exit_code == 0, result.stdout
        assert "No URL given" in result.stdout
        assert "upstream" in result.stdout
        assert "https://github.com/acme/widget" in result.stdout
        # And it must actually have dispatched, rather than printing
        # the message and failing on the way.
        handler.assert_called_once()

    def test_gerrit_checkout_stops_with_guidance(self, mocker):
        info = GitReviewInfo(
            host="gerrit.linuxfoundation.org", port=29418, project="releng/tool"
        )
        target = LocalTarget(
            source=ChangeSource.GERRIT,
            url="",
            remote="origin",
            root=Path("/tmp/tool"),
            gitreview=info,
            host=info.host,
            project=info.project,
        )
        mocker.patch(
            "dependamerge.cli._merge_target.detect_local_target", return_value=target
        )

        result = self.runner.invoke(app, ["merge", "--token", "test_token"])

        assert result.exit_code == 1
        # Naming the detected host and project is what makes this
        # actionable rather than a bare refusal.
        assert "Gerrit repository" in result.stdout
        assert "gerrit.linuxfoundation.org" in result.stdout
        assert "releng/tool" in result.stdout
        assert "topic:" in result.stdout

    def test_gerrit_guidance_names_a_remote_only_checkout(self, mocker):
        # No .gitreview, so the identity comes from the remote alone.
        target = LocalTarget(
            source=ChangeSource.GERRIT,
            url="",
            remote="origin",
            root=Path("/tmp/tool"),
            host="gerrit.example.org",
            project="releng/tool",
        )
        mocker.patch(
            "dependamerge.cli._merge_target.detect_local_target", return_value=target
        )

        result = self.runner.invoke(app, ["merge", "--token", "test_token"])

        assert "gerrit.example.org" in result.stdout
        assert "releng/tool" in result.stdout

    def test_outside_a_repository_explains_the_alternatives(self, mocker):
        mocker.patch(
            "dependamerge.cli._merge_target.detect_local_target", return_value=None
        )

        result = self.runner.invoke(app, ["merge", "--token", "test_token"])

        assert result.exit_code == 1
        assert "not a git repository" in result.stdout
        assert "owner/repo" in result.stdout

    def test_an_explicit_url_skips_inference_entirely(self, mocker):
        detect = mocker.patch("dependamerge.cli._merge_target.detect_local_target")
        # Stubbed so the assertion does not depend on a live request
        # succeeding or failing.
        mocker.patch("dependamerge.cli._merge_cmd._handle_repo_merge")

        self.runner.invoke(
            app, ["merge", "https://github.com/acme/widget", "--token", "t"]
        )

        detect.assert_not_called()


class TestCredentialsNeverReachTheLog:
    """A declined remote is logged without its credentials.

    The module contract is that no credential survives into output, and
    a debug log is output.  A remote this module *declines* may use any
    scheme, so redaction cannot assume http(s).
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ftp://user:password@host/repo.git", "ftp://***@host/repo.git"),
            ("https://tok@github.com/a/b.git", "https://***@github.com/a/b.git"),
            ("https://github.com/a/b.git", "https://github.com/a/b.git"),
        ],
    )
    def test_userinfo_is_redacted(self, raw, expected):
        assert _safe_for_log(raw) == expected

    def test_declined_remote_does_not_log_its_password(self, repo, caplog):
        _git(repo, "remote", "add", "origin", "ftp://user:hunter2@host/repo.git")

        with caplog.at_level(logging.DEBUG, logger="dependamerge.local_repo"):
            assert detect_local_target(repo) is None

        assert "hunter2" not in caplog.text
        assert "***" in caplog.text


class TestLocalTargetModel:
    """The returned value's own contract."""

    def test_is_gerrit_reflects_the_source(self, tmp_path):
        github = LocalTarget(
            source=ChangeSource.GITHUB, url="u", remote="origin", root=tmp_path
        )
        gerrit = LocalTarget(
            source=ChangeSource.GERRIT, url="", remote="origin", root=tmp_path
        )
        assert github.is_gerrit is False
        assert gerrit.is_gerrit is True
