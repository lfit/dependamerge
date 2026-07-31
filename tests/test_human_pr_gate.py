# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the human-authored source PR gate.

Regression cover for lfreleng-actions/dependamerge#395: --include-human-prs
had no bearing on the source PR check, so pointing dependamerge at a
human-authored PR demanded an --override SHA whether or not the flag was
given, and then exited 0 -- a silent no-op indistinguishable from
"nothing to merge" when scripted.
"""

from unittest.mock import MagicMock

import pytest
import typer

from dependamerge.cli import _generate_override_sha, _validate_automation_author
from dependamerge.error_codes import ExitCode

FIRST_COMMIT_LINE = "Fix: Env-mediate job status in run block"


def _ctx(*, author, include_human_prs=False, override=None):
    """A merge context whose source PR is authored by *author*."""
    source_pr = MagicMock()
    source_pr.author = author
    source_pr.title = "Fix: Env-mediate job status in run block"
    source_pr.body = "body"

    github_client = MagicMock()
    github_client.is_automation_author.side_effect = lambda a: a.endswith("[bot]")
    github_client.get_pull_request_commits.return_value = [FIRST_COMMIT_LINE]

    ctx = MagicMock()
    ctx.github_client = github_client
    ctx.source_pr = source_pr
    ctx.owner = "lfreleng-actions"
    ctx.repo_name = "tailscale-openstack-bastion-action"
    ctx.pr_number = 63
    ctx.include_human_prs = include_human_prs
    ctx.override = override
    return ctx


def _expected_sha(ctx):
    return _generate_override_sha(ctx.source_pr, FIRST_COMMIT_LINE)


class TestAutomationSource:
    """Automation PRs are unaffected by the gate."""

    def test_bot_author_passes_without_flags(self):
        _validate_automation_author(_ctx(author="pre-commit-ci[bot]"))

    def test_bot_author_does_not_consult_commits(self):
        # The gate should short-circuit before fetching commit messages;
        # an extra API call per PR is wasteful at organisation scale.
        ctx = _ctx(author="dependabot[bot]")
        _validate_automation_author(ctx)
        ctx.github_client.get_pull_request_commits.assert_not_called()


class TestHumanSourceWithoutAuthorisation:
    """The defect in #395: neither flag given must fail fast."""

    def test_exits_non_zero(self):
        ctx = _ctx(author="ModeSevenIndustrialSolutions")
        with pytest.raises(SystemExit) as excinfo:
            _validate_automation_author(ctx)
        # Previously this exited 0, so a scripted run could not tell
        # refusal apart from success.
        assert excinfo.value.code == ExitCode.VALIDATION_ERROR
        assert excinfo.value.code != 0

    def test_failure_names_the_flag(self, capsys):
        ctx = _ctx(author="ModeSevenIndustrialSolutions")
        with pytest.raises(SystemExit):
            _validate_automation_author(ctx)
        output = capsys.readouterr().out
        assert "--include-human-prs" in output

    def test_failure_still_offers_the_override_sha(self, capsys):
        ctx = _ctx(author="ModeSevenIndustrialSolutions")
        with pytest.raises(SystemExit):
            _validate_automation_author(ctx)
        assert _expected_sha(ctx) in capsys.readouterr().out


class TestHumanSourceWithIncludeHumanPrs:
    """--include-human-prs alone authorises a human-authored source."""

    def test_proceeds_without_override(self):
        _validate_automation_author(
            _ctx(author="ModeSevenIndustrialSolutions", include_human_prs=True)
        )

    def test_does_not_fetch_commits_when_no_override_to_check(self):
        # Deriving the override SHA costs an API call per pull request.
        # With --include-human-prs and no override there is nothing to
        # check it against, so the call is pure waste at org scale.
        ctx = _ctx(author="ModeSevenIndustrialSolutions", include_human_prs=True)
        _validate_automation_author(ctx)
        ctx.github_client.get_pull_request_commits.assert_not_called()

    def test_still_fetches_commits_when_an_override_needs_checking(self):
        ctx = _ctx(author="ModeSevenIndustrialSolutions", include_human_prs=True)
        ctx.override = _expected_sha(ctx)
        ctx.github_client.get_pull_request_commits.reset_mock()
        _validate_automation_author(ctx)
        ctx.github_client.get_pull_request_commits.assert_called_once()

    def test_says_why_it_proceeded(self, capsys):
        ctx = _ctx(author="ModeSevenIndustrialSolutions", include_human_prs=True)
        _validate_automation_author(ctx)
        output = capsys.readouterr().out
        assert "--include-human-prs" in output
        assert "ModeSevenIndustrialSolutions" in output

    def test_no_prompt_is_issued(self, monkeypatch):
        # The gate must never block on input: with --no-confirm the run
        # has to proceed unattended.  Any prompt here would hang CI.
        def _explode(*args, **kwargs):
            raise AssertionError("the author gate must not prompt")

        monkeypatch.setattr(typer, "prompt", _explode)
        monkeypatch.setattr(typer, "confirm", _explode)
        _validate_automation_author(
            _ctx(author="ModeSevenIndustrialSolutions", include_human_prs=True)
        )

    def test_invalid_override_still_rejected(self):
        # --include-human-prs authorises the class of PR; it does not
        # excuse a SHA that points at a different PR entirely.
        ctx = _ctx(
            author="ModeSevenIndustrialSolutions",
            include_human_prs=True,
            override="0000000000000000",
        )
        with pytest.raises(SystemExit) as excinfo:
            _validate_automation_author(ctx)
        assert excinfo.value.code == ExitCode.VALIDATION_ERROR

    def test_matching_override_is_accepted(self):
        ctx = _ctx(author="ModeSevenIndustrialSolutions", include_human_prs=True)
        ctx.override = _expected_sha(ctx)
        _validate_automation_author(ctx)


class TestHumanSourceWithOverrideOnly:
    """--override keeps working on its own, for existing invocations."""

    def test_matching_override_proceeds(self):
        ctx = _ctx(author="ModeSevenIndustrialSolutions")
        ctx.override = _expected_sha(ctx)
        _validate_automation_author(ctx)

    def test_matching_override_points_at_the_flag(self, capsys):
        ctx = _ctx(author="ModeSevenIndustrialSolutions")
        ctx.override = _expected_sha(ctx)
        _validate_automation_author(ctx)
        assert "--include-human-prs" in capsys.readouterr().out

    def test_mismatched_override_exits_validation_error(self):
        ctx = _ctx(author="ModeSevenIndustrialSolutions", override="deadbeefdeadbeef")
        with pytest.raises(SystemExit) as excinfo:
            _validate_automation_author(ctx)
        assert excinfo.value.code == ExitCode.VALIDATION_ERROR
