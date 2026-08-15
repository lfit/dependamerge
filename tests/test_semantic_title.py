# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for Dependabot title / commit-subject alignment.

See ``semantic_title`` for the problem. These tests weight heavily
towards the *guards*: the remedy rewrites somebody's pull request title,
so the cases that must NOT trigger matter more than the one that must.

The corpus figures quoted here come from 112 real single-commit
Dependabot mismatches sampled across ``lfreleng-actions``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from dependamerge.merge_manager import MergeStatus  # noqa: F401  (import sanity)
from dependamerge.models import PullRequestInfo
from dependamerge.semantic_title import (
    describe_title_change,
    is_semantic_check_name,
    single_commit_subject,
    version_fragment_removed,
)
from tests.conftest import make_merge_manager

REPO = "lfreleng-actions/example-action"
OWNER, NAME = REPO.split("/")

# Real pairs observed in the 503-PR run.
MID_STRING = (
    "Chore: Bump cryptography from 49.0.0 to 50.0.0 in the uv group across 1 directory",
    "Chore: Bump cryptography in the uv group across 1 directory",
)
MID_STRING_PATH = (
    "CI(deps): Bump github-security-report from 0.8.0 to 0.10.0 in /.github/runtime-pin",
    "CI(deps): Bump github-security-report in /.github/runtime-pin",
)
SUFFIX = (
    "CI(actions): Bump lfit/releng-reusable-workflows/.github/workflows/"
    "reuse-openssf-scorecard.yaml from 0.9.1 to 0.10.1",
    "CI(actions): Bump lfit/releng-reusable-workflows/.github/workflows/"
    "reuse-openssf-scorecard.yaml",
)
# Version drift, not truncation: the title was updated to a newer release
# while the commit kept the old one.  The check is right to fail this.
DRIFT = (
    "Chore: Bump dependamerge from 0.9.2 to 0.10.0",
    "Chore: Bump dependamerge from 0.9.2 to 0.9.3",
)


def _pr(title: str = MID_STRING[0], author: str = "dependabot[bot]") -> PullRequestInfo:
    return PullRequestInfo(
        number=283,
        title=title,
        body="bump",
        author=author,
        head_sha="c0ffee11" * 5,
        base_branch="main",
        head_branch="dependabot/x",
        state="open",
        mergeable=True,
        mergeable_state="blocked",
        behind_by=None,
        files_changed=[],
        repository_full_name=REPO,
        html_url=f"https://github.com/{REPO}/pull/283",
        reviews=[],
        review_comments=[],
    )


def _commit(subject: str, parents: int = 1) -> dict[str, Any]:
    return {
        "parents": [{"sha": "x"}] * parents,
        "commit": {"message": f"{subject}\n\nbody text"},
    }


def _run(
    name: str, conclusion: str, at: str = "2026-08-13T14:00:00Z"
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "completed_at": at,
    }


# --------------------------------------------------------------------------
# Check-name recognition
# --------------------------------------------------------------------------


class TestSemanticCheckName:
    @pytest.mark.parametrize(
        "name",
        [
            "Semantic Pull Request",
            "Semantic Pull Request 🛠️",
            "Semantic Pull Request / Semantic Pull Request",
            "semantic pull request",
        ],
    )
    def test_recognised(self, name: str) -> None:
        assert is_semantic_check_name(name)

    @pytest.mark.parametrize(
        "name",
        ["DCO", "Zizmor Scan 🌈", "AI Slop Scan 🧹", "pre-commit.ci - pr", ""],
    )
    def test_not_recognised(self, name: str) -> None:
        assert not is_semantic_check_name(name)


# --------------------------------------------------------------------------
# Commit subject extraction
# --------------------------------------------------------------------------


class TestSingleCommitSubject:
    def test_single_commit(self) -> None:
        assert single_commit_subject([_commit("Chore: Bump x")]) == "Chore: Bump x"

    def test_subject_is_first_line_only(self) -> None:
        assert single_commit_subject([_commit("Chore: Bump x")]) == "Chore: Bump x"

    def test_two_commits_is_none(self) -> None:
        """A title cannot be said to match "the" commit when there are two."""
        assert single_commit_subject([_commit("a"), _commit("b")]) is None

    def test_merge_commits_ignored(self) -> None:
        commits = [_commit("Merge branch", parents=2), _commit("Chore: Bump x")]
        assert single_commit_subject(commits) == "Chore: Bump x"

    def test_no_commits_is_none(self) -> None:
        assert single_commit_subject([]) is None

    def test_malformed_entries_ignored(self) -> None:
        # Deliberately ill-typed input: the helper parses API payloads and
        # must tolerate a shape GitHub never promised.
        malformed: list[Any] = ["nonsense", {"commit": None}]
        assert single_commit_subject(malformed) is None


# --------------------------------------------------------------------------
# The elision rule
# --------------------------------------------------------------------------


class TestVersionFragmentRemoved:
    def test_mid_string_elision(self) -> None:
        assert version_fragment_removed(*MID_STRING) is not None

    def test_mid_string_elision_with_path(self) -> None:
        assert version_fragment_removed(*MID_STRING_PATH) is not None

    def test_suffix_truncation(self) -> None:
        """Already handled upstream, but must still be recognised here."""
        assert version_fragment_removed(*SUFFIX) is not None

    def test_version_drift_rejected(self) -> None:
        """The case the semantic check exists to catch: do not paper over it."""
        assert version_fragment_removed(*DRIFT) is None

    @pytest.mark.parametrize(
        ("title", "subject"),
        [
            ("Feat: Add support for X", "Chore: Add support for X"),
            (
                "Chore: Bump requests from 1.0 to 2.0",
                "Chore: Bump urllib3 from 1.0 to 2.0",
            ),
            ("Chore: Bump foo from 1 to 2 in /a", "Chore: Bump foo in /b"),
            ("Chore: Bump foo", "Chore: Bump foo from 1 to 2"),
            ("", ""),
            ("Chore: Bump foo", "Chore: Bump foo"),
            # Cut begins inside a token: removing the span would splice
            # "xabc" onto what followed rather than excise a whole
            # fragment.  Both ends must sit on whitespace.
            ("Chore: Bump xabcfrom 1 to 2 y", "Chore: Bump xabc y"),
            ("Chore: Bump x from 1 to 2abc", "Chore: Bump xabc"),
        ],
    )
    def test_rejected(self, title: str, subject: str) -> None:
        assert version_fragment_removed(title, subject) is None

    def test_accepts_a_span_abutting_the_title_end(self) -> None:
        """Nothing follows the cut, so there is nothing to splice against."""
        assert (
            version_fragment_removed("Chore: Bump foo from 1 to 2", "Chore: Bump foo")
            is not None
        )

    def test_describes_the_removed_fragment(self) -> None:
        note = describe_title_change(*MID_STRING)
        assert "from 49.0.0 to 50.0.0" in note


# --------------------------------------------------------------------------
# Orchestration: the guards
# --------------------------------------------------------------------------


class TestAlignSemanticTitle:
    @staticmethod
    def _wire(mgr, client, *, runs=None, subject=MID_STRING[1], statuses=None) -> None:
        client.get_check_runs_for_ref = AsyncMock(
            return_value=runs
            if runs is not None
            else [_run("Semantic Pull Request 🛠️", "failure"), _run("DCO", "success")]
        )
        client.get_failing_status_contexts = AsyncMock(
            return_value=statuses if statuses is not None else []
        )
        client.get_pull_request_commits = AsyncMock(return_value=[_commit(subject)])
        client.update_pull_request_title = AsyncMock()

    @pytest.mark.asyncio
    async def test_failing_commit_status_blocks_alignment(self) -> None:
        """pre-commit.ci and DCO report as statuses, not check runs.

        Reading only check runs would let the rewrite proceed while a
        required status context was genuinely failing.
        """
        mgr, client = make_merge_manager()
        self._wire(mgr, client, statuses=["pre-commit.ci - pr"])

        assert await mgr._align_semantic_title(_pr()) is False
        client.update_pull_request_title.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_patch_leaves_pr_eligible_for_retry(self) -> None:
        """A transient write failure is not an alignment.

        Recording the attempt regardless would strand the PR for the rest
        of the run over one failed request.
        """
        mgr, client = make_merge_manager()
        pr = _pr()
        self._wire(mgr, client)
        client.update_pull_request_title = AsyncMock(side_effect=RuntimeError("boom"))

        assert await mgr._align_semantic_title(pr) is False
        assert f"{REPO}#283" not in mgr._semantic_title_aligned

        # A later attempt in the same run succeeds.
        client.update_pull_request_title = AsyncMock()
        assert await mgr._align_semantic_title(pr) is True

    @pytest.mark.asyncio
    async def test_aligns_and_reports_success(self) -> None:
        mgr, client = make_merge_manager()
        pr = _pr()
        self._wire(mgr, client)

        assert await mgr._align_semantic_title(pr) is True
        client.update_pull_request_title.assert_awaited_once_with(
            OWNER, NAME, 283, MID_STRING[1]
        )
        assert pr.title == MID_STRING[1]

    @pytest.mark.asyncio
    async def test_disabled_by_flag(self) -> None:
        mgr, client = make_merge_manager(fix_semantic_title=False)
        self._wire(mgr, client)

        assert await mgr._align_semantic_title(_pr()) is False
        client.update_pull_request_title.assert_not_called()

    @pytest.mark.asyncio
    async def test_preview_mode_never_writes(self) -> None:
        mgr, client = make_merge_manager(preview_mode=True)
        self._wire(mgr, client)

        assert await mgr._align_semantic_title(_pr()) is False
        client.update_pull_request_title.assert_not_called()

    @pytest.mark.asyncio
    async def test_human_authored_pr_untouched(self) -> None:
        """Rewriting a person's title is intrusive; bots only."""
        mgr, client = make_merge_manager()
        self._wire(mgr, client)

        assert await mgr._align_semantic_title(_pr(author="a-human")) is False
        client.update_pull_request_title.assert_not_called()

    @pytest.mark.asyncio
    async def test_other_failing_check_blocks_alignment(self) -> None:
        """Never mask a real failure by fixing the title."""
        mgr, client = make_merge_manager()
        self._wire(
            mgr,
            client,
            runs=[
                _run("Semantic Pull Request 🛠️", "failure"),
                _run("Python Tests", "failure"),
            ],
        )

        assert await mgr._align_semantic_title(_pr()) is False
        client.update_pull_request_title.assert_not_called()

    @pytest.mark.asyncio
    async def test_nothing_failing_is_a_no_op(self) -> None:
        mgr, client = make_merge_manager()
        self._wire(mgr, client, runs=[_run("DCO", "success")])

        assert await mgr._align_semantic_title(_pr()) is False
        client.update_pull_request_title.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancelled_duplicate_does_not_trigger_alignment(self) -> None:
        """A superseded cancelled run is not a failure.

        ``check_runs`` resolves duplicates by latest, so a cancelled run
        sitting beside a later success must not look like a semantic
        failure and provoke a title rewrite.
        """
        mgr, client = make_merge_manager()
        self._wire(
            mgr,
            client,
            runs=[
                _run("Semantic Pull Request 🛠️", "cancelled", "2026-08-13T14:04:35Z"),
                _run("Semantic Pull Request 🛠️", "success", "2026-08-13T14:04:58Z"),
            ],
        )

        assert await mgr._align_semantic_title(_pr()) is False
        client.update_pull_request_title.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_commits_blocks_alignment(self) -> None:
        mgr, client = make_merge_manager()
        self._wire(mgr, client)
        client.get_pull_request_commits = AsyncMock(
            return_value=[_commit("a"), _commit("b")]
        )

        assert await mgr._align_semantic_title(_pr()) is False
        client.update_pull_request_title.assert_not_called()

    @pytest.mark.asyncio
    async def test_version_drift_blocks_alignment(self) -> None:
        """Rewriting here would hide genuine drift the check caught."""
        mgr, client = make_merge_manager()
        self._wire(mgr, client, subject=DRIFT[1])

        assert await mgr._align_semantic_title(_pr(title=DRIFT[0])) is False
        client.update_pull_request_title.assert_not_called()

    @pytest.mark.asyncio
    async def test_title_already_matching_is_a_no_op(self) -> None:
        mgr, client = make_merge_manager()
        self._wire(mgr, client, subject=MID_STRING[0])

        assert await mgr._align_semantic_title(_pr()) is False
        client.update_pull_request_title.assert_not_called()

    @pytest.mark.asyncio
    async def test_attempted_at_most_once_per_pr(self) -> None:
        """A check that keeps failing must not drive a rewrite loop."""
        mgr, client = make_merge_manager()
        pr = _pr()
        self._wire(mgr, client)

        assert await mgr._align_semantic_title(pr) is True
        # Reset the title as though the check failed again anyway.
        pr.title = MID_STRING[0]
        assert await mgr._align_semantic_title(pr) is False
        assert client.update_pull_request_title.await_count == 1

    @pytest.mark.asyncio
    async def test_api_failure_is_reported_not_raised(self) -> None:
        mgr, client = make_merge_manager()
        self._wire(mgr, client)
        client.update_pull_request_title = AsyncMock(side_effect=RuntimeError("nope"))

        assert await mgr._align_semantic_title(_pr()) is False

    @pytest.mark.asyncio
    async def test_check_run_read_failure_is_survivable(self) -> None:
        mgr, client = make_merge_manager()
        self._wire(mgr, client)
        client.get_check_runs_for_ref = AsyncMock(side_effect=RuntimeError("boom"))

        assert await mgr._align_semantic_title(_pr()) is False
        client.update_pull_request_title.assert_not_called()
