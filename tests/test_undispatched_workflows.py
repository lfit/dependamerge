# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for bounding waits on required workflows that never started.

A ruleset can require a workflow that GitHub queues but never
dispatches --- typically one hosted in the org's ``.github`` repository.
The PR then reports "Required workflows … are not satisfied" forever.

Two costs follow, both observed in the 503-PR run analysed in
``docs/BULK_RUN_PERFORMANCE_AUDIT.md``:

- the wait loop spends its entire ``merge_timeout`` learning nothing;
- because the striped scheduler serialises a repository's PRs, siblings
  repeat the discovery one after another. ``workflows-template#29``
  burned 300 s and failed, then ``#30`` began its own fresh 300 s wait
  for the identical reason.

These tests pin both the detection and the propagation, and --- as
importantly --- that an *ambiguous* answer still waits.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dependamerge.models import PullRequestInfo
from dependamerge.rule_violations import (
    is_rule_violation,
    required_status_check_names,
    required_workflow_names,
    violation_verb,
)
from tests.conftest import make_merge_manager

REPO = "lfreleng-actions/workflows-template"
OWNER, NAME = REPO.split("/")

VIOLATION = (
    "Failed to merge PR #29 in lfreleng-actions/workflows-template. "
    "GitHub: Repository rule violations found "
    "Required workflows 'AI Slop Scan 🧹, Zizmor Scan 🌈' are not satisfied"
)


def _pr(number: int = 29) -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
        title="CI(actions): bump something",
        body=None,
        author="dependabot[bot]",
        head_sha="f00dcafe" * 5,
        base_branch="main",
        head_branch="dependabot/x",
        state="open",
        mergeable=True,
        mergeable_state="blocked",
        behind_by=None,
        files_changed=[],
        repository_full_name=REPO,
        html_url=f"https://github.com/{REPO}/pull/{number}",
        reviews=[],
        review_comments=[],
    )


# --------------------------------------------------------------------------
# Parsing the violation string
# --------------------------------------------------------------------------


class TestRuleViolationParsing:
    def test_extracts_workflow_names(self) -> None:
        assert required_workflow_names(VIOLATION) == [
            "AI Slop Scan 🧹",
            "Zizmor Scan 🌈",
        ]

    def test_collapses_repeated_names(self) -> None:
        reason = "Required workflows 'A, B, A' are not satisfied"
        assert required_workflow_names(reason) == ["A", "B"]

    def test_extracts_status_check_names(self) -> None:
        reason = (
            'Repository rule violations found Required status check "DCO" is failing.'
        )
        assert required_status_check_names(reason) == ["DCO"]

    def test_distinguishes_failed_from_unsatisfied(self) -> None:
        assert violation_verb("Required workflows 'A' are not satisfied") == (
            "not satisfied"
        )
        assert violation_verb("Required workflows 'A' failed") == "failed"

    def test_recognises_a_rule_violation(self) -> None:
        assert is_rule_violation(VIOLATION)
        assert not is_rule_violation("Merge already in progress")

    @pytest.mark.parametrize("reason", ["", "something else entirely"])
    def test_returns_nothing_for_unrelated_text(self, reason: str) -> None:
        assert required_workflow_names(reason) == []
        assert required_status_check_names(reason) == []


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


class TestWorkflowsNeverDispatched:
    @pytest.mark.asyncio
    async def test_missing_workflow_is_detected(self) -> None:
        mgr, client = make_merge_manager()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )

        missing = await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION)

        assert missing == ["AI Slop Scan 🧹", "Zizmor Scan 🌈"]

    @pytest.mark.asyncio
    async def test_dispatched_workflow_is_not_reported(self) -> None:
        """If it started, waiting is worthwhile."""
        mgr, client = make_merge_manager()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"AI Slop Scan 🧹", "Zizmor Scan 🌈", "DCO"}
        )

        assert (
            await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION) == []
        )

    @pytest.mark.asyncio
    async def test_no_runs_at_all_is_treated_as_unknown(self) -> None:
        """An empty answer may mean the lookup failed; on doubt, wait."""
        mgr, client = make_merge_manager()
        client.get_workflow_run_names_for_sha = AsyncMock(return_value=set())

        assert (
            await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION) == []
        )

    @pytest.mark.asyncio
    async def test_lookup_failure_falls_back_to_waiting(self) -> None:
        mgr, client = make_merge_manager()
        client.get_workflow_run_names_for_sha = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        assert (
            await mgr._workflows_never_dispatched(_pr(), OWNER, NAME, VIOLATION) == []
        )

    @pytest.mark.asyncio
    async def test_unparseable_error_is_ignored(self) -> None:
        mgr, client = make_merge_manager()
        client.get_workflow_run_names_for_sha = AsyncMock()

        assert (
            await mgr._workflows_never_dispatched(
                _pr(), OWNER, NAME, "Merge already in progress"
            )
            == []
        )
        client.get_workflow_run_names_for_sha.assert_not_called()


# --------------------------------------------------------------------------
# Propagation to sibling PRs
# --------------------------------------------------------------------------


class TestSiblingPropagation:
    @pytest.mark.asyncio
    async def test_sibling_skips_the_lookup_entirely(self) -> None:
        """The second PR must not repeat the first PR's discovery."""
        mgr, client = make_merge_manager()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )

        first = await mgr._workflows_never_dispatched(_pr(29), OWNER, NAME, VIOLATION)
        assert first == ["AI Slop Scan 🧹", "Zizmor Scan 🌈"]
        assert client.get_workflow_run_names_for_sha.await_count == 1

        second = await mgr._workflows_never_dispatched(_pr(30), OWNER, NAME, VIOLATION)

        assert second == ["AI Slop Scan 🧹", "Zizmor Scan 🌈"]
        # No second lookup: the repository's finding was reused.
        assert client.get_workflow_run_names_for_sha.await_count == 1

    @pytest.mark.asyncio
    async def test_a_different_workflow_is_still_checked(self) -> None:
        """Reuse must not swallow a name nobody has looked up yet."""
        mgr, client = make_merge_manager()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "Zizmor Scan 🌈"}
        )

        await mgr._workflows_never_dispatched(_pr(29), OWNER, NAME, VIOLATION)
        assert client.get_workflow_run_names_for_sha.await_count == 1

        other = "Required workflows 'Semantic Pull Request 🛠️' are not satisfied"
        await mgr._workflows_never_dispatched(_pr(30), OWNER, NAME, other)

        assert client.get_workflow_run_names_for_sha.await_count == 2

    @pytest.mark.asyncio
    async def test_findings_do_not_leak_between_repositories(self) -> None:
        mgr, client = make_merge_manager()
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )

        await mgr._workflows_never_dispatched(_pr(29), OWNER, NAME, VIOLATION)
        other = _pr(1)
        other.repository_full_name = "lfreleng-actions/other-repo"

        await mgr._workflows_never_dispatched(other, OWNER, "other-repo", VIOLATION)

        assert client.get_workflow_run_names_for_sha.await_count == 2


# --------------------------------------------------------------------------
# The wait itself
# --------------------------------------------------------------------------


class TestWaitIsBounded:
    @pytest.mark.asyncio
    async def test_wait_returns_immediately_when_nothing_was_dispatched(self) -> None:
        mgr, client = make_merge_manager()
        pr = _pr()
        mgr._last_merge_exception[f"{OWNER}/{NAME}#29"] = Exception(VIOLATION)
        client.get_workflow_run_names_for_sha = AsyncMock(
            return_value={"DCO", "CodeQL"}
        )
        # Would be consulted only if the method proceeded to wait.
        mgr._wait_for_auto_merge = AsyncMock(  # type: ignore[method-assign]
            return_value=(False, False)
        )

        merged = await mgr._wait_for_required_workflows_and_retry(pr, OWNER, NAME)

        assert merged is False
        mgr._wait_for_auto_merge.assert_not_awaited()
