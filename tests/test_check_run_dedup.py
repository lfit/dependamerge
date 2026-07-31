# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for check-run deduplication.

The motivating defect: GitHub can attach two runs of the same check to a
single commit when a duplicate workflow event causes ``concurrency`` to
cancel a superseded run.  Reading every run as authoritative made a
healthy commit look broken, which blocked automerge until someone forced
a rebase -- retrying never helped, because the cancelled run stays
attached to that SHA permanently.
"""

from collections.abc import Mapping
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from dependamerge.check_runs import (
    failing_check_names,
    latest_check_run_per_name,
)
from dependamerge.github_async import GitHubAsync
from dependamerge.github_service import GitHubService


def _rest_router(*, check_runs):
    """Route ``analyze_block_reason``'s GETs with an approved PR.

    Everything except the check runs is benign, so any block reason the
    method returns comes from the check-run handling under test.
    """

    async def _get(url: str):
        if url.endswith("/reviews"):
            return [{"state": "APPROVED", "user": {}}]
        if url.endswith("/comments"):
            return []
        if "/check-runs" in url:
            return {"check_runs": check_runs}
        if url.endswith("/status"):
            return {"statuses": []}
        if url == "/repos/owner/repo":
            return {"default_branch": "main"}
        if url.endswith("/pulls/123"):
            return {"base": {"ref": "main"}}
        return {}

    return _get


def _run(name, conclusion, completed_at=None, status="completed"):
    run = {"name": name, "conclusion": conclusion, "status": status}
    if completed_at is not None:
        run["completed_at"] = completed_at
    return run


class TestSupersededRuns:
    """A cancelled run superseded by a successful one must not block."""

    def test_cancelled_then_success_is_not_failing(self):
        # Observed on lfreleng-actions/harden-runner-block-action#36:
        # both runs on head 06e8015, one second apart.
        runs = [
            _run("Audit changes", "cancelled", "2026-07-30T14:43:37Z"),
            _run("Audit changes", "success", "2026-07-30T14:44:05Z"),
        ]
        assert failing_check_names(runs) == []

    def test_order_in_payload_does_not_matter(self):
        runs = [
            _run("Audit changes", "success", "2026-07-30T14:44:05Z"),
            _run("Audit changes", "cancelled", "2026-07-30T14:43:37Z"),
        ]
        assert failing_check_names(runs) == []

    def test_success_wins_when_timestamps_absent(self):
        # The GraphQL rollup historically exposed no timestamps; the
        # tie-break must still resolve the supersede race.
        runs = [
            _run("Zizmor Scan", "cancelled"),
            _run("Zizmor Scan", "success"),
        ]
        assert failing_check_names(runs) == []

    def test_newest_run_wins_even_when_it_failed(self):
        # A re-run that fails must still block: newest wins, not
        # "any success anywhere".
        runs = [
            _run("Python Tests", "success", "2026-07-30T10:00:00Z"),
            _run("Python Tests", "failure", "2026-07-30T11:00:00Z"),
        ]
        assert failing_check_names(runs) == ["Python Tests"]

    def test_timestamped_run_preferred_over_untimestamped(self):
        runs = [
            _run("Build", "cancelled"),
            _run("Build", "success", "2026-07-30T11:00:00Z"),
        ]
        assert failing_check_names(runs) == []

    def test_success_wins_when_only_the_failure_is_timestamped(self):
        # Partial timestamps leave the order genuinely unknown.  An
        # earlier revision preferred whichever run carried a timestamp,
        # which let a superseded cancel beat the success that replaced
        # it and reintroduced the phantom failure.
        runs = [
            _run("Build", "success"),
            _run("Build", "cancelled", "2026-07-30T14:43:37Z"),
        ]
        assert failing_check_names(runs) == []

    def test_success_wins_when_only_the_success_is_timestamped(self):
        runs = [
            _run("Build", "cancelled"),
            _run("Build", "success", "2026-07-30T14:44:05Z"),
        ]
        assert failing_check_names(runs) == []

    def test_identical_timestamps_resolve_to_success(self):
        # Duplicate runs readily complete within the same second.
        runs = [
            _run("Build", "cancelled", "2026-07-30T14:43:37Z"),
            _run("Build", "success", "2026-07-30T14:43:37Z"),
        ]
        assert failing_check_names(runs) == []


class TestGenuineFailures:
    """Deduplication must not mask real problems."""

    def test_single_failure_still_reported(self):
        assert failing_check_names([_run("Lint", "failure")]) == ["Lint"]

    def test_lone_cancelled_run_still_blocks(self):
        # Nothing superseded it, so it is the latest word on that check.
        assert failing_check_names([_run("Lint", "cancelled")]) == ["Lint"]

    def test_timed_out_still_blocks(self):
        assert failing_check_names([_run("Slow", "timed_out")]) == ["Slow"]

    def test_names_are_independent(self):
        runs = [
            _run("A", "cancelled", "2026-07-30T10:00:00Z"),
            _run("A", "success", "2026-07-30T10:00:05Z"),
            _run("B", "failure", "2026-07-30T10:00:00Z"),
        ]
        assert failing_check_names(runs) == ["B"]

    def test_each_name_reported_at_most_once(self):
        runs = [
            _run("Flaky", "failure", "2026-07-30T10:00:00Z"),
            _run("Flaky", "failure", "2026-07-30T10:00:05Z"),
        ]
        assert failing_check_names(runs) == ["Flaky"]

    def test_reporting_order_follows_first_appearance(self):
        runs = [
            _run("Second", "failure"),
            _run("First", "failure"),
        ]
        assert failing_check_names(runs) == ["Second", "First"]


class TestMalformedInput:
    """Hostile or incomplete payloads must not raise."""

    def test_unnamed_runs_ignored(self):
        assert (
            failing_check_names([_run("", "failure"), {"conclusion": "failure"}]) == []
        )

    def test_non_mapping_entries_ignored(self):
        # API payloads are untrusted; the guard is deliberate, so the
        # cast documents that this input is hostile by design.
        hostile = cast(
            "list[Mapping[str, Any]]", [None, "nonsense", _run("A", "failure")]
        )
        assert failing_check_names(hostile) == ["A"]

    def test_unparseable_timestamp_does_not_raise(self):
        runs = [
            _run("A", "cancelled", "not-a-date"),
            _run("A", "success", "also-not-a-date"),
        ]
        # Both timestamps unusable, so the success tie-break applies.
        assert failing_check_names(runs) == []

    def test_naive_and_aware_timestamps_do_not_raise(self):
        # Comparing an offset-naive datetime with an offset-aware one
        # raises TypeError in Python.  A payload that omits the offset
        # must not crash a run; GitHub reports UTC, so naive values are
        # read as UTC.
        runs = [
            _run("A", "cancelled", "2026-07-30T14:43:37"),
            _run("A", "success", "2026-07-30T14:44:05Z"),
        ]
        assert failing_check_names(runs) == []

    def test_naive_timestamps_still_order_correctly(self):
        runs = [
            _run("A", "success", "2026-07-30T10:00:00"),
            _run("A", "failure", "2026-07-30T11:00:00Z"),
        ]
        assert failing_check_names(runs) == ["A"]

    def test_missing_conclusion_treated_as_not_failing(self):
        # An in-progress run has no conclusion yet; pending is handled
        # elsewhere and must not be reported as a failure here.
        assert failing_check_names([_run("A", None, status="in_progress")]) == []

    def test_latest_helper_returns_one_entry_per_name(self):
        runs = [
            _run("A", "cancelled", "2026-07-30T10:00:00Z"),
            _run("A", "success", "2026-07-30T10:00:05Z"),
        ]
        latest = latest_check_run_per_name(runs)
        assert set(latest) == {"A"}
        assert latest["A"]["conclusion"] == "success"


class TestGraphQLRollupExtraction:
    """The GraphQL path must apply the same rule."""

    @staticmethod
    def _pr(context_nodes):
        return {
            "commits": {
                "nodes": [
                    {
                        "commit": {
                            "statusCheckRollup": {"contexts": {"nodes": context_nodes}}
                        }
                    }
                ]
            }
        }

    def test_superseded_check_run_not_reported(self):
        pr = self._pr(
            [
                {
                    "__typename": "CheckRun",
                    "name": "Audit Workflows",
                    "conclusion": "CANCELLED",
                    "completedAt": "2026-07-30T14:43:37Z",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Audit Workflows",
                    "conclusion": "SUCCESS",
                    "completedAt": "2026-07-30T14:44:04Z",
                },
            ]
        )
        assert GitHubService._extract_failing_checks(pr) == []

    def test_status_contexts_still_reported(self):
        # Commit statuses are already latest-per-context, so they bypass
        # deduplication and must survive it unchanged.
        pr = self._pr(
            [
                {
                    "__typename": "StatusContext",
                    "context": "pre-commit.ci - pr",
                    "state": "FAILURE",
                }
            ]
        )
        assert GitHubService._extract_failing_checks(pr) == ["pre-commit.ci - pr"]

    def test_mixed_superseded_run_and_failing_status(self):
        pr = self._pr(
            [
                {
                    "__typename": "CheckRun",
                    "name": "Audit changes",
                    "conclusion": "CANCELLED",
                    "completedAt": "2026-07-30T14:43:37Z",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Audit changes",
                    "conclusion": "SUCCESS",
                    "completedAt": "2026-07-30T14:44:05Z",
                },
                {
                    "__typename": "StatusContext",
                    "context": "pre-commit.ci - pr",
                    "state": "FAILURE",
                },
            ]
        )
        assert GitHubService._extract_failing_checks(pr) == ["pre-commit.ci - pr"]

    def test_no_commits_returns_empty(self):
        assert GitHubService._extract_failing_checks({}) == []


class TestRestCheckRunPath:
    """The REST path is what produced the reports seen in the field."""

    @pytest.mark.asyncio
    async def test_superseded_cancelled_run_does_not_block(self) -> None:
        # Reproduces lfreleng-actions/nexus-staging-action#52, reported
        # as "Required workflows failed - Package Hardening Audit" while
        # a successful run of that very check sat on the same commit.
        async with GitHubAsync(token="t") as api:
            api.get = AsyncMock(  # type: ignore[method-assign]
                side_effect=_rest_router(
                    check_runs=[
                        {
                            "name": "Audit package manager hardening",
                            "status": "completed",
                            "conclusion": "cancelled",
                            "completed_at": "2026-07-30T13:38:44Z",
                        },
                        {
                            "name": "Audit package manager hardening",
                            "status": "completed",
                            "conclusion": "success",
                            "completed_at": "2026-07-30T13:39:03Z",
                        },
                    ]
                )
            )
            api.get_required_status_checks = AsyncMock(return_value=[])  # type: ignore[method-assign]

            result = await api.analyze_block_reason("owner", "repo", 123, "abc123")

        assert result is None or "Audit package manager hardening" not in result

    @pytest.mark.asyncio
    async def test_unsuperseded_failure_still_blocks(self) -> None:
        async with GitHubAsync(token="t") as api:
            api.get = AsyncMock(  # type: ignore[method-assign]
                side_effect=_rest_router(
                    check_runs=[
                        {
                            "name": "Audit Workflows",
                            "status": "completed",
                            "conclusion": "failure",
                            "completed_at": "2026-07-30T13:39:03Z",
                        },
                    ]
                )
            )
            api.get_required_status_checks = AsyncMock(return_value=[])  # type: ignore[method-assign]

            result = await api.analyze_block_reason("owner", "repo", 123, "abc123")

        assert result is not None and "Audit Workflows" in result

    @pytest.mark.asyncio
    async def test_unnamed_failing_run_is_not_surfaced(self) -> None:
        # An unnamed run cannot be matched against a required-check
        # rule.  Reporting it produced output naming a check called
        # "unknown", which tells the operator nothing actionable.
        async with GitHubAsync(token="t") as api:
            api.get = AsyncMock(  # type: ignore[method-assign]
                side_effect=_rest_router(
                    check_runs=[
                        {"status": "completed", "conclusion": "failure"},
                        {"name": None, "status": "completed", "conclusion": "failure"},
                    ]
                )
            )
            api.get_required_status_checks = AsyncMock(return_value=[])  # type: ignore[method-assign]

            result = await api.analyze_block_reason("owner", "repo", 123, "abc123")

        assert result is None or "unknown" not in result
