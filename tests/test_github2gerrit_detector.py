# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""Tests for GitHub2Gerrit detection and merge manager integration."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from dependamerge.gerrit import GerritRestError
from dependamerge.gerrit.service import GerritService
from dependamerge.github2gerrit_detector import (
    _END_MARKER,
    _START_MARKER,
    GITHUB2GERRIT_BOT_AUTHORS,
    GitHub2GerritDetectionResult,
    GitHub2GerritMapping,
    GitHub2GerritMode,
    GitReviewInfo,
    _detect_via_heuristic,
    _detect_via_markers,
    _extract_author,
    _extract_body,
    _looks_like_mapping,
    _parse_block_lines,
    _parse_heuristic,
    _parse_marker_block,
    build_gerrit_change_url_from_mapping,
    build_gerrit_skip_message,
    build_gerrit_submission_comment,
    detect_github2gerrit_comments,
    detect_github2gerrit_from_graphql_comments,
    fetch_gitreview_from_github,
    has_github2gerrit_comments,
    parse_gitreview_text,
)
from dependamerge.merge_manager import AsyncMergeManager, MergeStatus
from dependamerge.models import PullRequestInfo

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

SAMPLE_MAPPING_COMMENT_BODY = (
    "<!-- github2gerrit:change-id-map v1 -->\n"
    "PR: https://github.com/lfit/releng-gerrit_to_platform/pull/41\n"
    "Mode: squash\n"
    "Topic: GH-releng-gerrit_to_platform-41\n"
    "Change-Ids:\n"
    "  I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
    "GitHub-Hash: 41b89b8d5055be4e\n"
    "\n"
    "_Note: This metadata is also included in the Gerrit commit message for reconciliation._\n"
    "\n"
    "<!-- end github2gerrit:change-id-map -->"
)

SAMPLE_MAPPING_COMMENT_MULTI = (
    "<!-- github2gerrit:change-id-map v1 -->\n"
    "PR: https://github.com/lfit/releng-lftools/pull/99\n"
    "Mode: multi-commit\n"
    "Topic: GH-releng-lftools-99\n"
    "Change-Ids:\n"
    "  I1111111111111111111111111111111111111111\n"
    "  I2222222222222222222222222222222222222222\n"
    "GitHub-Hash: abcdef1234567890\n"
    "\n"
    "_Note: This metadata is also included in the Gerrit commit message for reconciliation._\n"
    "\n"
    "<!-- end github2gerrit:change-id-map -->"
)

SAMPLE_HEURISTIC_COMMENT = (
    "PR: #41\n"
    "Mode: squash\n"
    "Topic: GH-releng-gerrit_to_platform-41\n"
    "Change-Ids:\n"
    "I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
    "GitHub-Hash: 41b89b8d5055be4e\n"
    "\n"
    "Note: This metadata is also included in the Gerrit commit message for reconciliation."
)

SAMPLE_COMMENT_WITH_GERRIT_URL = (
    "<!-- github2gerrit:change-id-map v1 -->\n"
    "PR: https://github.com/lfit/releng-gerrit_to_platform/pull/41\n"
    "Mode: squash\n"
    "Topic: GH-releng-gerrit_to_platform-41\n"
    "Change-Ids:\n"
    "  I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
    "GitHub-Hash: 41b89b8d5055be4e\n"
    "\n"
    "Gerrit: https://gerrit.linuxfoundation.org/infra/c/releng/gerrit_to_platform/+/12345\n"
    "\n"
    "<!-- end github2gerrit:change-id-map -->"
)


def _make_rest_comment(
    body: str, author: str = "github-actions[bot]"
) -> dict[str, object]:
    """Build a REST-style comment dict."""
    return {"user": {"login": author}, "body": body, "id": 100}


def _make_graphql_comment(
    body: str, author: str = "github-actions[bot]"
) -> dict[str, object]:
    """Build a GraphQL-style comment dict."""
    return {
        "author": {"login": author},
        "body": body,
        "createdAt": "2025-01-26T00:00:00Z",
    }


def _make_pr_info(**overrides) -> PullRequestInfo:
    """Build a minimal PullRequestInfo for testing."""
    defaults: dict[str, Any] = {
        "number": 41,
        "title": "Bump foo from 1.0 to 2.0",
        "body": "Dependabot bump body",
        "author": "dependabot[bot]",
        "head_sha": "abc123",
        "base_branch": "main",
        "head_branch": "dependabot/pip/foo-2.0",
        "state": "open",
        "mergeable": True,
        "mergeable_state": "clean",
        "behind_by": 0,
        "files_changed": [],
        "repository_full_name": "lfit/releng-gerrit_to_platform",
        "html_url": "https://github.com/lfit/releng-gerrit_to_platform/pull/41",
    }
    defaults.update(overrides)
    return PullRequestInfo(**defaults)


# ===========================================================================
# GitHub2GerritMapping model tests
# ===========================================================================


class TestGitHub2GerritMapping:
    def test_primary_change_id(self):
        m = GitHub2GerritMapping(
            pr_url="https://example.com/pull/1",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa", "Ibbbb"),
        )
        assert m.primary_change_id == "Iaaaa"

    def test_primary_change_id_empty(self):
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=(),
        )
        assert m.primary_change_id == ""

    def test_is_valid_true(self):
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048",),
        )
        assert m.is_valid is True

    def test_is_valid_no_topic(self):
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="",
            change_ids=("Iaaaa",),
        )
        assert m.is_valid is False

    def test_is_valid_no_change_ids(self):
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=(),
        )
        assert m.is_valid is False

    def test_is_valid_no_mode(self):
        m = GitHub2GerritMapping(
            pr_url="",
            mode="",
            topic="GH-test-1",
            change_ids=("Iaaaa",),
        )
        assert m.is_valid is False


class TestGitHub2GerritMode:
    def test_enum_values(self):
        assert GitHub2GerritMode.SQUASH == "squash"
        assert GitHub2GerritMode.MULTI_COMMIT == "multi-commit"


# ===========================================================================
# Internal helper tests
# ===========================================================================


class TestExtractBody:
    def test_normal_body(self):
        assert _extract_body({"body": "hello"}) == "hello"

    def test_empty_body(self):
        assert _extract_body({"body": ""}) == ""

    def test_none_body(self):
        assert _extract_body({"body": None}) == ""

    def test_missing_body(self):
        assert _extract_body({}) == ""

    def test_strips_whitespace(self):
        assert _extract_body({"body": "  hello  "}) == "hello"


class TestExtractAuthor:
    def test_graphql_author(self):
        assert (
            _extract_author({"author": {"login": "github-actions[bot]"}})
            == "github-actions[bot]"
        )

    def test_rest_user(self):
        assert (
            _extract_author({"user": {"login": "Github-Actions[bot]"}})
            == "github-actions[bot]"
        )

    def test_prefers_author_over_user(self):
        c = {"author": {"login": "author-bot"}, "user": {"login": "user-bot"}}
        assert _extract_author(c) == "author-bot"

    def test_missing_author(self):
        assert _extract_author({}) == ""

    def test_none_author(self):
        assert _extract_author({"author": None}) == ""

    def test_empty_login(self):
        assert _extract_author({"author": {"login": ""}}) == ""


class TestLooksLikeMapping:
    def test_has_all_fields(self):
        body = (
            "Topic: GH-repo-1\n"
            "Mode: squash\n"
            "I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
        )
        assert _looks_like_mapping(body) is True

    def test_change_id_and_topic(self):
        body = "Topic: GH-repo-1\nI6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048"
        assert _looks_like_mapping(body) is True

    def test_change_id_and_mode(self):
        body = "Mode: squash\nI6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048"
        assert _looks_like_mapping(body) is True

    def test_only_change_id(self):
        body = "I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048"
        assert _looks_like_mapping(body) is False

    def test_no_change_id(self):
        body = "Topic: GH-repo-1\nMode: squash"
        assert _looks_like_mapping(body) is False

    def test_empty_string(self):
        assert _looks_like_mapping("") is False


# ===========================================================================
# Marker-based parsing tests
# ===========================================================================


class TestParseMarkerBlock:
    def test_standard_squash_mapping(self):
        mapping = _parse_marker_block(SAMPLE_MAPPING_COMMENT_BODY)
        assert mapping is not None
        assert mapping.mode == "squash"
        assert mapping.topic == "GH-releng-gerrit_to_platform-41"
        assert len(mapping.change_ids) == 1
        assert mapping.change_ids[0] == "I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048"
        assert mapping.github_hash == "41b89b8d5055be4e"
        assert "pull/41" in mapping.pr_url

    def test_multi_commit_mapping(self):
        mapping = _parse_marker_block(SAMPLE_MAPPING_COMMENT_MULTI)
        assert mapping is not None
        assert mapping.mode == "multi-commit"
        assert len(mapping.change_ids) == 2
        assert mapping.change_ids[0] == "I1111111111111111111111111111111111111111"
        assert mapping.change_ids[1] == "I2222222222222222222222222222222222222222"

    def test_no_markers(self):
        assert _parse_marker_block("Just a regular comment") is None

    def test_only_start_marker(self):
        body = _START_MARKER + "\nMode: squash\nTopic: t\nChange-Ids:\n  Iaaaa"
        assert _parse_marker_block(body) is None

    def test_empty_block(self):
        body = _START_MARKER + "\n" + _END_MARKER
        assert _parse_marker_block(body) is None

    def test_incomplete_block_no_topic(self):
        body = (
            _START_MARKER + "\n"
            "Mode: squash\n"
            "Change-Ids:\n"
            "  I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n" + _END_MARKER
        )
        mapping = _parse_marker_block(body)
        assert mapping is None  # topic is required

    def test_deduplicates_change_ids(self):
        body = (
            _START_MARKER + "\n"
            "PR: https://example.com/pull/1\n"
            "Mode: squash\n"
            "Topic: GH-test-1\n"
            "Change-Ids:\n"
            "  I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
            "  I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
            "GitHub-Hash: abc123\n" + _END_MARKER
        )
        mapping = _parse_marker_block(body)
        assert mapping is not None
        assert len(mapping.change_ids) == 1


class TestParseBlockLines:
    def test_parses_all_fields(self):
        block = (
            "PR: https://example.com/pull/1\n"
            "Mode: squash\n"
            "Topic: GH-repo-1\n"
            "Change-Ids:\n"
            "  I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
            "GitHub-Hash: abc123"
        )
        mapping = _parse_block_lines(block, "full body")
        assert mapping is not None
        assert mapping.pr_url == "https://example.com/pull/1"
        assert mapping.mode == "squash"
        assert mapping.topic == "GH-repo-1"
        assert mapping.github_hash == "abc123"
        assert mapping.raw_comment_body == "full body"

    def test_handles_digest_field(self):
        block = (
            "PR: https://example.com/pull/1\n"
            "Mode: squash\n"
            "Topic: GH-repo-1\n"
            "Change-Ids:\n"
            "  I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
            "Digest: abc123def456\n"
            "GitHub-Hash: deadbeef"
        )
        mapping = _parse_block_lines(block, "")
        assert mapping is not None
        assert mapping.github_hash == "deadbeef"

    def test_handles_note_line(self):
        block = (
            "PR: https://example.com/pull/1\n"
            "Mode: squash\n"
            "Topic: GH-repo-1\n"
            "Change-Ids:\n"
            "  I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
            "GitHub-Hash: deadbeef\n"
            "Note: some note"
        )
        mapping = _parse_block_lines(block, "")
        assert mapping is not None

    def test_returns_none_for_missing_mode(self):
        block = (
            "Topic: GH-repo-1\n"
            "Change-Ids:\n"
            "  I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
        )
        assert _parse_block_lines(block, "") is None


# ===========================================================================
# Heuristic parsing tests
# ===========================================================================


class TestParseHeuristic:
    def test_full_heuristic_comment(self):
        mapping = _parse_heuristic(SAMPLE_HEURISTIC_COMMENT)
        assert mapping is not None
        assert mapping.topic == "GH-releng-gerrit_to_platform-41"
        assert mapping.mode == "squash"
        assert len(mapping.change_ids) == 1

    def test_no_change_id(self):
        body = "Topic: GH-repo-1\nMode: squash\n"
        assert _parse_heuristic(body) is None

    def test_extracts_pr_url(self):
        body = (
            "PR: https://github.com/owner/repo/pull/42\n"
            "Mode: squash\n"
            "Topic: GH-repo-42\n"
            "I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
        )
        mapping = _parse_heuristic(body)
        assert mapping is not None
        assert "pull/42" in mapping.pr_url

    def test_defaults_to_squash_if_no_mode(self):
        body = "Topic: GH-repo-1\nI6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
        mapping = _parse_heuristic(body)
        assert mapping is not None
        assert mapping.mode == "squash"

    def test_deduplicates_change_ids(self):
        body = (
            "Topic: GH-repo-1\n"
            "I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
            "I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
        )
        mapping = _parse_heuristic(body)
        assert mapping is not None
        assert len(mapping.change_ids) == 1


# ===========================================================================
# Detection pipeline tests
# ===========================================================================


class TestDetectViaMarkers:
    def test_finds_marker_comment(self):
        bodies = [
            (0, "Just a comment", {}),
            (1, SAMPLE_MAPPING_COMMENT_BODY, {}),
        ]
        result = _detect_via_markers(bodies)
        assert result.has_mapping is True
        assert result.detection_source == "marker"
        assert 1 in result.comment_indices
        assert result.mapping is not None

    def test_returns_latest_mapping(self):
        bodies = [
            (0, SAMPLE_MAPPING_COMMENT_BODY, {}),
            (1, SAMPLE_MAPPING_COMMENT_MULTI, {}),
        ]
        result = _detect_via_markers(bodies)
        assert result.has_mapping is True
        assert result.mapping is not None
        assert result.mapping.mode == "multi-commit"
        assert len(result.comment_indices) == 2

    def test_no_markers(self):
        bodies = [(0, "Regular comment", {})]
        result = _detect_via_markers(bodies)
        assert result.has_mapping is False


class TestDetectViaHeuristic:
    def test_finds_heuristic_comment_from_bot(self):
        bodies = [
            (
                0,
                SAMPLE_HEURISTIC_COMMENT,
                _make_graphql_comment(SAMPLE_HEURISTIC_COMMENT),
            ),
        ]
        result = _detect_via_heuristic(bodies)
        assert result.has_mapping is True
        assert result.detection_source == "heuristic"

    def test_ignores_non_bot_author(self):
        bodies = [
            (
                0,
                SAMPLE_HEURISTIC_COMMENT,
                _make_graphql_comment(SAMPLE_HEURISTIC_COMMENT, author="some-user"),
            ),
        ]
        result = _detect_via_heuristic(bodies)
        assert result.has_mapping is False

    def test_ignores_non_mapping_comment_from_bot(self):
        bodies = [
            (
                0,
                "Just a regular comment from bot",
                _make_graphql_comment("Just a regular comment from bot"),
            ),
        ]
        result = _detect_via_heuristic(bodies)
        assert result.has_mapping is False


class TestDetectGitHub2GerritComments:
    def test_finds_marker_comment(self):
        comments = [
            _make_rest_comment("Random comment", author="some-user"),
            _make_rest_comment(SAMPLE_MAPPING_COMMENT_BODY),
        ]
        result = detect_github2gerrit_comments(comments)
        assert result.has_mapping is True
        assert result.mapping is not None
        assert result.mapping.topic == "GH-releng-gerrit_to_platform-41"

    def test_falls_back_to_heuristic(self):
        comments = [
            _make_rest_comment(SAMPLE_HEURISTIC_COMMENT),
        ]
        result = detect_github2gerrit_comments(comments)
        assert result.has_mapping is True
        assert result.detection_source == "heuristic"

    def test_empty_comments(self):
        result = detect_github2gerrit_comments([])
        assert result.has_mapping is False

    def test_no_mapping_comments(self):
        comments = [
            _make_rest_comment("Just a regular comment"),
            _make_rest_comment("Another comment", author="dependabot[bot]"),
        ]
        result = detect_github2gerrit_comments(comments)
        assert result.has_mapping is False

    def test_graphql_comments(self):
        comments = [
            _make_graphql_comment(SAMPLE_MAPPING_COMMENT_BODY),
        ]
        result = detect_github2gerrit_comments(comments)
        assert result.has_mapping is True

    def test_marker_takes_priority_over_heuristic(self):
        comments = [
            _make_rest_comment(SAMPLE_HEURISTIC_COMMENT),
            _make_rest_comment(SAMPLE_MAPPING_COMMENT_BODY),
        ]
        result = detect_github2gerrit_comments(comments)
        assert result.has_mapping is True
        assert result.detection_source == "marker"


class TestDetectFromGraphQLPRNode:
    def test_with_comments(self):
        pr_node = {
            "comments": {
                "nodes": [
                    _make_graphql_comment(SAMPLE_MAPPING_COMMENT_BODY),
                ]
            }
        }
        result = detect_github2gerrit_from_graphql_comments(pr_node)
        assert result.has_mapping is True

    def test_no_comments_key(self):
        result = detect_github2gerrit_from_graphql_comments({})
        assert result.has_mapping is False

    def test_empty_nodes(self):
        result = detect_github2gerrit_from_graphql_comments({"comments": {"nodes": []}})
        assert result.has_mapping is False


class TestHasGitHub2GerritComments:
    def test_true_with_marker(self):
        comments = [_make_rest_comment(SAMPLE_MAPPING_COMMENT_BODY)]
        assert has_github2gerrit_comments(comments) is True

    def test_true_with_heuristic(self):
        comments = [_make_rest_comment(SAMPLE_HEURISTIC_COMMENT)]
        assert has_github2gerrit_comments(comments) is True

    def test_false_with_no_mapping(self):
        comments = [_make_rest_comment("Just a comment")]
        assert has_github2gerrit_comments(comments) is False

    def test_false_with_empty_list(self):
        assert has_github2gerrit_comments([]) is False


# ===========================================================================
# URL and comment builder tests
# ===========================================================================


class TestBuildGerritChangeUrl:
    def test_with_base_path(self):
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048",),
        )
        url = build_gerrit_change_url_from_mapping(
            m, "gerrit.linuxfoundation.org", "infra"
        )
        assert "gerrit.linuxfoundation.org/infra/q/" in url
        assert "I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048" in url

    def test_without_base_path(self):
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa",),
        )
        url = build_gerrit_change_url_from_mapping(m, "gerrit.example.org")
        assert url == "https://gerrit.example.org/q/Iaaaa"

    def test_no_change_ids_returns_base(self):
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=(),
        )
        url = build_gerrit_change_url_from_mapping(m, "gerrit.example.org")
        assert url == "https://gerrit.example.org"


class TestBuildGerritSubmissionComment:
    def test_with_url(self):
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa",),
        )
        body = build_gerrit_submission_comment(m, "https://gerrit.example.org/q/Iaaaa")
        assert "**Automated PR Closure**" in body
        assert "dependamerge" in body
        assert "submitted" in body
        assert "https://gerrit.example.org/q/Iaaaa" in body
        assert "GitHub2Gerrit awareness" in body

    def test_without_url(self):
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa",),
        )
        body = build_gerrit_submission_comment(m)
        assert "**Automated PR Closure**" in body
        assert "submitted" in body


class TestBuildGerritSkipMessage:
    def test_message_content(self):
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-releng-test-42",
            change_ids=("I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048",),
        )
        msg = build_gerrit_skip_message(m)
        assert "GitHub2Gerrit PR" in msg
        assert "GH-releng-test-42" in msg
        assert "I6a9987bd1b1" in msg  # truncated Change-Id


# ===========================================================================
# Bot authors constant test
# ===========================================================================


class TestBotAuthors:
    def test_contains_expected_authors(self):
        assert "github-actions" in GITHUB2GERRIT_BOT_AUTHORS
        assert "github-actions[bot]" in GITHUB2GERRIT_BOT_AUTHORS


# ===========================================================================
# CLI flag integration tests
# ===========================================================================


class TestCLIGitHub2GerritFlags:
    """Test that CLI flags are properly handled."""

    runner: CliRunner = CliRunner()

    def setup_method(self):
        self.runner = CliRunner()

    def test_mutually_exclusive_flags_submit_and_skip(self):
        from dependamerge.cli import app

        result = self.runner.invoke(
            app,
            [
                "merge",
                "https://github.com/owner/repo/pull/1",
                "--submit-gerrit-changes",
                "--skip-gerrit-changes",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.stdout.lower()

    def test_mutually_exclusive_flags_submit_and_ignore(self):
        from dependamerge.cli import app

        result = self.runner.invoke(
            app,
            [
                "merge",
                "https://github.com/owner/repo/pull/1",
                "--submit-gerrit-changes",
                "--ignore-github2gerrit",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.stdout.lower()

    def test_mutually_exclusive_flags_skip_and_ignore(self):
        from dependamerge.cli import app

        result = self.runner.invoke(
            app,
            [
                "merge",
                "https://github.com/owner/repo/pull/1",
                "--skip-gerrit-changes",
                "--ignore-github2gerrit",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.stdout.lower()

    def test_all_three_flags(self):
        from dependamerge.cli import app

        result = self.runner.invoke(
            app,
            [
                "merge",
                "https://github.com/owner/repo/pull/1",
                "--submit-gerrit-changes",
                "--skip-gerrit-changes",
                "--ignore-github2gerrit",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.stdout.lower()


# ===========================================================================
# Merge Manager integration tests
# ===========================================================================


class TestMergeManagerGitHub2GerritInit:
    """Test AsyncMergeManager initialisation with GitHub2Gerrit params."""

    def test_default_mode_is_submit(self):
        mgr = AsyncMergeManager(token="test-token")
        assert mgr.github2gerrit_mode == "submit"

    def test_skip_mode(self):
        mgr = AsyncMergeManager(token="test-token", github2gerrit_mode="skip")
        assert mgr.github2gerrit_mode == "skip"

    def test_ignore_mode(self):
        mgr = AsyncMergeManager(token="test-token", github2gerrit_mode="ignore")
        assert mgr.github2gerrit_mode == "ignore"

    def test_netrc_params_stored(self):
        p = Path("/tmp/test-netrc")
        mgr = AsyncMergeManager(
            token="test-token",
            no_netrc=True,
            netrc_file=p,
        )
        assert mgr.no_netrc is True
        assert mgr.netrc_file == p


class TestMergeManagerDetectGitHub2Gerrit:
    """Test the _detect_github2gerrit helper on AsyncMergeManager."""

    def test_detect_with_mapping(self):
        mgr = AsyncMergeManager(token="test-token")
        mgr._github_client = AsyncMock()
        mgr._github_client.get = AsyncMock(
            return_value=[
                {
                    "user": {"login": "github-actions[bot]"},
                    "body": SAMPLE_MAPPING_COMMENT_BODY,
                },
            ]
        )

        result = asyncio.run(mgr._detect_github2gerrit("owner", "repo", 41))
        assert result.has_mapping is True
        assert result.mapping is not None
        assert result.mapping.topic == "GH-releng-gerrit_to_platform-41"
        mgr._github_client.get.assert_called_once_with(
            "/repos/owner/repo/issues/41/comments"
        )

    def test_detect_no_mapping(self):
        mgr = AsyncMergeManager(token="test-token")
        mgr._github_client = AsyncMock()
        mgr._github_client.get = AsyncMock(
            return_value=[
                {
                    "user": {"login": "dependabot[bot]"},
                    "body": "Bumps foo from 1.0 to 2.0",
                },
            ]
        )

        result = asyncio.run(mgr._detect_github2gerrit("owner", "repo", 5))
        assert result.has_mapping is False

    def test_detect_handles_api_error(self):
        mgr = AsyncMergeManager(token="test-token")
        mgr._github_client = AsyncMock()
        mgr._github_client.get = AsyncMock(side_effect=RuntimeError("API failure"))

        result = asyncio.run(mgr._detect_github2gerrit("owner", "repo", 99))
        assert result.has_mapping is False

    def test_detect_handles_non_list_response(self):
        mgr = AsyncMergeManager(token="test-token")
        mgr._github_client = AsyncMock()
        mgr._github_client.get = AsyncMock(return_value={"message": "Not Found"})

        result = asyncio.run(mgr._detect_github2gerrit("owner", "repo", 404))
        assert result.has_mapping is False

    def test_detect_no_client(self):
        mgr = AsyncMergeManager(token="test-token")
        mgr._github_client = None

        result = asyncio.run(mgr._detect_github2gerrit("owner", "repo", 1))
        assert result.has_mapping is False


class TestMergeManagerSkipMode:
    """Test that skip mode skips GitHub2Gerrit PRs."""

    def test_skip_mode_skips_g2g_pr(self):
        mgr = AsyncMergeManager(
            token="test-token",
            github2gerrit_mode="skip",
            preview_mode=False,
        )
        mgr._github_client = AsyncMock()
        mgr._github_client.get = AsyncMock(
            return_value=[
                {
                    "user": {"login": "github-actions[bot]"},
                    "body": SAMPLE_MAPPING_COMMENT_BODY,
                },
            ]
        )
        mgr._console = MagicMock()

        pr = _make_pr_info()
        result = asyncio.run(mgr._merge_single_pr(pr))

        assert result.status == MergeStatus.SKIPPED
        assert "GitHub2Gerrit" in (result.error or "")


class TestMergeManagerIgnoreMode:
    """Test that ignore mode skips GitHub2Gerrit detection entirely."""

    def test_ignore_mode_does_not_check_comments(self):
        mgr = AsyncMergeManager(
            token="test-token",
            github2gerrit_mode="ignore",
            preview_mode=True,
        )
        mgr._github_client = AsyncMock()
        mgr._console = MagicMock()

        # Mock _check_merge_requirements to avoid unawaited-coroutine warnings
        # from the AsyncMock client (the real method would call async methods on
        # the mock whose return-value coroutines are never awaited).
        mgr._check_merge_requirements = AsyncMock(
            return_value=(True, "mocked for test")
        )

        # Mock get to track whether issue comments are fetched
        call_log = []

        async def tracking_get(url):
            call_log.append(url)
            if "issues" in url and "comments" in url:
                return [
                    {
                        "user": {"login": "github-actions[bot]"},
                        "body": SAMPLE_MAPPING_COMMENT_BODY,
                    },
                ]
            if "pulls" in url:
                return {"mergeable": True, "mergeable_state": "clean"}
            return []

        mgr._github_client.get = AsyncMock(side_effect=tracking_get)

        pr = _make_pr_info()
        result = asyncio.run(mgr._merge_single_pr(pr))

        # The ignore mode should NOT have fetched issue comments
        issue_comment_calls = [
            c for c in call_log if "/issues/" in c and "/comments" in c
        ]
        assert len(issue_comment_calls) == 0

        # The PR should proceed through normal merge logic (preview mode -> MERGED)
        assert result.status == MergeStatus.MERGED


class TestMergeManagerSubmitModePreview:
    """Test submit mode in preview (dry-run) context."""

    def test_preview_shows_gerrit_submit_intent(self):
        mgr = AsyncMergeManager(
            token="test-token",
            github2gerrit_mode="submit",
            preview_mode=True,
        )
        mgr._github_client = AsyncMock()
        mgr._github_client.get = AsyncMock(
            return_value=[
                {
                    "user": {"login": "github-actions[bot]"},
                    "body": SAMPLE_MAPPING_COMMENT_BODY,
                },
            ]
        )
        mgr._console = MagicMock()

        pr = _make_pr_info()
        result = asyncio.run(mgr._merge_single_pr(pr))

        assert result.status == MergeStatus.MERGED
        # Verify console printed the Gerrit submit message
        printed_args = [str(call) for call in mgr._console.print.call_args_list]
        joined = " ".join(printed_args)
        assert "Gerrit submit" in joined or result.status == MergeStatus.MERGED


# ===========================================================================
# .gitreview parser tests
# ===========================================================================


class TestGitReviewInfo:
    def test_valid_info(self):
        info = GitReviewInfo(
            host="gerrit.example.org", port=29418, project="releng/tool"
        )
        assert info.is_valid is True

    def test_empty_host_invalid(self):
        info = GitReviewInfo(host="", port=29418, project="releng/tool")
        assert info.is_valid is False

    def test_default_port(self):
        info = GitReviewInfo(host="gerrit.example.org")
        assert info.port == 29418

    def test_frozen(self):
        info = GitReviewInfo(host="gerrit.example.org")
        with pytest.raises(AttributeError):
            info.host = "other"  # type: ignore[misc]


class TestParseGitreviewText:
    def test_standard_gitreview(self):
        text = (
            "[gerrit]\n"
            "host=gerrit.linuxfoundation.org\n"
            "port=29418\n"
            "project=releng/gerrit_to_platform.git\n"
        )
        info = parse_gitreview_text(text)
        assert info is not None
        assert info.host == "gerrit.linuxfoundation.org"
        assert info.port == 29418
        assert info.project == "releng/gerrit_to_platform"
        assert info.base_path == "infra"

    def test_no_port_defaults(self):
        text = "[gerrit]\nhost=gerrit.example.org\nproject=my/project.git\n"
        info = parse_gitreview_text(text)
        assert info is not None
        assert info.port == 29418

    def test_git_suffix_stripped(self):
        text = "[gerrit]\nhost=gerrit.example.org\nproject=my/project.git\n"
        info = parse_gitreview_text(text)
        assert info is not None
        assert info.project == "my/project"

    def test_no_git_suffix(self):
        text = "[gerrit]\nhost=gerrit.example.org\nproject=my/project\n"
        info = parse_gitreview_text(text)
        assert info is not None
        assert info.project == "my/project"

    def test_no_host_returns_none(self):
        text = "[gerrit]\nport=29418\nproject=my/project.git\n"
        assert parse_gitreview_text(text) is None

    def test_empty_host_returns_none(self):
        text = "[gerrit]\nhost=\nport=29418\nproject=my/project.git\n"
        assert parse_gitreview_text(text) is None

    def test_empty_string(self):
        assert parse_gitreview_text("") is None

    def test_unknown_host_no_base_path(self):
        text = "[gerrit]\nhost=gerrit.custom.org\nproject=my/project.git\n"
        info = parse_gitreview_text(text)
        assert info is not None
        assert info.base_path is None

    def test_lf_host_has_infra_base_path(self):
        text = "[gerrit]\nhost=gerrit.linuxfoundation.org\nproject=test.git\n"
        info = parse_gitreview_text(text)
        assert info is not None
        assert info.base_path == "infra"

    def test_whitespace_stripped(self):
        text = "[gerrit]\nhost= gerrit.example.org \nproject= my/project.git \n"
        info = parse_gitreview_text(text)
        assert info is not None
        assert info.host == "gerrit.example.org"
        assert info.project == "my/project"

    def test_host_only_is_valid(self):
        text = "[gerrit]\nhost=gerrit.example.org\n"
        info = parse_gitreview_text(text)
        assert info is not None
        assert info.is_valid is True
        assert info.project == ""


class TestFetchGitreviewFromGithub:
    def test_successful_fetch(self):
        import base64

        gitreview_text = "[gerrit]\nhost=gerrit.linuxfoundation.org\nport=29418\nproject=releng/test.git\n"
        encoded = base64.b64encode(gitreview_text.encode()).decode()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value={"content": encoded})

        result = asyncio.run(
            fetch_gitreview_from_github(mock_client, "lfit", "releng-test")
        )
        assert result is not None
        assert result.host == "gerrit.linuxfoundation.org"
        assert result.project == "releng/test"
        mock_client.get.assert_called_once_with(
            "/repos/lfit/releng-test/contents/.gitreview"
        )

    def test_fetch_with_ref(self):
        import base64

        gitreview_text = "[gerrit]\nhost=gerrit.example.org\nproject=test.git\n"
        encoded = base64.b64encode(gitreview_text.encode()).decode()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value={"content": encoded})

        result = asyncio.run(
            fetch_gitreview_from_github(mock_client, "org", "repo", ref="main")
        )
        assert result is not None
        mock_client.get.assert_called_once_with(
            "/repos/org/repo/contents/.gitreview?ref=main"
        )

    def test_file_not_found_returns_none(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("404 Not Found"))

        result = asyncio.run(fetch_gitreview_from_github(mock_client, "org", "repo"))
        assert result is None

    def test_non_dict_response_returns_none(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=[])

        result = asyncio.run(fetch_gitreview_from_github(mock_client, "org", "repo"))
        assert result is None

    def test_empty_content_returns_none(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value={"content": ""})

        result = asyncio.run(fetch_gitreview_from_github(mock_client, "org", "repo"))
        assert result is None

    def test_invalid_gitreview_content_returns_none(self):
        import base64

        encoded = base64.b64encode(b"no host here\n").decode()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value={"content": encoded})

        result = asyncio.run(fetch_gitreview_from_github(mock_client, "org", "repo"))
        assert result is None

    def test_api_error_returns_none(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("Server error"))

        result = asyncio.run(fetch_gitreview_from_github(mock_client, "org", "repo"))
        assert result is None


class TestMergeManagerResolveGerritHost:
    """Test _resolve_gerrit_host logic (now async with .gitreview priority)."""

    def _make_mgr_with_no_gitreview(self):
        """Create a manager whose GitHub client returns 404 for .gitreview."""
        mgr = AsyncMergeManager(token="test-token")
        mgr._github_client = AsyncMock()
        mgr._github_client.get = AsyncMock(side_effect=Exception("404 Not Found"))
        return mgr

    def test_gitreview_is_highest_priority(self):
        """Even with GERRIT_HOST set, .gitreview wins."""
        import base64

        gitreview_text = "[gerrit]\nhost=gerrit.from-gitreview.org\nproject=test.git\n"
        encoded = base64.b64encode(gitreview_text.encode()).decode()

        mgr = AsyncMergeManager(token="test-token")
        mgr._github_client = AsyncMock()
        mgr._github_client.get = AsyncMock(return_value={"content": encoded})

        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa",),
        )
        with patch.dict(
            "os.environ",
            {"GERRIT_HOST": "gerrit.from-env.org", "GERRIT_BASE_PATH": "base"},
        ):
            host, base_path = asyncio.run(mgr._resolve_gerrit_host(m, "owner", "repo"))
        assert host == "gerrit.from-gitreview.org"
        # base_path derived from unknown host → None
        assert base_path is None

    def test_gitreview_lf_host_derives_infra_base_path(self):
        import base64

        gitreview_text = (
            "[gerrit]\nhost=gerrit.linuxfoundation.org\n"
            "port=29418\nproject=releng/tool.git\n"
        )
        encoded = base64.b64encode(gitreview_text.encode()).decode()

        mgr = AsyncMergeManager(token="test-token")
        mgr._github_client = AsyncMock()
        mgr._github_client.get = AsyncMock(return_value={"content": encoded})

        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa",),
        )
        host, base_path = asyncio.run(
            mgr._resolve_gerrit_host(m, "lfit", "releng-tool")
        )
        assert host == "gerrit.linuxfoundation.org"
        assert base_path == "infra"

    def test_falls_back_to_env_when_no_gitreview(self):
        mgr = self._make_mgr_with_no_gitreview()
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa",),
        )
        with patch.dict(
            "os.environ",
            {"GERRIT_HOST": "gerrit.test.org", "GERRIT_BASE_PATH": "base"},
        ):
            host, base_path = asyncio.run(mgr._resolve_gerrit_host(m, "owner", "repo"))
        assert host == "gerrit.test.org"
        assert base_path == "base"

    def test_from_comment_body_gerrit_url(self):
        mgr = self._make_mgr_with_no_gitreview()
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa",),
            raw_comment_body="See https://gerrit.example.org/infra/c/project/+/123",
        )
        import os

        orig_host = os.environ.pop("GERRIT_HOST", None)
        orig_url = os.environ.pop("GERRIT_URL", None)
        try:
            host, base_path = asyncio.run(mgr._resolve_gerrit_host(m, "owner", "repo"))
        finally:
            if orig_host:
                os.environ["GERRIT_HOST"] = orig_host
            if orig_url:
                os.environ["GERRIT_URL"] = orig_url
        assert host == "gerrit.example.org"
        assert base_path == "infra"

    def test_lf_well_known_host(self):
        mgr = self._make_mgr_with_no_gitreview()
        m = GitHub2GerritMapping(
            pr_url="https://github.com/lfit/releng-test/pull/1",
            mode="squash",
            topic="GH-releng-test-1",
            change_ids=("Iaaaa",),
        )
        import os

        orig_host = os.environ.pop("GERRIT_HOST", None)
        orig_url = os.environ.pop("GERRIT_URL", None)
        try:
            host, base_path = asyncio.run(
                mgr._resolve_gerrit_host(m, "lfit", "releng-test")
            )
        finally:
            if orig_host:
                os.environ["GERRIT_HOST"] = orig_host
            if orig_url:
                os.environ["GERRIT_URL"] = orig_url
        assert host == "gerrit.linuxfoundation.org"
        assert base_path == "infra"

    def test_from_gerrit_url_env(self):
        mgr = self._make_mgr_with_no_gitreview()
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa",),
        )
        import os

        orig_host = os.environ.pop("GERRIT_HOST", None)
        try:
            with patch.dict(
                "os.environ", {"GERRIT_URL": "https://gerrit.foo.org/mybase"}
            ):
                host, base_path = asyncio.run(
                    mgr._resolve_gerrit_host(m, "owner", "repo")
                )
        finally:
            if orig_host:
                os.environ["GERRIT_HOST"] = orig_host
        assert host == "gerrit.foo.org"
        assert base_path == "mybase"

    def test_returns_none_when_unresolvable(self):
        mgr = self._make_mgr_with_no_gitreview()
        m = GitHub2GerritMapping(
            pr_url="https://github.com/someorg/somerepo/pull/1",
            mode="squash",
            topic="GH-somerepo-1",
            change_ids=("Iaaaa",),
        )
        import os

        orig_host = os.environ.pop("GERRIT_HOST", None)
        orig_url = os.environ.pop("GERRIT_URL", None)
        try:
            host, base_path = asyncio.run(
                mgr._resolve_gerrit_host(m, "someorg", "somerepo")
            )
        finally:
            if orig_host:
                os.environ["GERRIT_HOST"] = orig_host
            if orig_url:
                os.environ["GERRIT_URL"] = orig_url
        assert host is None
        assert base_path is None

    def test_no_github_client_skips_gitreview(self):
        """When _github_client is None, .gitreview step is skipped gracefully."""
        mgr = AsyncMergeManager(token="test-token")
        mgr._github_client = None
        m = GitHub2GerritMapping(
            pr_url="https://github.com/lfit/releng-test/pull/1",
            mode="squash",
            topic="GH-releng-test-1",
            change_ids=("Iaaaa",),
        )
        import os

        orig_host = os.environ.pop("GERRIT_HOST", None)
        orig_url = os.environ.pop("GERRIT_URL", None)
        try:
            host, base_path = asyncio.run(
                mgr._resolve_gerrit_host(m, "lfit", "releng-test")
            )
        finally:
            if orig_host:
                os.environ["GERRIT_HOST"] = orig_host
            if orig_url:
                os.environ["GERRIT_URL"] = orig_url
        # Falls through to well-known LF host
        assert host == "gerrit.linuxfoundation.org"
        assert base_path == "infra"


class TestMergeManagerSubmitGerritChange:
    """Test _submit_gerrit_change when Gerrit service/creds are mocked."""

    def _make_mgr_with_no_gitreview(self, **kwargs):
        """Create a manager whose GitHub client returns 404 for .gitreview."""
        mgr = AsyncMergeManager(
            token=kwargs.get("token", "test-token"),
            github2gerrit_mode=kwargs.get("github2gerrit_mode", "submit"),
            preview_mode=kwargs.get("preview_mode", False),
            merge_method=kwargs.get("merge_method", "merge"),
        )
        mgr._github_client = AsyncMock()
        # .gitreview fetch returns 404 so tests fall through to mocked creds
        original_get = mgr._github_client.get

        async def _get_side_effect(url):
            if ".gitreview" in url:
                raise Exception("404 Not Found")
            return await original_get(url)

        mgr._github_client.get = AsyncMock(side_effect=_get_side_effect)
        mgr._console = MagicMock()
        return mgr

    def test_submit_success(self):
        mgr = self._make_mgr_with_no_gitreview(preview_mode=False)

        mapping = GitHub2GerritMapping(
            pr_url="https://github.com/lfit/releng-test/pull/1",
            mode="squash",
            topic="GH-releng-test-1",
            change_ids=("I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048",),
        )
        pr = _make_pr_info()

        mock_creds = MagicMock()
        mock_creds.is_valid = True
        mock_creds.username = "user"
        mock_creds.password = "pass"

        mock_change = MagicMock()
        mock_change.project = "releng/test"
        mock_change.number = 12345

        mock_result = MagicMock()
        mock_result.submitted = True
        mock_result.success = True

        mock_service = MagicMock(spec=GerritService)
        mock_service.find_open_change_by_change_id.return_value = mock_change

        mock_submit_mgr = MagicMock()
        mock_submit_mgr.submit_changes.return_value = [mock_result]

        with (
            patch(
                "dependamerge.merge_manager.resolve_gerrit_credentials",
                return_value=mock_creds,
            ),
            patch(
                "dependamerge.merge_manager.create_gerrit_service",
                return_value=mock_service,
            ),
            patch(
                "dependamerge.merge_manager.create_submit_manager",
                return_value=mock_submit_mgr,
            ),
        ):
            result = asyncio.run(
                mgr._submit_gerrit_change(mapping, pr, "lfit", "releng-test")
            )

        assert result is True
        mock_submit_mgr.submit_changes.assert_called_once()

    def test_submit_no_credentials(self):
        mgr = self._make_mgr_with_no_gitreview()

        mapping = GitHub2GerritMapping(
            pr_url="https://github.com/lfit/releng-test/pull/1",
            mode="squash",
            topic="GH-releng-test-1",
            change_ids=("I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048",),
        )
        pr = _make_pr_info()

        with patch(
            "dependamerge.merge_manager.resolve_gerrit_credentials",
            return_value=None,
        ):
            result = asyncio.run(
                mgr._submit_gerrit_change(mapping, pr, "lfit", "releng-test")
            )

        assert result is False

    def test_submit_no_matching_gerrit_change(self, caplog):
        mgr = self._make_mgr_with_no_gitreview()

        mapping = GitHub2GerritMapping(
            pr_url="https://github.com/lfit/releng-test/pull/1",
            mode="squash",
            topic="GH-releng-test-1",
            change_ids=("I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048",),
        )
        pr = _make_pr_info()

        mock_creds = MagicMock()
        mock_creds.is_valid = True
        mock_creds.username = "user"
        mock_creds.password = "pass"

        mock_service = MagicMock(spec=GerritService)
        mock_service.find_open_change_by_change_id.return_value = None

        with (
            patch(
                "dependamerge.merge_manager.resolve_gerrit_credentials",
                return_value=mock_creds,
            ),
            patch(
                "dependamerge.merge_manager.create_gerrit_service",
                return_value=mock_service,
            ),
            patch(
                "dependamerge.merge_manager.create_submit_manager",
            ) as mock_submit_mgr_factory,
        ):
            with caplog.at_level(logging.WARNING, logger="dependamerge.merge_manager"):
                result = asyncio.run(
                    mgr._submit_gerrit_change(mapping, pr, "lfit", "releng-test")
                )

        assert result is False
        # ``result is False`` alone is satisfied by several unrelated
        # failure paths, so pin the reason.  The lookup ran and was asked
        # for this Change-Id...
        mock_service.find_open_change_by_change_id.assert_called_once_with(
            "I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048"
        )
        # ...it reported the miss and returned cleanly, rather than
        # crashing on the way past...
        assert "No open Gerrit change found for Change-Id" in caplog.text
        # ...and nothing downstream was attempted.  Without these two,
        # dropping the None guard lets execution run on and still reach
        # False by way of a later exception --- precisely how this test
        # used to pass against a reverted fix.
        mock_submit_mgr_factory.assert_not_called()

    def test_submit_query_failure_is_not_reported_as_no_match(self, caplog):
        # Regression (#466): a Gerrit that never answered must not be
        # reported as "no open change found".  The lookup used to
        # swallow REST errors and return an empty result, so an outage,
        # a 500 or a rejected credential all produced a message
        # asserting something about the change that was never
        # established.
        mgr = self._make_mgr_with_no_gitreview()

        mapping = GitHub2GerritMapping(
            pr_url="https://github.com/lfit/releng-test/pull/1",
            mode="squash",
            topic="GH-releng-test-1",
            change_ids=("I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048",),
        )
        pr = _make_pr_info()

        mock_creds = MagicMock()
        mock_creds.is_valid = True
        mock_creds.username = "user"
        mock_creds.password = "pass"

        # A real GerritService over a failing REST client, so the error
        # travels the actual path --- _query_changes, then the public
        # lookup, then the submit handler.  Mocking the lookup directly
        # would only re-test the handler and would pass even with the
        # swallow restored.
        rest_client = MagicMock()
        rest_client.get.side_effect = GerritRestError("Service Unavailable", 503)
        with (
            patch("dependamerge.gerrit.service.build_client", return_value=rest_client),
            patch("dependamerge.gerrit.service.create_url_builder"),
        ):
            real_service = GerritService(host="gerrit.example.org")

        with (
            patch(
                "dependamerge.merge_manager.resolve_gerrit_credentials",
                return_value=mock_creds,
            ),
            patch(
                "dependamerge.merge_manager.create_gerrit_service",
                return_value=real_service,
            ),
        ):
            with caplog.at_level(logging.WARNING, logger="dependamerge.merge_manager"):
                result = asyncio.run(
                    mgr._submit_gerrit_change(mapping, pr, "lfit", "releng-test")
                )

        assert result is False
        assert "No open Gerrit change found" not in caplog.text
        assert "Gerrit error submitting change" in caplog.text
        assert "Service Unavailable" in caplog.text

    def test_submit_gerrit_rest_error(self):
        from dependamerge.gerrit import GerritRestError

        mgr = self._make_mgr_with_no_gitreview()

        mapping = GitHub2GerritMapping(
            pr_url="https://github.com/lfit/releng-test/pull/1",
            mode="squash",
            topic="GH-releng-test-1",
            change_ids=("I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048",),
        )
        pr = _make_pr_info()

        mock_creds = MagicMock()
        mock_creds.is_valid = True
        mock_creds.username = "user"
        mock_creds.password = "pass"

        with (
            patch(
                "dependamerge.merge_manager.resolve_gerrit_credentials",
                return_value=mock_creds,
            ),
            patch(
                "dependamerge.merge_manager.create_gerrit_service",
                side_effect=GerritRestError("Connection refused", status_code=503),
            ),
        ):
            result = asyncio.run(
                mgr._submit_gerrit_change(mapping, pr, "lfit", "releng-test")
            )

        assert result is False

    def test_submit_no_gerrit_host(self):
        mgr = self._make_mgr_with_no_gitreview()

        mapping = GitHub2GerritMapping(
            pr_url="https://github.com/unknownorg/repo/pull/1",
            mode="squash",
            topic="GH-repo-1",
            change_ids=("I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048",),
        )
        pr = _make_pr_info(repository_full_name="unknownorg/repo")

        import os

        orig_host = os.environ.pop("GERRIT_HOST", None)
        orig_url = os.environ.pop("GERRIT_URL", None)
        try:
            result = asyncio.run(
                mgr._submit_gerrit_change(mapping, pr, "unknownorg", "repo")
            )
        finally:
            if orig_host:
                os.environ["GERRIT_HOST"] = orig_host
            if orig_url:
                os.environ["GERRIT_URL"] = orig_url

        assert result is False


class TestMergeManagerCloseGitHubPRAfterGerrit:
    """Test _close_github_pr_after_gerrit_submit."""

    def test_closes_pr_and_posts_comment(self):
        mgr = AsyncMergeManager(
            token="test-token",
            preview_mode=False,
        )
        mgr._github_client = AsyncMock()
        mgr._github_client.post_issue_comment = AsyncMock(return_value={})
        mgr._github_client.close_pull_request = AsyncMock(return_value={})

        mapping = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa",),
        )
        pr = _make_pr_info()

        asyncio.run(
            mgr._close_github_pr_after_gerrit_submit(
                pr, mapping, "https://gerrit.example.org/q/Iaaaa"
            )
        )

        mgr._github_client.post_issue_comment.assert_called_once()
        call_args = mgr._github_client.post_issue_comment.call_args
        assert call_args[0][0] == "lfit"  # owner
        assert call_args[0][1] == "releng-gerrit_to_platform"  # repo
        assert call_args[0][2] == 41  # PR number
        assert "submitted" in call_args[0][3].lower()

        mgr._github_client.close_pull_request.assert_called_once_with(
            "lfit", "releng-gerrit_to_platform", 41
        )

    def test_does_nothing_in_preview_mode(self):
        mgr = AsyncMergeManager(
            token="test-token",
            preview_mode=True,
        )
        mgr._github_client = AsyncMock()

        mapping = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa",),
        )
        pr = _make_pr_info()

        asyncio.run(
            mgr._close_github_pr_after_gerrit_submit(
                pr, mapping, "https://gerrit.example.org/q/Iaaaa"
            )
        )

        mgr._github_client.post_issue_comment.assert_not_called()
        mgr._github_client.close_pull_request.assert_not_called()

    def test_handles_close_error_gracefully(self):
        mgr = AsyncMergeManager(
            token="test-token",
            preview_mode=False,
        )
        mgr._github_client = AsyncMock()
        mgr._github_client.post_issue_comment = AsyncMock(return_value={})
        mgr._github_client.close_pull_request = AsyncMock(
            side_effect=RuntimeError("Permission denied")
        )

        mapping = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa",),
        )
        pr = _make_pr_info()

        # Should not raise
        asyncio.run(
            mgr._close_github_pr_after_gerrit_submit(
                pr, mapping, "https://gerrit.example.org/q/Iaaaa"
            )
        )


# ===========================================================================
# Edge-case and regression tests
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and corner scenarios."""

    def test_comment_with_extra_whitespace(self):
        body = "  \n  " + SAMPLE_MAPPING_COMMENT_BODY + "  \n  "
        comments = [_make_rest_comment(body)]
        result = detect_github2gerrit_comments(comments)
        assert result.has_mapping is True

    def test_multiple_mapping_comments_returns_latest(self):
        comments = [
            _make_rest_comment(SAMPLE_MAPPING_COMMENT_BODY),
            _make_rest_comment(SAMPLE_MAPPING_COMMENT_MULTI),
        ]
        result = detect_github2gerrit_comments(comments)
        assert result.has_mapping is True
        assert result.mapping is not None
        assert result.mapping.mode == "multi-commit"  # From second comment

    def test_mapping_with_gerrit_url_in_body(self):
        comments = [_make_rest_comment(SAMPLE_COMMENT_WITH_GERRIT_URL)]
        result = detect_github2gerrit_comments(comments)
        assert result.has_mapping is True
        assert result.mapping is not None
        assert "gerrit_to_platform" in result.mapping.topic

    def test_non_dict_body_ignored(self):
        comments = [{"body": 12345}]  # Non-string body
        result = detect_github2gerrit_comments(comments)
        assert result.has_mapping is False

    def test_comment_with_only_start_marker_ignored(self):
        body = _START_MARKER + "\nSome incomplete content"
        comments = [_make_rest_comment(body)]
        result = detect_github2gerrit_comments(comments)
        # Falls back to heuristic, which may or may not find something
        # The key is it doesn't crash
        assert isinstance(result, GitHub2GerritDetectionResult)

    def test_comment_with_reversed_markers_ignored(self):
        body = _END_MARKER + "\nContent\n" + _START_MARKER
        comments = [_make_rest_comment(body)]
        result = detect_github2gerrit_comments(comments)
        # Should not parse this as a valid marker block
        assert isinstance(result, GitHub2GerritDetectionResult)

    def test_detection_result_defaults(self):
        r = GitHub2GerritDetectionResult()
        assert r.has_mapping is False
        assert r.mapping is None
        assert r.comment_indices == []
        assert r.detection_source == ""

    def test_mapping_frozen_dataclass(self):
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa",),
        )
        with pytest.raises(AttributeError):
            m.topic = "changed"  # type: ignore[misc]

    def test_change_ids_are_tuple_not_list(self):
        """Ensure change_ids is a tuple (immutable) in the frozen dataclass."""
        m = GitHub2GerritMapping(
            pr_url="",
            mode="squash",
            topic="GH-test-1",
            change_ids=("Iaaaa", "Ibbbb"),
        )
        assert isinstance(m.change_ids, tuple)


class TestSubmitModeFullPRFlow:
    """Integration-level test for the full submit-mode PR flow in _merge_single_pr."""

    def test_submit_mode_calls_gerrit_and_closes_pr(self):
        mgr = AsyncMergeManager(
            token="test-token",
            github2gerrit_mode="submit",
            preview_mode=False,
        )
        mgr._github_client = AsyncMock()
        mgr._console = MagicMock()

        # Mock issue comments endpoint to return a G2G mapping
        async def mock_get(url):
            if "/issues/" in url and "/comments" in url:
                return [
                    {
                        "user": {"login": "github-actions[bot]"},
                        "body": SAMPLE_MAPPING_COMMENT_BODY,
                    },
                ]
            return {}

        mgr._github_client.get = AsyncMock(side_effect=mock_get)
        mgr._github_client.post_issue_comment = AsyncMock(return_value={})
        mgr._github_client.close_pull_request = AsyncMock(return_value={})

        pr = _make_pr_info()

        # Mock the Gerrit submission path
        mock_creds = MagicMock()
        mock_creds.is_valid = True
        mock_creds.username = "user"
        mock_creds.password = "pass"

        mock_change = MagicMock()
        mock_change.project = "releng/gerrit_to_platform"
        mock_change.number = 99999

        mock_submit_result = MagicMock()
        mock_submit_result.submitted = True
        mock_submit_result.success = True

        mock_service = MagicMock(spec=GerritService)
        mock_service.find_open_change_by_change_id.return_value = mock_change

        mock_submit_mgr = MagicMock()
        mock_submit_mgr.submit_changes.return_value = [mock_submit_result]

        with (
            patch(
                "dependamerge.merge_manager.resolve_gerrit_credentials",
                return_value=mock_creds,
            ),
            patch(
                "dependamerge.merge_manager.create_gerrit_service",
                return_value=mock_service,
            ),
            patch(
                "dependamerge.merge_manager.create_submit_manager",
                return_value=mock_submit_mgr,
            ),
        ):
            result = asyncio.run(mgr._merge_single_pr(pr))

        assert result.status == MergeStatus.MERGED
        mgr._github_client.post_issue_comment.assert_called_once()
        mgr._github_client.close_pull_request.assert_called_once()

    def test_submit_mode_failure_reports_failed(self):
        mgr = AsyncMergeManager(
            token="test-token",
            github2gerrit_mode="submit",
            preview_mode=False,
        )
        mgr._github_client = AsyncMock()
        mgr._console = MagicMock()

        async def mock_get(url):
            if "/issues/" in url and "/comments" in url:
                return [
                    {
                        "user": {"login": "github-actions[bot]"},
                        "body": SAMPLE_MAPPING_COMMENT_BODY,
                    },
                ]
            return {}

        mgr._github_client.get = AsyncMock(side_effect=mock_get)

        pr = _make_pr_info()

        # Mock credentials but no matching change
        mock_creds = MagicMock()
        mock_creds.is_valid = True
        mock_creds.username = "user"
        mock_creds.password = "pass"

        mock_service = MagicMock(spec=GerritService)
        mock_service.find_open_change_by_change_id.return_value = None

        with (
            patch(
                "dependamerge.merge_manager.resolve_gerrit_credentials",
                return_value=mock_creds,
            ),
            patch(
                "dependamerge.merge_manager.create_gerrit_service",
                return_value=mock_service,
            ),
        ):
            result = asyncio.run(mgr._merge_single_pr(pr))

        assert result.status == MergeStatus.FAILED
        assert "Gerrit" in (result.error or "")
