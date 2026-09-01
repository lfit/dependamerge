# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from unittest.mock import AsyncMock, patch

import pytest

from dependamerge.github_client import GitHubClient
from dependamerge.models import PullRequestInfo


class TestGitHubClient:
    @patch.dict("os.environ", {"GITHUB_TOKEN": "test_token"})
    def test_init_with_env_token(self):
        client = GitHubClient()
        assert client.token == "test_token"

    def test_init_with_explicit_token(self):
        client = GitHubClient(token="explicit_token")
        assert client.token == "explicit_token"

    def test_init_without_token_raises_error(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GitHub token is required"):
                GitHubClient()

    def test_parse_pr_url_valid(self):
        client = GitHubClient(token="test_token")
        owner, repo, pr_number = client.parse_pr_url(
            "https://github.com/lfreleng-actions/python-project-name-action/pull/22"
        )
        assert owner == "lfreleng-actions"
        assert repo == "python-project-name-action"
        assert pr_number == 22

    def test_parse_pr_url_with_trailing_slash(self):
        client = GitHubClient(token="test_token")
        owner, repo, pr_number = client.parse_pr_url(
            "https://github.com/owner/repo/pull/123/"
        )
        assert owner == "owner"
        assert repo == "repo"
        assert pr_number == 123

    def test_parse_pr_url_invalid(self):
        client = GitHubClient(token="test_token")
        # An undeclared host is refused, and the message now names the
        # remedy rather than only calling the URL invalid — see
        # TestUndeclaredPullRequestUrlExplainsItself in
        # tests/test_github_enterprise.py.
        with pytest.raises(ValueError, match="not enabled for host"):
            client.parse_pr_url("https://invalid-url.com")

    def test_parse_pr_url_malformed_on_a_known_host(self):
        client = GitHubClient(token="test_token")
        # A permitted host with a path that is not a pull request keeps
        # the original wording, since nothing needs declaring.
        with pytest.raises(ValueError, match="Invalid GitHub PR URL"):
            client.parse_pr_url("https://github.com/acme/widget")

    def test_parse_pr_url_with_files_path(self):
        client = GitHubClient(token="test_token")
        owner, repo, pr_number = client.parse_pr_url(
            "https://github.com/lfreleng-actions/python-project-name-action/pull/23/files"
        )
        assert owner == "lfreleng-actions"
        assert repo == "python-project-name-action"
        assert pr_number == 23

    def test_parse_pr_url_with_commits_path(self):
        client = GitHubClient(token="test_token")
        owner, repo, pr_number = client.parse_pr_url(
            "https://github.com/owner/repo/pull/456/commits"
        )
        assert owner == "owner"
        assert repo == "repo"
        assert pr_number == 456

    def test_parse_pr_url_with_multiple_path_segments(self):
        client = GitHubClient(token="test_token")
        owner, repo, pr_number = client.parse_pr_url(
            "https://github.com/org/repository/pull/789/files/diff"
        )
        assert owner == "org"
        assert repo == "repository"
        assert pr_number == 789

    @patch("dependamerge.github_async.GitHubAsync")
    def test_get_pull_request_info(self, mock_async_class):
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock PR data
        pr_data = {
            "number": 22,
            "title": "Test PR",
            "body": "Test body",
            "user": {"login": "dependabot[bot]"},
            "head": {"sha": "abc123", "ref": "update-deps"},
            "base": {"ref": "main"},
            "state": "open",
            "mergeable": True,
            "mergeable_state": "clean",
            "html_url": "https://github.com/owner/repo/pull/22",
        }

        # Mock file data
        file_data = {
            "filename": "requirements.txt",
            "additions": 1,
            "deletions": 1,
            "changes": 2,
            "status": "modified",
        }

        # Mock different responses for different API calls
        def mock_get_side_effect(url):
            if url == "/repos/owner/repo/pulls/22":
                return pr_data
            elif url == "/repos/owner/repo/pulls/22/reviews":
                return []  # Empty reviews list
            else:
                return {}

        mock_async.get = AsyncMock(side_effect=mock_get_side_effect)

        # Create proper async iterator mock
        class MockAsyncIterator:
            def __init__(self, data):
                self.data = data
                self.called_with = None

            def __call__(self, *args, **kwargs):
                self.called_with = (args, kwargs)
                return self

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.data:
                    result = self.data.pop(0)
                    return result
                raise StopAsyncIteration

            def assert_called_once_with(self, *args, **kwargs):
                assert self.called_with == (args, kwargs)

        mock_async.get_paginated = MockAsyncIterator([[file_data]])

        client = GitHubClient(token="test_token")
        pr_info = client.get_pull_request_info("owner", "repo", 22)

        assert isinstance(pr_info, PullRequestInfo)
        assert pr_info.number == 22
        assert pr_info.title == "Test PR"
        assert pr_info.author == "dependabot[bot]"
        assert len(pr_info.files_changed) == 1
        assert pr_info.files_changed[0].filename == "requirements.txt"

        # Verify async methods were called properly (PR info and reviews)
        assert mock_async.get.call_count == 2
        mock_async.get.assert_any_call("/repos/owner/repo/pulls/22")
        mock_async.get.assert_any_call("/repos/owner/repo/pulls/22/reviews")
        mock_async.get_paginated.assert_called_once_with(
            "/repos/owner/repo/pulls/22/files", per_page=100
        )

    def test_is_automation_author(self):
        client = GitHubClient(token="test_token")

        assert client.is_automation_author("dependabot[bot]")
        assert client.is_automation_author("pre-commit-ci[bot]")
        assert client.is_automation_author("renovate[bot]")
        assert not client.is_automation_author("human-user")
        assert not client.is_automation_author("random-bot")

    @patch("dependamerge.github_async.GitHubAsync")
    def test_get_pull_request_commits(self, mock_async_class):
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock commit data
        commit_data = [
            {
                "commit": {
                    "message": "Fix bug in authentication\n\nDetailed description"
                }
            },
            {"commit": {"message": "Update tests"}},
        ]

        # Create proper async iterator mock
        class MockAsyncIterator:
            def __init__(self, data):
                self.data = data
                self.called_with = None

            def __call__(self, *args, **kwargs):
                self.called_with = (args, kwargs)
                return self

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.data:
                    result = self.data.pop(0)
                    return result
                raise StopAsyncIteration

            def assert_called_once_with(self, *args, **kwargs):
                assert self.called_with == (args, kwargs)

        mock_async.get_paginated = MockAsyncIterator([commit_data])

        client = GitHubClient(token="test_token")
        commits = client.get_pull_request_commits("owner", "repo", 22)

        assert len(commits) == 2
        assert commits[0] == "Fix bug in authentication\n\nDetailed description"
        assert commits[1] == "Update tests"

        # Verify async method was called properly
        mock_async.get_paginated.assert_called_once_with(
            "/repos/owner/repo/pulls/22/commits", per_page=100
        )

    @patch("dependamerge.github_async.GitHubAsync")
    def test_get_pull_request_commits_empty(self, mock_async_class):
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Create proper async iterator mock for empty commits
        class MockAsyncIterator:
            def __init__(self, data):
                self.data = data

            def __call__(self, *args, **kwargs):
                return self

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.data:
                    result = self.data.pop(0)
                    return result
                raise StopAsyncIteration

        mock_async.get_paginated = MockAsyncIterator([[]])

        client = GitHubClient(token="test_token")
        commits = client.get_pull_request_commits("owner", "repo", 22)

        assert len(commits) == 0
        assert commits == []

    @patch("dependamerge.github_async.GitHubAsync")
    def test_get_organization_repositories(self, mock_async_class):
        """Test getting organization repositories using async client."""
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock repository data
        repo_data = [
            {"full_name": "test-org/repo1"},
            {"full_name": "test-org/repo2"},
            {"full_name": "test-org/repo3"},
        ]

        # Create proper async iterator mock
        class MockAsyncIterator:
            def __init__(self, data):
                self.data = data
                self.called_with = None

            def __call__(self, *args, **kwargs):
                self.called_with = (args, kwargs)
                return self

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.data:
                    result = self.data.pop(0)
                    return result
                raise StopAsyncIteration

            def assert_called_once_with(self, *args, **kwargs):
                assert self.called_with == (args, kwargs)

        mock_async.get_paginated = MockAsyncIterator([repo_data])

        client = GitHubClient(token="test_token")
        repos = client.get_organization_repositories("test-org")

        assert len(repos) == 3
        assert "test-org/repo1" in repos
        assert "test-org/repo2" in repos
        assert "test-org/repo3" in repos

        # Verify async method was called properly
        mock_async.get_paginated.assert_called_once_with(
            "/orgs/test-org/repos", per_page=100
        )

    @patch("dependamerge.github_async.GitHubAsync")
    def test_approve_pull_request(self, mock_async_class):
        """Test pull request approval using async client."""
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_async.approve_pull_request = AsyncMock()

        client = GitHubClient(token="test_token")
        result = client.approve_pull_request("owner", "repo", 22, "LGTM")

        assert result is True
        mock_async.approve_pull_request.assert_called_once_with(
            "owner", "repo", 22, "LGTM"
        )

    @patch("dependamerge.github_async.GitHubAsync")
    def test_approve_pull_request_failure(self, mock_async_class):
        """Test pull request approval failure handling."""
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_async.approve_pull_request = AsyncMock(side_effect=Exception("API Error"))

        client = GitHubClient(token="test_token")
        result = client.approve_pull_request("owner", "repo", 22)

        assert result is False

    @patch("dependamerge.github_async.GitHubAsync")
    def test_merge_pull_request(self, mock_async_class):
        """Test pull request merging using async client."""
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_async.merge_pull_request = AsyncMock(return_value=True)

        client = GitHubClient(token="test_token")
        result = client.merge_pull_request("owner", "repo", 22, "squash")

        assert result is True
        mock_async.merge_pull_request.assert_called_once_with(
            "owner", "repo", 22, "squash"
        )

    @patch("dependamerge.github_async.GitHubAsync")
    def test_merge_pull_request_failure(self, mock_async_class):
        """Test pull request merging failure handling."""
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_async.merge_pull_request = AsyncMock(side_effect=Exception("API Error"))

        client = GitHubClient(token="test_token")
        result = client.merge_pull_request("owner", "repo", 22)

        assert result is False

    @patch("dependamerge.github_async.GitHubAsync")
    def test_fix_out_of_date_pr(self, mock_async_class):
        """Test fixing out of date PR using async client."""
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_async.update_branch = AsyncMock()

        client = GitHubClient(token="test_token")
        result = client.fix_out_of_date_pr("owner", "repo", 22)

        assert result is True
        mock_async.update_branch.assert_called_once_with("owner", "repo", 22)

    @patch("dependamerge.github_async.GitHubAsync")
    def test_analyze_block_reason_failing_checks(self, mock_async_class):
        """Test analyzing block reason with failing checks."""
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock the analyze_block_reason method to return expected result
        mock_async.analyze_block_reason = AsyncMock(
            return_value="Blocked by failing check: CI Tests"
        )

        client = GitHubClient(token="test_token")

        # Create a mock PR info
        pr_info = PullRequestInfo(
            number=22,
            title="Test PR",
            body="Test body",
            author="user",
            head_sha="abc123",
            base_branch="main",
            head_branch="feature",
            state="open",
            mergeable=True,
            mergeable_state="blocked",
            behind_by=0,
            files_changed=[],
            repository_full_name="owner/repo",
            html_url="https://github.com/owner/repo/pull/22",
        )

        reason = client._analyze_block_reason(pr_info)
        assert "failing check" in reason.lower()

    @patch("dependamerge.github_async.GitHubAsync")
    def test_analyze_block_reason_copilot_reviews(self, mock_async_class):
        """Test analyzing block reason with Copilot reviews requesting changes."""
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock the analyze_block_reason method to return expected result
        mock_async.analyze_block_reason = AsyncMock(
            return_value="Blocked by 1 unresolved Copilot reviews"
        )

        client = GitHubClient(token="test_token")

        pr_info = PullRequestInfo(
            number=23,
            title="Test PR",
            body="Test body",
            author="user",
            head_sha="abc123",
            base_branch="main",
            head_branch="feature",
            state="open",
            mergeable=True,
            mergeable_state="blocked",
            behind_by=0,
            files_changed=[],
            repository_full_name="owner/repo",
            html_url="https://github.com/owner/repo/pull/23",
        )

        reason = client._analyze_block_reason(pr_info)
        assert "copilot" in reason.lower()

    @patch("dependamerge.github_async.GitHubAsync")
    def test_analyze_block_reason_copilot_comments(self, mock_async_class):
        """Test analyzing block reason with unresolved Copilot comments."""
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock the analyze_block_reason method to return expected result
        mock_async.analyze_block_reason = AsyncMock(
            return_value="Blocked by 2 unresolved Copilot comments"
        )

        client = GitHubClient(token="test_token")

        pr_info = PullRequestInfo(
            number=24,
            title="Test PR",
            body="Test body",
            author="user",
            head_sha="abc123",
            base_branch="main",
            head_branch="feature",
            state="open",
            mergeable=True,
            mergeable_state="blocked",
            behind_by=0,
            files_changed=[],
            repository_full_name="owner/repo",
            html_url="https://github.com/owner/repo/pull/24",
        )

        reason = client._analyze_block_reason(pr_info)
        assert "copilot comments" in reason.lower()

    @patch("dependamerge.github_async.GitHubAsync")
    def test_analyze_block_reason_human_changes_requested(self, mock_async_class):
        """Test analyzing block reason with human reviewer requesting changes."""
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock the analyze_block_reason method to return expected result
        mock_async.analyze_block_reason = AsyncMock(
            return_value="Human reviewer requested changes"
        )

        client = GitHubClient(token="test_token")

        pr_info = PullRequestInfo(
            number=25,
            title="Test PR",
            body="Test body",
            author="user",
            head_sha="abc123",
            base_branch="main",
            head_branch="feature",
            state="open",
            mergeable=True,
            mergeable_state="blocked",
            behind_by=0,
            files_changed=[],
            repository_full_name="owner/repo",
            html_url="https://github.com/owner/repo/pull/25",
        )

        reason = client._analyze_block_reason(pr_info)
        assert "human reviewer" in reason.lower()
        assert "human reviewer requested changes" in reason.lower()

    @patch("dependamerge.github_async.GitHubAsync")
    def test_analyze_block_reason_multiple_failing_checks(self, mock_async_class):
        """Test analyzing block reason with multiple failing checks."""
        # Setup async mocks properly
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock the analyze_block_reason method to return expected result
        mock_async.analyze_block_reason = AsyncMock(
            return_value="Blocked by 3 failing checks"
        )

        client = GitHubClient(token="test_token")

        pr_info = PullRequestInfo(
            number=26,
            title="Test PR",
            body="Test body",
            author="user",
            head_sha="abc123",
            base_branch="main",
            head_branch="feature",
            state="open",
            mergeable=True,
            mergeable_state="blocked",
            behind_by=0,
            files_changed=[],
            repository_full_name="owner/repo",
            html_url="https://github.com/owner/repo/pull/26",
        )

        reason = client._analyze_block_reason(pr_info)
        assert "3" in reason and "failing check" in reason.lower()

    def test_should_attempt_merge_logic(self):
        """Test the _should_attempt_merge logic with different PR states."""
        client = GitHubClient(token="test_token")

        # Mock PR objects with different states
        class MockPR:
            def __init__(self, mergeable, mergeable_state):
                self.mergeable = mergeable
                self.mergeable_state = mergeable_state

        # Test cases: (mergeable, mergeable_state, expected_result)
        test_cases = [
            (True, "clean", True),  # Ready to merge
            (True, "blocked", True),  # Mergeable but blocked by protection
            (True, "draft", False),  # Draft PR should not be merged
            (False, "blocked", True),  # Might resolve after approval
            (False, "dirty", False),  # Has conflicts, needs manual fix
            (False, "behind", False),  # Behind base, needs update
            (None, "clean", True),  # GitHub still calculating, but looks clean
            (None, "blocked", True),  # GitHub still calculating, might work
            (None, "dirty", False),  # GitHub still calculating, but dirty
        ]

        for mergeable, mergeable_state, expected in test_cases:
            pr = MockPR(mergeable, mergeable_state)
            result = client._should_attempt_merge(pr)
            assert result == expected, (
                f"Failed for mergeable={mergeable}, mergeable_state={mergeable_state}"
            )
