# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Integration tests for async components to verify the async refactor works correctly.
These tests focus on the integration between GitHubClient, GitHubService, and GitHubAsync.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from dependamerge.github_client import GitHubClient
from dependamerge.github_service import GitHubService
from dependamerge.models import FileChange, PullRequestInfo


class TestAsyncIntegration:
    """Test integration between async components."""

    @pytest.mark.asyncio
    async def test_github_service_initialization(self):
        """Test that GitHubService initializes correctly."""
        service = GitHubService(token="test_token")
        assert service._api is not None
        await service.close()

    @pytest.mark.asyncio
    async def test_github_service_scan_organization_empty(self):
        """Test scanning an empty organization."""
        service = GitHubService(token="test_token")

        async def empty_repos(org):
            return
            yield  # This will never execute, making it an empty generator

        service._iter_org_repositories_with_open_prs = empty_repos

        result = await service.scan_organization("empty-org")

        assert result.organization == "empty-org"
        assert result.total_repositories == 0
        assert result.total_prs == 0
        assert len(result.unmergeable_prs) == 0

        await service.close()

    def test_github_client_sync_wrapper(self):
        """Test that GitHubClient properly wraps async operations."""
        client = GitHubClient(token="test_token")

        # Test URL parsing (sync method)
        owner, repo, pr_number = client.parse_pr_url(
            "https://github.com/owner/repo/pull/123"
        )
        assert owner == "owner"
        assert repo == "repo"
        assert pr_number == 123

    @patch("dependamerge.github_async.GitHubAsync")
    def test_github_client_get_pull_request_info_integration(self, mock_async_class):
        """Test the integration between GitHubClient and GitHubAsync for PR info."""
        # Setup async mocks
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock PR data
        pr_data = {
            "number": 42,
            "title": "Update dependencies",
            "body": "Updates all dependencies",
            "user": {"login": "dependabot[bot]"},
            "head": {"sha": "abc123", "ref": "update-deps"},
            "base": {"ref": "main"},
            "state": "open",
            "mergeable": True,
            "mergeable_state": "clean",
            "html_url": "https://github.com/owner/repo/pull/42",
        }

        # Mock file data
        file_data = {
            "filename": "package.json",
            "additions": 5,
            "deletions": 3,
            "changes": 8,
            "status": "modified",
        }

        # Mock different responses for different API calls
        def mock_get_side_effect(url):
            if url == "/repos/owner/repo/pulls/42":
                return pr_data
            elif url == "/repos/owner/repo/pulls/42/reviews":
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
        pr_info = client.get_pull_request_info("owner", "repo", 42)

        # Verify the result
        assert isinstance(pr_info, PullRequestInfo)
        assert pr_info.number == 42
        assert pr_info.title == "Update dependencies"
        assert pr_info.author == "dependabot[bot]"
        assert pr_info.head_sha == "abc123"
        assert pr_info.state == "open"
        assert pr_info.mergeable is True
        assert pr_info.mergeable_state == "clean"
        assert len(pr_info.files_changed) == 1

        # Verify file change
        file_change = pr_info.files_changed[0]
        assert file_change.filename == "package.json"
        assert file_change.additions == 5
        assert file_change.deletions == 3
        assert file_change.changes == 8
        assert file_change.status == "modified"

        # Verify async methods were called (PR info and reviews)
        assert mock_async.get.call_count == 2
        mock_async.get.assert_any_call("/repos/owner/repo/pulls/42")
        mock_async.get.assert_any_call("/repos/owner/repo/pulls/42/reviews")
        mock_async.get_paginated.assert_called_once_with(
            "/repos/owner/repo/pulls/42/files", per_page=100
        )

    @patch("dependamerge.github_service.GitHubService")
    def test_github_client_organization_scan_integration(self, mock_service_class):
        """Test integration between GitHubClient and GitHubService for org scanning."""
        # Setup service mock
        mock_service = AsyncMock()
        mock_scan_result = Mock()
        mock_scan_result.organization = "test-org"
        mock_scan_result.total_repositories = 1
        mock_scan_result.total_prs = 2
        mock_scan_result.unmergeable_prs = []
        mock_scan_result.errors = []

        mock_service.scan_organization = AsyncMock(return_value=mock_scan_result)
        mock_service.close = AsyncMock()
        mock_service_class.return_value = mock_service

        client = GitHubClient(token="test_token")
        result = client.scan_organization_for_unmergeable_prs("test-org")

        # Verify the result
        assert result.organization == "test-org"
        assert result.total_repositories == 1
        assert result.total_prs == 2

        # Verify service was used correctly.  ``host`` is asserted
        # explicitly: a service built without it addresses github.com,
        # which is right by luck on dotcom and silently wrong against a
        # GitHub Enterprise Server install.
        mock_service_class.assert_called_once_with(
            token="test_token", host="github.com", progress_tracker=None
        )
        mock_service.scan_organization.assert_called_once_with(
            "test-org", include_drafts=False
        )
        mock_service.close.assert_called_once()

    @patch("dependamerge.github_async.GitHubAsync")
    def test_github_client_async_operations(self, mock_async_class):
        """Test that GitHubClient handles async operations correctly."""
        # Setup async mocks
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock async methods
        mock_async.approve_pull_request = AsyncMock()
        mock_async.merge_pull_request = AsyncMock(return_value=True)
        mock_async.update_branch = AsyncMock()

        client = GitHubClient(token="test_token")

        # Test approve
        result = client.approve_pull_request("owner", "repo", 42, "LGTM")
        assert result is True
        mock_async.approve_pull_request.assert_called_once_with(
            "owner", "repo", 42, "LGTM"
        )

        # Test merge
        result = client.merge_pull_request("owner", "repo", 42, "squash")
        assert result is True
        mock_async.merge_pull_request.assert_called_once_with(
            "owner", "repo", 42, "squash"
        )

        # Test branch update
        result = client.fix_out_of_date_pr("owner", "repo", 42)
        assert result is True
        mock_async.update_branch.assert_called_once_with("owner", "repo", 42)

    def test_automation_author_detection(self):
        """Test automation author detection logic."""
        client = GitHubClient(token="test_token")

        # Test known automation authors
        assert client.is_automation_author("dependabot[bot]")
        assert client.is_automation_author("pre-commit-ci[bot]")
        assert client.is_automation_author("renovate[bot]")
        assert client.is_automation_author("github-actions[bot]")
        assert client.is_automation_author("allcontributors[bot]")

        # Test human authors
        assert not client.is_automation_author("john-doe")
        assert not client.is_automation_author("jane-smith")
        assert not client.is_automation_author("some-user")

    def test_pr_status_analysis(self):
        """Test PR status analysis logic."""
        client = GitHubClient(token="test_token")

        # Create test PR infos with different states
        test_cases = [
            # (mergeable, mergeable_state, state, expected_status_keywords)
            (True, "clean", "open", ["ready", "merge"]),
            (False, "dirty", "open", ["merge conflicts"]),
            (False, "behind", "open", ["rebase required"]),
            (True, "blocked", "open", ["requires approval", "blocked"]),
            (None, "draft", "open", ["draft"]),
            (None, None, "closed", ["closed"]),
        ]

        for mergeable, mergeable_state, state, expected_keywords in test_cases:
            pr_info = PullRequestInfo(
                number=1,
                title="Test PR",
                body="Test body",
                author="test-author",
                head_sha="abc123",
                base_branch="main",
                head_branch="feature",
                state=state,
                mergeable=mergeable,
                mergeable_state=mergeable_state,
                behind_by=0,
                files_changed=[],
                repository_full_name="owner/repo",
                html_url="https://github.com/owner/repo/pull/1",
            )

            status = client.get_pr_status_details(pr_info)

            # Check that at least one expected keyword is in the status
            has_expected_keyword = any(
                keyword.lower() in status.lower() for keyword in expected_keywords
            )
            assert has_expected_keyword, (
                f"Status '{status}' missing keywords {expected_keywords}"
            )

    def test_merge_attempt_logic(self):
        """Test the logic for determining when to attempt a merge."""
        client = GitHubClient(token="test_token")

        # Mock PR object
        class MockPR:
            def __init__(self, mergeable, mergeable_state):
                self.mergeable = mergeable
                self.mergeable_state = mergeable_state

        # Test cases: (mergeable, mergeable_state, should_attempt)
        test_cases = [
            (True, "clean", True),  # Ready to merge
            (True, "blocked", True),  # May resolve after approval
            (True, "draft", False),  # Draft should not be merged
            (False, "blocked", True),  # Might resolve after approval
            (False, "dirty", False),  # Has conflicts
            (False, "behind", False),  # Behind base branch
            (None, "clean", True),  # GitHub calculating, but looks clean
            (None, "blocked", True),  # GitHub calculating, might work
            (None, "dirty", False),  # GitHub calculating, but dirty
        ]

        for mergeable, mergeable_state, expected in test_cases:
            pr = MockPR(mergeable, mergeable_state)
            result = client._should_attempt_merge(pr)
            assert result == expected, (
                f"Wrong merge attempt decision for "
                f"mergeable={mergeable}, mergeable_state={mergeable_state}. "
                f"Expected {expected}, got {result}"
            )

    @pytest.mark.asyncio
    async def test_error_handling_in_async_context(self):
        """Test error handling in async contexts."""
        service = GitHubService(token="test_token")

        # Test that service handles errors gracefully during org scan
        with patch.object(
            service, "_iter_org_repositories", side_effect=Exception("Network error")
        ):
            # The service doesn't catch this exception at the scan_organization level
            # so we need to catch it ourselves to test proper error handling
            try:
                result = await service.scan_organization("test-org")
                # If we get here, the exception was handled internally
                assert result.organization == "test-org"
            except Exception as e:
                # This is expected behavior - the service propagates critical errors
                assert "Network error" in str(e)

        await service.close()

    def test_file_change_extraction(self):
        """Test file change data extraction and modeling."""
        # Test FileChange model
        file_change = FileChange(
            filename="requirements.txt",
            additions=5,
            deletions=2,
            changes=7,
            status="modified",
        )

        assert file_change.filename == "requirements.txt"
        assert file_change.additions == 5
        assert file_change.deletions == 2
        assert file_change.changes == 7
        assert file_change.status == "modified"

    @pytest.mark.asyncio
    async def test_service_lifecycle_management(self):
        """Test proper lifecycle management of async services."""
        # Test that services can be created and closed properly
        service = GitHubService(token="test_token")

        # Should be in valid initial state
        assert service._api is not None

        # Should be able to close cleanly
        await service.close()

        # Should be idempotent (can close multiple times)
        await service.close()

        # Should still be in valid state after closing
        assert service._api is not None

    def test_url_parsing_edge_cases(self):
        """Test URL parsing with various edge cases."""
        client = GitHubClient(token="test_token")

        # Test various valid URL formats
        test_urls = [
            ("https://github.com/owner/repo/pull/123", ("owner", "repo", 123)),
            ("https://github.com/owner/repo/pull/123/", ("owner", "repo", 123)),
            ("https://github.com/owner/repo/pull/123/files", ("owner", "repo", 123)),
            ("https://github.com/owner/repo/pull/123/commits", ("owner", "repo", 123)),
            (
                "https://github.com/my-org/my-repo-name/pull/456",
                ("my-org", "my-repo-name", 456),
            ),
        ]

        for url, expected in test_urls:
            result = client.parse_pr_url(url)
            assert result == expected, f"Failed to parse {url}"

        # Test invalid URLs
        invalid_urls = [
            "https://invalid-site.com/owner/repo/pull/123",
            "https://github.com/owner/repo/issues/123",  # Not a PR URL
            "https://github.com/owner/repo",  # No PR number
            "not-a-url",
            "",
            # Too few segments for /owner/repo/pull/N.  Splitting the
            # path and indexing backwards from "pull" wrapped around:
            # these returned ("pull", "7", 7) and ("7", "a", 7), so
            # close queried a repository nobody named.
            "https://github.com/pull/7",
            "https://github.com/a/pull/7",
            "https://github.com/pull",
        ]

        for invalid_url in invalid_urls:
            with pytest.raises(ValueError):
                client.parse_pr_url(invalid_url)

    @pytest.mark.asyncio
    async def test_analyze_block_reason_in_async_context(self):
        """Test that analyze_block_reason works correctly in async context without coroutine warnings."""
        from dependamerge.github_async import GitHubAsync

        # Mock review and check data
        reviews = [
            {"state": "CHANGES_REQUESTED", "user": {"login": "github-copilot[bot]"}}
        ]
        comments = [
            {"user": {"login": "github-copilot[bot]"}, "body": "Please fix this"}
        ]
        check_runs = {"check_runs": [{"conclusion": "failure", "name": "CI Tests"}]}

        async with GitHubAsync(token="test_token") as api:
            # Mock the API calls
            async def mock_get(url):
                if "/reviews" in url:
                    return reviews
                elif "/comments" in url:
                    return comments
                elif "/check-runs" in url:
                    return check_runs
                return {}

            api.get = mock_get

            # This should work without raising RuntimeWarning about unawaited coroutine
            result = await api.analyze_block_reason("owner", "repo", 123, "abc123")

            # Verify we got a meaningful result
            assert "failing check" in result.lower() or "blocked" in result.lower()
            assert isinstance(result, str)
            assert len(result) > 0

    def test_analyze_block_reason_sync_context_detection(self):
        """Test that GitHubClient._analyze_block_reason detects async context properly."""
        from dependamerge.models import PullRequestInfo

        client = GitHubClient(token="test_token")

        pr_info = PullRequestInfo(
            number=1,
            title="Test PR",
            body="Test body",
            author="test-author",
            head_sha="abc123",
            base_branch="main",
            head_branch="feature",
            state="open",
            mergeable=True,
            mergeable_state="blocked",
            behind_by=0,
            files_changed=[],
            repository_full_name="owner/repo",
            html_url="https://github.com/owner/repo/pull/1",
        )

        # In sync context (no event loop), this should work fine
        # It will return a fallback message since we can't actually make API calls
        result = client._analyze_block_reason(pr_info)

        # Should return a string without raising warnings
        assert isinstance(result, str)
        assert len(result) > 0
