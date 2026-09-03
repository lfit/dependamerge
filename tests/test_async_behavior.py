# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Comprehensive tests for async behavior patterns in the dependamerge codebase.
These tests verify that the async refactor is working correctly and all
async patterns are properly implemented.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from dependamerge.github_async import (
    GitHubAsync,
    GraphQLError,
)
from dependamerge.github_client import GitHubClient
from dependamerge.github_service import GitHubService
from dependamerge.models import FileChange, PullRequestInfo


class TestGitHubAsyncCore:
    """Test core async functionality of GitHubAsync client."""

    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self):
        """Test that async context manager works correctly."""
        async with GitHubAsync(token="test_token") as api:
            assert api._client is not None
            # Client should be ready to use
            assert hasattr(api, "_client")

    @pytest.mark.asyncio
    async def test_manual_lifecycle_management(self):
        """Test manual open/close lifecycle."""
        api = GitHubAsync(token="test_token")
        try:
            assert api._client is not None
            # Should be ready to use
            assert hasattr(api, "_client")
        finally:
            await api.aclose()

    @pytest.mark.asyncio
    async def test_concurrent_requests_with_semaphore(self):
        """Test that concurrent requests are properly limited by semaphore."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock successful responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"test": "data"}
            mock_response.headers = {"content-type": "application/json"}
            mock_client.request.return_value = mock_response

            api = GitHubAsync(token="test_token", max_concurrency=2)
            try:
                # Start multiple concurrent requests
                tasks = [
                    api.get("/test1"),
                    api.get("/test2"),
                    api.get("/test3"),
                    api.get("/test4"),
                ]

                # All should complete successfully
                results = await asyncio.gather(*tasks)
                assert len(results) == 4
                assert all(isinstance(r, dict) and r["test"] == "data" for r in results)

                # Verify requests were made
                assert mock_client.request.call_count == 4

            finally:
                await api.aclose()

    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self):
        """Test rate limiting and retry behavior."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # First request hits rate limit, second succeeds
            rate_limit_response = Mock()
            rate_limit_response.status_code = 429
            rate_limit_response.headers = {"retry-after": "1"}
            rate_limit_response.text = "API rate limit exceeded"

            success_response = Mock()
            success_response.status_code = 200
            success_response.json.return_value = {"success": True}
            success_response.headers = {"content-type": "application/json"}

            mock_client.request.side_effect = [rate_limit_response, success_response]

            api = GitHubAsync(token="test_token", requests_per_second=10.0)
            try:
                result = await api.get("/test")

                # Should have succeeded after retry
                assert isinstance(result, dict) and result["success"] is True

                # Should have made 2 requests (first failed, second succeeded)
                assert mock_client.request.call_count == 2

            finally:
                await api.aclose()

    @pytest.mark.asyncio
    async def test_graphql_error_handling(self):
        """Test GraphQL error handling and retries."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock GraphQL error response (non-transient error)
            error_response = Mock()
            error_response.status_code = 200
            error_response.json.return_value = {
                "errors": [{"message": "Field 'invalidField' doesn't exist"}]
            }
            error_response.headers = {"content-type": "application/json"}

            mock_client.request.return_value = error_response

            api = GitHubAsync(token="test_token")
            try:
                with pytest.raises(GraphQLError):
                    await api.graphql("query { viewer { login } }")
            finally:
                await api.aclose()

    @pytest.mark.asyncio
    async def test_secondary_rate_limit_detection(self):
        """Test secondary rate limit detection and handling."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock secondary rate limit response
            secondary_limit_response = Mock()
            secondary_limit_response.status_code = 403
            secondary_limit_response.text = (
                "You have exceeded a secondary rate limit. Please wait a few minutes..."
            )
            secondary_limit_response.headers = {"retry-after": "2"}

            success_response = Mock()
            success_response.status_code = 200
            success_response.json.return_value = {"recovered": True}
            success_response.headers = {"content-type": "application/json"}

            mock_client.request.side_effect = [
                secondary_limit_response,
                success_response,
            ]

            api = GitHubAsync(token="test_token")
            try:
                result = await api.get("/test")
                assert isinstance(result, dict) and result["recovered"] is True
                assert mock_client.request.call_count == 2
            finally:
                await api.aclose()


class TestGitHubServiceAsync:
    """Test async patterns in GitHubService."""

    @pytest.mark.asyncio
    async def test_service_initialization_and_cleanup(self):
        """Test proper async service initialization and cleanup."""
        service = GitHubService(token="test_token")

        # Should initialize without errors
        assert service._api is not None
        assert service._max_repo_tasks == 8
        assert service._max_page_tasks == 16

        # Should close cleanly
        await service.close()

    @pytest.mark.asyncio
    async def test_organization_scan_with_mocked_graphql(self):
        """Test organization scanning with mocked GraphQL responses."""
        service = GitHubService(token="test_token")

        try:
            # Mock the internal GraphQL calls
            with patch.object(service._api, "graphql") as mock_graphql:
                # Mock empty organization response
                mock_graphql.return_value = {
                    "organization": {
                        "repositories": {
                            "totalCount": 0,
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [],
                        }
                    }
                }

                result = await service.scan_organization("empty-org")

                assert result.organization == "empty-org"
                assert result.total_repositories == 0
                assert result.total_prs == 0
                assert len(result.unmergeable_prs) == 0
                assert len(result.errors) == 0

        finally:
            await service.close()

    @pytest.mark.asyncio
    async def test_find_similar_prs_async_behavior(self):
        """Test the find_similar_prs method async behavior."""
        service = GitHubService(token="test_token")

        try:
            # Create a mock PR to search for
            source_pr = PullRequestInfo(
                number=1,
                title="Bump requests from 2.28.0 to 2.28.1",
                body="Update requests dependency",
                author="dependabot[bot]",
                head_sha="abc123",
                base_branch="main",
                head_branch="dependabot/pip/requests-2.28.1",
                state="open",
                mergeable=True,
                mergeable_state="clean",
                behind_by=0,
                files_changed=[
                    FileChange(
                        filename="requirements.txt",
                        additions=1,
                        deletions=1,
                        changes=2,
                        status="modified",
                    )
                ],
                repository_full_name="owner/repo",
                html_url="https://github.com/owner/repo/pull/1",
            )

            # Mock comparator
            mock_comparator = Mock()

            # Mock the GraphQL response for organization scan
            with patch.object(service._api, "graphql") as mock_graphql:
                mock_graphql.return_value = {
                    "organization": {
                        "repositories": {
                            "totalCount": 0,
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [],
                        }
                    }
                }

                similar_prs = await service.find_similar_prs(
                    "test-org", source_pr, mock_comparator, only_automation=True
                )

                # Should return empty list for empty organization
                assert similar_prs == []

        finally:
            await service.close()

    @pytest.mark.asyncio
    async def test_concurrent_repository_processing(self):
        """Test that repositories are processed concurrently with bounded parallelism."""
        service = GitHubService(token="test_token", max_repo_tasks=2)

        try:
            # Mock repository data
            repo_nodes = [
                {"nameWithOwner": "org/repo1"},
                {"nameWithOwner": "org/repo2"},
                {"nameWithOwner": "org/repo3"},
            ]

            # Track processing order to verify concurrency
            processing_order = []

            async def mock_process_repo_delay(repo_name):
                processing_order.append(f"start-{repo_name}")
                await asyncio.sleep(0.1)  # Simulate work
                processing_order.append(f"end-{repo_name}")
                return [], 0, 1, []  # unmergeable_prs, total_prs, scanned_repos, errors

            with patch.object(service._api, "graphql") as mock_graphql:
                # Mock organization repositories response
                mock_graphql.return_value = {
                    "organization": {
                        "repositories": {
                            "totalCount": 3,
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": repo_nodes,
                        }
                    }
                }

                # Mock the repository processing function
                async def mock_fetch_repo_prs(owner, name):
                    await mock_process_repo_delay(f"{owner}/{name}")
                    return [], {"hasNextPage": False, "endCursor": None}

                with patch.object(
                    service,
                    "_fetch_repo_prs_first_page",
                    side_effect=mock_fetch_repo_prs,
                ):
                    result = await service.scan_organization("org")

                    # Should have processed all repos
                    assert result.scanned_repositories == 3

                    # Should show evidence of concurrent processing
                    # (starts should be interleaved with ends if concurrent)
                    assert len(processing_order) == 6
                    starts = [
                        item for item in processing_order if item.startswith("start-")
                    ]
                    assert len(starts) == 3

        finally:
            await service.close()


class TestGitHubClientAsyncIntegration:
    """Test async integration patterns in GitHubClient."""

    @patch("dependamerge.github_async.GitHubAsync")
    def test_sync_wrapper_for_async_get_pr_info(self, mock_async_class):
        """Test that sync wrapper properly handles async PR info retrieval."""
        # Setup async mock
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock PR data
        pr_data = {
            "number": 42,
            "title": "Test async PR",
            "body": "Testing async behavior",
            "user": {"login": "test-user"},
            "head": {"sha": "abc123", "ref": "test-branch"},
            "base": {"ref": "main"},
            "state": "open",
            "mergeable": True,
            "mergeable_state": "clean",
            "html_url": "https://github.com/owner/repo/pull/42",
        }

        # Mock different responses for different API calls
        def mock_get_side_effect(url):
            if url == "/repos/owner/repo/pulls/42":
                return pr_data
            elif url == "/repos/owner/repo/pulls/42/reviews":
                return []  # Empty reviews list
            else:
                return {}

        mock_async.get.side_effect = mock_get_side_effect

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

        mock_async.get_paginated = MockAsyncIterator([[]])

        client = GitHubClient(token="test_token")

        # This should work synchronously despite using async under the hood
        pr_info = client.get_pull_request_info("owner", "repo", 42)

        assert pr_info.number == 42
        assert pr_info.title == "Test async PR"
        assert pr_info.author == "test-user"

        # Verify async methods were called (PR info and reviews)
        assert mock_async.get.call_count == 2
        mock_async.get.assert_any_call("/repos/owner/repo/pulls/42")
        mock_async.get.assert_any_call("/repos/owner/repo/pulls/42/reviews")
        mock_async.get_paginated.assert_called_once_with(
            "/repos/owner/repo/pulls/42/files", per_page=100
        )

    @patch("dependamerge.github_async.GitHubAsync")
    def test_sync_wrapper_error_handling(self, mock_async_class):
        """Test that sync wrapper properly handles async exceptions."""
        # Setup async mock to raise exception
        mock_async = AsyncMock()
        mock_async_class.return_value.__aenter__ = AsyncMock(return_value=mock_async)
        mock_async_class.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_async.merge_pull_request.side_effect = Exception("Async error")

        client = GitHubClient(token="test_token")

        # Should handle async exception gracefully and return False
        result = client.merge_pull_request("owner", "repo", 42)
        assert result is False

    @patch("dependamerge.github_service.GitHubService")
    def test_organization_scan_async_integration(self, mock_service_class):
        """Test organization scan async integration through sync interface."""
        # Setup service mock
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock scan result
        mock_scan_result = Mock()
        mock_scan_result.organization = "test-org"
        mock_scan_result.total_repositories = 5
        mock_scan_result.total_prs = 10
        mock_scan_result.unmergeable_prs = []
        mock_scan_result.errors = []

        mock_service.scan_organization.return_value = mock_scan_result
        mock_service.close = AsyncMock()

        client = GitHubClient(token="test_token")

        # Should work synchronously
        result = client.scan_organization_for_unmergeable_prs("test-org")

        assert result.organization == "test-org"
        assert result.total_repositories == 5
        assert result.total_prs == 10

        # Verify async methods were called properly
        mock_service.scan_organization.assert_called_once_with(
            "test-org", include_drafts=False
        )
        mock_service.close.assert_called_once()


class TestAsyncErrorHandling:
    """Test error handling patterns in async code."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_api_errors(self):
        """Test graceful degradation when API calls fail."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock network error
            mock_client.request.side_effect = httpx.NetworkError("Connection failed")

            api = GitHubAsync(token="test_token")
            try:
                with pytest.raises(httpx.NetworkError):
                    await api.get("/test")
            finally:
                await api.aclose()

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in async requests."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock timeout error
            mock_client.request.side_effect = httpx.TimeoutException(
                "Request timed out"
            )

            api = GitHubAsync(token="test_token", timeout=5.0)
            try:
                with pytest.raises(httpx.TimeoutException):
                    await api.get("/test")
            finally:
                await api.aclose()

    @pytest.mark.asyncio
    async def test_service_error_aggregation(self):
        """Test that service properly aggregates errors from async operations."""
        service = GitHubService(token="test_token")

        try:
            # Mock GraphQL to return error on repository counting
            with patch.object(service._api, "graphql") as mock_graphql:
                mock_graphql.side_effect = Exception("GraphQL error")

                # Service doesn't catch all errors - some propagate up
                with pytest.raises(Exception) as exc_info:
                    await service.scan_organization("error-org")

                assert "GraphQL error" in str(exc_info.value)

        finally:
            await service.close()


class TestAsyncPaginationPatterns:
    """Test async pagination patterns."""

    @pytest.mark.asyncio
    async def test_paginated_get_async_iterator(self):
        """Test async pagination iterator works correctly."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock paginated responses - just test single page for now
            page1_response = Mock()
            page1_response.status_code = 200
            page1_response.json.return_value = [{"item": 1}, {"item": 2}]
            page1_response.headers = {
                "content-type": "application/json"
            }  # No next page link

            mock_client.request.return_value = page1_response

            api = GitHubAsync(token="test_token")
            try:
                all_items = []
                async for page in api.get_paginated("/test"):
                    all_items.extend(page)

                # Should have collected all items from the page
                assert len(all_items) == 2
                assert all_items[0]["item"] == 1
                assert all_items[1]["item"] == 2

                # Should have made 1 request
                assert mock_client.request.call_count == 1

            finally:
                await api.aclose()

    @pytest.mark.asyncio
    async def test_empty_pagination_handling(self):
        """Test handling of empty pagination responses."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock empty response
            empty_response = Mock()
            empty_response.status_code = 200
            empty_response.json.return_value = []
            empty_response.headers = {"content-type": "application/json"}

            mock_client.request.return_value = empty_response

            api = GitHubAsync(token="test_token")
            try:
                all_items = []
                async for page in api.get_paginated("/empty"):
                    all_items.extend(page)

                # Should handle empty response gracefully
                assert len(all_items) == 0
                assert mock_client.request.call_count == 1

            finally:
                await api.aclose()


class TestAsyncConcurrencyLimits:
    """Test concurrency limiting and semaphore behavior."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_requests(self):
        """Test that semaphore properly limits concurrent requests."""
        concurrent_requests = []
        max_seen_concurrent = 0

        async def mock_request_with_tracking(*args, **kwargs):
            concurrent_requests.append(1)
            nonlocal max_seen_concurrent
            current_concurrent = len(concurrent_requests)
            max_seen_concurrent = max(max_seen_concurrent, current_concurrent)

            await asyncio.sleep(0.1)  # Simulate work

            concurrent_requests.pop()

            # Mock response
            response = Mock()
            response.status_code = 200
            response.json.return_value = {"done": True}
            response.headers = {"content-type": "application/json"}
            return response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.request.side_effect = mock_request_with_tracking

            # Set low concurrency limit
            api = GitHubAsync(token="test_token", max_concurrency=3)
            try:
                # Start many concurrent requests
                tasks = [api.get(f"/test{i}") for i in range(10)]
                await asyncio.gather(*tasks)

                # Should never have exceeded the concurrency limit
                assert max_seen_concurrent <= 3

            finally:
                await api.aclose()


class TestAsyncRetryPatterns:
    """Test retry patterns and resilience."""

    @pytest.mark.asyncio
    async def test_transient_error_retry_behavior(self):
        """Test retry behavior for transient errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # First two requests fail with 503, third succeeds
            error_response = Mock()
            error_response.status_code = 503
            error_response.text = "Service temporarily unavailable"
            error_response.headers = {"content-type": "application/json"}

            success_response = Mock()
            success_response.status_code = 200
            success_response.json.return_value = {"recovered": True}
            success_response.headers = {"content-type": "application/json"}

            mock_client.request.side_effect = [
                error_response,
                error_response,
                success_response,
            ]

            api = GitHubAsync(token="test_token")
            try:
                result = await api.get("/test")

                # Should eventually succeed
                assert isinstance(result, dict) and result["recovered"] is True

                # Should have retried (3 total requests)
                assert mock_client.request.call_count == 3

            finally:
                await api.aclose()

    @pytest.mark.asyncio
    async def test_non_retryable_error_immediate_failure(self):
        """Test that non-retryable errors fail immediately."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock 404 error (not retryable)
            error_response = Mock()
            error_response.status_code = 404
            error_response.text = "Not found"
            error_response.headers = {"content-type": "application/json"}
            error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=error_response
            )

            mock_client.request.return_value = error_response

            api = GitHubAsync(token="test_token")
            try:
                with pytest.raises(httpx.HTTPStatusError):
                    await api.get("/nonexistent")

                # Should not retry (only 1 request)
                assert mock_client.request.call_count == 1

            finally:
                await api.aclose()
