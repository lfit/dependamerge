# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Tests for Gerrit REST client module.

This module tests the GerritRestClient's initialization, request handling,
retry behavior, and authentication using pygerrit2.
"""

from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import ConnectionError, HTTPError, Timeout

from dependamerge.gerrit.client import (
    GerritAuthError,
    GerritNotFoundError,
    GerritRestClient,
    GerritRestError,
    _calculate_backoff,
    _extract_status_code,
    _is_transient_error,
    _mask_secret,
    build_client,
)


class TestMaskSecret:
    """Tests for the _mask_secret helper function."""

    def test_empty_string(self):
        """Test masking empty string."""
        assert _mask_secret("") == ""

    def test_short_string(self):
        """Test masking short strings (4 or fewer chars)."""
        assert _mask_secret("abc") == "****"
        assert _mask_secret("abcd") == "****"

    def test_normal_string(self):
        """Test masking normal length strings."""
        assert _mask_secret("password123") == "pa*******23"
        assert _mask_secret("secret") == "se**et"

    def test_long_string(self):
        """Test masking long strings."""
        result = _mask_secret("verylongsecretpassword")
        assert result.startswith("ve")
        assert result.endswith("rd")
        assert "****" in result


class TestIsTransientError:
    """Tests for transient error detection."""

    def test_timeout_error(self):
        """Test that timeout errors are detected as transient."""
        exc = Exception("Connection timed out")
        assert _is_transient_error(exc) is True

    def test_connection_reset(self):
        """Test that connection reset is detected as transient."""
        exc = Exception("Connection reset by peer")
        assert _is_transient_error(exc) is True

    def test_service_unavailable(self):
        """Test that service unavailable is detected as transient."""
        exc = Exception("Service unavailable")
        assert _is_transient_error(exc) is True

    def test_non_transient_error(self):
        """Test that non-transient errors are not flagged."""
        exc = Exception("Invalid request")
        assert _is_transient_error(exc) is False


class TestCalculateBackoff:
    """Tests for backoff calculation."""

    def test_first_attempt(self):
        """Test backoff for first attempt."""
        delay = _calculate_backoff(0, base_delay=1.0, max_delay=30.0, jitter=0.0)
        assert delay == 1.0

    def test_exponential_growth(self):
        """Test that delay grows exponentially."""
        delay0 = _calculate_backoff(0, base_delay=1.0, max_delay=60.0, jitter=0.0)
        delay1 = _calculate_backoff(1, base_delay=1.0, max_delay=60.0, jitter=0.0)
        delay2 = _calculate_backoff(2, base_delay=1.0, max_delay=60.0, jitter=0.0)

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        delay = _calculate_backoff(10, base_delay=1.0, max_delay=30.0, jitter=0.0)
        assert delay == 30.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds variability to delay."""
        # With 50% jitter, delays should vary
        delays = [
            _calculate_backoff(1, base_delay=1.0, max_delay=30.0, jitter=0.5)
            for _ in range(10)
        ]
        # Base delay at attempt 1 is 2.0, with 50% jitter it can be 2.0-3.0
        assert all(2.0 <= d <= 3.0 for d in delays)
        # With random jitter, not all delays should be identical
        # (statistically very unlikely)
        assert len(set(delays)) > 1


class TestExtractStatusCode:
    """Tests for status code extraction from exceptions."""

    def test_extract_from_response_attribute(self):
        """Test extracting status code from response attribute."""
        exc = HTTPError()
        exc.response = MagicMock()
        exc.response.status_code = 404
        assert _extract_status_code(exc) == 404

    def test_extract_from_string_representation(self):
        """Test extracting status code from exception string."""
        exc = Exception("Server returned 503 Service Unavailable")
        assert _extract_status_code(exc) == 503

    def test_no_status_code(self):
        """Test when no status code can be extracted."""
        exc = Exception("Some random error")
        assert _extract_status_code(exc) is None


class TestGerritRestClientInit:
    """Tests for GerritRestClient initialization."""

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_basic_init(self, mock_api):
        """Test basic client initialization."""
        client = GerritRestClient(base_url="https://gerrit.example.org/")

        assert client.base_url == "https://gerrit.example.org/"
        assert client.is_authenticated is False
        mock_api.assert_called_once()

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_init_normalizes_base_url(self, mock_api):
        """Test that base URL is normalized to end with slash."""
        client = GerritRestClient(base_url="https://gerrit.example.org")
        assert client.base_url == "https://gerrit.example.org/"

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    @patch("dependamerge.gerrit.client.HTTPBasicAuth")
    def test_init_with_auth(self, mock_auth, mock_api):
        """Test client initialization with authentication."""
        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            auth=("user", "password"),
        )

        assert client.is_authenticated is True
        mock_auth.assert_called_once_with("user", "password")

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_init_with_empty_auth(self, mock_api):
        """Test that empty auth credentials are ignored."""
        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            auth=("", ""),
        )

        assert client.is_authenticated is False

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_init_with_partial_auth(self, mock_api):
        """Test that partial auth credentials are ignored."""
        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            auth=("user", ""),
        )

        assert client.is_authenticated is False

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_repr(self, mock_api):
        """Test string representation."""
        client = GerritRestClient(base_url="https://gerrit.example.org/")
        repr_str = repr(client)
        assert "gerrit.example.org" in repr_str


class TestGerritRestClientRequests:
    """Tests for GerritRestClient request methods."""

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_get_empty_path_raises(self, mock_api):
        """Test that empty path raises GerritRestError."""
        client = GerritRestClient(base_url="https://gerrit.example.org/")

        with pytest.raises(GerritRestError, match="path is required"):
            client.get("")

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_get_success(self, mock_api):
        """Test successful GET request."""
        mock_instance = MagicMock()
        mock_instance.get.return_value = {"key": "value"}
        mock_api.return_value = mock_instance

        client = GerritRestClient(base_url="https://gerrit.example.org/")
        result = client.get("/changes/12345")

        assert result == {"key": "value"}
        mock_instance.get.assert_called_once()

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_get_normalizes_path(self, mock_api):
        """Test that GET normalizes path to start with /."""
        mock_instance = MagicMock()
        mock_instance.get.return_value = {}
        mock_api.return_value = mock_instance

        client = GerritRestClient(base_url="https://gerrit.example.org/")
        client.get("changes/12345")

        # Should be called with path starting with /
        call_args = mock_instance.get.call_args
        assert call_args[0][0].startswith("/")

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_post_with_data(self, mock_api):
        """Test POST request with JSON data."""
        mock_instance = MagicMock()
        mock_instance.post.return_value = {"success": True}
        mock_api.return_value = mock_instance

        client = GerritRestClient(base_url="https://gerrit.example.org/")
        result = client.post("/changes/12345/review", {"labels": {"Code-Review": 2}})

        assert result == {"success": True}
        mock_instance.post.assert_called_once()
        # Verify data was passed
        call_kwargs = mock_instance.post.call_args[1]
        assert call_kwargs["data"] == {"labels": {"Code-Review": 2}}

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_post_without_data(self, mock_api):
        """Test POST request without data."""
        mock_instance = MagicMock()
        mock_instance.post.return_value = {}
        mock_api.return_value = mock_instance

        client = GerritRestClient(base_url="https://gerrit.example.org/")
        client.post("/changes/12345/submit")

        mock_instance.post.assert_called_once()

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_put_with_data(self, mock_api):
        """Test PUT request with JSON data."""
        mock_instance = MagicMock()
        mock_instance.put.return_value = {"updated": True}
        mock_api.return_value = mock_instance

        client = GerritRestClient(base_url="https://gerrit.example.org/")
        result = client.put("/changes/12345/topic", {"topic": "feature-x"})

        assert result == {"updated": True}
        mock_instance.put.assert_called_once()

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_delete(self, mock_api):
        """Test DELETE request."""
        mock_instance = MagicMock()
        mock_instance.delete.return_value = {}
        mock_api.return_value = mock_instance

        client = GerritRestClient(base_url="https://gerrit.example.org/")
        client.delete("/changes/12345/topic")

        mock_instance.delete.assert_called_once()


class TestGerritRestClientErrors:
    """Tests for GerritRestClient error handling."""

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_401_raises_auth_error(self, mock_api):
        """Test that 401 response raises GerritAuthError."""
        mock_instance = MagicMock()
        error = HTTPError("401 Unauthorized")
        error.response = MagicMock()
        error.response.status_code = 401
        mock_instance.get.side_effect = error
        mock_api.return_value = mock_instance

        client = GerritRestClient(base_url="https://gerrit.example.org/")

        with pytest.raises(GerritAuthError) as exc_info:
            client.get("/changes/12345")

        assert exc_info.value.status_code == 401

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_403_raises_auth_error(self, mock_api):
        """Test that 403 response raises GerritAuthError."""
        mock_instance = MagicMock()
        error = HTTPError("403 Forbidden")
        error.response = MagicMock()
        error.response.status_code = 403
        mock_instance.get.side_effect = error
        mock_api.return_value = mock_instance

        client = GerritRestClient(base_url="https://gerrit.example.org/")

        with pytest.raises(GerritAuthError) as exc_info:
            client.get("/changes/12345")

        assert exc_info.value.status_code == 403

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_404_raises_not_found_error(self, mock_api):
        """Test that 404 response raises GerritNotFoundError."""
        mock_instance = MagicMock()
        error = HTTPError("404 Not Found")
        error.response = MagicMock()
        error.response.status_code = 404
        mock_instance.get.side_effect = error
        mock_api.return_value = mock_instance

        client = GerritRestClient(base_url="https://gerrit.example.org/")

        with pytest.raises(GerritNotFoundError) as exc_info:
            client.get("/changes/99999")

        assert exc_info.value.status_code == 404

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_500_raises_rest_error(self, mock_api):
        """Test that 500 response raises GerritRestError."""
        mock_instance = MagicMock()
        error = HTTPError("500 Internal Server Error")
        error.response = MagicMock()
        error.response.status_code = 500
        mock_instance.get.side_effect = error
        mock_api.return_value = mock_instance

        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            max_attempts=1,  # Disable retries for this test
        )

        with pytest.raises(GerritRestError) as exc_info:
            client.get("/changes/12345")

        assert exc_info.value.status_code == 500

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_connection_error_raises_rest_error(self, mock_api):
        """Test that connection errors raise GerritRestError."""
        mock_instance = MagicMock()
        mock_instance.get.side_effect = ConnectionError("Connection refused")
        mock_api.return_value = mock_instance

        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            max_attempts=1,
        )

        with pytest.raises(GerritRestError):
            client.get("/changes/12345")

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_timeout_error_raises_rest_error(self, mock_api):
        """Test that timeout errors raise GerritRestError."""
        mock_instance = MagicMock()
        mock_instance.get.side_effect = Timeout("Request timed out")
        mock_api.return_value = mock_instance

        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            max_attempts=1,
        )

        with pytest.raises(GerritRestError):
            client.get("/changes/12345")


class TestGerritRestClientRetry:
    """Tests for retry behavior."""

    @patch("time.sleep")
    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_retry_on_503(self, mock_api, mock_sleep):
        """Test that 503 errors trigger retry."""
        mock_instance = MagicMock()

        # First call fails with 503, second succeeds
        error = HTTPError("503 Service Unavailable")
        error.response = MagicMock()
        error.response.status_code = 503

        mock_instance.get.side_effect = [error, {"key": "value"}]
        mock_api.return_value = mock_instance

        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            max_attempts=3,
        )
        result = client.get("/changes/12345")

        assert result == {"key": "value"}
        assert mock_instance.get.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("time.sleep")
    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_retry_on_transient_error(self, mock_api, mock_sleep):
        """Test that transient network errors trigger retry."""
        mock_instance = MagicMock()

        # First call fails with transient error, second succeeds
        mock_instance.get.side_effect = [
            ConnectionError("Connection reset by peer"),
            {"key": "value"},
        ]
        mock_api.return_value = mock_instance

        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            max_attempts=3,
        )
        result = client.get("/changes/12345")

        assert result == {"key": "value"}
        assert mock_instance.get.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("time.sleep")
    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_no_retry_on_401(self, mock_api, mock_sleep):
        """Test that 401 errors do not trigger retry."""
        mock_instance = MagicMock()
        error = HTTPError("401 Unauthorized")
        error.response = MagicMock()
        error.response.status_code = 401
        mock_instance.get.side_effect = error
        mock_api.return_value = mock_instance

        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            max_attempts=3,
        )

        with pytest.raises(GerritAuthError):
            client.get("/changes/12345")

        assert mock_instance.get.call_count == 1
        assert mock_sleep.call_count == 0

    @patch("time.sleep")
    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_no_retry_on_404(self, mock_api, mock_sleep):
        """Test that 404 errors do not trigger retry."""
        mock_instance = MagicMock()
        error = HTTPError("404 Not Found")
        error.response = MagicMock()
        error.response.status_code = 404
        mock_instance.get.side_effect = error
        mock_api.return_value = mock_instance

        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            max_attempts=3,
        )

        with pytest.raises(GerritNotFoundError):
            client.get("/changes/99999")

        assert mock_instance.get.call_count == 1
        assert mock_sleep.call_count == 0

    @patch("time.sleep")
    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_max_attempts_exhausted(self, mock_api, mock_sleep):
        """Test that error is raised after max attempts exhausted."""
        mock_instance = MagicMock()
        error = HTTPError("503 Service Unavailable")
        error.response = MagicMock()
        error.response.status_code = 503
        mock_instance.get.side_effect = error
        mock_api.return_value = mock_instance

        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            max_attempts=3,
        )

        with pytest.raises(GerritRestError):
            client.get("/changes/12345")

        assert mock_instance.get.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between attempts


class TestBuildClient:
    """Tests for the build_client factory function."""

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_build_client_basic(self, mock_api):
        """Test building client with just hostname."""
        client = build_client("gerrit.example.org")

        assert "gerrit.example.org" in client.base_url
        assert client.is_authenticated is False

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_build_client_with_base_path(self, mock_api):
        """Test building client with base path."""
        client = build_client("gerrit.example.org", base_path="infra")

        assert "gerrit.example.org/infra/" in client.base_url

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    @patch("dependamerge.gerrit.client.HTTPBasicAuth")
    def test_build_client_with_credentials(self, mock_auth, mock_api):
        """Test building client with explicit credentials."""
        client = build_client(
            "gerrit.example.org",
            username="testuser",
            password="testpass",
        )

        assert client.is_authenticated is True
        mock_auth.assert_called_once_with("testuser", "testpass")

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    @patch("dependamerge.gerrit.client.HTTPBasicAuth")
    def test_build_client_from_env(self, mock_auth, mock_api, monkeypatch):
        """Test building client with credentials from environment."""
        monkeypatch.setenv("GERRIT_USERNAME", "envuser")
        monkeypatch.setenv("GERRIT_PASSWORD", "envpass")

        client = build_client("gerrit.example.org")

        assert client.is_authenticated is True
        mock_auth.assert_called_once_with("envuser", "envpass")

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    @patch("dependamerge.gerrit.client.HTTPBasicAuth")
    def test_build_client_env_fallback(self, mock_auth, mock_api, monkeypatch):
        """Test credential fallback from GERRIT_HTTP_USER/PASSWORD."""
        monkeypatch.setenv("GERRIT_HTTP_USER", "httpuser")
        monkeypatch.setenv("GERRIT_HTTP_PASSWORD", "httppass")

        client = build_client("gerrit.example.org")

        assert client.is_authenticated is True
        mock_auth.assert_called_once_with("httpuser", "httppass")

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    @patch("dependamerge.gerrit.client.HTTPBasicAuth")
    def test_build_client_explicit_overrides_env(
        self, mock_auth, mock_api, monkeypatch
    ):
        """Test that explicit credentials override environment."""
        monkeypatch.setenv("GERRIT_USERNAME", "envuser")
        monkeypatch.setenv("GERRIT_PASSWORD", "envpass")

        client = build_client(
            "gerrit.example.org",
            username="explicit",
            password="credentials",
        )

        assert client.is_authenticated is True
        mock_auth.assert_called_once_with("explicit", "credentials")


class TestPygerrit2Integration:
    """Tests for pygerrit2 integration."""

    def test_pygerrit2_is_importable(self):
        """Test that pygerrit2 can be imported."""
        from pygerrit2 import GerritRestAPI, HTTPBasicAuth

        assert GerritRestAPI is not None
        assert HTTPBasicAuth is not None

    @patch("dependamerge.gerrit.client.GerritRestAPI")
    def test_client_uses_pygerrit2(self, mock_api):
        """Test that client uses pygerrit2 for requests."""
        mock_instance = MagicMock()
        mock_instance.get.return_value = {"test": "data"}
        mock_api.return_value = mock_instance

        client = GerritRestClient(base_url="https://gerrit.example.org/")
        result = client.get("/changes/")

        assert result == {"test": "data"}
        mock_api.assert_called_once()
        mock_instance.get.assert_called_once()
