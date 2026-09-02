# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Centralized error codes and exit codes for dependamerge.

This module defines standard exit codes and error messages to provide
consistent, user-friendly error reporting across the CLI and library.
"""

from __future__ import annotations

import logging
import sys
from enum import IntEnum
from typing import NoReturn

from rich.console import Console

from .url_parser.models import UrlParseError

log = logging.getLogger("dependamerge.error_codes")
console = Console()


class ExitCode(IntEnum):
    """Standard exit codes for dependamerge operations."""

    SUCCESS = 0
    """Operation completed successfully."""

    GENERAL_ERROR = 1
    """General operational failure."""

    CONFIGURATION_ERROR = 2
    """Configuration validation failed or missing required parameters."""

    GITHUB_API_ERROR = 3
    """GitHub API access failed due to permissions or authentication issues."""

    NETWORK_ERROR = 4
    """Network connectivity issues."""

    REPOSITORY_ERROR = 5
    """Git repository access or operation failed."""

    PR_STATE_ERROR = 6
    """Pull request is in invalid state for processing."""

    MERGE_ERROR = 7
    """Pull request merge operation failed."""

    VALIDATION_ERROR = 8
    """Input validation failed."""


# Error message templates
ERROR_MESSAGES = {
    ExitCode.GITHUB_API_ERROR: (
        "❌ GitHub API access failed; ensure GITHUB_TOKEN has required permissions"
    ),
    ExitCode.CONFIGURATION_ERROR: (
        "❌ Configuration validation failed; check required parameters"
    ),
    ExitCode.NETWORK_ERROR: (
        "❌ Network connectivity failed; check internet connection"
    ),
    ExitCode.REPOSITORY_ERROR: (
        "❌ Git repository access failed; check repository permissions"
    ),
    ExitCode.PR_STATE_ERROR: ("❌ Pull request cannot be processed in current state"),
    ExitCode.MERGE_ERROR: (
        "❌ Pull request merge failed; check branch protection rules"
    ),
    ExitCode.VALIDATION_ERROR: ("❌ Input validation failed; check parameter values"),
    ExitCode.GENERAL_ERROR: "❌ Operation failed; check logs for details",
}


class DependamergeError(Exception):
    """Base exception class for Dependamerge errors with exit codes."""

    def __init__(
        self,
        exit_code: ExitCode,
        message: str | None = None,
        details: str | None = None,
        original_exception: Exception | None = None,
    ):
        self.exit_code = exit_code
        self.message = message or ERROR_MESSAGES.get(
            exit_code, ERROR_MESSAGES[ExitCode.GENERAL_ERROR]
        )
        self.details = details
        self.original_exception = original_exception

        # Call parent constructor with the error message
        super().__init__(self.message)

    def display_and_exit(self) -> NoReturn:
        """Display the error message and exit with the appropriate code."""
        if self.original_exception:
            log.error(
                "Exit code %d: %s (Exception: %s)",
                self.exit_code,
                self.message,
                self.original_exception,
            )
            if self.details:
                log.error("Additional details: %s", self.details)
        else:
            log.error("Exit code %d: %s", self.exit_code, self.message)
            if self.details:
                log.error("Details: %s", self.details)

        # Display user-friendly error message
        console.print(self.message, style="red")

        if self.details:
            console.print(f"Details: {self.details}", style="dim red")

        sys.exit(int(self.exit_code))


def exit_with_error(
    exit_code: ExitCode,
    message: str | None = None,
    details: str | None = None,
    exception: Exception | None = None,
) -> NoReturn:
    """Exit with standardized error message and code.

    Args:
        exit_code: Standard exit code from ExitCode enum
        message: Override default error message (optional)
        details: Additional error details (optional)
        exception: Original exception for logging (optional)
    """
    error = DependamergeError(exit_code, message, details, exception)
    error.display_and_exit()


def exit_for_github_api_error(
    message: str | None = None,
    details: str | None = None,
    exception: Exception | None = None,
) -> NoReturn:
    """Exit with GitHub API error code and message."""
    default_msg = (
        "❌ GitHub API query failed; provide a GITHUB_TOKEN with the "
        "required permissions"
    )
    error = DependamergeError(
        ExitCode.GITHUB_API_ERROR,
        message=message or default_msg,
        details=details,
        original_exception=exception,
    )
    error.display_and_exit()


def exit_for_configuration_error(
    message: str | None = None,
    details: str | None = None,
    exception: Exception | None = None,
) -> NoReturn:
    """Exit with configuration error code and message."""
    error = DependamergeError(
        ExitCode.CONFIGURATION_ERROR,
        message=message,
        details=details,
        original_exception=exception,
    )
    error.display_and_exit()


def exit_for_pr_state_error(
    pr_number: int,
    pr_state: str,
    details: str | None = None,
) -> NoReturn:
    """Exit with PR state error code and message."""
    message = f"❌ Pull request #{pr_number} is {pr_state} and cannot be processed"
    error = DependamergeError(
        ExitCode.PR_STATE_ERROR,
        message=message,
        details=details,
    )
    error.display_and_exit()


def exit_for_pr_not_found(
    pr_number: int,
    repository: str,
    details: str | None = None,
    exception: Exception | None = None,
) -> NoReturn:
    """Exit with PR not found error (GitHub API error)."""
    message = f"❌ Pull request #{pr_number} not found in repository {repository}"
    error = DependamergeError(
        ExitCode.GITHUB_API_ERROR,
        message=message,
        details=details,
        original_exception=exception,
    )
    error.display_and_exit()


def exit_for_merge_error(
    pr_number: int,
    repository: str,
    details: str | None = None,
    exception: Exception | None = None,
) -> NoReturn:
    """Exit with merge error code and message."""
    message = f"❌ Failed to merge pull request #{pr_number} in {repository}"
    error = DependamergeError(
        ExitCode.MERGE_ERROR,
        message=message,
        details=details,
        original_exception=exception,
    )
    error.display_and_exit()


def is_github_api_permission_error(exception: Exception) -> bool:
    """Check if exception indicates GitHub API permission/authentication issues.

    Args:
        exception: Exception to check

    Returns:
        True if the exception indicates GitHub API permission issues
    """
    # A URL that could not be parsed is never a credentials problem, and
    # this check has to come first because the patterns below match
    # *substrings of the message*.  The undeclared-host error explains
    # that hosts are declared "so a mistyped URL cannot send your token
    # somewhere unintended"; the bare word "token" in that sentence was
    # enough to report it as "provide a GITHUB_TOKEN with the required
    # permissions", which sent the operator to regenerate a credential
    # that was never at fault and buried the message saying what to do.
    if isinstance(exception, UrlParseError):
        return False

    error_str = str(exception).lower()

    github_permission_patterns = [
        "resource not accessible by integration",
        "bad credentials",
        "requires authentication",
        "api rate limit exceeded",
        "forbidden",
        "unauthorized",
        "token",
        "permission",
        "401",
        "403",
        "404",  # Sometimes permissions manifest as 404
    ]

    return any(pattern in error_str for pattern in github_permission_patterns)


def is_network_error(exception: Exception) -> bool:
    """Check if exception indicates network connectivity issues.

    Args:
        exception: Exception to check

    Returns:
        True if the exception indicates network issues
    """
    error_str = str(exception).lower()

    network_patterns = [
        "network is unreachable",
        "connection refused",
        "connection timed out",
        "connection reset",
        "name resolution failed",
        "no route to host",
        "network unreachable",
        "timeout",
        "connection error",
        "dns",
    ]

    return any(pattern in error_str for pattern in network_patterns)


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception indicates GitHub rate limiting.

    Args:
        exception: Exception to check

    Returns:
        True if the exception indicates rate limiting
    """
    error_str = str(exception).lower()

    rate_limit_patterns = [
        "rate limit",
        "api rate limit exceeded",
        "secondary rate limit",
        "abuse detection",
        "too many requests",
        "429",
    ]

    return any(pattern in error_str for pattern in rate_limit_patterns)


def convert_git_error(git_error: Exception) -> DependamergeError:
    """Convert GitError to DependamergeError.

    Args:
        git_error: The GitError to convert

    Returns:
        DependamergeError with REPOSITORY_ERROR exit code
    """
    return DependamergeError(
        exit_code=ExitCode.REPOSITORY_ERROR,
        message=f"❌ Git operation failed: {git_error}",
        original_exception=git_error,
    )


def convert_github_api_error(api_error: Exception) -> DependamergeError:
    """Convert GitHub API errors to DependamergeError.

    Args:
        api_error: The GitHub API error to convert

    Returns:
        DependamergeError with appropriate exit code
    """
    if is_rate_limit_error(api_error):
        return DependamergeError(
            exit_code=ExitCode.GITHUB_API_ERROR,
            message="❌ GitHub API rate limit exceeded; please wait and try again",
            details="Consider using a token with higher rate limits",
            original_exception=api_error,
        )

    if is_github_api_permission_error(api_error):
        return DependamergeError(
            exit_code=ExitCode.GITHUB_API_ERROR,
            message="❌ GitHub API permission denied; check token permissions",
            details="Ensure GITHUB_TOKEN has read/write access to repositories",
            original_exception=api_error,
        )

    return DependamergeError(
        exit_code=ExitCode.GITHUB_API_ERROR,
        message=f"❌ GitHub API error: {api_error}",
        original_exception=api_error,
    )


def convert_network_error(network_error: Exception) -> DependamergeError:
    """Convert network errors to DependamergeError.

    Args:
        network_error: The network error to convert

    Returns:
        DependamergeError with NETWORK_ERROR exit code
    """
    return DependamergeError(
        exit_code=ExitCode.NETWORK_ERROR,
        message="❌ Network connectivity failed; check internet connection",
        details=str(network_error),
        original_exception=network_error,
    )


def map_exception_to_exit_code(exception: Exception) -> ExitCode:
    """Map common exception types to appropriate exit codes.

    Args:
        exception: Exception to categorize

    Returns:
        Appropriate ExitCode for the exception type
    """
    if is_github_api_permission_error(exception):
        return ExitCode.GITHUB_API_ERROR

    if is_network_error(exception):
        return ExitCode.NETWORK_ERROR

    if is_rate_limit_error(exception):
        return ExitCode.GITHUB_API_ERROR

    exception_type = type(exception).__name__

    if "GitError" in exception_type:
        return ExitCode.REPOSITORY_ERROR

    if any(name in exception_type for name in ["RateLimit", "GraphQL", "GitHub"]):
        return ExitCode.GITHUB_API_ERROR

    if "Configuration" in exception_type:
        return ExitCode.CONFIGURATION_ERROR

    if "Validation" in exception_type:
        return ExitCode.VALIDATION_ERROR

    # Default to general error
    return ExitCode.GENERAL_ERROR
