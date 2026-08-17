# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Confirmation and override SHA generation and validation.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from ..gerrit import (
    GerritChangeInfo,
)
from ..models import PullRequestInfo

if TYPE_CHECKING:
    pass


def _generate_override_sha(
    pr_info: PullRequestInfo, commit_message_first_line: str
) -> str:
    """
    Generate a SHA hash based on PR author info and commit message.

    Args:
        pr_info: Pull request information containing author details
        commit_message_first_line: First line of the commit message to use as salt

    Returns:
        First 16 hex characters of a SHA256 hash string.
    """
    # Create a string combining author info and commit message first line
    combined_data = f"{pr_info.author}:{commit_message_first_line.strip()}"

    # Generate SHA256 hash
    sha_hash = hashlib.sha256(combined_data.encode("utf-8")).hexdigest()

    return sha_hash[:16]


def _generate_gerrit_override_sha(change: GerritChangeInfo) -> str:
    """
    Generate a SHA hash based on Gerrit change owner and subject.

    Args:
        change: Gerrit change information containing owner and subject

    Returns:
        SHA256 hash string
    """
    combined_data = f"{change.owner.strip()}:{change.subject.strip()}"
    sha_hash = hashlib.sha256(combined_data.encode("utf-8")).hexdigest()
    return sha_hash[:16]


def _generate_gerrit_continue_sha(change: GerritChangeInfo) -> str:
    """
    Generate a SHA hash for continuing after a Gerrit preview.

    Mirrors _generate_continue_sha for the GitHub path: the value is
    derived from the source change so the confirmation string is unique
    per batch and cannot be replayed against a different change.

    Args:
        change: Source Gerrit change information

    Returns:
        First 16 hex characters of a SHA256 hash string.
    """
    combined_data = (
        f"continue:{change.project}#{change.number}:{change.subject.strip()}"
    )
    sha_hash = hashlib.sha256(combined_data.encode("utf-8")).hexdigest()
    return sha_hash[:16]


def _validate_override_sha(
    provided_sha: str, pr_info: PullRequestInfo, commit_message_first_line: str
) -> bool:
    """
    Validate that the provided SHA matches the expected one for this PR.

    Args:
        provided_sha: SHA provided by user via --override flag
        pr_info: Pull request information
        commit_message_first_line: First line of commit message

    Returns:
        True if SHA is valid, False otherwise
    """
    expected_sha = _generate_override_sha(pr_info, commit_message_first_line)
    return provided_sha == expected_sha


def _generate_continue_sha(
    pr_info: PullRequestInfo, commit_message_first_line: str
) -> str:
    """
    Generate a SHA hash for continuing after preview evaluation.

    Args:
        pr_info: Source pull request information
        commit_message_first_line: First line of the commit message

    Returns:
        SHA256 hash string for continuation
    """
    # Create a string combining source PR info for preview continuation
    combined_data = f"continue:{pr_info.repository_full_name}#{pr_info.number}:{commit_message_first_line.strip()}"

    # Generate SHA256 hash
    sha_hash = hashlib.sha256(combined_data.encode("utf-8")).hexdigest()

    return sha_hash[:16]
