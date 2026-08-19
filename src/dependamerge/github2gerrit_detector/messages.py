# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Gerrit URL and GitHub comment text built from a GitHub2Gerrit mapping.

These helpers turn a parsed
:class:`~dependamerge.github2gerrit_detector.models.GitHub2GerritMapping`
into user-facing output: the Gerrit change URL to link to, the comment body
posted on the GitHub PR after submission, and the short skip reason shown in
logs and the CLI.
"""

from __future__ import annotations

from .models import GitHub2GerritMapping


def build_gerrit_change_url_from_mapping(
    mapping: GitHub2GerritMapping,
    gerrit_host: str,
    gerrit_base_path: str | None = None,
) -> str:
    """
    Build a Gerrit web change URL from mapping metadata.

    This constructs a URL suitable for posting as a comment on the GitHub PR
    after the Gerrit change has been submitted.  URL construction is delegated
    to :class:`~dependamerge.gerrit.urls.GerritUrlBuilder` to ensure the
    base path is handled consistently.

    Args:
        mapping: The parsed mapping containing Change-IDs and topic.
        gerrit_host: Gerrit server hostname.
        gerrit_base_path: Optional base path (e.g., ``"infra"``).

    Returns:
        A Gerrit change URL string.  If the exact change number is not
        available in the mapping, returns a search URL using the Change-ID.
    """
    from dependamerge.gerrit.urls import GerritUrlBuilder

    builder = GerritUrlBuilder(
        host=gerrit_host, base_path=gerrit_base_path, auto_discover=False
    )

    # Use the primary Change-ID for the search URL
    change_id = mapping.primary_change_id
    if change_id:
        return builder.web_url(f"q/{change_id}")
    return builder.web_url()


def build_gerrit_submission_comment(
    mapping: GitHub2GerritMapping,
    gerrit_url: str | None = None,
) -> str:
    """
    Build the GitHub PR comment body to post after submitting in Gerrit.

    This follows the comment conventions established by github2gerrit-action
    for consistency.

    Args:
        mapping: The parsed mapping.
        gerrit_url: Optional Gerrit change URL to include.

    Returns:
        Formatted comment body.
    """
    lines = [
        "**Automated PR Closure**",
        "",
        "This pull request has been automatically closed by dependamerge.",
        "",
    ]

    if gerrit_url:
        lines.extend(
            [
                "The corresponding Gerrit change has been reviewed (+2) "
                + "and submitted ✅",
                "",
                f"Gerrit change URL: {gerrit_url}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The corresponding Gerrit change has been reviewed (+2) "
                + "and submitted ✅",
                "",
            ]
        )

    lines.extend(
        [
            "The changes from this PR are now part of the main codebase "
            + "in Gerrit.",
            "",
            "---",
            "*This is an automated action performed by dependamerge "
            + "(GitHub2Gerrit awareness).*",
        ]
    )

    return "\n".join(lines)


def build_gerrit_skip_message(
    mapping: GitHub2GerritMapping,
) -> str:
    """
    Build a human-readable skip reason for PRs with GitHub2Gerrit mappings.

    Args:
        mapping: The parsed mapping.

    Returns:
        A short descriptive string for log/UI output.
    """
    change_id_short = (
        mapping.primary_change_id[:12] if mapping.primary_change_id else "unknown"
    )
    return f"GitHub2Gerrit PR (topic: {mapping.topic}, Change-Id: {change_id_short}...)"
