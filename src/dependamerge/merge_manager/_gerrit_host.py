# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Resolution of the Gerrit host and credentials for a repository.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import os
import re

from ..github2gerrit_detector import (
    GitHub2GerritMapping,
    fetch_gitreview_from_github,
)
from ._base import _MergeManagerBase


class _GerritHostMixin(_MergeManagerBase):
    """Resolution of the Gerrit host and credentials for a repository."""

    async def _resolve_gerrit_host(
        self,
        mapping: GitHub2GerritMapping,
        repo_owner: str,
        repo_name: str,
    ) -> tuple[str | None, str | None]:
        """
        Determine the Gerrit host and base path for a GitHub2Gerrit PR.

        Resolution priority (highest first):

        1. ``.gitreview`` file in the repository (canonical source of truth)
        2. ``GERRIT_HOST`` / ``GERRIT_BASE_PATH`` environment variables
        3. Gerrit URL embedded in the mapping comment body
        4. Well-known host conventions (e.g. ``lfit`` → LF Gerrit)
        5. ``GERRIT_URL`` environment variable

        The ``.gitreview`` file is treated as definitive because every
        repository that uses GitHub2Gerrit is required to have one, and it
        records the exact Gerrit host, port, and project path.

        Args:
            mapping: The parsed GitHub2Gerrit mapping from the PR comment.
            repo_owner: Repository owner (org or user).
            repo_name: Repository name.

        Returns:
            Tuple of (host, base_path) or (None, None) if unresolvable.
        """
        # 1. .gitreview file — highest priority / source of truth
        if self._github_client is not None:
            gitreview_info = await fetch_gitreview_from_github(
                self._github_client, repo_owner, repo_name
            )
            if gitreview_info and gitreview_info.is_valid:
                self.log.info(
                    "Resolved Gerrit host from .gitreview in %s/%s: %s (base_path=%s)",
                    repo_owner,
                    repo_name,
                    gitreview_info.host,
                    gitreview_info.base_path,
                )
                return gitreview_info.host, gitreview_info.base_path

        # 2. Explicit environment variables
        env_host = os.getenv("GERRIT_HOST", "").strip()
        env_base_path = os.getenv("GERRIT_BASE_PATH", "").strip() or None
        if env_host:
            return env_host, env_base_path

        # 3. Gerrit URL embedded in the mapping comment body
        if mapping.raw_comment_body:
            gerrit_url_match = re.search(
                r"https?://([^/\s]+)(?:/([\w-]+))?/c/",
                mapping.raw_comment_body,
            )
            if gerrit_url_match:
                host = gerrit_url_match.group(1)
                base_path = (
                    gerrit_url_match.group(2) if gerrit_url_match.group(2) else None
                )
                return host, base_path

        # 4. Well-known LF Gerrit host
        if (
            mapping.pr_url and "github.com/lfit/" in mapping.pr_url
        ) or repo_owner == "lfit":
            return "gerrit.linuxfoundation.org", "infra"

        # 5. GERRIT_URL environment variable (catch-all)
        gerrit_url = os.getenv("GERRIT_URL", "").strip()
        if gerrit_url:
            url_match = re.match(r"https?://([^/]+)(?:/([\w-]+))?/?$", gerrit_url)
            if url_match:
                return url_match.group(1), url_match.group(2) if url_match.group(
                    2
                ) else None

        return None, None
