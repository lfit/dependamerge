# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The rebase call and its conflict parsing behind ``GerritService``.

:class:`_GerritRebaseMixin` carries the Gerrit rebase endpoint call and
the defensive parser that pulls conflicting file names out of the HTTP
409 body Gerrit returns when the rebase cannot be applied cleanly.

It lives here rather than in ``dependamerge.gerrit.service`` purely to
keep that module reviewable.  Nothing in here references
``create_url_builder`` or ``build_client``: those names are only
resolved in ``service``'s own namespace, so that patching them there
stays effective.  Every attribute this mixin reads is established by
``GerritService.__init__``.  The logger name is unchanged so records
keep reporting as ``dependamerge.gerrit.service``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import logging
from typing import Any

from dependamerge.gerrit.client import GerritRestClient, GerritRestError

log = logging.getLogger("dependamerge.gerrit.service")


class _GerritRebaseMixin:
    """Rebase behaviour shared into ``GerritService``."""

    # Established by GerritService.__init__.
    _client: GerritRestClient

    def rebase_change(
        self,
        change_number: int,
        base: str | None = None,
    ) -> dict[str, Any]:
        """
        Attempt to rebase a change onto the target branch.

        This calls the Gerrit rebase endpoint which will:
        - Succeed if the change can be cleanly rebased
        - Return HTTP 409 with conflict details if there are merge conflicts

        Args:
            change_number: The Gerrit change number.
            base: Optional base revision to rebase onto. If None, rebases
                 onto the target branch HEAD.

        Returns:
            A dict with rebase result:
            - success: bool - whether rebase succeeded
            - change_info: dict | None - updated change info if successful
            - conflict: bool - whether there was a merge conflict
            - conflicting_files: list[str] - list of files with conflicts
            - error: str | None - error message if failed

        Note:
            Unlike GitHub's update_branch, Gerrit's rebase creates a new
            patchset with the rebased content if successful.
        """
        endpoint = f"/changes/{change_number}/rebase"
        log.debug("Attempting rebase: %s", endpoint)

        data: dict[str, Any] = {}
        if base:
            data["base"] = base

        try:
            result = self._client.post(endpoint, data=data if data else None)
            log.info("Successfully rebased change %d", change_number)
            return {
                "success": True,
                "change_info": result,
                "conflict": False,
                "conflicting_files": [],
                "error": None,
            }
        except GerritRestError as exc:
            # HTTP 409 indicates a merge conflict
            if exc.status_code == 409:
                conflicting_files = self._parse_conflict_files(exc.response_body or "")
                log.warning(
                    "Rebase failed for change %d: merge conflict in %s",
                    change_number,
                    conflicting_files,
                )
                return {
                    "success": False,
                    "change_info": None,
                    "conflict": True,
                    "conflicting_files": conflicting_files,
                    "error": exc.response_body or "Merge conflict during rebase",
                }
            # Other errors
            log.error("Rebase failed for change %d: %s", change_number, exc)
            return {
                "success": False,
                "change_info": None,
                "conflict": False,
                "conflicting_files": [],
                "error": str(exc),
            }

    def _parse_conflict_files(self, response_body: str) -> list[str]:
        """
        Parse conflicting file names from Gerrit's 409 response.

        The response format is typically:
        "The change could not be rebased due to a conflict during merge.

        merge conflict(s):
        path/to/file1.txt
        path/to/file2.txt"

        Returns:
            List of conflicting file paths. May be empty if parsing fails
            or the response format is unexpected.
        """
        files: list[str] = []
        if not response_body:
            # Nothing to parse; log at debug level to aid diagnostics without being noisy.
            log.debug(
                "Gerrit conflict response body is empty when parsing conflict files."
            )
            return files

        # Look for the "merge conflict(s):" marker
        lines = response_body.strip().splitlines()
        in_conflict_section = False
        marker_found = False

        for line in lines:
            line = line.strip()
            if not line:
                # Skip empty lines; if we're already in the conflict section,
                # treat a blank line as the end of that section.
                if in_conflict_section:
                    break
                continue
            if "merge conflict" in line.lower():
                in_conflict_section = True
                marker_found = True
                continue
            if in_conflict_section:
                # Each subsequent non-empty line is treated as a conflicting file.
                files.append(line)

        if not marker_found:
            # The response did not contain the expected marker; format may have changed.
            log.warning(
                "Failed to find 'merge conflict' marker in Gerrit response when "
                "parsing conflict files. Raw body: %r",
                response_body,
            )
        elif not files:
            # Marker was present but no files were parsed – response format may differ.
            log.warning(
                "No conflicting files parsed from Gerrit response after the "
                "'merge conflict' marker. Raw body: %r",
                response_body,
            )

        return files
