# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The Gerrit-side steps a github2gerrit submission is built from.

Each step here talks to a Gerrit server through one of the factories the
package re-exports, and none of them needs to know about the GitHub pull
request the change mirrors.  :class:`_GerritSubmitMixin` composes them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..netrc import NetrcParseError
from ._base import _MergeManagerBase

# Annotation-only, so nothing is bound at run time that could shadow a
# substitution made on the package (see tests/test_patch_targets.py).
if TYPE_CHECKING:
    from ..gerrit import GerritChangeInfo, GerritSubmitResult
    from ..github2gerrit_detector import GitHub2GerritMapping
    from ..netrc import GerritCredentials


class _GerritSubmitStepsMixin(_MergeManagerBase):
    """The individual Gerrit calls a submission is assembled from."""

    def _resolve_gerrit_submit_credentials(
        self, gerrit_host: str, mapping: GitHub2GerritMapping
    ) -> GerritCredentials | None:
        """
        Resolve usable Gerrit credentials for ``gerrit_host``.

        Returns None, having warned, when nothing valid is available.
        """
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        try:
            credentials = _mm.resolve_gerrit_credentials(
                host=gerrit_host,
                use_netrc=not self.no_netrc,
                netrc_file=self.netrc_file,
            )
        except NetrcParseError as exc:
            self.log.warning("Error parsing .netrc for Gerrit: %s", exc)
            credentials = None

        if credentials is None or not credentials.is_valid:
            self.log.warning(
                "No Gerrit credentials found for %s. Cannot submit "
                "GitHub2Gerrit change (topic: %s).",
                gerrit_host,
                mapping.topic,
            )
            return None

        return credentials

    def _find_gerrit_change(
        self,
        gerrit_target: tuple[str, str | None],
        credentials: GerritCredentials,
        change_id: str,
    ) -> GerritChangeInfo | None:
        """
        Query Gerrit for the open change carrying ``change_id``.

        Returns None, having warned, when the query matches nothing.
        """
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        gerrit_host, gerrit_base_path = gerrit_target

        service = _mm.create_gerrit_service(
            host=gerrit_host,
            base_path=gerrit_base_path,
            username=credentials.username,
            password=credentials.password,
        )

        # Query Gerrit for the change using the primary Change-ID
        changes = service._query_changes(
            query=f"change:{change_id} status:open",
            limit=5,
            offset=0,
            options=[
                "CURRENT_REVISION",
                "LABELS",
                "DETAILED_LABELS",
                "SUBMIT_REQUIREMENTS",
            ],
        )

        if not changes:
            self.log.warning(
                "No open Gerrit change found for Change-Id %s on %s",
                change_id,
                gerrit_host,
            )
            return None

        # Use the first matching change
        gerrit_change = changes[0]
        self.log.info(
            "Found Gerrit change %s #%d for Change-Id %s",
            gerrit_change.project,
            gerrit_change.number,
            change_id,
        )
        return gerrit_change

    def _run_gerrit_submit(
        self,
        gerrit_target: tuple[str, str | None],
        credentials: GerritCredentials,
        gerrit_change: GerritChangeInfo,
    ) -> list[GerritSubmitResult]:
        """Apply +2 Code-Review to ``gerrit_change`` and submit it."""
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        gerrit_host, gerrit_base_path = gerrit_target

        submit_manager = _mm.create_submit_manager(
            host=gerrit_host,
            base_path=gerrit_base_path,
            username=credentials.username,
            password=credentials.password,
        )

        return submit_manager.submit_changes(
            [(gerrit_change, None)],
            review_labels={"Code-Review": 2},
            dry_run=self.preview_mode,
        )
