# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Entry point and failure confirmation for one pull request.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._models import (
    MergeResult,
    MergeStatus,
    _merged_from_payload,
)


class _SinglePrMixin(_MergeManagerBase):
    """Entry point and failure confirmation for one pull request."""

    async def _merge_single_pr(self, pr_info: PullRequestInfo) -> MergeResult:
        """Merge a single PR, then confirm any failure is real.

        Thin wrapper over :meth:`_merge_single_pr_impl`.  It exists
        because a reported failure is frequently not one: GitHub
        auto-merge routinely completes a PR moments after this tool
        stops waiting for it.  In the 503-PR run analysed in
        ``docs/BULK_RUN_PERFORMANCE_AUDIT.md``, **21 of the 34 reported
        failures had in fact merged**, most within two minutes of being
        reported.

        Wrapping rather than editing the end of ``_merge_single_pr_impl``
        is deliberate: that method has several early ``return`` paths
        (permission denied, already merged, conflict handling), and a
        check placed before its final ``return`` would miss them.
        """
        result = await self._merge_single_pr_impl(pr_info)
        return await self._confirm_failure(pr_info, result)

    async def _confirm_failure(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> MergeResult:
        """Re-read a failed PR once and correct the outcome if it landed.

        Costs a single GET, and only for PRs that are about to be
        reported as failures --- a rounding error against the run's
        total API budget, in exchange for not telling the user a merged
        PR failed.

        Best-effort by construction: any error here leaves the original
        result untouched, because the verification must never be able to
        turn a reportable failure into a crash.
        """
        if result.status != MergeStatus.FAILED:
            return result
        if self.preview_mode or self._github_client is None:
            return result
        if pr_info.repository_full_name in self._permission_failed_repos:
            # The token cannot act on this repository, so no merge was
            # ever dispatched and the PR cannot have landed.  Skipping
            # also preserves the point of the fast-fail path: one failed
            # repository must not cost an API call per remaining PR.
            return result

        try:
            owner, repo = pr_info.repository_full_name.split("/", 1)
        except ValueError:
            return result

        try:
            refreshed = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
            )
        except asyncio.CancelledError:
            # Cancellation must propagate; a shutdown in flight is not a
            # verification failure.
            raise
        except Exception as exc:
            self.log.debug(
                "Could not verify reported failure for %s: %s", pr_info.html_url, exc
            )
            return result

        if not isinstance(refreshed, dict):
            return result

        # Tri-state: ``None`` means the payload could not tell us.  Treat
        # it as unknown throughout, so an ambiguous response can neither
        # invent a merge nor assert the absence of one.
        merged = _merged_from_payload(refreshed)

        if merged:
            self.log.info(
                "Reported failure for %s was stale; the PR merged at %s",
                pr_info.html_url,
                refreshed.get("merged_at") or "an unknown time",
            )
            self._pr_status(f"✅ Merged: {pr_info.html_url}", level="debug")
            result.status = MergeStatus.MERGED
            # The recorded reason described a state that no longer holds.
            # Keep it as a note rather than an error so the summary does
            # not show a merged PR carrying a failure message.
            if result.error:
                result.warning = f"merged after being reported as: {result.error}"
                result.error = None
            pr_info.state = "closed"
            return result

        if merged is False and refreshed.get("state") == "closed":
            self.log.info(
                "Reported failure for %s was stale; the PR is closed unmerged",
                pr_info.html_url,
            )
            result.status = MergeStatus.CLOSED
            pr_info.state = "closed"
            return result

        # Either still open, or closed with merged-ness unknown.  Keep the
        # original failure: reporting CLOSED here would assert "did not
        # merge" from a value that never said so.
        return result
