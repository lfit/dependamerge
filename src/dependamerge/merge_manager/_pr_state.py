# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Re-reading pull request state around a retry.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from typing import Any

import dependamerge.merge_manager as _pkg

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._models import (
    _merged_from_payload,
)


class _PrStateMixin(_MergeManagerBase):
    """Re-reading pull request state around a retry."""

    async def _recheck_pr_before_retry(
        self, owner: str, repo: str, pr_info: PullRequestInfo, attempt: int
    ) -> bool | None:
        """Re-fetch PR state before a retry attempt.

        Returns True if the PR was already merged (skip retry), False if it
        was closed without merging (abort retry), or None to proceed.
        """
        if self._github_client is None:
            raise RuntimeError("GitHub client not initialized")
        try:
            current_pr_data = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
            )
            if isinstance(current_pr_data, dict):
                current_state = current_pr_data.get("state")
                # Shared derivation: prefer the explicit ``merged`` bool,
                # fall back to ``merged_at``, ``None`` when neither is
                # usable so an ambiguous closed PR proceeds rather than
                # aborting.  See ``_merged_from_payload``.
                current_merged: bool | None = _merged_from_payload(current_pr_data)

                if current_state == "closed" and current_merged:
                    self.log.info(
                        f"✅ PR {owner}/{repo}#{pr_info.number} was already merged, skipping retry"
                    )
                    return True
                elif current_state == "closed" and current_merged is False:
                    self.log.info(
                        f"⚠️ PR {owner}/{repo}#{pr_info.number} was closed without merging, aborting retry"
                    )
                    # This will be caught by the outer merge logic and formatted consistently
                    return False
        except asyncio.CancelledError:
            # Cancellation must propagate so an in-flight shutdown is not
            # swallowed by the broad handler below.
            raise
        except Exception as state_check_error:
            self.log.debug(
                f"Failed to check PR state before retry {attempt + 1}: {state_check_error}",
                exc_info=True,
            )
        return None

    async def _fetch_pr_state(
        self, owner: str, repo: str, number: int
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Read one PR's state, batched with any concurrent reads.

        Routes through :class:`~dependamerge.pr_poller.PullRequestStatePoller`
        so the wait loops' polling costs one GraphQL query per tick for the
        whole run rather than one REST request per parked PR --- the
        difference between ~6 and ~360 calls/minute at 60 parked PRs.  See
        ``docs/BULK_RUN_PERFORMANCE_AUDIT.md`` §2.1.

        Falls back to a direct REST read when no poller is configured,
        which is the case for a manager used outside its async context
        manager (notably unit tests).  The returned shape is identical
        either way.
        """
        if self._pr_poller is not None:
            return await self._pr_poller.fetch(owner, repo, number)
        if self._github_client is None:
            return None
        return await self._github_client.get(f"/repos/{owner}/{repo}/pulls/{number}")

    async def _refresh_pr_mergeable(
        self, owner: str, repo: str, pr_info: PullRequestInfo, pr_key: str
    ) -> None:
        """Best-effort refresh of ``pr_info`` mergeable fields from the API."""
        try:
            if self._github_client:
                refreshed = await self._fetch_pr_state(owner, repo, pr_info.number)
                if isinstance(refreshed, dict):
                    pr_info.mergeable = refreshed.get("mergeable")
                    pr_info.mergeable_state = refreshed.get("mergeable_state")
                    self.log.debug(
                        f"Refreshed {pr_key}: mergeable={pr_info.mergeable}, "
                        f"mergeable_state={pr_info.mergeable_state}"
                    )
        except asyncio.CancelledError:
            # Cancellation must propagate so an in-flight shutdown is not
            # swallowed by the broad handler below.
            raise
        except Exception as refresh_err:
            self.log.debug(
                f"Failed to refresh PR state for {pr_key}: {refresh_err}",
                exc_info=True,
            )

    async def _await_in_progress_merge(
        self, owner: str, repo: str, pr_info: PullRequestInfo, pr_key: str
    ) -> bool:
        """Watch a PR GitHub says it is already merging; report success.

        Returns ``True`` when the PR reaches a merged state within
        :data:`MERGE_IN_PROGRESS_TIMEOUT_SECONDS`, ``False`` otherwise
        (including when it closes unmerged, which the caller reports).

        The wait is parked, so it holds no concurrency slot.
        """
        if self._github_client is None:
            return False
        if self._no_wait:
            return False

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _pkg.MERGE_IN_PROGRESS_TIMEOUT_SECONDS
        if self._run_deadline is not None:
            deadline = min(deadline, self._run_deadline)

        self.log.info(
            "GitHub reports a merge already in progress for %s; "
            "waiting up to %.0fs for it to complete",
            pr_key,
            max(0.0, deadline - loop.time()),
        )
        self._track_pr_state(pr_info, "waiting")
        async with self._waiting_lock:
            self._waiting_prs[pr_key] = deadline
        try:
            async with _pkg.parked():
                first_poll = True
                while True:
                    interval = (
                        _pkg.MERGE_IN_PROGRESS_FIRST_POLL_SECONDS
                        if first_poll
                        else _pkg.MERGE_IN_PROGRESS_POLL_SECONDS
                    )
                    remaining = max(0.0, deadline - loop.time())
                    await asyncio.sleep(min(interval, remaining))
                    try:
                        refreshed = await self._fetch_pr_state(
                            owner, repo, pr_info.number
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        self.log.debug(
                            "Failed to poll in-progress merge for %s: %s", pr_key, exc
                        )
                        refreshed = None
                    if isinstance(refreshed, dict):
                        merged = _merged_from_payload(refreshed)
                        if merged:
                            self.log.info("Merge completed for %s", pr_key)
                            pr_info.state = "closed"
                            return True
                        if merged is False and refreshed.get("state") == "closed":
                            pr_info.state = "closed"
                            return False
                        # Unknown merged-ness: keep polling rather than
                        # concluding the merge did not happen.
                    # Checked *after* polling, so an already-expired
                    # deadline (a run-wide ``max_wait`` about to elapse)
                    # still gets exactly one confirmation rather than
                    # reporting a failure this method exists to prevent.
                    if loop.time() >= deadline:
                        break
                    first_poll = False
        finally:
            async with self._waiting_lock:
                self._waiting_prs.pop(pr_key, None)
            self._track_pr_state(pr_info, None)

        self.log.debug("In-progress merge for %s did not complete in time", pr_key)
        return False

    async def _blocked_pr_became_clean(
        self, owner: str, repo: str, pr_info: PullRequestInfo, pr_key: str
    ) -> bool:
        """Wait for post-approval propagation, refresh state, report clean.

        Returns True when the PR's ``mergeable_state`` became ``clean`` (so
        the merge should be retried); False otherwise.  Refresh failures are
        swallowed at debug level.
        """
        try:
            if self._github_client:
                if self._post_approval_delay <= 0:
                    retry_delay = 0.0
                else:
                    retry_delay = self._post_approval_delay + 2.0
                self.log.info(
                    f"Post-approval propagation retry for {pr_key}, "
                    f"waiting {retry_delay}s before re-checking…"
                )
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)
                refreshed = await self._github_client.get(
                    f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
                )
                if isinstance(refreshed, dict):
                    new_state = refreshed.get("mergeable_state")
                    new_mergeable = refreshed.get("mergeable")
                    self.log.info(
                        f"Refreshed {pr_key}: mergeable={new_mergeable}, "
                        f"mergeable_state={new_state}"
                    )
                    pr_info.mergeable = new_mergeable
                    pr_info.mergeable_state = new_state
                    if new_state == "clean":
                        # Approval has propagated — retry the merge
                        return True
        except asyncio.CancelledError:
            # Cancellation must propagate so an in-flight shutdown is not
            # swallowed by the broad handler below.
            raise
        except Exception as refresh_err:
            self.log.debug(
                f"Failed to refresh PR state for {pr_key}: {refresh_err}",
                exc_info=True,
            )
        return False
