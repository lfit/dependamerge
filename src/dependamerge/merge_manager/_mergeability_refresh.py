# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The bounded wait for GitHub to recompute mergeability.
"""

from __future__ import annotations

import asyncio

from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _MergeabilityRefreshMixin(_MergeManagerBase):
    """Waiting out GitHub's asynchronous mergeability computation."""

    async def _refresh_pr_mergeability(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> None:
        """Refresh ``pr_info`` with the PR's current live merge state.

        The batch of PRs is fetched once up front, so a worker may act
        on a snapshot taken seconds-to-minutes earlier.  In a
        repo-scoped run this is routinely wrong: merging one PR can
        immediately make a sibling PR ``dirty`` (the classic
        ``uv.lock`` / workflow-pin conflict) or ``behind``.  A
        concurrent ``dependamerge`` run elsewhere in the org can do the
        same.

        This method is the **post-failure** half of the conflict-
        detection pair, called from ``_merge_single_pr`` *after* a
        repo-scoped merge attempt returns falsy and always **outside**
        the per-repo dispatch lock.  It catches the case where a PR
        turned ``dirty`` *during* our own merge window (a sibling
        merged between the pre-dispatch check and the merge call) so a
        freshly-conflicted PR is routed to conflict recovery rather
        than reported as a generic merge failure.  The complementary
        **pre-dispatch** check is the single-GET ``_is_pr_dirty_now``,
        which runs *inside* the dispatch lock; the polling done here
        deliberately stays off that lock so GitHub's recompute window
        never serialises the whole repo batch.

        GitHub recomputes ``mergeable`` / ``mergeable_state``
        asynchronously after the base branch moves, reporting
        ``mergeable=None`` and ``mergeable_state="unknown"`` (or an
        empty string) in the gap — usually for a few seconds.  When we
        catch the PR in that window we poll up to
        :data:`MERGEABILITY_REFRESH_TIMEOUT_SECONDS` for a concrete
        value so the merge decision is made against real data.

        Mutates ``pr_info`` in place (``state``, ``mergeable``,
        ``mergeable_state``, ``head_sha``).  Best-effort: any API error
        leaves the existing snapshot untouched so the caller's
        downstream logic still runs.
        """
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        if not self._github_client:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _mm.MERGEABILITY_REFRESH_TIMEOUT_SECONDS
        # Poll cadence for the "still computing" window.  Kept short
        # (GitHub usually settles in ~5s) but never longer than the
        # configured recheck interval.
        poll_interval = min(2.0, self._merge_recheck_interval)

        while True:
            try:
                data = await self._fetch_pr_state(owner, repo, pr_info.number)
            except Exception as exc:
                self.log.debug(
                    "Mergeability refresh failed for %s/%s#%s: %s",
                    owner,
                    repo,
                    pr_info.number,
                    exc,
                )
                return

            if not isinstance(data, dict):
                return

            state = data.get("state")
            if isinstance(state, str) and state:
                pr_info.state = state
            head_sha = (data.get("head") or {}).get("sha")
            if head_sha:
                pr_info.head_sha = head_sha

            mergeable = data.get("mergeable")
            mergeable_state = data.get("mergeable_state")

            # A closed PR will never resolve to a concrete mergeable
            # value; record what we have and let the caller's
            # closed-PR handling take over.
            if state == "closed":
                pr_info.mergeable = mergeable
                pr_info.mergeable_state = mergeable_state
                return

            # GitHub signals "still computing" with a null ``mergeable``
            # and an ``unknown``/empty ``mergeable_state``.  Keep
            # polling until it settles or the deadline passes.
            still_computing = mergeable is None or mergeable_state in (
                None,
                "",
                "unknown",
            )
            if not still_computing:
                pr_info.mergeable = mergeable
                pr_info.mergeable_state = mergeable_state
                return

            now = loop.time()
            if now >= deadline:
                # Timed out waiting for GitHub to settle.  Record the
                # latest values we did get (even if still computing) so
                # downstream logic sees GitHub's current best answer
                # rather than the older snapshot.  Reaching here means we
                # are still in the recompute window, where GitHub signals
                # "still computing" with ``mergeable=None`` and a
                # ``mergeable_state`` of ``None``, ``""`` or
                # ``"unknown"``.  Normalise any of those to a concrete
                # ``"unknown"`` and always record it, so a stale concrete
                # state (e.g. ``clean``) is never left in place —
                # consistent with the ``still_computing`` check above,
                # which treats ``None``/``""``/``"unknown"`` alike.
                if mergeable is not None:
                    pr_info.mergeable = mergeable
                pr_info.mergeable_state = mergeable_state or "unknown"
                self.log.debug(
                    "Mergeability for %s/%s#%s still computing after %.0fs; "
                    "proceeding with mergeable=%s state=%s",
                    owner,
                    repo,
                    pr_info.number,
                    _mm.MERGEABILITY_REFRESH_TIMEOUT_SECONDS,
                    pr_info.mergeable,
                    pr_info.mergeable_state,
                )
                return

            await asyncio.sleep(min(poll_interval, deadline - now))
