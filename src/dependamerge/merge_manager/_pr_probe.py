# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Direct probes of a pull request's current state on GitHub.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio

import dependamerge.merge_manager as _pkg

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._models import (
    _merged_from_payload,
)


class _PrProbeMixin(_MergeManagerBase):
    """Direct probes of a pull request's current state on GitHub."""

    async def _is_pr_already_merged(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Return ``True`` if the PR has been merged externally.

        Called when the PR was already closed at fetch time to
        distinguish two outcomes:

        * The PR was merged while we were processing it (a
          concurrent ``dependamerge`` run at org scope, a human
          admin, or auto-merge landing mid-flight) — classify as
          ``SKIPPED`` because there is no remaining work.
        * The PR was closed without merging (superseded, no longer
          needed, or closed by a human) — callers classify as
          ``CLOSED``, which also needs no operator follow-up.

        Any API error during the recheck (network, rate limit,
        permission, unexpected payload) degrades to ``False`` so
        the caller falls back to its non-merged path.  The intent
        here is to upgrade the user experience for known benign
        races, not to mask genuine errors.
        """
        state, merged = await self._fetch_pr_state_now(pr_info, owner, repo)
        return state == "closed" and merged is True

    async def _fetch_pr_state_now(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> tuple[str | None, bool | None]:
        """Best-effort fetch of a PR's current ``(state, merged)``.

        Returns ``(None, None)`` on any API error or unexpected
        payload so callers can fall back to their existing paths.
        Used to distinguish merged-externally (SKIPPED) from
        closed-without-merge (CLOSED — e.g. dependabot decided the
        update is no longer needed after sibling merges advanced the
        base branch) after a merge attempt fails.
        """
        if not self._github_client:
            return None, None
        try:
            pr_data = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
            )
        except Exception as e:
            self.log.debug(
                "Failed to recheck %s/%s#%s state: %s",
                owner,
                repo,
                pr_info.number,
                e,
            )
            return None, None
        if not isinstance(pr_data, dict):
            return None, None
        state = pr_data.get("state")
        if not isinstance(state, str):
            return None, None
        # Shared derivation (see ``_merged_from_payload``): an
        # unrecoverable payload degrades this call to "unknown" rather
        # than asserting the PR did not merge.
        merged = _merged_from_payload(pr_data)
        if merged is None:
            return None, None
        return state, merged

    async def _is_pr_dirty_now(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Best-effort single GET: ``True`` only if the PR is *concretely* dirty.

        Called inside the per-repo dispatch lock, immediately before the
        merge is dispatched, to catch a PR that an earlier sibling merge
        in the same repo-scoped batch has turned ``dirty`` (the classic
        shared-``uv.lock`` conflict) since the one-shot fetch snapshot.
        Routing such a PR straight to conflict recovery avoids
        dispatching a doomed merge that 405s and then churns
        :meth:`_merge_pr_with_retry`'s retry loop against the stale
        ``clean`` snapshot (which misreads the 405 as a transient error
        on a mergeable PR and sleeps under the dispatch lock before
        re-fetching).

        Deliberately a *single* GET with no recompute poll — unlike
        :meth:`_refresh_pr_mergeability`, which polls GitHub's
        "still computing" window and therefore runs only *after* a
        failed merge, off the lock.  The dispatch lock is the one point
        serialised *and* ordered after a sibling merge, so polling here
        would serialise the whole repo batch.  We therefore act only on
        a *concrete* ``dirty``; a still-computing, closed, non-dirty, or
        errored result returns ``False`` so the merge attempt proceeds
        and the off-lock post-failure refresh settles anything that
        turns out to be a fresh conflict.

        Mutates ``pr_info`` (``mergeable``, ``mergeable_state``,
        ``head_sha``) only when it confirms a concrete ``dirty``, so the
        conflict handler and failure summary report accurate state.  The
        snapshot is otherwise left untouched — in particular a transient
        ``unknown`` never overwrites a concrete ``clean``, preserving the
        transient-405-on-``clean`` retry path in
        :meth:`_merge_pr_with_retry`.
        """
        if not self._github_client:
            return False
        try:
            data = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
            )
        except Exception as exc:
            self.log.debug(
                "Pre-dispatch dirty check failed for %s/%s#%s: %s",
                owner,
                repo,
                pr_info.number,
                exc,
            )
            return False
        if not isinstance(data, dict):
            return False
        # A closed PR is handled by the caller's closed-PR path, not as
        # a conflict.
        if data.get("state") == "closed":
            return False
        if data.get("mergeable_state") != "dirty":
            return False
        # Concrete conflict: record it so ``_handle_merge_conflict`` and
        # the failure summary act on accurate, current state.
        pr_info.mergeable = data.get("mergeable")
        pr_info.mergeable_state = "dirty"
        head_sha = (data.get("head") or {}).get("sha")
        if head_sha:
            pr_info.head_sha = head_sha
        return True

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
        if not self._github_client:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _pkg.MERGEABILITY_REFRESH_TIMEOUT_SECONDS
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
                    _pkg.MERGEABILITY_REFRESH_TIMEOUT_SECONDS,
                    pr_info.mergeable,
                    pr_info.mergeable_state,
                )
                return

            await asyncio.sleep(min(poll_interval, deadline - now))
