# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Re-reading a pull request's state from GitHub.

GitHub computes mergeability asynchronously and reports merges as
eventually consistent, so nearly every decision in the merge loop is
preceded by a deliberate refresh.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._types import _merged_from_payload


class _PullRequestStateMixin(_MergeManagerBase):
    """Refreshing the locally held view of a pull request."""

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
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        if self._github_client is None:
            return False
        if self._no_wait:
            return False

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _mm.MERGE_IN_PROGRESS_TIMEOUT_SECONDS
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
            async with _mm.parked():
                first_poll = True
                while True:
                    interval = (
                        _mm.MERGE_IN_PROGRESS_FIRST_POLL_SECONDS
                        if first_poll
                        else _mm.MERGE_IN_PROGRESS_POLL_SECONDS
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
