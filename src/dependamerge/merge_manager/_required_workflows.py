# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The wait for required workflows to appear and then retry.
"""

from __future__ import annotations

import asyncio

from ..github_async import PermissionError as GitHubPermissionError
from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _RequiredWorkflowWaitMixin(_MergeManagerBase):
    """Waiting for required workflows before retrying a merge."""

    async def _wait_for_required_workflows_and_retry(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Wait out still-executing required workflows, then retry the merge.

        Recovery for a merge rejected with 405 "Repository rule
        violations found … Required workflows '…' are not satisfied"
        (see :meth:`_merge_error_indicates_pending_workflows`): the
        ruleset-required workflows are still running, so the rejection
        is pending, not terminal.  Alternates between waiting for the
        PR to leave a blocked/unstable state and re-attempting the
        merge, all under one shared deadline of ``merge_timeout`` (the
        same shared-budget pattern conflict recovery uses for its
        rebase/checks phases) so the recovery can never double-spend
        the wait budget.  Each cycle dispatches a *single* merge call
        rather than :meth:`_merge_pr_with_retry`, whose internal
        retry ladder would multiply the number of merge attempts.

        The loop also covers the stale-``clean`` snapshot case (the
        REST ``mergeable_state`` can lag behind or be blind to ruleset
        workflows): when the wait exits immediately because the cached
        state already looks mergeable, the failed re-merge plus the
        interval sleep below degrade gracefully into a bounded
        merge-poll loop.

        Returns True when a retry merged the PR (or it merged on its
        own during the wait); False otherwise, leaving the last stored
        merge exception in place for the caller's failure reporting.
        """
        if self._github_client is None or self.preview_mode or self._no_wait:
            return False

        pr_key = f"{owner}/{repo}#{pr_info.number}"

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._merge_timeout
        if self._run_deadline is not None:
            deadline = min(deadline, self._run_deadline)

        merge_method = self._pr_merge_methods.get(
            f"{owner}/{repo}", self.default_merge_method
        )

        self._pr_status(
            f"⏳ Waiting: {pr_info.html_url} [required workflows still running]",
            level="info",
        )

        # Refresh so the wait loop starts from live state rather than
        # the (possibly stale-``clean``) snapshot that let the doomed
        # merge dispatch in the first place.
        try:
            refreshed = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
            )
            if isinstance(refreshed, dict):
                if refreshed.get("mergeable") is not None:
                    pr_info.mergeable = refreshed.get("mergeable")
                refreshed_state = refreshed.get("mergeable_state")
                if refreshed_state not in (None, "", "unknown"):
                    pr_info.mergeable_state = refreshed_state
                refreshed_head = (refreshed.get("head") or {}).get("sha")
                if refreshed_head:
                    pr_info.head_sha = refreshed_head
        except Exception as exc:
            self.log.debug(
                "Failed to refresh %s before required-workflows wait: %s",
                pr_key,
                exc,
            )

        # Only now, against the *live* head SHA, ask whether waiting can
        # help at all.  Checking before the refresh above would test a
        # snapshot that may predate a force-push, and a workflow absent
        # from the old head says nothing about the new one.
        #
        # The rejection itself is evidence about the commit it rejected,
        # so it is only usable while the PR still points at that commit.
        # After a force-push the new head's workflows may not have
        # dispatched *yet* --- exactly the transient this path must not
        # call terminal --- so a moved head means we simply wait and let
        # the retry produce a rejection for the commit we are on.
        last_error = ""
        if self._last_merge_exception_head.get(pr_key) == pr_info.head_sha:
            last_error = str(self._last_merge_exception.get(pr_key) or "")
        if last_error and await self._stop_for_undispatched_workflows(
            pr_info, owner, repo, last_error, deadline
        ):
            return False
        # The head this question has already been put for.  A rejection
        # arriving later for a *different* head is fresh evidence about
        # a commit never judged, and without asking again the loop would
        # wait out the whole timeout on a workflow that will never run.
        judged_head = pr_info.head_sha if last_error else None

        self._track_pr_state(pr_info, "waiting")
        try:
            while True:
                closed, merged_during = await self._wait_for_auto_merge(
                    pr_info,
                    owner,
                    repo,
                    continue_states=("blocked", "unstable"),
                    deadline=deadline,
                    measures_checks=True,
                )
                if closed:
                    return merged_during
                try:
                    merged = await self._github_client.merge_pull_request(
                        owner, repo, pr_info.number, merge_method
                    )
                except GitHubPermissionError:
                    raise
                except Exception as exc:
                    self._last_merge_exception[pr_key] = exc
                    self._last_merge_exception_head[pr_key] = pr_info.head_sha
                    if not self._merge_error_indicates_pending_workflows(str(exc)):
                        # The rejection reason changed (e.g. a workflow
                        # finished and *failed*) — terminal; let the
                        # caller classify and report it.
                        return False
                    merged = False
                    if pr_info.head_sha != judged_head:
                        # A head we have not judged --- either the
                        # refresh above found a force-push and skipped
                        # the question, or one landed during the wait.
                        # This rejection is evidence for the commit we
                        # are actually on, so ask now.
                        judged_head = pr_info.head_sha
                        if await self._stop_for_undispatched_workflows(
                            pr_info, owner, repo, str(exc), deadline
                        ):
                            return False
                if merged:
                    return True
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return False
                await asyncio.sleep(min(self._merge_recheck_interval, remaining))
        finally:
            self._track_pr_state(pr_info, None)
