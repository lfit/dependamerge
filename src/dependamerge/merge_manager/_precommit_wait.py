# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Reading the commit status, and waiting for pre-commit.ci to answer.

The two belong together: waiting *is* reading the same endpoint on a
loop, and both need the same view of what "the status" is --- every
context on the head commit, across every page.

Separated from the retrigger decision in :mod:`_precommit_ci`, which
asks whether a reading means the run has stalled and nudges it when it
has.  Nothing here decides anything; it only reports what GitHub says
and how long it is prepared to keep asking.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._precommit_status import precommit_outcome

#: Status contexts fetched per request.  The combined-status endpoint
#: defaults to 30, so a repository with a moderate integration set can
#: push ``pre-commit.ci - pr`` onto a second page --- where reading one
#: page misreads an errored run as one that never reported.
_STATUSES_PER_PAGE = 100

#: How many pages to read.  A cap keeps a pathological commit from
#: costing the run unbounded requests.
_MAX_STATUS_PAGES = 10


class _PrecommitWaitMixin(_MergeManagerBase):
    """Reading pre-commit.ci's status, and polling until it settles."""

    async def _combined_statuses(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Every status context on the PR's head commit, across pages.

        Returns ``(fetched, statuses)``.  ``fetched`` is False when the
        commit status could not be read at all, which suppresses the
        retrigger --- an unreadable commit is not evidence of a stall.

        Paginated because the endpoint defaults to 30 entries.  On a
        repository with more integrations than that, a single page can
        omit ``pre-commit.ci - pr`` entirely, and this path reads absence
        as "never reported": an errored run would then look missing,
        leaving an ``unknown`` PR unrepaired and a ``blocked`` one nudged
        without the incident timestamp that scopes the suppression.
        """
        if not self._github_client:
            return False, []
        statuses: list[dict[str, Any]] = []
        page = 1
        while page <= _MAX_STATUS_PAGES:
            try:
                data = await self._github_client.get(
                    f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}"
                    f"/status?per_page={_STATUSES_PER_PAGE}&page={page}"
                )
            except Exception as e:
                self.log.debug(
                    "Failed to fetch commit status for pre-commit.ci check on %s#%s "
                    "(sha=%s); skipping retrigger: %s",
                    pr_info.repository_full_name,
                    pr_info.number,
                    pr_info.head_sha,
                    e,
                )
                return False, []
            if not isinstance(data, dict):
                return False, []
            batch = [s for s in (data.get("statuses") or []) if isinstance(s, dict)]
            statuses.extend(batch)
            if len(batch) < _STATUSES_PER_PAGE:
                break
            page += 1
        return True, statuses

    async def _await_precommit_ci(
        self,
        repo_owner: str,
        repo_name: str,
        pr_info: PullRequestInfo,
        since: datetime | None = None,
    ) -> bool:
        """Poll the head commit until pre-commit.ci reports a result.

        pre-commit.ci can take up to five minutes to run and report
        back, so we need a generous timeout to avoid prematurely marking
        PRs as unmergeable when the check simply hasn't finished yet.
        The whole poll is a wait on an external service, so the worker's
        concurrency slot is released for its duration (``parked()``).

        The poll honours the run-wide ceiling ``--max-wait`` sets, in
        the same way as the auto-merge and required-workflow waits: a
        stale pre-commit status previously parked the worker for the
        full ``merge_timeout`` even under ``--max-wait 0``, which
        promises never to block.
        """
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        if self._no_wait:
            self.log.debug(
                "Not waiting for pre-commit.ci on %s#%s (--max-wait 0)",
                pr_info.repository_full_name,
                pr_info.number,
            )
            return False

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._merge_timeout
        if self._run_deadline is not None:
            deadline = min(deadline, self._run_deadline)

        max_polls = self._merge_poll_max_attempts
        async with _mm.parked():
            for attempt in range(max_polls):
                # Sleep no longer than the time remaining, matching
                # ``_wait_for_auto_merge`` and the required-workflow
                # wait.  Checking the deadline without clamping would
                # still overshoot the ceiling by up to a full interval.
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(self._merge_recheck_interval, remaining))
                outcome = await self._poll_precommit_status(
                    repo_owner, repo_name, pr_info, since=since
                )
                if outcome is not None:
                    return outcome

                if attempt == max_polls - 1:
                    self.log.debug(
                        f"Still waiting for pre-commit.ci on "
                        f"{pr_info.repository_full_name}#{pr_info.number} "
                        f"({(attempt + 1) * self._merge_recheck_interval:.0f}s elapsed)"
                    )

        self.log.warning(
            f"Timed out waiting for pre-commit.ci on "
            f"{pr_info.repository_full_name}#{pr_info.number}"
        )
        return False

    async def _poll_precommit_status(
        self,
        repo_owner: str,
        repo_name: str,
        pr_info: PullRequestInfo,
        since: datetime | None = None,
    ) -> bool | None:
        """Take one reading of pre-commit.ci, or None while still pending.

        *since* is the reading the retrigger was posted against; a
        terminal status no newer than it has not been replaced yet and
        reads as pending.
        """
        if not self._github_client:
            return None
        fetched, statuses = await self._combined_statuses(
            repo_owner, repo_name, pr_info
        )
        if not fetched:
            return None
        outcome = precommit_outcome({"statuses": statuses}, since)
        if outcome is True:
            self._pr_status(
                f"✅ pre-commit.ci passed: {pr_info.html_url}",
                level="info",
            )
            return True
        if outcome is False:
            self._pr_status(
                f"❌ pre-commit.ci failed: {pr_info.html_url}",
                level="warning",
            )
            return False
        return None
