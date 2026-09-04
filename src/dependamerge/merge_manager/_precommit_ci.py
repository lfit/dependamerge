# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Re-triggering a stalled pre-commit.ci run.

Reading the status is :mod:`_precommit_status`; deciding whether that
reading means the run has stalled, and nudging it when it has, is here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..models import PullRequestInfo
from ._precommit_status import (
    PRECOMMIT_CONTEXT,
    find_precommit_status,
    has_precommit_trigger_comment,
    parse_timestamp,
    pending_age,
)
from ._precommit_wait import _PrecommitWaitMixin

#: Comments fetched per request when looking for an existing nudge.
_COMMENTS_PER_PAGE = 100

#: How many pages to read.  A cap keeps a pathological thread from
#: costing the run unbounded requests; a trigger comment beyond a
#: thousand is not worth the search.
_MAX_COMMENT_PAGES = 10


class _PrecommitCiMixin(_PrecommitWaitMixin):
    """Nudging pre-commit.ci when its check has gone stale."""

    async def _trigger_stale_precommit_ci(
        self, pr_info: PullRequestInfo, *, treat_missing_as_stuck: bool = True
    ) -> bool:
        """Detect and retrigger a stuck pre-commit.ci run by posting a comment.

        pre-commit.ci uses the commit status API and sometimes gets
        stuck — either never reporting a status at all, or leaving the
        ``pre-commit.ci - pr`` context in ``pending`` indefinitely.
        Either way the PR stays blocked when that context is a required
        status check.  Posting ``pre-commit.ci run`` triggers a fresh
        run.

        A run is treated as stuck when the status is missing entirely,
        when pre-commit.ci reported ``error`` (it could not complete a
        run, as distinct from the hooks failing), or when it has been
        ``pending`` for longer than
        :data:`PRECOMMIT_CI_STUCK_PENDING_SECONDS` (a slow-but-normal
        run within that window is left alone).

        Args:
            pr_info: Pull request information
            treat_missing_as_stuck: Whether a status pre-commit.ci has
                not reported at all counts as stuck.  True only when
                GitHub has settled the PR's mergeable state: an absent
                status is indistinguishable from one that has not
                propagated yet, so on a PR GitHub is still working out
                it means "too early to tell" rather than "stalled".

        Returns:
            True if a retrigger comment was posted and the status check
            subsequently completed, False otherwise.
        """
        if not self._github_client:
            return False

        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)

        # 1. Check whether pre-commit.ci is a required status check
        if not await self._precommit_ci_required(repo_owner, repo_name, pr_info):
            return False

        # 2. Inspect the existing pre-commit.ci status.  Retrigger when
        #    it is missing entirely, when pre-commit.ci reported an
        #    ``error``, or when it has been ``pending`` past the stuck
        #    threshold; leave a reported ``success`` / ``failure``, and
        #    a pending run still within its normal window.
        now = datetime.now(timezone.utc)
        fetched, precommit_status = await self._precommit_status(
            repo_owner, repo_name, pr_info
        )
        if not fetched:
            return False
        reported_at = parse_timestamp(
            (precommit_status or {}).get("updated_at")
            or (precommit_status or {}).get("created_at")
        )
        if precommit_status is None:
            if not treat_missing_as_stuck:
                self.log.debug(
                    "pre-commit.ci has not reported on %s#%s and GitHub has "
                    "not settled the PR yet; too early to call it stuck.",
                    pr_info.repository_full_name,
                    pr_info.number,
                )
                return False
        elif not self._precommit_run_is_stuck(precommit_status, now, pr_info):
            return False

        # 3. The run is stale (missing, errored, or stuck pending) ---
        # check for an existing trigger comment before posting a
        # duplicate (avoids spam if dependamerge runs repeatedly while
        # the status is still not progressing).  Scoped to the current
        # incident where the status carries a timestamp, so a nudge for
        # an episode that has since resolved cannot silence this one.
        if await self._precommit_already_triggered(
            repo_owner,
            repo_name,
            pr_info,
            since=reported_at,
        ):
            return False

        self._pr_status(
            f"🔄 Re-triggering pre-commit.ci: {pr_info.html_url}",
            level="info",
        )

        if not await self._post_precommit_trigger(repo_owner, repo_name, pr_info):
            return False

        # 4. Poll for the status to appear (up to ~5 minutes).  The
        # reading that prompted the nudge is passed along so the poll
        # cannot mistake it for the answer: an ``error`` sits on the
        # commit until pre-commit.ci replaces it, and the first poll
        # would otherwise return that same error and end the wait on the
        # very failure being retried.
        return await self._await_precommit_ci(
            repo_owner, repo_name, pr_info, since=reported_at
        )

    async def _precommit_ci_required(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> bool:
        """Report whether pre-commit.ci blocks merges on the base branch.

        A failure to read the required checks counts as "not required":
        without evidence that the context blocks the merge there is
        nothing worth retriggering.
        """
        if not self._github_client:
            return False
        try:
            required_checks = await self._github_client.get_required_status_checks(
                repo_owner, repo_name, pr_info.base_branch or "main"
            )
            required_contexts = [
                c.get("context", "") for c in required_checks if isinstance(c, dict)
            ]
            if PRECOMMIT_CONTEXT not in required_contexts:
                return False
        except Exception:
            return False
        return True

    async def _precommit_status(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> tuple[bool, dict[str, Any] | None]:
        """Read the pre-commit.ci status on the PR's head commit.

        Returns a ``(fetched, status)`` pair.  ``fetched`` is False when
        the commit status could not be read at all, which suppresses the
        retrigger; a True with a ``None`` status means pre-commit.ci has
        reported nothing yet.
        """
        fetched, statuses = await self._combined_statuses(
            repo_owner, repo_name, pr_info
        )
        if not fetched:
            return False, None
        return True, find_precommit_status({"statuses": statuses})

    def _precommit_run_is_stuck(
        self,
        precommit_status: dict[str, Any],
        now: datetime,
        pr_info: PullRequestInfo,
    ) -> bool:
        """Judge a reported pre-commit.ci status as stuck or healthy."""
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        state = precommit_status.get("state")
        if state == "error":
            # The two terminal states mean different things, the way the
            # commit status API intends: ``failure`` is the hooks
            # reporting a genuine problem with the change, ``error`` is
            # pre-commit.ci failing to complete a run at all --- an
            # upload that 5xx'd, a container that died.  The first
            # reports the same result however often it is re-run; the
            # second routinely clears on the next attempt, and until it
            # does the PR stays blocked on a verdict nobody reached.
            #
            # Age is not consulted.  A terminal state does not become
            # more stuck with time, and there is nothing left to await.
            self.log.info(
                "pre-commit.ci on %s#%s reported an infrastructure error; "
                "treating as stuck.",
                pr_info.repository_full_name,
                pr_info.number,
            )
            return True
        if state != "pending":
            # ``success`` or ``failure`` is an answer, not a stall.
            # Re-running a genuine hook failure would post a comment and
            # then wait five minutes to be told the same thing.
            return False
        # Pending: only stuck once it has been pending longer than
        # the threshold.
        age = pending_age(precommit_status, now)
        if age is None or age < _mm.PRECOMMIT_CI_STUCK_PENDING_SECONDS:
            # Still within the normal window (or no timestamp to
            # judge by) --- leave the run to finish.
            return False
        self.log.info(
            "pre-commit.ci on %s#%s pending for %.0fs; treating as stuck.",
            pr_info.repository_full_name,
            pr_info.number,
            age,
        )
        return True

    async def _precommit_already_triggered(
        self,
        repo_owner: str,
        repo_name: str,
        pr_info: PullRequestInfo,
        since: datetime | None = None,
    ) -> bool:
        """Report whether this incident has already been nudged.

        *since* is the moment the status being reacted to was reported.
        A trigger comment older than that belongs to an earlier episode
        --- pre-commit.ci can error again on a later head after an
        intervening success --- and must not suppress a fresh nudge.
        Without a timestamp to compare against, any comment suppresses,
        which is the previous behaviour and the safe direction for a
        status that has never reported at all.

        Every page of comments is read.  A single page would let a PR
        with a long discussion hide the nudge already posted for this
        incident, and then post a second one --- the one guarantee this
        check exists to make.
        """
        if not self._github_client:
            return False
        page = 1
        while page <= _MAX_COMMENT_PAGES:
            try:
                comments = await self._github_client.get(
                    f"/repos/{repo_owner}/{repo_name}/issues/{pr_info.number}"
                    f"/comments?per_page={_COMMENTS_PER_PAGE}&page={page}"
                )
            except Exception:
                # If we fail to list comments, continue and attempt to
                # post the trigger anyway.
                return False
            if not isinstance(comments, list) or not comments:
                return False
            if has_precommit_trigger_comment(comments, since):
                self.log.info(
                    "Found existing pre-commit.ci trigger comment on "
                    f"{pr_info.repository_full_name}#{pr_info.number}; "
                    "skipping duplicate comment."
                )
                return True
            if len(comments) < _COMMENTS_PER_PAGE:
                return False
            page += 1
        return False

    async def _post_precommit_trigger(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> bool:
        """Post ``pre-commit.ci run``, reporting whether it was accepted."""
        if not self._github_client:
            return False
        try:
            await self._github_client.post_issue_comment(
                repo_owner, repo_name, pr_info.number, "pre-commit.ci run"
            )
            self._record_retrigger()
        except Exception as e:
            self.log.warning(
                f"Failed to post pre-commit.ci trigger comment on "
                f"{pr_info.repository_full_name}#{pr_info.number}: {e}"
            )
            return False
        return True
