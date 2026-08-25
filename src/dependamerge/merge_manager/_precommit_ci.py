# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Re-triggering a stalled pre-commit.ci run.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase

_PRECOMMIT_CONTEXT = "pre-commit.ci - pr"


def _find_precommit_status(status_data: Any) -> dict[str, Any] | None:
    """Pick the pre-commit.ci entry out of a combined status response."""
    if not isinstance(status_data, dict):
        return None
    for s in status_data.get("statuses", []):
        if isinstance(s, dict) and s.get("context") == _PRECOMMIT_CONTEXT:
            return s
    return None


def _pending_age(precommit_status: dict[str, Any], now: datetime) -> float | None:
    """Seconds the status has been pending, or None when unknowable.

    Uses ``updated_at`` (when pre-commit.ci set the pending status),
    falling back to ``created_at``.
    """
    raw_ts = precommit_status.get("updated_at") or precommit_status.get("created_at")
    if not isinstance(raw_ts, str) or not raw_ts:
        return None
    try:
        ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        return (now - ts).total_seconds()
    except (ValueError, TypeError):
        # ``ValueError``: unparsable timestamp.
        # ``TypeError``: a timestamp lacking tz info parses to a naive
        # datetime, which cannot be subtracted from the tz-aware
        # ``now``.  Either way, degrade to ``None`` (fail closed)
        # rather than abort the run.
        return None


def _precommit_outcome(status_data: Any) -> bool | None:
    """Report a settled pre-commit.ci result, or None while pending.

    Scans past pending entries rather than stopping at the first match,
    so a later context with a final state is still honoured.
    """
    if not isinstance(status_data, dict):
        return None
    for s in status_data.get("statuses", []):
        if not isinstance(s, dict):
            continue
        if s.get("context") != _PRECOMMIT_CONTEXT:
            continue
        state = s.get("state")
        if state == "success":
            return True
        if state in ("failure", "error"):
            return False
        # state == "pending" — keep polling
    return None


def _has_precommit_trigger_comment(comments: Any) -> bool:
    """Report whether a ``pre-commit.ci run`` comment already exists."""
    if not isinstance(comments, list):
        return False
    for c in comments:
        if not isinstance(c, dict):
            continue
        body = c.get("body")
        if isinstance(body, str) and body.strip() == "pre-commit.ci run":
            return True
    return False


class _PrecommitCiMixin(_MergeManagerBase):
    """Nudging pre-commit.ci when its check has gone stale."""

    async def _trigger_stale_precommit_ci(self, pr_info: PullRequestInfo) -> bool:
        """Detect and retrigger a stuck pre-commit.ci run by posting a comment.

        pre-commit.ci uses the commit status API and sometimes gets
        stuck — either never reporting a status at all, or leaving the
        ``pre-commit.ci - pr`` context in ``pending`` indefinitely.
        Either way the PR stays blocked when that context is a required
        status check.  Posting ``pre-commit.ci run`` triggers a fresh
        run.

        A run is treated as stuck when the status is missing entirely,
        or when it has been ``pending`` for longer than
        :data:`PRECOMMIT_CI_STUCK_PENDING_SECONDS` (a slow-but-normal
        run within that window is left alone).

        Args:
            pr_info: Pull request information

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
        #    it is missing entirely or has been ``pending`` past the
        #    stuck threshold; leave any other state (success / failure
        #    / error, or a pending run still within its normal window).
        now = datetime.now(timezone.utc)
        fetched, precommit_status = await self._precommit_status(
            repo_owner, repo_name, pr_info
        )
        if not fetched:
            return False
        if precommit_status is not None and not self._precommit_run_is_stuck(
            precommit_status, now, pr_info
        ):
            return False

        # 3. The run is stale (missing, or stuck pending) — check for
        # an existing trigger comment before posting a duplicate
        # (avoids spam if dependamerge runs repeatedly while the
        # status is still not progressing).
        if await self._precommit_already_triggered(repo_owner, repo_name, pr_info):
            return False

        self._pr_status(
            f"🔄 Re-triggering pre-commit.ci: {pr_info.html_url}",
            level="info",
        )

        if not await self._post_precommit_trigger(repo_owner, repo_name, pr_info):
            return False

        # 4. Poll for the status to appear (up to ~5 minutes)
        return await self._await_precommit_ci(repo_owner, repo_name, pr_info)

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
            if _PRECOMMIT_CONTEXT not in required_contexts:
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
        if not self._github_client:
            return False, None
        try:
            status_data = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}/status"
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
            return False, None
        return True, _find_precommit_status(status_data)

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
        if state != "pending":
            # A reported, non-pending result (success / failure /
            # error) is not stale — nothing to retrigger.
            return False
        # Pending: only stuck once it has been pending longer than
        # the threshold.
        pending_age = _pending_age(precommit_status, now)
        if pending_age is None or pending_age < _mm.PRECOMMIT_CI_STUCK_PENDING_SECONDS:
            # Still within the normal window (or no timestamp to
            # judge by) — leave the run to finish.
            return False
        self.log.info(
            "pre-commit.ci on %s#%s pending for %.0fs; treating as stuck.",
            pr_info.repository_full_name,
            pr_info.number,
            pending_age,
        )
        return True

    async def _precommit_already_triggered(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> bool:
        """Report whether a trigger comment has already been posted."""
        if not self._github_client:
            return False
        try:
            comments = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/issues/{pr_info.number}/comments?per_page=100"
            )
            if _has_precommit_trigger_comment(comments):
                self.log.info(
                    "Found existing pre-commit.ci trigger comment on "
                    f"{pr_info.repository_full_name}#{pr_info.number}; "
                    "skipping duplicate comment."
                )
                return True
        except Exception:
            # If we fail to list comments, continue and attempt to post the
            # trigger anyway.
            pass
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

    async def _await_precommit_ci(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
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
                    repo_owner, repo_name, pr_info
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
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> bool | None:
        """Take one reading of pre-commit.ci, or None while still pending."""
        if not self._github_client:
            return None
        try:
            status_data = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}/status"
            )
            outcome = _precommit_outcome(status_data)
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
        except Exception as e:
            self.log.debug(
                "Failed to poll pre-commit.ci status for %s: %s",
                f"{pr_info.repository_full_name}#{pr_info.number}",
                e,
            )
        return None
