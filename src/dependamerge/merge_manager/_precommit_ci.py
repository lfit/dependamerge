# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Re-triggering a pre-commit.ci run that has hung in pending.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import dependamerge.merge_manager as _pkg

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._constants import (
    PRECOMMIT_CI_STUCK_PENDING_SECONDS,
)

if TYPE_CHECKING:
    from datetime import datetime

    from ..github_async import GitHubAsync

# The commit status context pre-commit.ci reports a pull request run
# under; the ``- ci`` variant covers pushes and never gates a PR.
_PRECOMMIT_CONTEXT = "pre-commit.ci - pr"


class _PrecommitCiMixin(_MergeManagerBase):
    """Re-triggering a pre-commit.ci run that has hung in pending."""

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
        client = self._github_client
        if not client:
            return False

        owner, repo = pr_info.repository_full_name.split("/", 1)

        if not await self._precommit_ci_is_required(client, owner, repo, pr_info):
            return False
        if not await self._precommit_run_is_stale(client, owner, repo, pr_info):
            return False
        if await self._precommit_trigger_already_posted(client, owner, repo, pr_info):
            return False
        if not await self._post_precommit_trigger(client, owner, repo, pr_info):
            return False
        return await self._await_precommit_completion(client, owner, repo, pr_info)

    async def _precommit_ci_is_required(
        self,
        client: GitHubAsync,
        repo_owner: str,
        repo_name: str,
        pr_info: PullRequestInfo,
    ) -> bool:
        """Report whether pre-commit.ci gates merges on the base branch.

        A retrigger is only worth its side effects when the context is a
        required status check; a run that cannot block the merge is not
        this method's problem.  Kept separate because a branch
        protection lookup that fails must be indistinguishable from "not
        required" — in both cases the caller leaves the PR untouched.
        """
        try:
            required_checks = await client.get_required_status_checks(
                repo_owner, repo_name, pr_info.base_branch or "main"
            )
            required_contexts = [
                c.get("context", "") for c in required_checks if isinstance(c, dict)
            ]
            return _PRECOMMIT_CONTEXT in required_contexts
        except Exception:
            return False

    async def _precommit_run_is_stale(
        self,
        client: GitHubAsync,
        repo_owner: str,
        repo_name: str,
        pr_info: PullRequestInfo,
    ) -> bool:
        """Report whether the pre-commit.ci status warrants a retrigger.

        Stale means the context is missing from the head commit
        entirely, or has sat in ``pending`` for longer than
        :data:`PRECOMMIT_CI_STUCK_PENDING_SECONDS`.  Any reported result
        (success / failure / error) and any pending run still inside
        that window are left to stand.  Isolating the judgement keeps
        the read-only detection legible apart from the side effects the
        caller goes on to perform.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        try:
            status_data = await client.get(
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
            return False

        precommit_status: dict[str, Any] | None = None
        if isinstance(status_data, dict):
            for s in status_data.get("statuses", []):
                if isinstance(s, dict) and s.get("context") == _PRECOMMIT_CONTEXT:
                    precommit_status = s
                    break

        if precommit_status is None:
            return True

        if precommit_status.get("state") != "pending":
            # A reported, non-pending result (success / failure /
            # error) is not stale — nothing to retrigger.
            return False

        pending_age = self._precommit_pending_age(precommit_status, now)
        if pending_age is None or pending_age < PRECOMMIT_CI_STUCK_PENDING_SECONDS:
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

    @staticmethod
    def _precommit_pending_age(
        precommit_status: dict[str, Any], now: datetime
    ) -> float | None:
        """Return how long a pending pre-commit.ci status has stood.

        ``updated_at`` records when pre-commit.ci set the pending state,
        with ``created_at`` as the fallback.  ``None`` means "no usable
        timestamp", which the caller reads as too little evidence to
        declare the run stuck.  Separate so that fail-closed rule — and
        the two distinct ways a timestamp can be unusable — has one
        home rather than being buried in the staleness branch.
        """
        from datetime import datetime

        raw_ts = precommit_status.get("updated_at") or precommit_status.get(
            "created_at"
        )
        if not isinstance(raw_ts, str) or not raw_ts:
            return None
        try:
            ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            return (now - ts).total_seconds()
        except (ValueError, TypeError):
            # ``ValueError``: unparsable timestamp.
            # ``TypeError``: a timestamp lacking tz info parses to a
            # naive datetime, which cannot be subtracted from the
            # tz-aware ``now``.  Either way, degrade to ``None`` (fail
            # closed) rather than abort the run.
            return None

    async def _precommit_trigger_already_posted(
        self,
        client: GitHubAsync,
        repo_owner: str,
        repo_name: str,
        pr_info: PullRequestInfo,
    ) -> bool:
        """Report whether a ``pre-commit.ci run`` comment is already there.

        dependamerge may reach a stuck PR repeatedly while the status
        still fails to progress, and without this guard every pass would
        add another identical comment.  Failing to list the comments is
        not fatal — the caller posts the trigger anyway — so the error
        path deliberately answers False.
        """
        try:
            comments = await client.get(
                f"/repos/{repo_owner}/{repo_name}/issues/{pr_info.number}/comments?per_page=100"
            )
            if isinstance(comments, list):
                for c in comments:
                    if not isinstance(c, dict):
                        continue
                    body = c.get("body")
                    if isinstance(body, str) and body.strip() == "pre-commit.ci run":
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
        self,
        client: GitHubAsync,
        repo_owner: str,
        repo_name: str,
        pr_info: PullRequestInfo,
    ) -> bool:
        """Post the ``pre-commit.ci run`` comment that starts a fresh run.

        The comment is the only supported way to retrigger pre-commit.ci
        — dependabot's ``recreate`` macro does not reach it.  Separated
        as the one step that writes to the PR, and it reports whether
        the write landed so the caller never waits for a run it failed
        to ask for.
        """
        self._pr_status(
            f"🔄 Re-triggering pre-commit.ci: {pr_info.html_url}",
            level="info",
        )

        try:
            await client.post_issue_comment(
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

    async def _await_precommit_completion(
        self,
        client: GitHubAsync,
        repo_owner: str,
        repo_name: str,
        pr_info: PullRequestInfo,
    ) -> bool:
        """Poll for the retriggered run to report (up to ~5 minutes).

        pre-commit.ci can take up to five minutes to run and report
        back, so we need a generous timeout to avoid prematurely marking
        PRs as unmergeable when the check simply hasn't finished yet.
        The whole poll is a wait on an external service, so the worker's
        concurrency slot is released for its duration (``parked()``).
        Held apart from the trigger because it is the only blocking
        phase, and the only one that gives the slot up.

        Returns:
            True once pre-commit.ci reports success; False on a reported
            failure or error, and on timeout.
        """
        max_polls = self._merge_poll_max_attempts
        async with _pkg.parked():
            for attempt in range(max_polls):
                await asyncio.sleep(self._merge_recheck_interval)
                outcome = await self._poll_precommit_status(
                    client, repo_owner, repo_name, pr_info
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
        client: GitHubAsync,
        repo_owner: str,
        repo_name: str,
        pr_info: PullRequestInfo,
    ) -> bool | None:
        """Read the pre-commit.ci status once, for a single poll pass.

        Answers True or False once pre-commit.ci reports a terminal
        state, and ``None`` while the run is still pending or the status
        could not be read — the caller keeps polling in both of those
        cases.  Split out so one transient API failure costs an attempt
        rather than ending the wait.
        """
        try:
            status_data = await client.get(
                f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}/status"
            )
            if isinstance(status_data, dict):
                for s in status_data.get("statuses", []):
                    if not isinstance(s, dict):
                        continue
                    if s.get("context") != _PRECOMMIT_CONTEXT:
                        continue
                    state = s.get("state")
                    if state == "success":
                        self._pr_status(
                            f"✅ pre-commit.ci passed: {pr_info.html_url}",
                            level="info",
                        )
                        return True
                    elif state in ("failure", "error"):
                        self._pr_status(
                            f"❌ pre-commit.ci failed: {pr_info.html_url}",
                            level="warning",
                        )
                        return False
                    # state == "pending" — keep polling
        except Exception as e:
            self.log.debug(
                "Failed to poll pre-commit.ci status for %s: %s",
                f"{pr_info.repository_full_name}#{pr_info.number}",
                e,
            )
        return None
