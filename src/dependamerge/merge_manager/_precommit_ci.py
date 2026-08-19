# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Re-triggering a pre-commit.ci run that has hung in pending.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from typing import Any

import dependamerge.merge_manager as _pkg

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._constants import (
    PRECOMMIT_CI_STUCK_PENDING_SECONDS,
)


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
        if not self._github_client:
            return False

        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)
        precommit_context = "pre-commit.ci - pr"

        # 1. Check whether pre-commit.ci is a required status check
        try:
            required_checks = await self._github_client.get_required_status_checks(
                repo_owner, repo_name, pr_info.base_branch or "main"
            )
            required_contexts = [
                c.get("context", "") for c in required_checks if isinstance(c, dict)
            ]
            if precommit_context not in required_contexts:
                return False
        except Exception:
            return False

        # 2. Inspect the existing pre-commit.ci status.  Retrigger when
        #    it is missing entirely or has been ``pending`` past the
        #    stuck threshold; leave any other state (success / failure
        #    / error, or a pending run still within its normal window).
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
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
            return False

        precommit_status: dict[str, Any] | None = None
        if isinstance(status_data, dict):
            for s in status_data.get("statuses", []):
                if isinstance(s, dict) and s.get("context") == precommit_context:
                    precommit_status = s
                    break

        if precommit_status is not None:
            state = precommit_status.get("state")
            if state != "pending":
                # A reported, non-pending result (success / failure /
                # error) is not stale — nothing to retrigger.
                return False
            # Pending: only stuck once it has been pending longer than
            # the threshold.  Use ``updated_at`` (when pre-commit.ci
            # set the pending status), falling back to ``created_at``.
            raw_ts = precommit_status.get("updated_at") or precommit_status.get(
                "created_at"
            )
            pending_age: float | None = None
            if isinstance(raw_ts, str) and raw_ts:
                try:
                    ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                    pending_age = (now - ts).total_seconds()
                except (ValueError, TypeError):
                    # ``ValueError``: unparsable timestamp.
                    # ``TypeError``: a timestamp lacking tz info parses
                    # to a naive datetime, which cannot be subtracted
                    # from the tz-aware ``now``.  Either way, degrade to
                    # ``None`` (fail closed) rather than abort the run.
                    pending_age = None
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

        # 3. The run is stale (missing, or stuck pending) — check for
        # an existing trigger comment before posting a duplicate
        # (avoids spam if dependamerge runs repeatedly while the
        # status is still not progressing).
        try:
            comments = await self._github_client.get(
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
                        return False
        except Exception:
            # If we fail to list comments, continue and attempt to post the
            # trigger anyway.
            pass

        self._pr_status(
            f"🔄 Re-triggering pre-commit.ci: {pr_info.html_url}",
            level="info",
        )

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

        # 4. Poll for the status to appear (up to ~5 minutes)
        # pre-commit.ci can take up to five minutes to run and report back,
        # so we need a generous timeout to avoid prematurely marking PRs as
        # unmergeable when the check simply hasn't finished yet.  The
        # whole poll is a wait on an external service, so the worker's
        # concurrency slot is released for its duration (``parked()``).
        max_polls = self._merge_poll_max_attempts
        async with _pkg.parked():
            for attempt in range(max_polls):
                await asyncio.sleep(self._merge_recheck_interval)
                try:
                    status_data = await self._github_client.get(
                        f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}/status"
                    )
                    if isinstance(status_data, dict):
                        for s in status_data.get("statuses", []):
                            if not isinstance(s, dict):
                                continue
                            if s.get("context") != precommit_context:
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
