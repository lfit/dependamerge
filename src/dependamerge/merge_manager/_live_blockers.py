# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Naming the conditions that are blocking a merge, read live.

A merge rejection names the rules GitHub evaluated at the instant the
merge was attempted, and it names them as *prose*.  That prose is not a
reliable account of what is wrong now: it lists conditions that had not
finished as readily as ones that failed, and it can omit the condition
that is actually holding the merge.

Reading the state directly avoids both problems.  Two sources are
needed, because GitHub reports through two mechanisms that do not
overlap:

- **check runs**, which is where Actions workflows report;
- **commit status contexts**, which is where pre-commit.ci, DCO and
  other integrations report.

A view built from check runs alone cannot see ``pre-commit.ci - pr`` at
all, which is how a rejection came to list three workflows that had
passed while saying nothing about the status context that was failing.
"""

from __future__ import annotations

import asyncio

from ..check_runs import failing_check_names
from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _LiveBlockerMixin(_MergeManagerBase):
    """Deriving the blocking set from live check and status state."""

    async def _live_blocking_conditions(self, pr_info: PullRequestInfo) -> list[str]:
        """Conditions blocking *pr_info* right now, most certain first.

        Status contexts are filtered to those the base branch actually
        requires, so an advisory integration that reports a failure
        cannot be presented as the reason a merge was refused.  Check
        runs are not filtered: a required *workflow* declared by a
        ruleset never appears among the required status contexts, so
        filtering on that list would discard the very names a ruleset
        rejection is about.

        Best-effort throughout.  An empty list means "nothing could be
        established", not "nothing is wrong", so callers must keep
        whatever reason they already had rather than treat it as an
        all-clear.
        """
        if self._github_client is None:
            return []
        try:
            owner, repo = pr_info.repository_full_name.split("/", 1)
        except ValueError:
            return []

        required = await self._required_context_names(owner, repo, pr_info)
        contexts = await self._failing_context_names(owner, repo, pr_info)
        checks = await self._failing_check_run_names(owner, repo, pr_info)

        blocking = [name for name in contexts if name.strip().lower() in required]
        named = {name.strip().lower() for name in blocking}
        return [f"required status check: {name}" for name in blocking] + [
            f"failing check: {name}"
            for name in checks
            if name.strip().lower() not in named
        ]

    async def _required_context_names(
        self, owner: str, repo: str, pr_info: PullRequestInfo
    ) -> set[str]:
        """Lower-cased status contexts the PR's base branch requires."""
        if self._github_client is None:
            return set()
        try:
            required = await self._github_client.get_required_status_checks(
                owner, repo, pr_info.base_branch or "main"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.debug(
                "Could not read required checks for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return set()
        if not isinstance(required, list):
            return set()
        return {
            str(entry.get("context", "")).strip().lower()
            for entry in required
            if isinstance(entry, dict) and entry.get("context")
        }

    async def _failing_context_names(
        self, owner: str, repo: str, pr_info: PullRequestInfo
    ) -> list[str]:
        """Commit status contexts whose latest state is failing."""
        if self._github_client is None:
            return []
        try:
            contexts = await self._github_client.get_failing_status_contexts(
                owner, repo, pr_info.head_sha
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.debug(
                "Could not read status contexts for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return []
        if not isinstance(contexts, list):
            return []
        return [name for name in contexts if isinstance(name, str) and name.strip()]

    async def _failing_check_run_names(
        self, owner: str, repo: str, pr_info: PullRequestInfo
    ) -> list[str]:
        """Check names whose *latest* run did not succeed.

        Deduplicated through :func:`check_runs.failing_check_names`, so a
        cancelled run superseded by a successful one of the same name is
        not reported as a blocker.
        """
        if self._github_client is None:
            return []
        try:
            runs = await self._github_client.get_check_runs_for_ref(
                owner, repo, pr_info.head_sha
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.debug(
                "Could not read check runs for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return []
        if not isinstance(runs, list):
            return []
        return failing_check_names(runs)
