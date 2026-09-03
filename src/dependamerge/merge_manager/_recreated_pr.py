# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Waiting for a freshly recreated pull request to become ready.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ..github_async import PermissionError as GitHubPermissionError
from ..models import PullRequestInfo
from ..url_parser import pull_request_url_for
from ._base import _MergeManagerBase
from ._types import RecreateOutcome, RecreateResult

if TYPE_CHECKING:
    pass


class _RecreatedPullRequestMixin(_MergeManagerBase):
    """Polling a recreated pull request until its checks settle."""

    async def _wait_for_recreated_pr_checks(
        self,
        repo_owner: str,
        repo_name: str,
        new_number: int,
        pr_data: dict[str, Any],
        deadline: float | None = None,
    ) -> RecreateResult:
        """
        Wait for the recreated PR's status checks to complete.

        Auto-merge is armed on the replacement **before** this poll
        starts, so it can complete between iterations.  The wait
        therefore distinguishes the terminal states rather than only
        "mergeable" versus "keep polling": ready, already merged,
        abandoned (closed unmerged, or conflicted), and still pending.
        Reporting a merged replacement as a timeout would under-report a
        success, which is the most damaging direction to be wrong in.

        Giving up is **not** the same as finding nothing.  Once
        auto-merge is armed the replacement is expected to merge on its
        own, so a ceiling, an exhausted poll budget or ``--max-wait 0``
        all yield ``PENDING`` carrying that PR, which the caller reports
        as ``AUTO_MERGE_PENDING``.  ``NONE`` is reserved for the cases
        where nothing was acted on --- including a replacement we could
        not arm, which will not merge by itself.

        Args:
            repo_owner: Repository owner.
            repo_name: Repository name.
            new_number: The PR number of the recreated pull request.
            pr_data: The initial PR data dict from the GitHub API.
            deadline: Monotonic deadline shared with the enclosing
                recreate poll.  Passed in rather than derived here so
                the recreate path spends **one** budget in total; a
                fresh one would let a single PR consume a second full
                ``merge_timeout`` on top of whatever the outer loop had
                already spent.

        Returns:
            A :class:`RecreateResult` naming the terminal state.
        """
        if not self._github_client:
            return RecreateResult.none()

        full_name = f"{repo_owner}/{repo_name}"
        html_url = pr_data.get(
            "html_url", pull_request_url_for(self.host, full_name, new_number)
        )

        self._pr_status(
            f"🔍 Found recreated PR, waiting for checks: {html_url}",
            level="info",
        )

        replacement = self._recreated_pr_stub(
            repo_owner, repo_name, new_number, pr_data
        )
        armed = await self._auto_merge_recreated_pr(repo_owner, repo_name, replacement)

        def _gave_up(reason: str) -> RecreateResult:
            """Stop waiting without claiming the replacement failed."""
            if not armed:
                # Nothing will happen on its own, so there is no
                # in-flight merge to report as pending.
                self.log.warning(
                    "Stopped waiting on recreated PR %s#%s (%s); "
                    "auto-merge was not enabled",
                    full_name,
                    new_number,
                    reason,
                )
                return RecreateResult.none()
            self._pr_status(
                f"⏳ Recreated PR left to auto-merge: {html_url} [{reason}]",
                level="info",
            )
            return RecreateResult(RecreateOutcome.PENDING, replacement)

        # ``--max-wait 0`` promises never to block.  Auto-merge has been
        # armed above, so GitHub still completes the merge later --- the
        # same fire-and-forget shape the other waits use.
        if self._no_wait:
            return _gave_up("--max-wait 0")

        # Poll for the new PR to become mergeable
        loop = asyncio.get_running_loop()
        max_check_polls = self._merge_poll_max_attempts
        for check_attempt in range(max_check_polls):
            if deadline is None:
                await asyncio.sleep(self._merge_recheck_interval)
            else:
                # Clamped to the remaining budget so a replacement found
                # near ``--max-wait`` cannot extend the run past it by a
                # further interval.
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return _gave_up("wait ceiling reached")
                await asyncio.sleep(min(self._merge_recheck_interval, remaining))
            outcome = await self._poll_recreated_pr(
                full_name, new_number, html_url, check_attempt
            )
            if outcome is not None:
                return outcome

        return _gave_up("timed out waiting for checks")

    def _recreated_pr_stub(
        self,
        repo_owner: str,
        repo_name: str,
        new_number: int,
        pr_data: dict[str, Any],
    ) -> PullRequestInfo:
        """Build a minimal ``PullRequestInfo`` for the replacement.

        The full object needs a files fetch this path does not need, so
        the stub carries what arming auto-merge and reporting an
        outcome require.  Shared by both, so a ``PENDING`` result names
        exactly the PR auto-merge was armed on.
        """
        full_name = f"{repo_owner}/{repo_name}"
        return PullRequestInfo(
            number=new_number,
            node_id=pr_data.get("node_id"),
            title=pr_data.get("title", ""),
            body=pr_data.get("body"),
            author=((pr_data.get("user") or {}).get("login", "")),
            head_sha=((pr_data.get("head") or {}).get("sha", "")),
            base_branch=((pr_data.get("base") or {}).get("ref", "")),
            head_branch=((pr_data.get("head") or {}).get("ref", "")),
            state="open",
            mergeable=None,
            mergeable_state=None,
            behind_by=None,
            files_changed=[],
            repository_full_name=full_name,
            html_url=pr_data.get(
                "html_url", pull_request_url_for(self.host, full_name, new_number)
            ),
        )

    async def _auto_merge_recreated_pr(
        self,
        repo_owner: str,
        repo_name: str,
        replacement: PullRequestInfo,
    ) -> bool:
        """Approve the replacement and arm auto-merge, reporting **both**.

        Arming auto-merge is a commitment to let GitHub finish the merge
        once branch protection is satisfied, so the head must already
        carry this run's approval --- otherwise, on a repository that
        requires reviews, auto-merge waits forever.  The approval in
        :meth:`_merge_recreated_pr` is reached only for ``READY``, so a
        replacement we stop waiting on has no other chance to get one.

        The approve-and-arm helper is deliberately **not** used here.
        It swallows non-permission approval failures and returns the
        *arming* result, so a ``True`` from it evidences only that
        auto-merge is active --- a distinction that does not matter
        where a real merge is attempted afterwards, but does here,
        because ``PENDING`` is a claim about what will happen without
        us.  Approving separately keeps the two signals apart.

        Note the approval signal is **whether it raised**, not what it
        returned.  ``_ensure_pr_approved`` returns ``True`` only when a
        *new* review was submitted and ``False`` when the head is
        already sufficiently approved --- by us or by anyone else ---
        so treating ``False`` as "not approved" would reject exactly
        the replacements that are ready to go.  Genuine failures arrive
        as exceptions.

        Returns True only when approval is satisfied **and** auto-merge
        is active; anything less cannot be reported as pending.  Typed
        permission errors propagate, as they do in the shared helper, so
        the caller's dedicated handler can report them.
        """
        if not replacement.node_id:
            return False

        pr_key = f"{repo_owner}/{repo_name}#{replacement.number}"
        try:
            return await self._approve_and_arm_replacement(
                repo_owner, repo_name, replacement
            )
        finally:
            # Discard where it was added, rather than after the caller's
            # outcome handling.  ``_ensure_pr_approved`` registers the
            # replacement, but ``_merge_single_pr_impl``'s cleanup knows
            # only about the *original* PR, so anything that leaves this
            # path early --- an exception or cancellation while arming,
            # polling or merging --- would strand the key.  A stale
            # entry makes a later run on a reused manager treat a new
            # head as already approved and skip approval recovery.
            #
            # Nothing in the recreate path reads the key: the ``READY``
            # merge approves via ``_approve_pr`` directly, and GitHub
            # itself de-duplicates a repeat approval.
            self._recently_approved.discard(pr_key)

    async def _approve_and_arm_replacement(
        self,
        repo_owner: str,
        repo_name: str,
        replacement: PullRequestInfo,
    ) -> bool:
        """Approve then arm, reporting whether both succeeded."""
        try:
            # No propagation delay: we are arming auto-merge, not
            # dispatching a merge, and GitHub re-checks branch
            # protection when the required checks finish.
            await self._ensure_pr_approved(
                replacement, repo_owner, repo_name, propagation_delay=False
            )
            approved = True
        except GitHubPermissionError:
            raise
        except Exception as exc:
            self.log.warning(
                "Could not approve recreated PR %s/%s#%s: %s",
                repo_owner,
                repo_name,
                replacement.number,
                exc,
            )
            approved = False

        armed = await self._enable_auto_merge_for_pr(replacement, repo_owner, repo_name)

        if armed and not approved:
            # Arming still helps --- the PR may merge if the repository
            # needs no review --- but we cannot promise it, so the
            # caller must not report it as pending.
            self.log.warning(
                "Auto-merge armed on %s/%s#%s but approval failed; "
                "not reporting it as pending.",
                repo_owner,
                repo_name,
                replacement.number,
            )

        return armed and approved
