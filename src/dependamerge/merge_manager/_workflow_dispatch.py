# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Detection of required workflows GitHub never dispatched.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio

import dependamerge.merge_manager as _pkg

from ..models import PullRequestInfo
from ..rule_violations import workflow_name_fragments
from ._base import _MergeManagerBase


class _WorkflowDispatchMixin(_MergeManagerBase):
    """Detection of required workflows GitHub never dispatched."""

    async def _workflows_never_dispatched(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        error_text: str,
        deadline: float | None = None,
    ) -> list[str]:
        """Required workflows named in *error_text* that GitHub never started.

        A ruleset can require a workflow that never runs --- most often
        because the workflow lives in the org's ``.github`` repository and
        GitHub queued it but never dispatched it.  The PR then reports
        "Required workflows … are not satisfied" forever, and the wait
        loop spends its entire ``merge_timeout`` discovering that nothing
        will change.

        The answer is always computed against **this PR's head SHA**.  An
        earlier sibling's finding is not reused: absence is a fact about
        one commit, not about the repository, so a workflow missing from
        PR #29's head says nothing about PR #30's.  Reusing it would skip
        a wait that could have succeeded and report a failure instead ---
        and the lookup it would save is a single request, against a wait
        worth five minutes.

        Absence must also be observed **twice**, either side of a short
        delay.  One snapshot cannot distinguish "never dispatched" from
        "dispatched a moment ago and not yet visible"; concluding the
        former turns an ordinary dispatch delay into a terminal merge
        failure.  Two requests and a few seconds is a cheap premium
        against a five-minute wait, and the condition this detects
        persists for hours.

        Returns the names with no workflow run at all.  An empty list
        means either that every named workflow has started --- so waiting
        is worthwhile --- or that the lookup failed, which is deliberately
        indistinguishable here: on doubt, wait.
        """
        if self._github_client is None:
            return []
        # The *raw* fragments, duplicates intact: ``_unmatched_names``
        # rejoins spans of them against the runs that dispatched, and a
        # collapsed sequence cannot reconstruct a name like
        # ``Build, Build``.
        names = workflow_name_fragments(error_text)
        if not names:
            return []

        missing = await self._bounded_absent_runs(pr_info, owner, repo, names, deadline)
        if not missing:
            return []
        return await self._confirm_absent_workflow_runs(
            pr_info, owner, repo, missing, deadline
        )

    async def _bounded_absent_runs(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        names: list[str],
        deadline: float | None,
    ) -> list[str]:
        """:meth:`_absent_workflow_runs` inside the caller's budget.

        The lookup retries and paginates, so an unbounded call can run
        past a nearly exhausted ``max_wait`` while holding a worker
        slot.  Overrunning reads as unknown, like every other doubt on
        this path, so the caller waits rather than reporting a terminal
        failure it never actually established.
        """
        if deadline is None:
            return await self._absent_workflow_runs(pr_info, owner, repo, names)
        budget = deadline - asyncio.get_running_loop().time()
        if budget <= 0:
            return []
        try:
            return await asyncio.wait_for(
                self._absent_workflow_runs(pr_info, owner, repo, names), budget
            )
        except asyncio.TimeoutError:
            # Not the builtin ``TimeoutError``: the two are only the same
            # class from Python 3.11, and on 3.10 ``wait_for`` raises
            # ``asyncio.exceptions.TimeoutError``, so catching the builtin
            # would let it escape and fail the merge outright.
            self.log.debug(
                "Listing workflow runs for %s outlasted the remaining "
                "budget; treating as unknown and waiting",
                pr_info.html_url,
            )
            return []

    async def _confirm_absent_workflow_runs(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        missing: list[str],
        deadline: float | None,
    ) -> list[str]:
        """Re-observe *missing* after a pause, within the run's budget.

        A workflow dispatched moments ago may simply not be visible yet,
        so absence is observed twice.  Both halves of that cost must fit
        the caller's deadline --- the pause *and* the requests after it,
        which retry and paginate.  Budgeting only the pause would let a
        request start against an already-expired deadline and hold a
        worker slot past the run's ceiling.

        Whenever the two observations cannot fit, or the second outlasts
        what remains, the answer stays unknown and the caller waits: the
        safe direction, since calling a live workflow dead reports a
        terminal failure on a PR that would have merged.
        """
        delay = _pkg.UNDISPATCHED_CONFIRM_DELAY_SECONDS
        loop = asyncio.get_running_loop()
        if (
            deadline is not None
            and deadline - loop.time()
            < delay + _pkg.UNDISPATCHED_CONFIRM_LOOKUP_SECONDS
        ):
            self.log.debug(
                "Not enough budget to confirm undispatched workflows for "
                "%s; treating as unknown and waiting",
                pr_info.html_url,
            )
            return []
        async with _pkg.parked():
            await asyncio.sleep(delay)
        return await self._reobserve_absent_runs(
            pr_info, owner, repo, missing, deadline
        )

    async def _reobserve_absent_runs(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        missing: list[str],
        deadline: float | None,
    ) -> list[str]:
        """The second observation, valid only if the head has not moved.

        Dependabot force-pushes when it rebases, and the confirmation
        pause is long enough to straddle one.  Both observations would
        then describe a commit the PR has abandoned, and reporting
        "never dispatched" for an abandoned head --- while the live one
        may be running its workflows perfectly --- is the exact mistake
        this confirmation exists to prevent.

        The head is checked *after* the lookup rather than before, which
        costs the same one request but covers a wider window: a
        force-push during the lookup is caught as well as one during the
        pause.  It is also skipped entirely when nothing is missing,
        since that answer means "keep waiting" whatever the head is
        doing --- so the common case gets cheaper, not dearer.

        A moved head is treated as inconclusive rather than restarted
        on the new SHA: restarting could be chased indefinitely by a
        branch under active rebase, and one extra wait is far cheaper
        than one false terminal failure.
        """
        still_missing = await self._bounded_absent_runs(
            pr_info, owner, repo, missing, deadline
        )
        if not still_missing:
            return []
        if not await self._head_sha_unchanged(pr_info, owner, repo, deadline):
            self.log.debug(
                "Head of %s moved during confirmation; treating as unknown and waiting",
                pr_info.html_url,
            )
            return []
        return still_missing

    async def _head_sha_unchanged(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        deadline: float | None = None,
    ) -> bool:
        """Whether the PR still points at the commit already observed.

        An unreadable answer counts as *changed*: this gates a terminal
        verdict, so doubt must resolve towards waiting like every other
        ambiguity on this path.  That covers running out of budget too
        --- the lookup before this one may have consumed it all, and
        starting a request past the run's ceiling would hold the
        reacquired worker slot beyond ``max_wait``.
        """
        if self._github_client is None:
            return False
        budget: float | None = None
        if deadline is not None:
            budget = deadline - asyncio.get_running_loop().time()
            if budget <= 0:
                return False
        try:
            request = self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
            )
            # A timeout lands in the handler below as any other failure
            # would, which is the answer we want: unknown, so wait.
            refreshed = (
                await request
                if budget is None
                else await asyncio.wait_for(request, budget)
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.debug(
                "Could not re-read the head of %s: %s", pr_info.html_url, exc
            )
            return False
        if not isinstance(refreshed, dict):
            return False
        head = (refreshed.get("head") or {}).get("sha")
        return bool(head) and head == pr_info.head_sha

    async def _absent_workflow_runs(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        names: list[str],
    ) -> list[str]:
        """Which of *names* have no workflow run on the PR's head SHA.

        Returns an empty list when the answer cannot be established, so
        an unusable response reads as "wait" rather than "never runs".
        """
        if self._github_client is None:
            return []
        try:
            dispatched = await self._github_client.get_workflow_run_names_for_sha(
                owner, repo, pr_info.head_sha
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.debug(
                "Could not list workflow runs for %s: %s", pr_info.html_url, exc
            )
            return []
        if not dispatched:
            # No runs at all is ambiguous: it can mean the lookup
            # returned nothing useful.  Treat as unknown and wait.
            return []
        return self._unmatched_names(names, dispatched)

    @staticmethod
    def _unmatched_names(names: list[str], dispatched: set[str]) -> list[str]:
        """Which of *names* no dispatched run accounts for.

        GitHub joins the required names with a comma, and a workflow
        name may itself contain one, so ``'Build, Test, Lint'`` splits
        into three fragments that might be three workflows, or two
        (``Build, Test`` and ``Lint``), or one.  Nothing in the message
        says which, and this path reads an unmatched name as "never
        dispatched" --- so guessing wrongly reports a terminal failure
        against a workflow that ran perfectly.

        The observed runs settle it without another request: any
        *contiguous span* of fragments that rejoins into a run which
        dispatched accounts for all of them.  Every matching span is
        applied, including overlapping ones.  Committing to one and
        skipping its overlaps would let an arbitrary tie-break reject a
        partition that does explain the list: with runs ``A``, ``A, B``
        and ``B, C``, fragments ``A|B|C`` are wholly explained by
        ``A`` + ``B, C``, but claiming ``A, B`` first would leave ``C``
        looking undispatched.  A fragment any span can account for is
        therefore accounted for, which keeps every ambiguity on the
        waiting side.  A one-fragment span is the ordinary case, which
        keeps a plain two-workflow violation behaving exactly as before.

        Both spacings are tried because only the separator GitHub emits
        is known for certain.
        """
        total = len(names)
        accounted = [False] * total
        for width in range(1, total + 1):
            for start in range(total - width + 1):
                span = names[start : start + width]
                if any(sep.join(span) in dispatched for sep in (", ", ",")):
                    accounted[start : start + width] = [True] * width
        return [name for name, seen in zip(names, accounted, strict=True) if not seen]
