# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Handling of pull requests that cannot be merged as they stand.

Also the preview (dry-run) simulation, which answers the same
question without touching the repository.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..models import PullRequestInfo
from ._live_blockers import _LiveBlockerMixin
from ._types import MergeResult, MergeStatus, _merged_from_payload


class _NotMergeableMixin(_LiveBlockerMixin):
    """Reacting to a pull request that is not mergeable."""

    #: ``mergeable_state`` values that mean GitHub would accept a merge
    #: as things stand.  ``unstable`` qualifies because only *optional*
    #: checks are failing, which does not block a merge, and
    #: ``has_hooks`` because pre-receive hooks do not either.  Both are
    #: already treated as mergeable by ``_should_attempt_merge``, so
    #: reading them as blocking here would contradict the gate that
    #: decides whether to try in the first place.
    _MERGEABLE_NOW_STATES = frozenset({"clean", "has_hooks", "unstable"})

    async def _confirm_failure(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> MergeResult:
        """Re-read a failed PR once and correct the outcome if it moved on.

        Costs a single GET, and only for PRs that are about to be
        reported as failures --- a rounding error against the run's
        total API budget, in exchange for not telling the user a merged
        PR failed.

        The same payload settles all three corrections --- merged,
        closed unmerged, and a rejection whose stated cause has since
        cleared --- so the extra accuracy costs no extra request.

        Best-effort by construction: any error here leaves the original
        result untouched, because the verification must never be able to
        turn a reportable failure into a crash.
        """
        if result.status != MergeStatus.FAILED:
            return result
        if self.preview_mode or self._github_client is None:
            return result
        if pr_info.repository_full_name in self._permission_failed_repos:
            # The token cannot act on this repository, so no merge was
            # ever dispatched and the PR cannot have landed.  Skipping
            # also preserves the point of the fast-fail path: one failed
            # repository must not cost an API call per remaining PR.
            return result

        try:
            owner, repo = pr_info.repository_full_name.split("/", 1)
        except ValueError:
            return result

        try:
            refreshed = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
            )
        except asyncio.CancelledError:
            # Cancellation must propagate; a shutdown in flight is not a
            # verification failure.
            raise
        except Exception as exc:
            self.log.debug(
                "Could not verify reported failure for %s: %s", pr_info.html_url, exc
            )
            return result

        if not isinstance(refreshed, dict):
            return result

        # Tri-state: ``None`` means the payload could not tell us.  Treat
        # it as unknown throughout, so an ambiguous response can neither
        # invent a merge nor assert the absence of one.
        merged = _merged_from_payload(refreshed)

        if merged:
            self.log.info(
                "Reported failure for %s was stale; the PR merged at %s",
                pr_info.html_url,
                refreshed.get("merged_at") or "an unknown time",
            )
            self._pr_status(f"✅ Merged: {pr_info.html_url}", level="debug")
            result.status = MergeStatus.MERGED
            # The recorded reason described a state that no longer holds.
            # Keep it as a note rather than an error so the summary does
            # not show a merged PR carrying a failure message.
            if result.error:
                result.warning = f"merged after being reported as: {result.error}"
                result.error = None
            pr_info.state = "closed"
            return result

        if merged is False and refreshed.get("state") == "closed":
            self.log.info(
                "Reported failure for %s was stale; the PR is closed unmerged",
                pr_info.html_url,
            )
            result.status = MergeStatus.CLOSED
            pr_info.state = "closed"
            return result

        # Either still open, or closed with merged-ness unknown.
        # Reporting CLOSED here would assert "did not merge" from a value
        # that never said so, but an open PR may still have outlived the
        # reason recorded against it.
        return await self._settle_stale_failure(pr_info, result, refreshed)

    async def _settle_stale_failure(
        self,
        pr_info: PullRequestInfo,
        result: MergeResult,
        refreshed: dict[str, Any],
    ) -> MergeResult:
        """Withdraw a failure whose stated cause has stopped being true.

        The recorded reason describes what was unsatisfied at the instant
        the merge was refused, which routinely includes required checks
        that had merely not finished.  Nothing re-read that between the
        rejection and the report, so a pull request that went green
        moments later was still printed as failed, under a cause that no
        longer held --- and an operator sent to investigate it found
        nothing wrong.

        Reads ``mergeable_state`` from the payload :meth:`_confirm_failure`
        has already fetched, so settling this costs no further request.

        ``unknown`` stays a failure.  GitHub computes mergeability in the
        background and reports ``unknown`` until it has, so the value is
        an absence of evidence rather than evidence of a block --- but it
        is equally not evidence that the PR would merge, and only that
        would justify withdrawing a reported failure.

        A closed pull request never qualifies, whatever its mergeable
        state says.  :meth:`_confirm_failure` reaches here for one that
        is closed with merged-ness unknown --- a trimmed payload carrying
        neither ``merged`` nor ``merged_at`` --- and such a payload can
        still report ``clean``.  Telling an operator to re-run and merge
        a closed PR would be advice they cannot act on.

        Only a failure GitHub itself produced by *refusing the merge* is
        eligible.  The run reports ``FAILED`` for its own troubles too
        --- an unhandled exception, a rebase that did not complete --- and
        those say nothing about mergeability.  Rewriting one because the
        PR happens to read ``clean`` would bury the actionable error in a
        note the summary never prints, and advise a re-run that would
        fail identically.
        """
        if not result.merge_refused:
            return result
        if refreshed.get("state") == "closed":
            return result
        state = refreshed.get("mergeable_state")
        if not isinstance(state, str):
            return result
        if state not in self._MERGEABLE_NOW_STATES:
            return await self._name_the_live_blocker(pr_info, result, state, refreshed)
        if refreshed.get("mergeable") is not True:
            # Only affirmative mergeability withdraws a failure.  A
            # ``null`` means GitHub is still computing --- the same
            # absence of evidence that keeps ``unknown`` a failure, and
            # the reason ``_state_is_waitable`` waits unless it sees
            # ``True``.  A payload asserting both ``clean`` and ``False``
            # contradicts itself, and is refused by the same rule.
            return result

        self.log.info(
            "Reported failure for %s is stale; the PR is now %s and would merge",
            pr_info.html_url,
            state,
        )
        self._pr_status(
            f"⏱️ Unsettled: {pr_info.html_url} [{state}]",
            level="debug",
        )
        result.status = MergeStatus.UNSETTLED
        if result.error and result.warning is None:
            # Kept as a note, the way a stale reason is kept on a PR that
            # turned out to have merged: it describes a state that no
            # longer holds, so the summary must not show it as the cause.
            #
            # Only when nothing is noted yet.  The end-of-run pass calls
            # this a second time, by which point ``error`` may already be
            # a live reading --- and overwriting the note with that would
            # discard the rejection this is meant to preserve.
            result.warning = f"the merge was refused as: {result.error}"
        result.error = (
            f"not settled during the run; now {state} and expected to merge on a re-run"
        )
        return result

    async def _name_the_live_blocker(
        self,
        pr_info: PullRequestInfo,
        result: MergeResult,
        state: str,
        refreshed: dict[str, Any],
    ) -> MergeResult:
        """Replace a rejection's prose with what is blocking the PR now.

        The recorded reason is GitHub's own rejection message, preferred
        over state-based inference because it usually carries the
        actionable cause.  On a ruleset rejection it frequently does not:
        it lists the rules that were evaluated, including ones that had
        merely not finished, and it can omit the condition actually
        holding the merge.  python-workflows#84 was reported against
        three workflows that had all passed, while ``pre-commit.ci - pr``
        --- a status context, invisible to any check-runs-only view ---
        was the sole blocker.

        The head and base come from *refreshed* rather than from the
        snapshot taken before the merge was attempted.  A dependabot
        rebase moves the head, and this tool requests those rebases, so
        reading the snapshot's commit would report conditions belonging
        to a commit nobody is trying to merge.  The same reasoning keeps
        the head current during the wait (``_wait.py``).

        Only ``blocked`` is re-examined.  A conflicted, behind or draft
        PR is already described accurately by its state, so reading its
        checks would spend requests without adding anything.

        The rejection is kept as a note.  It records what GitHub said at
        the time, which is worth having when the live reading and the
        message disagree, but it is no longer offered as the cause.
        """
        if state != "blocked":
            return result
        head = (refreshed.get("head") or {}).get("sha")
        head_sha = head if isinstance(head, str) and head else pr_info.head_sha
        base = (refreshed.get("base") or {}).get("ref")
        base_branch = (
            base if isinstance(base, str) and base else (pr_info.base_branch or "main")
        )
        rejection = result.error or ""

        blocking, also_failing, complete = await self._live_blocking_conditions(
            pr_info,
            head_sha=head_sha,
            base_branch=base_branch,
            rejection=rejection,
        )
        reason = self._compose_blocker_reason(blocking, also_failing, complete)
        if reason is None:
            # Nothing could be established.  An empty reading is not an
            # all-clear, so the reason already recorded stands.
            return result
        if result.error and result.warning is None:
            # See :meth:`_settle_stale_failure`: the note records what
            # GitHub said, and the end-of-run pass must not replace it
            # with the live reading this pass is about to supersede.
            result.warning = f"the merge was refused as: {result.error}"
        result.error = reason
        return result

    @staticmethod
    def _compose_blocker_reason(
        blocking: list[str], also_failing: list[str], complete: bool
    ) -> str | None:
        """Word the live reading without over-claiming what it proves.

        Only conditions a rule shows to be required are presented as
        blocking.  A check that is merely failing is still worth naming
        --- it is very often the cause --- but calling it the blocker
        would repeat the mistake this whole change exists to correct,
        in the opposite direction.

        A proven blocker stands on its own evidence, so it is reported
        whether or not every probe answered.  An unproven one is not:
        with a probe missing, "these are failing" may be the visible
        half of a picture whose other half held the real cause, and
        replacing the rejection with it would trade a stale reason for a
        confidently wrong one.
        """
        if blocking and also_failing:
            return (
                "blocked by "
                + "; ".join(blocking)
                + "; also failing: "
                + ", ".join(also_failing)
            )
        if blocking:
            return "blocked by " + "; ".join(blocking)
        if also_failing and complete:
            return "failing checks: " + ", ".join(also_failing)
        return None

    async def _handle_not_mergeable_pr(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> MergeResult:
        """Classify and report a PR that failed the mergeability gate.

        Extracted from ``_merge_single_pr`` to keep that method's
        branch count manageable. Produces a detailed skip/block
        reason, sets ``result`` accordingly, and returns it.
        """
        # Get detailed status for a more informative skip message
        # Use async method to avoid event loop conflicts
        repo_owner, repo_name = pr_info.repository_full_name.split("/")

        # Check if blocked to get more detailed status
        if pr_info.mergeable_state == "blocked" and self._github_client:
            try:
                detailed_status = await self._github_client.analyze_block_reason(
                    repo_owner,
                    repo_name,
                    pr_info.number,
                    pr_info.head_sha,
                    base_branch=pr_info.base_branch,
                )
            except Exception:
                detailed_status = f"Blocked (state: {pr_info.mergeable_state})"
        else:
            # For non-blocked states, provide basic status
            if pr_info.mergeable_state == "dirty":
                detailed_status = "Merge conflicts"
            elif pr_info.mergeable_state == "behind":
                detailed_status = "Rebase required (out of date)"
            elif pr_info.mergeable_state == "draft":
                detailed_status = "Draft PR"
            else:
                detailed_status = f"Not mergeable (state: {pr_info.mergeable_state})"

        # Use the detailed status as the skip reason, with fallback
        if detailed_status and detailed_status != "Status unclear":
            skip_reason = detailed_status.lower()
        else:
            # Fallback to basic mapping if detailed status is unclear
            # aislop-ignore-next-line ai-slop/python-repetitive-dispatch -- branches set distinct reasons with sub-conditions, not a uniform table
            if pr_info.mergeable_state == "dirty":
                skip_reason = "merge conflicts"
            elif pr_info.mergeable_state == "behind":
                skip_reason = "behind"
            elif pr_info.mergeable_state == "blocked":
                if pr_info.mergeable is True:
                    skip_reason = "blocked, requires review"
                else:
                    skip_reason = "blocked by failing checks"
            elif pr_info.mergeable_state == "unstable":
                skip_reason = "unstable"
            elif pr_info.mergeable is False:
                skip_reason = "not mergeable"
            else:
                skip_reason = "unknown"

        # Determine if this is truly blocked (unmergeable) or just skipped
        if pr_info.mergeable_state == "dirty" or (
            pr_info.mergeable_state == "behind" and pr_info.mergeable is False
        ):
            result.status = MergeStatus.BLOCKED
            icon = "🛑"
            status = "Blocked"
        else:
            result.status = MergeStatus.SKIPPED
            icon = "⏭️"
            status = "Skipped"

        self._pr_status(
            f"{icon} {status}: {pr_info.html_url} [{skip_reason}]",
            level="info",
        )

        result.error = f"PR is not mergeable (state: {pr_info.mergeable_state}, mergeable: {pr_info.mergeable})"

        # For the result error (used in CLI output), use the detailed status if it's more informative
        if detailed_status and detailed_status != "Status unclear":
            result.error = detailed_status

        return result

    def _simulate_preview_merge(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> None:
        """Simulate the Step 6 merge outcome for preview mode.

        Mutates ``result`` in place.  No console output: preview
        progress is conveyed by the Rich tracker counters (with
        preview-accurate labels such as "Mergeable") and the per-PR
        outcomes/reasons are reported in the end-of-run summary, so
        per-PR lines here go to the log only.
        """
        if pr_info.mergeable_state == "behind" and not self.fix_out_of_date:
            result.status = MergeStatus.SKIPPED
            result.error = "PR is behind base branch and --no-fix option is set"
            self._pr_status(
                f"\u23ed\ufe0f Skipped: {pr_info.html_url} [behind, rebase disabled]",
                level="debug",
            )
        elif pr_info.mergeable_state == "behind" and self.fix_out_of_date:
            # Behind PRs merge directly unless branch protection
            # requires up-to-date heads, in which case the real run
            # refreshes the branch first — either way the PR counts
            # as mergeable.
            result.status = MergeStatus.MERGED
            # Use ``warning`` (not ``error``) so the MERGED result
            # does not carry a contradictory error message.
            result.warning = "behind base branch"
            self._pr_status(
                f"\u2611\ufe0f Approve/merge: {pr_info.html_url} [behind base branch]",
                level="debug",
            )
        elif pr_info.mergeable_state == "dirty":
            result.status = MergeStatus.BLOCKED
            result.error = "PR has merge conflicts"
            self._pr_status(
                f"\U0001f6d1 Blocked: {pr_info.html_url} [merge conflicts]",
                level="debug",
            )
        elif pr_info.mergeable is False and pr_info.mergeable_state == "blocked":
            result.status = MergeStatus.BLOCKED
            result.error = "PR blocked by failing checks"
            self._pr_status(
                f"\U0001f6d1 Blocked: {pr_info.html_url} [blocked by failing checks]",
                level="debug",
            )
        else:
            # Simulate successful merge in preview mode
            result.status = MergeStatus.MERGED
            self._pr_status(
                f"\u2611\ufe0f Approve/merge: {pr_info.html_url}",
                level="debug",
            )
