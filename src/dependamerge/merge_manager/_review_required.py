# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Recovery when a merge is refused for want of an approving review.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from ..models import PullRequestInfo
from ..rule_violations import violation_verb
from ._base import _MergeManagerBase


class _ReviewRequiredMixin(_MergeManagerBase):
    """Recovery when a merge is refused for want of an approving review."""

    async def _approve_and_retry_if_review_required(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Approve-on-demand recovery after a failed direct merge.

        This is the merge-path approve-on-demand trigger: rather than
        approving every PR up-front, we attempt the merge first and only
        approve when GitHub rejects it *specifically* because our review
        is missing.  This avoids approving PRs that did not need it (e.g.
        a PR that fails for an unrelated reason).

        Called only after a direct merge attempt returned ``False``.  It
        consults :meth:`analyze_block_reason`; if (and only if) the PR is
        blocked pending approval and we have not already approved it this
        run, it approves the current head and retries the merge once.

        Returns:
            True if the approve-then-retry merged the PR, False otherwise
            (including when the failure was not approval-related, so the
            caller can proceed to its normal failure handling).
        """
        if self.preview_mode or self._github_client is None:
            return False

        pr_key = f"{owner}/{repo}#{pr_info.number}"
        if pr_key in self._recently_approved:
            # We already approved this PR this run; the retry machinery in
            # _merge_pr_with_retry has already had its post-approval
            # propagation retry, so a missing approval is not the cause.
            return False

        # Prefer GitHub's own merge-rejection body over the heuristic
        # block-reason classifier.  When the merge endpoint refuses the
        # merge it states the *authoritative* reason in the response body
        # (captured in the stored exception), e.g. "Repository rule
        # violations found Waiting on required approvals from <team>".
        # The heuristic ``analyze_block_reason`` only reports a single,
        # highest-priority reason and ranks the missing-approval condition
        # below required-status checks, so an unrelated or false-positive
        # "missing required status" (e.g. a DCO check GitHub does not
        # actually gate the merge on) masks the real cause and the
        # approve-on-demand recovery never fires.  Trust GitHub first.
        #
        # This authoritative check runs *regardless* of mergeable_state:
        # that field lags and is blind to repository rulesets, so GitHub
        # can reject a merge for a missing required approval even while the
        # cached state is not ``blocked``.  Gating it behind a ``blocked``
        # state would strand exactly the PRs this recovery exists to save.
        last_exception = self._last_merge_exception.get(pr_key)
        if last_exception is not None and self._merge_error_indicates_missing_approval(
            str(last_exception)
        ):
            self.log.debug(
                "Merge for %s was rejected by GitHub pending required "
                "approval; approving on demand and retrying",
                pr_key,
            )
            approved = await self._ensure_pr_approved(pr_info, owner, repo)
            if not approved:
                return False
            return await self._merge_pr_with_retry(pr_info, owner, repo)

        # Fall back to the heuristic block-reason classifier only when the
        # cached state actually shows the PR as ``blocked``.  A missing
        # review manifests as that state; a merge that fails from any other
        # state (e.g. a transient 405 on a ``clean`` PR) without an
        # authoritative approval signal above is not an approval problem,
        # so don't probe or approve it — let the caller's classifier
        # handle it.
        if pr_info.mergeable_state != "blocked":
            return False

        try:
            block_reason = await self._github_client.analyze_block_reason(
                owner,
                repo,
                pr_info.number,
                pr_info.head_sha,
                base_branch=pr_info.base_branch,
            )
        except Exception as exc:
            self.log.debug(
                "approve-on-demand block-reason check failed for %s: %s",
                pr_key,
                exc,
            )
            return False

        if not block_reason or "requires approval" not in block_reason.lower():
            # The merge failed for some reason other than a missing
            # review — do not approve; let the caller classify and report.
            return False

        self.log.debug(
            "Merge for %s was blocked pending approval; approving on "
            "demand and retrying",
            pr_key,
        )
        approved = await self._ensure_pr_approved(pr_info, owner, repo)
        if not approved:
            return False
        return await self._merge_pr_with_retry(pr_info, owner, repo)

    @staticmethod
    def _merge_error_indicates_missing_approval(error_text: str) -> bool:
        """Detect a missing-required-approval signal in a merge error body.

        GitHub's merge endpoint reports the authoritative rejection reason
        in its response body, which is preserved in the exception text
        raised by ``merge_pull_request``.  A merge blocked solely because
        our approving review is missing is recoverable: we can approve the
        head and retry.  This recognises the phrasings GitHub uses for
        both repository rulesets and classic branch protection, e.g.:

        - "Waiting on required approvals from <team>" (ruleset)
        - "At least 1 approving review is required by reviewers with
          write access." (branch protection)
        - "Required review ... review required"

        It deliberately does *not* match "changes requested" wording,
        which an approval cannot clear.
        """
        if not error_text:
            return False
        text = error_text.lower()
        return (
            "required approval" in text
            or "approving review" in text
            or "review required" in text
        )

    @staticmethod
    def _merge_error_indicates_pending_workflows(error_text: str) -> bool:
        """Detect still-executing required workflows in a merge error body.

        GitHub's merge endpoint rejects with 405 "Repository rule
        violations found … Required workflows 'X' are not satisfied"
        while ruleset-required workflows are still *executing* on the
        head commit — a pending condition that clears by itself once
        they finish.  That is distinct from the failure variant
        ("Required workflows 'X' … fail…"), where a workflow ran and
        reported failure: terminal, retrying cannot help.

        Only the clause starting at the ``required workflow`` wording
        is inspected, because the enhanced exception text always begins
        with "Failed to merge PR …" — matching ``fail`` against the
        whole message would classify every rejection as terminal.
        The outcome is read through :func:`violation_verb`, which looks
        only at the text *after* the quoted names, so a workflow called
        "Fail Fast Lint" no longer suppresses this recovery path.  Two
        parsers reading the same string is exactly the drift
        ``rule_violations`` exists to prevent.
        """
        if not error_text:
            return False
        # Trim the PR-state context ``_validate_merge_result`` appends
        # after GitHub's detail before anything reads it: that context
        # can itself say "blocked by failing checks", which the outcome
        # parser would take for the workflows' verdict.
        cut = error_text.lower().find(" (pr state:")
        detail = error_text if cut == -1 else error_text[:cut]
        lowered = detail.lower()
        idx = lowered.find("required workflow")
        if idx == -1:
            return False
        return "not satisfied" in lowered[idx:] and violation_verb(detail) != "failed"
