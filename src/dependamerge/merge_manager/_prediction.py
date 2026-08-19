# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Predicting the outcome of a merge without attempting it.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import Any

from ._base import _MergeManagerBase


class _PredictionMixin(_MergeManagerBase):
    """Predicting the outcome of a merge without attempting it."""

    async def _predict_merge_outcome(
        self, owner: str, repo: str, pr_number: int, merge_method: str
    ) -> tuple[bool, str]:
        """Best-effort, read-only prediction of whether a PR would merge.

        This is a **preview-only** probe used to render the dry-run
        evaluation.  It inspects the PR's ``mergeable`` / ``mergeable_state``
        and, for ``blocked`` PRs, consults :meth:`analyze_block_reason` to
        produce a one-line verdict.

        It deliberately has **no authority over the real merge**: GitHub's
        ``mergeable_state`` can lag the true state, and repository rulesets
        are invisible to this code path, so a confident "would block"
        verdict here can still be wrong.  The execution path therefore does
        not gate on this prediction — it attempts the merge and treats
        GitHub's actual response (Step 6 and ``_merge_pr_with_retry``) as
        authoritative.  Only the preview path calls this method.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            merge_method: Merge method to test

        Returns:
            Tuple of (can_merge: bool, reason: str)
        """
        if not self._github_client:
            return False, "GitHub client not initialized"

        try:
            # Check organization-level restrictions (cached per org)
            await self._get_org_settings(owner)

            # Note: Removed DCO signoff check as web_commit_signoff_required only affects
            # web-based commits, not PR merges. DCO enforcement for PRs is handled by
            # status checks/apps, not repository settings.

            pr_data = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_number}"
            )

            if isinstance(pr_data, dict):
                verdict = await self._predict_from_pr_data(
                    owner, repo, pr_number, pr_data
                )
                if verdict is not None:
                    return verdict

            return True, "merge capability test passed"

        except Exception as e:
            return self._prediction_from_error(owner, repo, pr_number, e)

    async def _predict_from_pr_data(
        self, owner: str, repo: str, pr_number: int, pr_data: dict[str, Any]
    ) -> tuple[bool, str] | None:
        """Derive a verdict from a PR payload's mergeability fields.

        Separate from :meth:`_predict_merge_outcome` so the caller keeps
        showing only the API calls the ``try`` block guards, while the
        state machine over ``mergeable`` / ``mergeable_state`` reads on
        its own.  It is called from inside that ``try``, so anything
        raised here is still swallowed by the same handler.

        Returns ``None`` when no branch reaches a verdict — including the
        fixable ``behind`` case — which the caller reads as "nothing
        objected" and answers with the pass-through result.
        """
        mergeable_state = pr_data.get("mergeable_state", "unknown")
        mergeable = pr_data.get("mergeable")
        head_sha = (pr_data.get("head") or {}).get("sha", "")

        self.log.debug(
            f"PR {owner}/{repo}#{pr_number} REST API status: mergeable={mergeable}, mergeable_state={mergeable_state}"
        )

        # Check for specific blocking conditions that indicate protection rules
        if mergeable_state == "blocked" and mergeable is False:
            # Before declaring the PR unmergeable, analyze WHY it's blocked.
            # If the only blocker is "requires approval", the tool is about to
            # provide that approval — so we should allow the merge to proceed.
            # Note: we only call analyze_block_reason when mergeable is False
            # to avoid unnecessary API traffic; when mergeable is True/None the
            # code falls through to the pass-through return at the end.
            block_reason = await self._analyze_block_reason_for_prediction(
                owner, repo, pr_number, head_sha, pr_data
            )
            return self._blocked_prediction(owner, repo, pr_number, block_reason)
        elif mergeable_state == "dirty":
            return False, "merge conflicts"
        elif mergeable_state == "behind":
            if not self.fix_out_of_date:
                return (
                    False,
                    "PR is behind base branch and --no-fix option is set",
                )
            # Otherwise it's fixable
        elif mergeable is False and mergeable_state == "unknown":
            # This often indicates hidden branch protection rules
            return self._hidden_protection_prediction(owner, repo, pr_number)

        return None

    async def _analyze_block_reason_for_prediction(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        head_sha: str,
        pr_data: dict[str, Any],
    ) -> str:
        """Ask the client why a PR is blocked, tolerating failure.

        Kept apart from the branch that consumes the answer because it
        owns its own ``except``: a failed analysis must degrade to an
        empty reason rather than abort the prediction, and burying that
        inner handler inside the state machine made both harder to read.
        Returns the empty string when the analysis is skipped or fails.
        """
        block_reason = ""
        if head_sha and self._github_client:
            try:
                block_reason = await self._github_client.analyze_block_reason(
                    owner,
                    repo,
                    pr_number,
                    head_sha,
                    base_branch=(pr_data.get("base") or {}).get("ref"),
                )
                self.log.debug(
                    f"PR {owner}/{repo}#{pr_number} block reason: {block_reason}"
                )
            except Exception as analyze_err:
                self.log.debug(
                    f"Could not analyze block reason for {owner}/{repo}#{pr_number}: {analyze_err}"
                )

        return block_reason

    def _blocked_prediction(
        self, owner: str, repo: str, pr_number: int, block_reason: str
    ) -> tuple[bool, str]:
        """Turn an analysed block reason into a verdict.

        Split out because it is pure decision-making over an already
        fetched reason: no I/O, so it stays testable and readable apart
        from the API calls that produced its input.
        """
        # If the PR is only blocked because it needs approval, allow it
        # through — the tool will approve it before attempting merge.
        if "requires approval" in block_reason.lower():
            self.log.info(
                f"PR {owner}/{repo}#{pr_number} is blocked pending approval — tool will approve before merge"
            )
            return True, "PR blocked pending approval (tool will approve)"

        # For other blocking reasons, check force level
        if self.force_level in ["code-owners", "protection-rules", "all"]:
            self.log.info(
                f"Force level '{self.force_level}' bypassing branch protection rules for {owner}/{repo}#{pr_number}"
            )
            return True, "branch protection bypassed by force level"
        return (
            False,
            f"branch protection rules prevent merge ({block_reason or 'blocked'})",
        )

    def _hidden_protection_prediction(
        self, owner: str, repo: str, pr_number: int
    ) -> tuple[bool, str]:
        """Decide the verdict for ``mergeable=False`` with no known state.

        GitHub reports this combination when protection rules it will not
        name are refusing the merge.  Extracted so the force-level
        override that applies to it sits next to the equivalent override
        in :meth:`_blocked_prediction` rather than nested in the chain.
        """
        if self.force_level in ["code-owners", "protection-rules", "all"]:
            self.log.info(
                f"Force level '{self.force_level}' bypassing hidden branch protection rules for {owner}/{repo}#{pr_number}"
            )
            return True, "hidden branch protection bypassed by force level"
        return (
            False,
            "cannot update protected ref - organization or branch protection rules prevent merge",
        )

    def _prediction_from_error(
        self, owner: str, repo: str, pr_number: int, exc: Exception
    ) -> tuple[bool, str]:
        """Classify an exception raised while probing the PR into a verdict.

        This is the whole body of :meth:`_predict_merge_outcome`'s
        ``except`` clause.  It is a second, independent decision tree —
        over error text rather than PR state — so it lives on its own;
        the caller still catches the exception, keeping the handler's
        scope unchanged.
        """
        error_msg = str(exc)
        self.log.debug(
            f"Exception in _predict_merge_outcome for {owner}/{repo}#{pr_number}: {error_msg}"
        )

        if self._error_is_dco_related(error_msg):
            # This error comes from GitHub API, not our code - but these PRs are actually mergeable
            # The DCO requirement doesn't apply to API merges, only web-based commits
            self.log.info(
                f"Ignoring DCO-related error for {owner}/{repo}#{pr_number} - API merges are allowed"
            )
            return True, "DCO enforcement not applicable to API merges"

        if "protected ref" in error_msg.lower() or "cannot update" in error_msg.lower():
            if self.force_level in ["code-owners", "protection-rules", "all"]:
                self.log.info(
                    f"Force level '{self.force_level}' bypassing protected ref error for {owner}/{repo}#{pr_number}"
                )
                return True, "protected ref error bypassed by force level"
            return (
                False,
                "cannot update protected ref - organization or branch protection rules prevent merge",
            )
        elif "403" in error_msg:
            if self.force_level == "all":
                self.log.info(
                    f"Force level 'all' attempting to bypass permissions error for {owner}/{repo}#{pr_number}"
                )
                return True, "permissions error bypassed by force level"
            return (
                False,
                "insufficient permissions or branch protection rules prevent merge",
            )
        elif "405" in error_msg:
            return False, "merge method not allowed by repository settings"
        else:
            # Unknown error during test - assume it's mergeable
            self.log.debug(f"Test merge capability failed with unknown error: {exc}")
            return True, "test merge capability failed - assuming mergeable"

    @staticmethod
    def _error_is_dco_related(error_msg: str) -> bool:
        """Report whether an error message describes a DCO/signoff refusal.

        Look for specific DCO-related errors in the GitHub API response.
        DCO errors typically come as 422 validation errors with specific
        messages, but some arrive without a status code, so both shapes
        are matched.  A standalone predicate keeps that pattern list out
        of :meth:`_prediction_from_error`'s branch chain.
        """
        if "422" in error_msg and (
            "commit signoff required" in error_msg.lower()
            or "commits must have verified signatures" in error_msg.lower()
            or (
                "dco" in error_msg.lower()
                and ("required" in error_msg.lower() or "sign" in error_msg.lower())
            )
        ):
            return True
        # Catch DCO errors that don't include status codes
        if "commit signoff required" in error_msg.lower():
            return True
        return False
