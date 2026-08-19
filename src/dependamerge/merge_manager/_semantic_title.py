# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Alignment of an automation pull request title with its commit.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio

from ..bot_identity import is_automation_author
from ..check_runs import failing_check_names
from ..models import PullRequestInfo
from ..semantic_title import (
    describe_title_change,
    is_semantic_check_name,
    single_commit_subject,
    version_fragment_removed,
)
from ._base import _MergeManagerBase


class _SemanticTitleMixin(_MergeManagerBase):
    """Alignment of an automation pull request title with its commit."""

    async def _align_semantic_title(self, pr_info: PullRequestInfo) -> bool:
        """Repair a Dependabot title/commit-subject mismatch; report success.

        Dependabot shortens the commit subject by cutting the
        `` from <old> to <new>`` fragment while the PR title keeps it.
        When the org's ``Semantic Pull Request`` check enforces
        ``validateSingleCommitMatchesPrTitle``, that mismatch fails a
        required check the PR can never satisfy on its own, so the merge
        waits out its full timeout and reports a failure.  Setting the
        title to the commit subject fixes it, and GitHub's
        ``pull_request.edited`` event re-runs the check without a
        force-push or a full CI rerun.

        Rewriting somebody's pull request title is intrusive, so this is
        deliberately narrow.  It acts only when every one of the
        following holds:

        * the feature is enabled and the run is not a preview;
        * the author is an automation bot;
        * the semantic check is the **only** failing check, so a real
          failure is never masked;
        * the PR has exactly one non-merge commit;
        * the title differs from that commit's subject by exactly one
          elided version fragment --- not by the versions themselves,
          which is genuine drift the check is right to catch;
        * no alignment has already been attempted for this PR, so a
          check that keeps failing cannot drive a loop.
        """
        if not self.fix_semantic_title or self.preview_mode:
            return False
        if self._github_client is None:
            return False
        if not is_automation_author(pr_info.author):
            return False

        pr_key = f"{pr_info.repository_full_name}#{pr_info.number}"
        if pr_key in self._semantic_title_aligned:
            return False

        owner, repo = pr_info.repository_full_name.split("/", 1)

        try:
            runs = await self._github_client.get_check_runs_for_ref(
                owner, repo, pr_info.head_sha
            )
            status_failures = await self._github_client.get_failing_status_contexts(
                owner, repo, pr_info.head_sha
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.debug("Could not read checks for %s: %s", pr_key, exc)
            return False

        # ``failing_check_names`` resolves superseded duplicates, so a
        # cancelled run sitting beside a successful one does not read as
        # a failure here.  Commit *statuses* are consulted as well:
        # pre-commit.ci and DCO report through that API rather than as
        # check runs, and missing them would let a title rewrite proceed
        # while another required check is genuinely failing.
        failing = failing_check_names(runs if isinstance(runs, list) else [])
        failing += [name for name in status_failures if name not in failing]
        if not failing or not all(is_semantic_check_name(n) for n in failing):
            return False

        try:
            commits = await self._github_client.get_pull_request_commits(
                owner, repo, pr_info.number
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.debug("Could not read commits for %s: %s", pr_key, exc)
            return False

        subject = single_commit_subject(commits)
        if subject is None or subject == pr_info.title:
            return False
        if version_fragment_removed(pr_info.title, subject) is None:
            self.log.debug(
                "Not aligning %s: title and subject differ by more than an "
                "elided version fragment",
                pr_key,
            )
            return False

        original_title = pr_info.title
        try:
            await self._github_client.update_pull_request_title(
                owner, repo, pr_info.number, subject
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.warning("Could not align title for %s: %s", pr_key, exc)
            return False

        # Recorded only on success.  A transient failure here means no
        # alignment happened, so the PR should stay eligible rather than
        # waiting out the full merge timeout for want of one retry.  A
        # *successful* rewrite is what must never repeat.
        self._semantic_title_aligned.add(pr_key)
        pr_info.title = subject
        self._record_retrigger()
        self.log.info(
            "%s for %s (was %r)",
            describe_title_change(original_title, subject),
            pr_info.html_url,
            original_title,
        )
        self._pr_status(
            f"✏️ Aligned title: {pr_info.html_url} [semantic check]",
            level="debug",
        )
        return True
