# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Rebase strategies for behind pull requests.

This package centralises every code path dependamerge uses to bring
a PR up to date with its base branch before merging:

- :func:`should_use_local_rebase` decides between the local-git
  workflow (preserves verified commit signatures via the user's
  ``~/.gitconfig``) and the GitHub REST ``update-branch``
  endpoint (fast but produces unsigned commits).
- :func:`local_rebase_pr` runs the local clone + rebase +
  force-push-with-lease against a secure temp workspace.
- :func:`perform_step5_rebase` is the top-level dispatcher used by
  :class:`AsyncMergeManager._merge_single_pr` Step 5.  It calls
  :func:`should_use_local_rebase` to decide between paths, then
  delegates to either ``_run_local_path`` or ``_run_rest_path``
  (private helpers).  ``_run_rest_path`` runs the REST
  ``update-branch`` call and the post-rebase polling loop
  (``_poll_post_rebase``) that waits for GitHub to recompute
  mergeability.
- Git network operations authenticate via the ``token`` argument of
  the :mod:`dependamerge.git_ops` helpers (GIT_ASKPASS-based), so the
  token never appears in remote URLs, argv, or ``.git/config``.

The dispatcher takes a :class:`RebaseContext` rather than a full
``AsyncMergeManager`` reference, which keeps the rebase logic
testable in isolation (no need to construct a manager + GitHub
client + progress tracker just to exercise a decision tree).

The local-rebase path is the headline reason this package exists:
``PUT /repos/{owner}/{repo}/pulls/{n}/update-branch`` creates a
server-side merge commit whose committer is the calling token's
GitHub user, which is *not* signed with the user's local SSH/GPG
key.  On repos whose branch protection requires verified
signatures, the resulting commit loses its ``Verified`` badge and
becomes un-mergeable.  ``pre-commit-ci[bot]`` PRs are particularly
affected because that bot has no comment macro for recreating a PR
with a re-signed commit (https://github.com/pre-commit-ci/issues/issues/41).
The local path solves this by shelling out to ``git`` so the
user's signing config is honoured.

Layout:
    context: ``RebaseContext``, ``Step5Outcome`` and the tracker shims
    decide: the local-vs-REST gate and its ``BaseRef`` argument
    local_rebase: the local clone + rebase + force-push workflow
    paths: the local, dependabot-macro and REST Step 5 paths
    polling: the post-``update-branch`` wait loop
    dispatch: ``perform_step5_rebase``
"""

from __future__ import annotations

# Compatibility re-exports.  Every name below has always been reachable
# as ``dependamerge.rebase.<name>`` because the single module this
# package replaced imported it at top level.  The redundant ``as``
# aliases mark them as deliberate: they are kept for callers and test
# patches, not used here.
from .. import git_ops as git_ops
from ..bot_identity import is_dependabot as is_dependabot
from ..git_ops import GitError as GitError
from ..git_ops import add_remote as add_remote
from ..git_ops import checkout as checkout
from ..git_ops import clone as clone
from ..git_ops import ensure_git_available as ensure_git_available
from ..git_ops import fetch as fetch
from ..git_ops import fetch_branch as fetch_branch
from ..git_ops import push_force_with_lease as push_force_with_lease
from ..git_ops import rebase as rebase
from ..git_ops import rebase_abort as rebase_abort
from ..git_ops import secure_rmtree as secure_rmtree
from ..models import PullRequestInfo as PullRequestInfo
from ..slot_lease import parked as parked
from .context import (
    RebaseContext,
    Step5Outcome,
    _record_rebase,
    _set_tracker_state,
)
from .decide import (
    BaseRef,
    authed_clone_url,
    should_use_local_rebase,
)
from .dispatch import perform_step5_rebase
from .local_rebase import local_rebase_pr
from .paths import (
    _run_dependabot_macro_path,
    _run_local_path,
    _run_rest_path,
)
from .polling import (
    _log_blocked_timeout,
    _log_post_rebase_status,
    _poll_post_rebase,
    _poll_should_continue,
)

__all__ = [
    "BaseRef",
    "RebaseContext",
    "Step5Outcome",
    "_log_blocked_timeout",
    "_log_post_rebase_status",
    "_poll_post_rebase",
    "_poll_should_continue",
    "_record_rebase",
    "_run_dependabot_macro_path",
    "_run_local_path",
    "_run_rest_path",
    "_set_tracker_state",
    "authed_clone_url",
    "local_rebase_pr",
    "perform_step5_rebase",
    "should_use_local_rebase",
]
