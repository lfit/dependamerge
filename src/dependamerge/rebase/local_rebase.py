# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""The local clone + rebase + force-push-with-lease workflow.

:func:`local_rebase_pr` is the entry point, and the reason this
package exists: ``PUT /repos/{owner}/{repo}/pulls/{n}/update-branch``
produces a commit signed by nobody the branch protection recognises,
whereas shelling out to ``git`` honours the operator's own signing
configuration.

The work is split three ways.  :mod:`.local_plan` resolves the PR into
a plan (which remote is ``origin``, which is ``upstream``, is this a
fork); :mod:`.local_workspace` runs the git steps against that plan;
and this module owns the workspace lifecycle that brackets them.
Nothing in any of the three talks to the GitHub API.

``local_rebase_pr`` is reached through this module object by
:mod:`dependamerge.rebase.paths`, so
``dependamerge.rebase.local_rebase`` is the canonical target when
patching it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import git_ops
from ..git_ops import ensure_git_available, secure_rmtree
from ..models import PullRequestInfo
from .local_plan import _build_rebase_plan
from .local_workspace import _rebase_and_push


async def local_rebase_pr(
    *,
    pr_info: PullRequestInfo,
    owner: str,
    repo: str,
    token: str,
    log: logging.Logger,
    host: str,
) -> bool:
    """Rebase a PR locally and force-push the result.

    Clones the head repo into a secure temp workspace, fetches the
    base branch (from upstream when the PR is from a fork), runs
    ``git rebase``, and force-pushes with lease back to the head
    repo.  All git invocations inherit the user's ``~/.gitconfig``,
    so signing config is respected.

    ``host`` is required rather than defaulted.  This function is
    re-exported for compatibility, and a default would map silently to
    github.com through ``clone_url_for`` --- so an Enterprise pull
    request whose clone URL is absent would be cloned, and force-pushed
    to, on the wrong server.  A caller that has no host should say
    ``"github.com"`` and mean it.

    Returns True only if every step succeeds.  On any failure (no
    ``git`` on PATH, conflict during rebase, network error, push
    rejected) the workspace is cleaned up and False is returned;
    the caller should fall through to the auto-merge path so we
    never leave a half-applied state.
    """
    # Ensure ``git`` is on PATH before we start. ``GitError`` is
    # also raised when git is missing entirely.
    try:
        ensure_git_available()
    except Exception as exc:
        log.debug("Local rebase unavailable (no git on PATH?): %s", exc)
        return False

    plan = _build_rebase_plan(
        pr_info=pr_info, owner=owner, repo=repo, log=log, host=host
    )
    if plan is None:
        return False

    # Use a per-PR workspace under a secure temp parent so
    # concurrent rebases (--concurrency=N) don't collide.
    workspace_parent = Path(
        git_ops.create_secure_tempdir(prefix="dependamerge-localrebase-")
    )
    workspace = (
        workspace_parent
        / f"{(plan.head_full or plan.base_full).replace('/', '__')}__pr_{pr_info.number}"
    )
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        return _rebase_and_push(plan=plan, workspace=workspace, token=token, log=log)
    finally:
        # Always clean up. The workspace contains a clone of the
        # user's repository, so we want it gone even on success.
        try:
            secure_rmtree(workspace_parent)
        except Exception as exc:
            log.debug(
                "Local rebase: failed to clean up workspace %s: %s",
                workspace_parent,
                exc,
            )
