# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""The git steps the local-rebase workflow runs inside a workspace.

Everything here operates on an already-created directory and an
already-resolved :class:`~dependamerge.rebase.local_plan._RebasePlan`:
clone the head repo, fetch the base branch from the right remote,
rebase (retrying once against unshallowed remotes), and force-push with
lease.  Creating and removing the workspace is the caller's job.

Every helper returns rather than raises on a git failure, because the
caller's contract is to fall through to the auto-merge path instead of
leaving a PR half-rebased.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..git_ops import (
    GitError,
    add_remote,
    checkout,
    clone,
    fetch,
    fetch_branch,
    push_force_with_lease,
    rebase,
    rebase_abort,
    run_git,
)
from .local_plan import _RebasePlan


def _rebase_and_push(
    *,
    plan: _RebasePlan,
    workspace: Path,
    token: str,
    log: logging.Logger,
) -> bool:
    """Clone, rebase and force-push inside an already-created workspace.

    Returns True only when every git step succeeded.  Cleanup of
    ``workspace`` is the caller's responsibility.
    """
    # Clone the head repo at the PR's head branch. Shallow
    # clone keeps disk + network footprint low for what are
    # typically tiny dependency-update PRs.
    try:
        clone(
            plan.origin_url,
            workspace,
            branch=plan.head_branch,
            depth=50,
            single_branch=True,
            no_tags=True,
            filter_blobs=True,
            logger=log.debug,
            token=token,
        )
    except GitError as exc:
        log.debug("Local rebase: clone failed for %s: %s", plan.html_url, exc)
        return False

    rebase_onto = _fetch_base_branch(
        plan=plan, workspace=workspace, token=token, log=log
    )
    if rebase_onto is None:
        return False

    # Make sure we are on the head branch (defensive against
    # detached HEAD after clone --branch).
    try:
        checkout(plan.head_branch, cwd=workspace, create=False, logger=log.debug)
    except GitError:
        # Already on the branch, or branch missing locally;
        # rebase will surface the real problem if any.
        pass

    if not _rebase_onto_base(
        plan=plan,
        workspace=workspace,
        rebase_onto=rebase_onto,
        token=token,
        log=log,
    ):
        return False

    # Force-push with lease to the head repo. We push back to
    # ``origin`` because the head ref always lives there (even
    # for forks, the head repo *is* the fork).
    try:
        push_force_with_lease(
            "origin",
            plan.head_branch,
            plan.head_branch,
            cwd=workspace,
            logger=log.debug,
            token=token,
        )
    except GitError as exc:
        log.debug(
            "Local rebase: force-push failed for %s: %s",
            plan.html_url,
            exc,
        )
        return False

    log.debug("Local rebase succeeded for %s", plan.html_url)
    return True


def _fetch_base_branch(
    *,
    plan: _RebasePlan,
    workspace: Path,
    token: str,
    log: logging.Logger,
) -> str | None:
    """Fetch the base branch and return the ref to rebase onto, or None.

    Fetches from upstream when the PR is from a fork, from origin
    otherwise. We need it available locally before we can rebase
    onto it.

    Uses :func:`fetch_branch` rather than the bare ``fetch``
    form: the ``--single-branch`` clone restricts the
    remote's configured refspec to the PR head branch, so a
    bare ``git fetch origin <base>`` would only populate
    ``FETCH_HEAD`` and a subsequent ``git rebase origin/<base>``
    would fail with ``fatal: invalid upstream 'origin/<base>'``.
    """
    try:
        if plan.is_fork:
            add_remote("upstream", plan.upstream_url, cwd=workspace, logger=log.debug)
            fetch_branch(
                "upstream",
                plan.base_branch,
                cwd=workspace,
                depth=50,
                logger=log.debug,
                token=token,
            )
            return f"upstream/{plan.base_branch}"
        fetch_branch(
            "origin",
            plan.base_branch,
            cwd=workspace,
            depth=50,
            logger=log.debug,
            token=token,
        )
        return f"origin/{plan.base_branch}"
    except GitError as exc:
        log.debug("Local rebase: fetch failed for %s: %s", plan.html_url, exc)
        return None


def _rebase_onto_base(
    *,
    plan: _RebasePlan,
    workspace: Path,
    rebase_onto: str,
    token: str,
    log: logging.Logger,
) -> bool:
    """Run ``git rebase``, retrying once against unshallowed remotes.

    ``git rebase`` runs with ``check=False`` (see
    ``git_ops.rebase``), so a non-zero exit does *not* raise
    ``GitError``; we have to inspect ``returncode``
    ourselves. Conflicts are the most common cause of a
    non-zero exit here, but other failures (corrupt index,
    invalid base ref, etc.) hit the same path — surface
    stderr/stdout in debug output so the cause is visible
    to anyone investigating, then abort the rebase to leave
    the workspace in a clean state before cleanup.
    """
    rebase_result = rebase(
        rebase_onto,
        cwd=workspace,
        autostash=False,
        interactive=False,
        logger=log.debug,
    )
    if rebase_result.returncode == 0:
        return True

    # Shallow clones (depth=50) can miss the merge base
    # for PRs whose branch point is older than 50 commits.
    # ``git rebase`` reports this as a generic non-zero
    # exit; the diagnostic is visible in stderr but we
    # can't reliably distinguish it without parsing
    # locale-dependent text.
    #
    # Recovery: abort the failed rebase, unshallow both
    # remotes, and retry the rebase. If that also fails,
    # the cause is genuine (conflicts, corrupt index,
    # etc.) and we return False as before so the caller
    # falls through to the auto-merge path.
    log.debug(
        "Local rebase: rebase exited non-zero for %s "
        "(rc=%d, stderr=%r, stdout=%r); attempting "
        "unshallow + retry.",
        plan.html_url,
        rebase_result.returncode,
        rebase_result.stderr,
        rebase_result.stdout,
    )
    try:
        rebase_abort(cwd=workspace, logger=log.debug)
    except Exception:
        # Cleanup abort is best-effort; ignore so the original
        # rebase failure still drives the retry below.
        pass

    if not _unshallow_remotes(plan=plan, workspace=workspace, token=token, log=log):
        return False

    rebase_result = rebase(
        rebase_onto,
        cwd=workspace,
        autostash=False,
        interactive=False,
        logger=log.debug,
    )
    if rebase_result.returncode != 0:
        log.debug(
            "Local rebase: rebase still failing after unshallow "
            "for %s (rc=%d, stderr=%r); aborting.",
            plan.html_url,
            rebase_result.returncode,
            rebase_result.stderr,
        )
        try:
            rebase_abort(cwd=workspace, logger=log.debug)
        except Exception:
            # Cleanup abort is best-effort; ignore and return
            # failure to the caller regardless.
            pass
        return False
    log.debug(
        "Local rebase: succeeded after unshallow for %s",
        plan.html_url,
    )
    return True


def _unshallow_remotes(
    *,
    plan: _RebasePlan,
    workspace: Path,
    token: str,
    log: logging.Logger,
) -> bool:
    """Deepen the clone to full history.

    Git's shallow state is **repository-wide, not per-remote**, but
    ``--unshallow`` only completes the history reachable from the remote
    being fetched.  Both halves of that matter here:

    * Once the repository *is* complete, a second ``--unshallow`` is a
      fatal error --- ``fatal: --unshallow on a complete repository does
      not make sense``, exit 128.  The resulting :class:`GitError` was
      caught below, logged at debug and reported as ``False``, so for
      every fork pull request the workspace preparation silently
      declared the local rebase impossible.

    * But ``origin --unshallow`` does not necessarily complete the
      repository.  ``upstream/<base>`` is fetched at ``depth=50``
      (see :func:`_fetch_base_branch`), and when the base has advanced
      further than that since the fork point, origin cannot supply the
      missing ancestors.  The repository then stays shallow, a plain
      ``upstream`` fetch sees an up-to-date ref and deepens nothing, and
      the rebase still has no merge base.

    So unshallow origin, then **ask git** whether the repository is
    still shallow and deepen upstream accordingly.  Verified against
    genuinely diverged remotes: after ``origin --unshallow`` the
    repository remains shallow, ``upstream --unshallow`` then succeeds,
    and a merge base exists --- whereas a plain fetch leaves
    ``merge-base`` empty.

    Returns False when a fetch failed, in which case the caller must
    abandon the rebase.
    """
    try:
        fetch(
            "origin",
            cwd=workspace,
            unshallow=True,
            logger=log.debug,
            token=token,
        )
        if plan.is_fork:
            if _is_shallow(workspace=workspace, log=log):
                # Origin could not supply upstream's missing ancestors.
                fetch(
                    "upstream",
                    cwd=workspace,
                    unshallow=True,
                    logger=log.debug,
                    token=token,
                )
            else:
                # Already complete --- ``--unshallow`` here would be
                # fatal, so fetch normally to pick up any new refs.
                fetch(
                    "upstream",
                    cwd=workspace,
                    logger=log.debug,
                    token=token,
                )
    except GitError as exc:
        log.debug(
            "Local rebase: unshallow failed for %s: %s",
            plan.html_url,
            exc,
        )
        return False
    return True


def _is_shallow(*, workspace: Path, log: logging.Logger) -> bool:
    """Report whether the working repository still has a shallow boundary.

    Asked of git rather than inferred, because whether a single
    ``--unshallow`` completed the repository depends on how far the two
    remotes have diverged.  A failure to answer is treated as "not
    shallow", which routes to the plain fetch: that can leave a rebase
    without a merge base, but it cannot abort the whole preparation the
    way an unwarranted ``--unshallow`` would.
    """
    try:
        result = run_git(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=workspace,
            logger=log.debug,
        )
    except GitError as exc:
        log.debug("Local rebase: could not read shallow state: %s", exc)
        return False
    return result.stdout.strip() == "true"
