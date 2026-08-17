# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Network-facing git wrappers that accept a token.

These are the only helpers that contact a remote, and the only ones that
take a ``token`` argument.  They reach ``run_git`` through the ``process``
module object rather than importing the function directly, so that
substituting ``dependamerge.git_ops.process.run_git`` is observed by every
caller instead of only by those that happened to import it late.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from . import process
from .paths import PathLike

__all__ = [
    "clone",
    "fetch",
    "fetch_branch",
    "push_force_with_lease",
]


def clone(
    url: str,
    dest: PathLike,
    *,
    branch: str | None = None,
    depth: int | None = 50,
    single_branch: bool = True,
    no_tags: bool = True,
    filter_blobs: bool = True,
    quiet: bool = True,
    logger: Callable[[str], None] | None = None,
    token: str | None = None,
) -> None:
    """Clone a repository with defaults optimized for speed and safety.

    Authentication: pass ``token`` rather than embedding credentials in
    ``url``. The token is supplied via a GIT_ASKPASS helper so it never
    appears in argv and is never persisted to the clone's .git/config
    (git stores ``url`` verbatim as ``remote.origin.url``).
    """
    args = ["git", "clone"]
    if quiet:
        args.append("--quiet")
    if depth and depth > 0:
        args.extend(["--depth", str(depth)])
    if single_branch:
        args.append("--single-branch")
    if no_tags:
        args.append("--no-tags")
    if filter_blobs:
        args.extend(["--filter=blob:none"])
    if branch:
        args.extend(["--branch", branch])
    args.extend([url, str(dest)])

    process.run_git(args, logger=logger, token=token)


def fetch(
    remote: str,
    refspecs: str | Sequence[str] = (),
    *,
    cwd: PathLike,
    depth: int | None = None,
    unshallow: bool = False,
    prune: bool = False,
    logger: Callable[[str], None] | None = None,
    token: str | None = None,
) -> None:
    """Fetch refs with optional shallow/unshallow behavior.

    .. warning::

       Passing a *bare* branch name (e.g. ``fetch("origin", "main")``)
       only populates ``FETCH_HEAD``; it does **not** update
       ``refs/remotes/<remote>/<branch>`` unless the remote's
       configured fetch refspec already covers that branch.  After a
       ``--single-branch`` clone (which is :func:`clone`'s default)
       the configured refspec covers only the branch the clone
       targeted, so a subsequent ``fetch("origin", "main")`` leaves
       ``origin/main`` *undefined* locally and any downstream
       ``rebase`` / ``merge`` / ``rev-list`` against ``origin/main``
       fails with ``fatal: invalid upstream 'origin/main'``.

       When the caller wants the fetched branch to be usable as a
       remote-tracking ref (the usual case), use :func:`fetch_branch`
       instead, which always writes through an explicit refspec
       mapping.  Callers that genuinely want
       ``FETCH_HEAD``-only semantics (e.g. preparing a one-shot
       ``git merge FETCH_HEAD``) can keep using the bare form here.

    Authentication: pass ``token`` rather than embedding credentials in
    the remote URL. It is supplied to git via a GIT_ASKPASS helper
    (see :func:`run_git`) so it never reaches argv or ``.git/config``.
    """
    args = ["git", "fetch", remote]
    if prune:
        args.append("--prune")
    if unshallow:
        args.append("--unshallow")
    if depth and depth > 0:
        args.extend(["--depth", str(depth)])
    if isinstance(refspecs, str):
        if refspecs:
            args.append(refspecs)
    else:
        args.extend(list(refspecs))
    process.run_git(args, cwd=cwd, logger=logger, token=token)


def fetch_branch(
    remote: str,
    branch: str,
    *,
    cwd: PathLike,
    depth: int | None = None,
    force: bool = True,
    logger: Callable[[str], None] | None = None,
    token: str | None = None,
) -> None:
    """Fetch ``branch`` from ``remote`` into ``refs/remotes/<remote>/<branch>``.

    Wraps :func:`fetch` with an explicit refspec mapping so the
    remote-tracking ref always lands locally, regardless of the
    remote's configured fetch refspec.  This is the safe form to
    use after a ``--single-branch`` clone (which is :func:`clone`'s
    default) when subsequent code needs to refer to
    ``<remote>/<branch>`` — e.g. as the target of
    :func:`rebase` / :func:`rev_list_count` / a ``log <r>/<b>..HEAD``
    invocation.

    A bare ``git fetch <remote> <branch>`` would only populate
    ``FETCH_HEAD`` in that scenario and the downstream rebase would
    fail with ``fatal: invalid upstream '<remote>/<branch>'``
    — see :func:`fetch` for the full background.

    Args:
        remote: Remote name (e.g. ``"origin"`` or ``"upstream"``).
        branch: Branch name on the remote (no ``refs/heads/`` prefix).
        cwd: Working directory in which to run ``git``.
        depth: Optional shallow-fetch depth.  ``None`` means
            "inherit the existing depth" (no ``--depth`` flag).
        force: When True (the default), prepend ``+`` to the
            refspec so the remote-tracking ref is updated even when
            the remote has been force-pushed (the common case for
            dependency-update bot branches that get re-pushed).
        logger: Optional logger callback.
        token: Optional secret supplied to git via GIT_ASKPASS
            (see :func:`run_git`).
    """
    prefix = "+" if force else ""
    refspec = f"{prefix}refs/heads/{branch}:refs/remotes/{remote}/{branch}"
    fetch(remote, refspec, cwd=cwd, depth=depth, logger=logger, token=token)


def push_force_with_lease(
    remote: str,
    src_ref: str,
    dst_ref: str,
    *,
    cwd: PathLike,
    logger: Callable[[str], None] | None = None,
    token: str | None = None,
) -> None:
    process.run_git(
        ["git", "push", "--force-with-lease", remote, f"{src_ref}:{dst_ref}"],
        cwd=cwd,
        logger=logger,
        token=token,
    )
