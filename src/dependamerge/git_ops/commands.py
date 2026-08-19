# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Porcelain wrappers for local repository operations.

These helpers never talk to a remote, so none of them take a token.
The network-facing wrappers (clone/fetch/fetch_branch/
push_force_with_lease) live in the package root alongside the public
re-exports.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .paths import PathLike
from .process import GitResult, run_git


def add_remote(
    name: str,
    url: str,
    *,
    cwd: PathLike,
    logger: Callable[[str], None] | None = None,
) -> None:
    run_git(["git", "remote", "add", name, url], cwd=cwd, logger=logger)


def checkout(
    branch: str,
    *,
    cwd: PathLike,
    create: bool = False,
    track: str | None = None,
    logger: Callable[[str], None] | None = None,
) -> None:
    """Checkout a branch; optionally create and set upstream."""
    args = ["git", "checkout"]
    if create:
        args.append("-B")
    args.append(branch)
    run_git(args, cwd=cwd, logger=logger)
    if track:
        run_git(
            ["git", "branch", "--set-upstream-to", track, branch],
            cwd=cwd,
            logger=logger,
        )


def rebase(
    onto: str,
    *,
    cwd: PathLike,
    autostash: bool = True,
    interactive: bool = False,
    logger: Callable[[str], None] | None = None,
) -> GitResult:
    """
    Start a rebase onto the provided branch/ref.

    If interactive=True, inherits stdio to allow editor/mergetool usage during conflicts.
    """
    args = ["git", "rebase"]
    if autostash:
        args.append("--autostash")
    args.append(onto)
    return run_git(args, cwd=cwd, interactive=interactive, check=False, logger=logger)


def rebase_continue(
    *,
    cwd: PathLike,
    interactive: bool = False,
    logger: Callable[[str], None] | None = None,
) -> GitResult:
    return run_git(
        ["git", "rebase", "--continue"],
        cwd=cwd,
        interactive=interactive,
        check=False,
        logger=logger,
    )


def rebase_abort(
    *,
    cwd: PathLike,
    logger: Callable[[str], None] | None = None,
) -> None:
    run_git(["git", "rebase", "--abort"], cwd=cwd, logger=logger)


def status_porcelain(
    *,
    cwd: PathLike,
    logger: Callable[[str], None] | None = None,
) -> str:
    """Return porcelain status output."""
    res = run_git(["git", "status", "--porcelain"], cwd=cwd, check=True, logger=logger)
    return res.stdout


def list_conflicted_files(
    *,
    cwd: PathLike,
    logger: Callable[[str], None] | None = None,
) -> list[str]:
    """
    Parse 'git status --porcelain' to list conflicted files.

    Conflicted XY codes include: DD, AU, UD, UA, DU, AA, UU
    """
    out = status_porcelain(cwd=cwd, logger=logger)
    conflicted = []
    for line in out.splitlines():
        if not line:
            continue
        # Format: XY <path>
        code = line[:2]
        path = line[3:].strip()
        if code in {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}:
            conflicted.append(path)
    return conflicted


def add(
    paths: str | Sequence[str],
    *,
    cwd: PathLike,
    logger: Callable[[str], None] | None = None,
) -> None:
    if isinstance(paths, str):
        args = ["git", "add", "--", paths]
    else:
        args = ["git", "add", "--", *paths]
    run_git(args, cwd=cwd, logger=logger)


def add_all(
    *,
    cwd: PathLike,
    logger: Callable[[str], None] | None = None,
) -> None:
    run_git(["git", "add", "-A"], cwd=cwd, logger=logger)


def commit_amend_no_edit(
    *,
    cwd: PathLike,
    no_verify: bool = False,
    logger: Callable[[str], None] | None = None,
) -> None:
    args = ["git", "commit", "--amend", "--no-edit"]
    if no_verify:
        args.append("--no-verify")
    run_git(args, cwd=cwd, logger=logger)


def rev_list_count(
    range_expr: str,
    *,
    cwd: PathLike,
    logger: Callable[[str], None] | None = None,
) -> int:
    """Return the number of commits in the given revision range (e.g., 'base..HEAD')."""
    res = run_git(["git", "rev-list", "--count", range_expr], cwd=cwd, logger=logger)
    try:
        return int((res.stdout or "0").strip())
    except ValueError:
        return 0
