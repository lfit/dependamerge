# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
git subprocess invocation with token redaction and askpass credentials.

Holds the single ``run_git()`` entrypoint used by every helper in this
package, the result/error types it produces, and the GIT_ASKPASS
context manager that hands a secret to git without it ever reaching
argv, a remote URL, or ``.git/config``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from .paths import PathLike, create_secure_tempdir, secure_rmtree
from .redaction import (
    _ASKPASS_DEFAULT_USERNAME,
    _ASKPASS_SCRIPT,
    _ASKPASS_TOKEN_ENV,
    _ASKPASS_USERNAME_ENV,
    _build_git_env,
    _redact,
    _redact_seq,
)


@dataclass
class GitResult:
    """Result of a git command execution."""

    returncode: int
    stdout: str
    stderr: str
    args: tuple[str, ...]


class GitError(RuntimeError):
    """Raised when a git command fails with non-zero exit code."""

    def __init__(
        self,
        message: str,
        *,
        args: Sequence[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        redacted_cmd = _redact(" ".join(args))
        redacted_out = _redact(stdout or "")
        redacted_err = _redact(stderr or "")
        super().__init__(
            f"{message}\n  cmd: {redacted_cmd}\n  exit: {returncode}\n  stderr: {redacted_err.strip()}"
        )
        # SECURITY: Redact args_vec to prevent token leakage if callers
        # inspect exception attributes. Command args may contain tokens
        # embedded in clone URLs (e.g., x-access-token:<token>@host).
        self.args_vec = tuple(_redact(str(a)) for a in args)
        self.returncode = returncode
        self.stdout = redacted_out
        self.stderr = redacted_err


@contextmanager
def git_askpass_env(
    token: str,
    *,
    username: str = _ASKPASS_DEFAULT_USERNAME,
) -> Iterator[dict[str, str]]:
    """Yield env overrides that supply ``token`` to git via GIT_ASKPASS.

    Creates a short-lived helper script in a 0700 temporary directory.
    The script contains no secret material; the token travels only in
    the child process environment (``DM_GIT_ASKPASS_TOKEN``), so it is:

    - not visible in process listings (argv),
    - not embedded in remote URLs,
    - not written to ``.git/config`` by this mechanism.

    This does not override a user-configured ``credential.helper``
    (e.g. osxkeychain, manager-core, credential-store), which may
    still persist credentials at rest independently of this code.

    Platform note: the helper uses a ``#!/bin/sh`` shebang, so it runs
    wherever ``/bin/sh`` is available (Unix/macOS) and under Git for
    Windows, which executes shebang scripts via its bundled MSYS
    shell.

    Args:
        token: Secret used to answer git password prompts.
        username: Non-secret username used to answer username prompts
            (default ``x-access-token``, suitable for GitHub tokens).

    Yields:
        Environment overrides to merge into the git invocation, including
        ``GIT_TERMINAL_PROMPT=0`` so git never falls back to a TTY prompt.
    """
    askpass_dir = Path(create_secure_tempdir(prefix="dependamerge-askpass-"))
    askpass_path = askpass_dir / "askpass.sh"
    try:
        askpass_path.write_text(_ASKPASS_SCRIPT, encoding="utf-8")
        os.chmod(askpass_path, 0o700)
        yield {
            "GIT_ASKPASS": str(askpass_path),
            _ASKPASS_TOKEN_ENV: token,
            _ASKPASS_USERNAME_ENV: username,
            "GIT_TERMINAL_PROMPT": "0",
        }
    finally:
        # Best-effort cleanup. The helper script holds no secret (the
        # token lives only in the child process environment), so a
        # failed removal is not a credential leak; suppress teardown
        # errors so they cannot mask an exception propagating from
        # within the context.
        with suppress(Exception):
            secure_rmtree(askpass_dir)


def ensure_git_available() -> None:
    """Ensure 'git' is available on PATH; raise GitError if not."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise GitError(
                "git is not available or failed to run",
                args=("git", "--version"),
                returncode=result.returncode,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
    except FileNotFoundError as e:
        raise GitError(
            "git executable not found on PATH",
            args=("git", "--version"),
            returncode=127,
            stdout="",
            stderr=str(e),
        ) from e


def _spawn_git(
    args: Sequence[str],
    *,
    cwd: PathLike | None,
    env: dict[str, str],
    interactive: bool,
    timeout: float | None,
) -> GitResult:
    """
    Spawn the git subprocess and capture its outcome.

    Split out of run_git() so the entrypoint stays readable; exit-code
    checking and timeout translation remain with the caller.
    """
    if interactive:
        retcode = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            check=False,
            timeout=timeout,
        ).returncode
        stdout_str = ""
        stderr_str = ""
    else:
        cp = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        retcode = cp.returncode
        stdout_str = str(cp.stdout or "")
        stderr_str = str(cp.stderr or "")

    return GitResult(
        returncode=retcode,
        stdout=stdout_str,
        stderr=stderr_str,
        args=tuple(str(a) for a in args),
    )


def run_git(
    args: Sequence[str],
    *,
    cwd: PathLike | None = None,
    env_overrides: dict[str, str] | None = None,
    interactive: bool = False,
    check: bool = True,
    timeout: float | None = None,
    logger: Callable[[str], None] | None = None,
    lfs_skip: bool = True,
    token: str | None = None,
) -> GitResult:
    """
    Run a git command safely with redaction.

    Args:
        args: Full command, starting with 'git' (e.g., ["git","status","--porcelain"]).
        cwd: Working directory for the command.
        env_overrides: Environment overrides to merge.
        interactive: If True, inherit stdin/stdout/stderr (no capture) - for user sessions.
        check: If True, raise GitError on non-zero exit code.
        timeout: Optional timeout in seconds.
        logger: Optional logger callable receiving a redacted command line string.
        lfs_skip: If True, set GIT_LFS_SKIP_SMUDGE=1 by default.
        token: Optional secret supplied to git via a GIT_ASKPASS helper
            (see :func:`git_askpass_env`). Use this instead of embedding
            credentials in URLs so secrets never reach argv or .git/config.

    Returns:
        GitResult with stdout/stderr captured (empty when interactive=True).

    Raises:
        GitError if check=True and the command fails.
    """
    if not args or args[0] != "git":
        raise ValueError("run_git requires args to start with 'git'")

    # Explicit None check: an empty string is an (invalid) token the
    # caller passed deliberately, so route it through askpass and let
    # git surface the auth failure rather than silently skipping auth.
    if token is not None:
        with git_askpass_env(token) as askpass_overrides:
            merged = dict(env_overrides or {})
            # Askpass settings win over caller overrides: an explicit
            # token request must not be silently defeated by a stale
            # GIT_ASKPASS value in env_overrides.
            merged.update(askpass_overrides)
            return run_git(
                args,
                cwd=cwd,
                env_overrides=merged,
                interactive=interactive,
                check=check,
                timeout=timeout,
                logger=logger,
                lfs_skip=lfs_skip,
            )

    env = _build_git_env(env_overrides, lfs_skip=lfs_skip)

    cmd_str = shlex.join(_redact_seq([str(a) for a in args]))  # type: ignore[arg-type]
    if logger:
        logger(f"$ {cmd_str}")

    try:
        result = _spawn_git(
            args,
            cwd=cwd,
            env=env,
            interactive=interactive,
            timeout=timeout,
        )

        if check and result.returncode != 0:
            raise GitError(
                "git command failed",
                args=result.args,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return result
    except subprocess.TimeoutExpired as e:
        raise GitError(
            "git command timed out",
            args=tuple(str(a) for a in args),
            returncode=124,
            stdout="",
            stderr=str(e),
        ) from e
