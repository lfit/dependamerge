# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
git_ops: Safe wrappers around common git operations with token redaction.

This module provides:
- A single run_git() entrypoint that redacts secrets in logs and exceptions
- High-level helpers for clone/fetch/checkout/rebase/push flows
- GIT_ASKPASS-based credential passing so tokens never appear in argv,
  remote URLs, or on-disk git configuration
- Utilities for secure temporary workspaces

Design goals:
- Never leak credentials or tokens in logs or exceptions
- Never persist credentials to disk (no tokens in .git/config remote URLs)
- Reasonable defaults for automation (no prompts, fast clones, skip LFS smudge)
- Allow interactive passes (inherit stdio) when the caller needs terminal UI

Token passing:
    Network-facing helpers (clone/fetch/fetch_branch/push_force_with_lease)
    and run_git() accept an optional ``token`` argument. When provided, the
    token is supplied to git through a temporary GIT_ASKPASS helper script
    that reads the secret from the child process environment. The token is
    therefore never present in the git command line (visible via ps /
    /proc/<pid>/cmdline), never embedded in the remote URL, and never
    written to .git/config.
"""

from __future__ import annotations

# ``subprocess`` is re-exported (not used here) so callers that reach
# for -- or substitute -- ``git_ops.subprocess`` keep working as they
# did while this package was a single module.
import subprocess

from .commands import (
    add,
    add_all,
    add_remote,
    checkout,
    commit_amend_no_edit,
    list_conflicted_files,
    rebase,
    rebase_abort,
    rebase_continue,
    rev_list_count,
    status_porcelain,
)
from .paths import (
    PathLike,
    _chmod_tree_safe,
    create_secure_tempdir,
    secure_rmtree,
)
from .process import (
    GitError,
    GitResult,
    _spawn_git,
    ensure_git_available,
    git_askpass_env,
    run_git,
)
from .redaction import (
    _ASKPASS_DEFAULT_USERNAME,
    _ASKPASS_SCRIPT,
    _ASKPASS_TOKEN_ENV,
    _ASKPASS_USERNAME_ENV,
    _BASIC_AUTH_IN_URL,
    _TOKEN_PATTERNS,
    _X_ACCESS_TOKEN_IN_URL,
    _build_git_env,
    _redact,
    _redact_seq,
    redact_text,
)
from .remote import (
    clone,
    fetch,
    fetch_branch,
    push_force_with_lease,
)

# Public API
__all__ = [
    "GitError",
    "GitResult",
    "ensure_git_available",
    "git_askpass_env",
    "redact_text",
    "run_git",
    "clone",
    "add_remote",
    "fetch",
    "checkout",
    "rebase",
    "rebase_continue",
    "rebase_abort",
    "status_porcelain",
    "list_conflicted_files",
    "add",
    "add_all",
    "commit_amend_no_edit",
    "push_force_with_lease",
    "rev_list_count",
    "create_secure_tempdir",
    "secure_rmtree",
]
