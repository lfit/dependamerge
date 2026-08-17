# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Filesystem helpers for git workspaces.

Provides the shared ``PathLike`` alias plus utilities for creating
temporary working directories with restrictive permissions and for
removing them again without tripping over read-only paths.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path

# Type aliases
PathLike = str | Path


def create_secure_tempdir(prefix: str = "dependamerge-") -> str:
    """
    Create a temporary directory with restrictive permissions (0700).

    Returns:
        Absolute path to the created directory.
    """
    path = tempfile.mkdtemp(prefix=prefix)
    try:
        os.chmod(path, 0o700)
    except OSError:
        # Best effort; continue even if chmod fails (Windows, etc.)
        pass
    return path


def _chmod_tree_safe(
    path: PathLike, file_mode: int = 0o600, dir_mode: int = 0o700
) -> None:
    """Best-effort to ensure paths are writable/removable by adjusting modes."""
    try:
        p = Path(path)
        if not p.exists():
            return
        for root, dirs, files in os.walk(p, topdown=False):
            for name in files:
                fp = Path(root) / name
                try:
                    os.chmod(fp, file_mode)
                except OSError:
                    # Best-effort: skip files we cannot chmod; the
                    # later rmtree retry handles stubborn paths.
                    pass
            for name in dirs:
                dp = Path(root) / name
                try:
                    os.chmod(dp, dir_mode)
                except OSError:
                    # Best-effort: skip dirs we cannot chmod.
                    pass
        try:
            os.chmod(p, dir_mode)
        except OSError:
            # Best-effort: ignore failure to chmod the tree root.
            pass
    except OSError:
        # Ignore any errors; deletion attempts will proceed anyway
        pass


def secure_rmtree(path: PathLike) -> None:
    """
    Remove a directory tree, attempting to scrub permissions first.

    Note: This does not guarantee cryptographically secure wiping of file
    contents. It makes a best effort to avoid permission-related failures
    and to remove all files. For true secure deletion, additional OS-level
    facilities are required and platform-dependent.
    """
    _chmod_tree_safe(path)
    try:
        shutil.rmtree(path)
    except Exception:
        # Retry with onerror handler that adjusts perms
        def _onerror(func, p, exc):
            try:
                st = os.lstat(p)
                if stat.S_ISDIR(st.st_mode):
                    os.chmod(p, 0o700)
                else:
                    os.chmod(p, 0o600)
                func(p)
            except OSError:
                # Give up on this path
                pass

        shutil.rmtree(path, onerror=_onerror)
