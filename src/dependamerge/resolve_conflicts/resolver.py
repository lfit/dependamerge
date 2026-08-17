# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The interactive half of the fix workflow.

:class:`InteractiveResolver` takes a workspace that
:class:`~dependamerge.resolve_conflicts.FixOrchestrator` has already
cloned and walks a human through rebasing it onto the base branch: it
starts the rebase, hands each conflicted file to the user's editor or
mergetool, stages and continues until the rebase completes, amends
single-commit PRs so no extra commit appears, and force-pushes with lease.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ..git_ops import (
    GitError,
    add,
    add_all,
    commit_amend_no_edit,
    list_conflicted_files,
    push_force_with_lease,
    rebase,
    rebase_continue,
    rev_list_count,
    run_git,
)
from .models import FixOptions, PRContext


class InteractiveResolver:
    """
    Drives a manual conflict resolution process for a PR:
      - Start rebase onto base branch
      - On conflicts, for each conflicted file open user's editor or mergetool
      - Stage and continue rebase until clean
      - Amend commit when the PR is a single-commit change
      - Force push with lease to update the PR branch
    """

    def __init__(
        self,
        token: str,
        *,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self._token = token
        self._log = logger or (lambda m: None)

    def resolve(
        self, ctx: PRContext, workspace: Path, options: FixOptions
    ) -> tuple[bool, str]:
        """
        Resolve conflicts interactively in the given workspace.

        Returns:
            (success, message)
        """
        base_remote = (
            "upstream"
            if ctx.head_repo_full_name != ctx.base_repo_full_name
            else "origin"
        )
        base_ref = f"{base_remote}/{ctx.base_branch}"

        # Initial rebase attempt
        self._log(f"Rebasing onto {base_ref}")
        rb = rebase(
            base_ref,
            cwd=workspace,
            autostash=True,
            interactive=options.interactive,
            logger=self._log,
        )
        if rb.returncode == 0:
            # Clean rebase; proceed to post steps
            self._log("Rebase completed without conflicts.")
        else:
            self._log("Conflicts detected. Entering manual resolution loop.")
            # Loop until rebase completes or user aborts
            while True:
                conflicts = list_conflicted_files(cwd=workspace, logger=self._log)
                if not conflicts:
                    # Sometimes rebase stops without conflicts (e.g., needs staging)
                    # Try to continue directly.
                    cont = rebase_continue(
                        cwd=workspace, interactive=options.interactive, logger=self._log
                    )
                    if cont.returncode == 0:
                        break
                    # If still not continuing, give the user a chance to edit anything
                    self._open_editor_for_paths(workspace, [], options)
                    add_all(cwd=workspace, logger=self._log)
                    cont = rebase_continue(
                        cwd=workspace, interactive=options.interactive, logger=self._log
                    )
                    if cont.returncode == 0:
                        break
                    # If it still fails, abort with an error message
                    return False, "Rebase could not continue and no conflicts listed"

                # Present and resolve each conflicted file
                self._log(f"Conflicted files: {', '.join(conflicts)}")
                if options.mergetool:
                    # Prefer mergetool if requested/configured
                    for path in conflicts:
                        self._run_mergetool(workspace, path, options)
                        add(path, cwd=workspace, logger=self._log)
                else:
                    # Open editor for each file
                    self._open_editor_for_paths(workspace, conflicts, options)
                    add(conflicts, cwd=workspace, logger=self._log)

                # Attempt to continue
                cont = rebase_continue(
                    cwd=workspace, interactive=options.interactive, logger=self._log
                )
                if cont.returncode == 0:
                    break
                # If still conflicts, loop again

        # Post-rebase: decide on amend rule
        try:
            count_expr = f"{base_ref}..HEAD"
            commit_count = rev_list_count(count_expr, cwd=workspace, logger=self._log)
        except GitError:
            commit_count = 0

        if commit_count == 1:
            # Single-commit PR: amend to preserve no extra top commit (no message change)
            self._log(
                "Single-commit change detected; amending commit without editing message."
            )
            try:
                commit_amend_no_edit(cwd=workspace, logger=self._log)
            except GitError as e:
                # Non-fatal; continue to push anyway
                self._log(f"Warning: amend failed: {e}")

        # Force push to update PR head branch
        self._log(
            f"Pushing updated branch with --force-with-lease to origin {ctx.head_branch}"
        )
        try:
            push_force_with_lease(
                "origin",
                "HEAD",
                f"refs/heads/{ctx.head_branch}",
                cwd=workspace,
                logger=self._log,
                token=self._token,
            )
        except GitError as e:
            return False, f"Push failed: {e}"

        return (
            True,
            "Rebased, amended (if applicable), and force-pushed to trigger checks",
        )

    def _open_editor_for_paths(
        self, cwd: Path, paths: Sequence[str], options: FixOptions
    ) -> None:
        """
        Open the user's editor for the given file paths. If no paths provided,
        open the editor at the repository root to allow manual edits.
        """
        editor_cmd = self._pick_editor(options)
        if not editor_cmd:
            # As a last resort, print instructions
            self._log(
                "No editor found. Please resolve conflicts manually in the workspace and then continue."
            )
            return

        # If the editor is VS Code, ensure we wait for the window to close
        # (-w) so the rebase does not continue before the user has saved
        # their conflict resolutions.
        #
        # Match on the launcher's program name (basename without a Windows
        # extension) against the known VS Code commands rather than a naive
        # substring test: a plain ``"code" in cmd_parts[0]`` would also fire
        # on unrelated binaries such as ``encode``, ``xcode`` or ``mycode``,
        # while still missing path-qualified launchers like ``/usr/bin/code``.
        cmd_parts = shlex.split(editor_cmd)
        prog = (
            Path(cmd_parts[0]).name.lower().removesuffix(".cmd").removesuffix(".exe")
            if cmd_parts
            else ""
        )
        if prog in ("code", "code-insiders") and "-w" not in cmd_parts:
            cmd_parts.append("-w")

        if paths:
            for p in paths:
                self._run_editor(cmd_parts, cwd, p)
        else:
            # Open editor at repo root
            self._run_editor(cmd_parts, cwd, None)

    def _run_editor(
        self, cmd_parts: list[str], cwd: Path, rel_path: str | None
    ) -> None:
        args = list(cmd_parts)
        if rel_path:
            args.append(rel_path)
        self._log(f"Opening editor: {' '.join(args)}")
        subprocess.run(args, cwd=str(cwd), check=False)

    def _run_mergetool(self, cwd: Path, rel_path: str, options: FixOptions) -> None:
        # Prefer --no-prompt to block until the tool finishes for this file
        args = ["git", "mergetool", "--no-prompt", "--", rel_path]
        run_git(
            args,
            cwd=cwd,
            interactive=options.interactive,
            check=False,
            logger=self._log,
        )

    def _pick_editor(self, options: FixOptions) -> str | None:
        if options.editor:
            return options.editor
        # Environment-driven choice
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if editor:
            return editor
        # Platform defaults
        if sys.platform.startswith("win"):
            return "notepad"
        # POSIX default
        return "vi"
