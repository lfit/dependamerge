# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The fix workflow's top-level coordinator.

:class:`FixOrchestrator` owns the shape of a fix run: it opens a base
workspace directory, asks :class:`._FixPreparationMixin` to fetch PR
details and clone workspaces, drives
:class:`~dependamerge.resolve_conflicts.InteractiveResolver` over each
prepared workspace in turn, and tidies up afterwards.

Interactive resolution stays serial so the user only ever faces one editor
or mergetool session at a time; only the clone/fetch preparation runs in
parallel.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

from ..git_ops import (
    create_secure_tempdir,
    rebase_abort,
    secure_rmtree,
)
from .models import FixOptions, FixResult, PRContext, PRSelection
from .preparation import _FixPreparationMixin
from .resolver import InteractiveResolver


class FixOrchestrator(_FixPreparationMixin):
    """
    Coordinates fetching PR details, preparing workspaces, and running the interactive
    conflict resolution flow. Interactive resolution is executed serially; workspace
    preparation (clone/fetch) is parallelized for responsiveness.
    """

    def __init__(
        self,
        token: str,
        *,
        progress_tracker: object | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        if not token:
            raise ValueError("A GitHub token is required for fix operations.")
        self._token = token
        self._progress = progress_tracker
        self._logger = logger or (lambda m: None)

    def run(
        self, selections: Sequence[PRSelection], options: FixOptions
    ) -> list[FixResult]:
        """
        Perform the full fix process:
          - create or use secure base workdir
          - fetch PR details
          - prefetch (clone/fetch) workspaces in parallel
          - resolve each PR interactively in serial
          - push updates and cleanup
        """
        base_dir, temp_created = self._open_base_workdir(options)

        # Wrap in try/finally for cleanup
        try:
            self._safe_progress(
                "update_operation", "Fetching PR details for fix candidates..."
            )

            contexts = asyncio.run(self.fetch_pr_details(selections))

            prepared, to_prepare = self._partition_by_pushability(contexts)

            # Prefetch workspaces (clone & fetch) in parallel
            if to_prepare:
                self._safe_progress(
                    "update_operation", "Preparing workspaces (clone/fetch repos)..."
                )
                prepared += self._prepare_workspaces_parallel(
                    to_prepare, base_dir, options
                )
            else:
                self._log("No PRs eligible for workspace preparation.")

            return self._resolve_prepared(prepared, options)
        finally:
            self._cleanup_base_workdir(base_dir, temp_created, options)

    def _open_base_workdir(self, options: FixOptions) -> tuple[Path, bool]:
        """
        Resolve the directory that will hold every per-PR workspace.

        Returns:
            (base_dir, temp_created), where temp_created records whether the
            directory was created here and is therefore ours to remove.
        """
        # Create secure base workdir if not provided
        if options.workdir:
            base_dir = Path(options.workdir).absolute()
            base_dir.mkdir(parents=True, exist_ok=True)
            return base_dir, False

        base_dir = Path(create_secure_tempdir(prefix="dependamerge-")).absolute()
        self._log(f"Created secure temp workspace at {base_dir}")
        return base_dir, True

    def _partition_by_pushability(
        self, contexts: Sequence[PRContext]
    ) -> tuple[list[tuple[PRContext, Path | None, str | None]], list[PRContext]]:
        """
        Split fetched contexts into the ones worth preparing and the rest.

        Returns:
            (prepared, to_prepare), where prepared already carries the
            failure records for forks we have no permission to push to.
        """
        # Filter out PRs we cannot push to (forks without maintainer_can_modify)
        prepared: list[tuple[PRContext, Path | None, str | None]] = []
        to_prepare: list[PRContext] = []
        for ctx in contexts:
            if ctx.is_fork and not ctx.maintainer_can_modify:
                msg = "Skipping fork without maintainer-can-modify permission"
                self._log(f"{ctx.base_repo_full_name}#{ctx.pr_number}: {msg}")
                prepared.append((ctx, None, msg))
            else:
                to_prepare.append(ctx)

        return prepared, to_prepare

    def _resolve_prepared(
        self,
        prepared: Sequence[tuple[PRContext, Path | None, str | None]],
        options: FixOptions,
    ) -> list[FixResult]:
        """
        Walk the prepared workspaces through interactive conflict resolution.

        Returns:
            One FixResult per entry, including the ones whose preparation failed.
        """
        # Interactive resolution in serial to keep terminal clear
        resolver = InteractiveResolver(
            token=self._token, logger=options.logger or self._logger
        )

        results: list[FixResult] = []
        for ctx, workspace, prep_err in prepared:
            sel = ctx.selection
            if workspace is None:
                results.append(
                    FixResult(
                        selection=sel,
                        success=False,
                        message=prep_err or "Preparation failed",
                    )
                )
                continue

            self._safe_progress("suspend")
            self._log(
                f"Starting interactive rebase for {ctx.base_repo_full_name}#{ctx.pr_number} in {workspace}"
            )

            try:
                ok, msg = resolver.resolve(ctx, workspace, options)
                results.append(
                    FixResult(
                        selection=sel,
                        success=ok,
                        message=msg,
                        workspace=str(workspace),
                    )
                )
                self._log(f"{ctx.base_repo_full_name}#{ctx.pr_number}: {msg}")
            except KeyboardInterrupt:
                # Attempt to abort any in-progress rebase and record failure
                try:
                    rebase_abort(cwd=workspace)
                except Exception as abort_err:
                    # Cleanup abort is best-effort; the failure is
                    # already recorded below regardless.  Surface it
                    # through the orchestrator logger so an unexpected
                    # cleanup failure remains discoverable.
                    self._log(
                        f"{ctx.base_repo_full_name}#{ctx.pr_number}: "
                        f"rebase --abort cleanup failed: {abort_err}"
                    )
                results.append(
                    FixResult(
                        selection=sel,
                        success=False,
                        message="Aborted by user",
                        workspace=str(workspace),
                    )
                )
                self._log(f"{ctx.base_repo_full_name}#{ctx.pr_number}: Aborted by user")
            except Exception as e:
                results.append(
                    FixResult(
                        selection=sel,
                        success=False,
                        message=f"Error: {e}",
                        workspace=str(workspace),
                    )
                )
                self._log(f"{ctx.base_repo_full_name}#{ctx.pr_number}: Error: {e}")
            finally:
                self._safe_progress("resume")

        return results

    def _cleanup_base_workdir(
        self, base_dir: Path, temp_created: bool, options: FixOptions
    ) -> None:
        """Remove the base workspace when we created it and may discard it."""
        # Cleanup base temp directory if we created it and keep_temp is False
        if temp_created and not options.keep_temp:
            try:
                secure_rmtree(str(base_dir))
                self._log(f"Removed temp workspace at {base_dir}")
            except Exception as e:
                self._log(f"Warning: Failed to remove temp workspace {base_dir}: {e}")
