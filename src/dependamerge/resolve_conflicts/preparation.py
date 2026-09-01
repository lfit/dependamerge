# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The preparation phase of a fix run: PR lookup and workspace cloning.

:class:`_FixPreparationMixin` carries everything ``FixOrchestrator`` does
before a human is asked to resolve anything: fetching each PR's
repo/branch/permission details over REST, and cloning or fetching a
workspace per PR in parallel.  It also holds the best-effort progress and
logging plumbing those steps share, since neither a missing progress
tracker nor a failing injected logger may interrupt the flow.

It is a mixin rather than a separate collaborator so ``FixOrchestrator``
exposes exactly the method surface it always did.  Every attribute it
reads is established by ``FixOrchestrator.__init__``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from ..git_ops import (
    add_remote,
    checkout,
    clone,
    fetch_branch,
)
from ..github_async import GitHubAsync
from ..url_parser import default_github_host, derive_api_urls
from .models import FixOptions, PRContext, PRSelection

_LOG = logging.getLogger("dependamerge.resolve_conflicts")


class _FixPreparationMixin:
    """PR fetching and workspace preparation shared into ``FixOrchestrator``."""

    # Established by FixOrchestrator.__init__.
    _token: str
    _host: str
    _progress: object | None
    _logger: Callable[[str], None]

    def _safe_progress(self, method_name: str, *args: object) -> None:
        """Invoke a progress-tracker method if available; ignore UI errors.

        Progress display is best-effort: a missing tracker, a missing method,
        or a failing render must never interrupt the fix flow.
        """
        if not self._progress:
            return
        method = getattr(self._progress, method_name, None)
        if not callable(method):
            return
        try:
            method(*args)
        except Exception as exc:
            _LOG.debug("Progress %s failed: %s", method_name, exc, exc_info=True)

    async def fetch_pr_details(
        self, selections: Sequence[PRSelection]
    ) -> list[PRContext]:
        """
        Fetch PR details via REST (single GitHubAsync session) for all selections.

        Returns:
            A list of PRContext containing the necessary repo/branch/permission info.
        """
        contexts: list[PRContext] = []

        api_url, graphql_url = derive_api_urls(self._host or default_github_host())
        async with GitHubAsync(
            token=self._token, api_url=api_url, graphql_url=graphql_url
        ) as api:
            tasks = []
            for sel in selections:
                try:
                    owner, repo = sel.repository.split("/", 1)
                except ValueError:
                    self._log(
                        f"Skipping invalid repository full name: {sel.repository}"
                    )
                    continue

                tasks.append(self._fetch_one_pr(api, owner, repo, sel.pr_number))

            for coro in asyncio.as_completed(tasks):
                try:
                    ctx = await coro
                    if ctx:
                        contexts.append(ctx)
                except Exception as e:
                    self._log(f"Error fetching PR details: {e}")

        return contexts

    async def _fetch_one_pr(
        self, api: GitHubAsync, owner: str, repo: str, number: int
    ) -> PRContext | None:
        data = await api.get(f"/repos/{owner}/{repo}/pulls/{number}")
        if not isinstance(data, dict):
            return None

        base = data.get("base") or {}
        head = data.get("head") or {}
        base_repo = base.get("repo") or {}
        head_repo = head.get("repo") or {}

        base_branch = base.get("ref") or ""
        head_branch = head.get("ref") or ""
        base_full = base_repo.get("full_name") or f"{owner}/{repo}"
        head_full = head_repo.get("full_name") or base_full
        base_clone = base_repo.get("clone_url") or f"https://github.com/{base_full}.git"
        head_clone = head_repo.get("clone_url") or base_clone
        is_fork = bool(head_repo.get("fork")) if head_repo else False
        maint_mod = bool(data.get("maintainer_can_modify"))

        return PRContext(
            owner=owner,
            repo=repo,
            pr_number=number,
            base_branch=base_branch,
            head_branch=head_branch,
            base_repo_full_name=base_full,
            base_repo_clone_url=base_clone,
            head_repo_full_name=head_full,
            head_repo_clone_url=head_clone,
            is_fork=is_fork,
            maintainer_can_modify=maint_mod,
        )

    def _prepare_workspaces_parallel(
        self,
        contexts: Sequence[PRContext],
        base_dir: Path,
        options: FixOptions,
    ) -> list[tuple[PRContext, Path | None, str | None]]:
        """
        Clone/fetch repositories for contexts in parallel.

        Returns:
            List of tuples (context, workspace_path or None, error_message or None).
        """
        results: list[tuple[PRContext, Path | None, str | None]] = []

        def worker(ctx: PRContext) -> tuple[PRContext, Path | None, str | None]:
            try:
                ws = self._prepare_single_workspace(ctx, base_dir, options)
                return (ctx, ws, None)
            except Exception as e:
                return (ctx, None, str(e))

        max_workers = max(1, int(options.prefetch or 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(worker, c) for c in contexts]
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

        return results

    def _prepare_single_workspace(
        self,
        ctx: PRContext,
        base_dir: Path,
        options: FixOptions,
    ) -> Path:
        """
        Create a workspace directory and clone/fetch the necessary branches/remotes.

        Strategy:
        - Clone head repo (push target) at head_branch for PR.
        - If base repo differs, add 'upstream' remote and fetch base_branch.
        - If same repo, ensure base_branch is fetched from origin as well.
        """
        workspace_name = (
            f"{ctx.head_repo_full_name.replace('/', '__')}__pr_{ctx.pr_number}"
        )
        workspace = base_dir / workspace_name
        workspace.mkdir(parents=True, exist_ok=True)

        # Clone/fetch with clean (credential-free) URLs; the token is
        # supplied per-operation via GIT_ASKPASS so it never lands in
        # argv or the workspace's .git/config.
        origin_url = ctx.head_repo_clone_url
        upstream_url = ctx.base_repo_clone_url

        # Clone head repo
        self._log(f"Cloning {ctx.head_repo_full_name}@{ctx.head_branch} -> {workspace}")
        clone(
            origin_url,
            workspace,
            branch=ctx.head_branch,
            depth=50,
            single_branch=True,
            no_tags=True,
            filter_blobs=True,
            logger=self._log,
            token=self._token,
        )

        # Ensure we have base branch available for rebase
        if ctx.head_repo_full_name != ctx.base_repo_full_name:
            add_remote("upstream", upstream_url, cwd=workspace, logger=self._log)
            # Use ``fetch_branch`` so ``upstream/<base_branch>``
            # lands as a remote-tracking ref — the ``--single-branch``
            # clone above restricts the origin's configured refspec
            # to the PR head branch, so a bare
            # ``git fetch upstream <base>`` would only populate
            # ``FETCH_HEAD`` and the downstream
            # ``git rebase upstream/<base>`` in
            # :meth:`InteractiveResolver.resolve` would fail with
            # ``fatal: invalid upstream 'upstream/<base>'``.
            fetch_branch(
                "upstream",
                ctx.base_branch,
                cwd=workspace,
                depth=50,
                logger=self._log,
                token=self._token,
            )
        else:
            # Same repo; fetch the base branch from origin into the
            # remote-tracking ref (see comment in the fork branch
            # above for why ``fetch_branch`` is required rather than
            # a bare ``fetch``).
            fetch_branch(
                "origin",
                ctx.base_branch,
                cwd=workspace,
                depth=50,
                logger=self._log,
                token=self._token,
            )

        # Ensure we are on the head branch explicitly (detached HEAD safety)
        checkout(ctx.head_branch, cwd=workspace, create=False, logger=self._log)

        return workspace

    def _log(self, msg: str) -> None:
        try:
            self._logger(msg)
        except Exception:
            # The injected logger failed; record the cause with context
            # via the module logger, then still emit the message on stdout
            # so interactive output is not lost.
            _LOG.warning("Injected logger failed; using stdout fallback", exc_info=True)
            # aislop-ignore-next-line ai-slop/python-print-debug -- deliberate stdout fallback
            print(msg)
