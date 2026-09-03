# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Pre-flight check that the token can actually merge.

A bulk run that discovers its token is read-only halfway through leaves a
partially merged batch, so the required permissions are probed against a
representative repository first and reported in full when missing.
"""

import asyncio
from collections import OrderedDict
from typing import Any

import typer

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
import dependamerge.cli as _pkg

from ..github_async import (
    GitHubAsync,
)
from ..github_async import (
    PermissionError as GitHubPermissionError,
)
from ..url_parser import default_github_host, derive_api_urls
from ._app import console
from ._context import _MergeContext
from ._merge_inputs import _source_pr_modifies_workflows


def _report_missing_permissions(
    missing_perms: list[str],
    perm_results: dict[str, dict[str, Any]],
) -> None:
    """Print grouped guidance for missing token permissions and exit.

    Groups failing operations by their guidance tuple so we don't repeat
    the same URL block once per failed operation.  In the common 401 /
    expired-token case every operation lands on the same guidance, which
    used to produce four near-identical six-line blocks; under the grouped
    renderer it collapses to a single list of operations followed by one
    guidance block.  The 403 case (per-operation scope guidance) naturally
    falls out as one group per distinct guidance tuple, so each operation
    still gets its own scope hint.
    """
    console.print("\n❌ Token Permission Check Failed:\n")
    groups: OrderedDict[
        tuple[str | None, str | None, str | None],
        list[tuple[str, str]],
    ] = OrderedDict()
    for op in missing_perms:
        result = perm_results[op]
        guidance = result.get("guidance") or {}
        key = (
            guidance.get("classic"),
            guidance.get("fine_grained"),
            guidance.get("fix"),
        )
        groups.setdefault(key, []).append((op, result.get("error", "")))

    for key, items in groups.items():
        classic, fine_grained, fix = key
        # List the failing operations in this group.
        for op, _err in items:
            console.print(f"   • {op}")
        # Show the shared error message once if every op in the group
        # reports the same one (the 401 case); otherwise show each
        # operation's message inline so distinct failures stay
        # distinguishable.
        distinct_errors = {err for _, err in items if err}
        if len(distinct_errors) == 1:
            console.print(f"\n   {next(iter(distinct_errors))}")
        elif distinct_errors:
            console.print()
            for op, err in items:
                if err:
                    console.print(f"   {op}: {err}")
        # Single guidance block for the whole group.
        if classic or fine_grained or fix:
            console.print()
            if classic:
                console.print(f"   Classic:       {classic}")
            if fine_grained:
                console.print(f"   Fine-grained:  {fine_grained}")
            if fix:
                console.print(f"   Fix:           {fix}")
        console.print()  # blank line between groups

    console.print("💡 Update your token permissions and try again.")
    raise typer.Exit(code=3)


def _check_merge_permissions(ctx: _MergeContext) -> None:
    """Pre-flight token permission check.

    Exits early when required permissions are missing.
    """
    console.print("🔍 Checking token permissions...")

    async def _check() -> dict[str, dict[str, Any]]:
        # The preflight has to address the same host as the merge that
        # follows: checked against api.github.com, an Enterprise token
        # reports every operation as missing and aborts the run before
        # the host-aware clients are ever built.
        api_url, graphql_url = derive_api_urls(ctx.host or default_github_host())
        async with GitHubAsync(
            token=ctx.token, api_url=api_url, graphql_url=graphql_url
        ) as client:
            operations = ["approve", "merge", "branch_protection"]
            if not ctx.no_fix:
                operations.append("update_branch")
            # Only assert the workflow scope when the source PR actually
            # touches workflow files.  The check is a no-op (passes) for
            # fine-grained / app tokens whose scopes cannot be introspected.
            if _source_pr_modifies_workflows(ctx):
                operations.append("merge_workflow")
            return await client.check_token_permissions(
                operations, ctx.owner, ctx.repo_name
            )

    try:
        perm_results = asyncio.run(_check())
        # Every permission probed by the pre-flight check is now
        # required by some part of the merge flow:
        #
        # * ``approve``, ``merge``, ``update_branch`` — obviously
        #   needed for the core merge actions.
        # * ``branch_protection`` — used by the signature-preserving
        #   local-rebase gate (``requires_commit_signatures``) and
        #   by Step 5.5's block-reason analysis.  A token without
        #   it degrades the merge flow in ways that are confusing
        #   to diagnose at runtime, so treat it as blocking up-front.
        #
        # There is consequently no advisory class — a missing
        # permission is always a hard failure.  If a future
        # operation is genuinely optional, partition
        # ``missing_perms`` into blocking and advisory subsets
        # (and re-introduce the ``⚠️ … (non-blocking)`` display
        # for the advisory subset) at that point.
        missing_perms = [
            op for op, result in perm_results.items() if not result["has_permission"]
        ]
        if missing_perms:
            _report_missing_permissions(missing_perms, perm_results)
        console.print("✅ Token has required permissions")
    except GitHubPermissionError as e:
        console.print(f"\n❌ Permission check failed: {e}")
        raise typer.Exit(code=3) from e
    except typer.Exit:
        # The hard-failure branch above raises typer.Exit(code=3)
        # to abort the run.  Without re-raising it here it would
        # be caught by the broad ``except Exception`` below and
        # silently downgraded to "Continuing anyway...", letting
        # the merge proceed against a token we already know lacks
        # the required permissions.
        raise
    except Exception as e:
        console.print(f"⚠️ Could not verify permissions: {e}")
        console.print("   Continuing anyway...")


def _maybe_check_merge_permissions(ctx: _MergeContext) -> None:
    """Run the write-permission pre-flight unless this is a dry run.

    A dry run performs no writes (no merge, approve, rebase, or branch
    update), so the merge/approve/update_branch/branch_protection scopes
    are not required.  Skipping the check lets the full analysis and
    preview run under a read-only token — which is exactly what the CI
    dry-run test matrix needs, since the default workflow token (and
    other low-privilege tokens) cannot satisfy the write scopes.
    """
    if ctx.dry_run:
        console.print(
            "🧪 Dry run: skipping token permission check (no changes will be made)"
        )
        return
    _pkg._check_merge_permissions(ctx)
