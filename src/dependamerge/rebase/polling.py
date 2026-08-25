# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""The post-``update-branch`` polling loop and its logging.

After the REST rebase path calls ``update_branch`` GitHub needs a
little while to recompute mergeability.  :func:`_poll_post_rebase`
waits it out, :func:`_poll_should_continue` holds the per-state
decisions that drive the loop, and the two ``_log_*`` helpers carry the
reporting so neither of the above has to.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..models import PullRequestInfo
from .context import RebaseContext


async def _poll_post_rebase(
    *,
    ctx: RebaseContext,
    pr_info: PullRequestInfo,
    owner: str,
    repo: str,
    auto_merge_ok: bool,
) -> tuple[bool | None, str | None]:
    """Poll the PR after ``update_branch`` until it stabilises.

    Returns the latest ``(mergeable, mergeable_state)`` observed.
    Updates ``pr_info.head_sha`` in place when the refresh shows a
    new head commit (so any subsequent ``analyze_block_reason()``
    call queries the rebased commit, not the pre-rebase one).
    """
    if ctx.github_client is None:
        return pr_info.mergeable, pr_info.mergeable_state
    client = ctx.github_client

    updated_mergeable: bool | None = pr_info.mergeable
    updated_mergeable_state: str | None = pr_info.mergeable_state

    for check_attempt in range(ctx.merge_poll_max_attempts):
        updated_pr_data: Any = await client.get(
            f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
        )

        if isinstance(updated_pr_data, dict):
            updated_mergeable = updated_pr_data.get("mergeable")
            updated_mergeable_state = updated_pr_data.get("mergeable_state")
            updated_head = (updated_pr_data.get("head") or {}).get("sha")
            if updated_head:
                pr_info.head_sha = updated_head
        else:
            updated_mergeable = None
            updated_mergeable_state = None

        if _poll_should_continue(
            ctx=ctx,
            pr_info=pr_info,
            attempt=check_attempt,
            mergeable_state=updated_mergeable_state,
            auto_merge_ok=auto_merge_ok,
        ):
            await asyncio.sleep(ctx.merge_recheck_interval)
            continue
        break

    return updated_mergeable, updated_mergeable_state


def _poll_should_continue(
    *,
    ctx: RebaseContext,
    pr_info: PullRequestInfo,
    attempt: int,
    mergeable_state: str | None,
    auto_merge_ok: bool,
) -> bool:
    """Return True when the post-rebase poll loop should keep waiting.

    Centralising the per-state decisions here keeps
    :func:`_poll_post_rebase` short and readable.
    """
    if mergeable_state == "clean":
        return False

    last_attempt = attempt >= ctx.merge_poll_max_attempts - 1

    if mergeable_state == "behind":
        if last_attempt:
            return False
        ctx.log.debug(
            "PR still processing rebase, waiting... (attempt %d/%d)",
            attempt + 1,
            ctx.merge_poll_max_attempts,
        )
        return True

    if mergeable_state == "blocked":
        if last_attempt:
            _log_blocked_timeout(ctx=ctx, pr_info=pr_info, auto_merge_ok=auto_merge_ok)
            return False
        ctx.log.debug(
            "PR status checks running after rebase, waiting... (attempt %d/%d)",
            attempt + 1,
            ctx.merge_poll_max_attempts,
        )
        return True

    if mergeable_state in (None, "", "unknown"):
        # GitHub is still computing mergeability (typically right
        # after update_branch).  Treat as transient and keep
        # polling until the deadline or a concrete state arrives —
        # breaking here would otherwise exit prematurely and (if
        # auto-merge enablement failed) fall through to a manual
        # merge attempt against the still-resolving PR state.
        #
        # All three values mean the same thing: GitHub returns null,
        # "" and "unknown" interchangeably while recomputing.  This
        # poll runs immediately after ``update_branch``, which is
        # precisely when a recompute is most likely, so answering
        # "unknown" rather than null is close to a coin flip --- and
        # testing only for null caught just some of the cases this
        # branch was written for.  The same triple is used by
        # ``_refresh_pr_mergeability``, ``_required_workflows`` and
        # ``_check_wait``; this is a fourth site agreeing with them
        # rather than holding a fourth opinion.
        if last_attempt:
            return False
        ctx.log.debug(
            "PR mergeable_state still computing after rebase, "
            "waiting... (attempt %d/%d)",
            attempt + 1,
            ctx.merge_poll_max_attempts,
        )
        return True

    # Any other concrete state ("dirty", "draft", "unstable", ...)
    # ends the poll loop immediately.
    return False


def _log_blocked_timeout(
    *,
    ctx: RebaseContext,
    pr_info: PullRequestInfo,
    auto_merge_ok: bool,
) -> None:
    """Log when the post-rebase poll times out blocked (log only)."""
    if auto_merge_ok:
        ctx.log.warning(
            "Auto-merge will complete: %s [timeout waiting for checks]",
            pr_info.html_url,
        )
    else:
        ctx.log.warning(
            "Proceeding without checks: %s [timeout waiting for checks]",
            pr_info.html_url,
        )


def _log_post_rebase_status(
    *,
    ctx: RebaseContext,
    pr_info: PullRequestInfo,
) -> None:
    """Log the post-rebase status based on the final mergeable_state."""
    state = pr_info.mergeable_state
    if state == "clean":
        ctx.log.debug("Rebased: %s", pr_info.html_url)
    elif state == "behind":
        ctx.log.debug("Rebased: %s [still behind after rebase]", pr_info.html_url)
    elif state == "blocked":
        ctx.log.debug("Rebased: %s [waiting for status checks]", pr_info.html_url)
    else:
        ctx.log.debug("Rebased: %s [state=%s]", pr_info.html_url, state)
