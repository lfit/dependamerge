# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""The three ways Step 5 can bring a pull request up to date.

- :func:`_run_local_path` clones and rebases locally, preserving the
  operator's commit signature.
- :func:`_run_dependabot_macro_path` asks dependabot to rebase itself,
  which it does with its own signing key.
- :func:`_run_rest_path` calls REST ``update-branch`` and waits for
  GitHub to recompute mergeability.

``local_rebase_pr`` is deliberately reached through the
:mod:`~dependamerge.rebase.local_rebase` module object rather than
imported by name, so that patching
``dependamerge.rebase.local_rebase.local_rebase_pr`` is observed here.
Importing the function directly would bind it into this module's
globals at import time and make such a patch silently ineffective —
the tests would keep passing while the real workflow shelled out to
``git``.
"""

from __future__ import annotations

import asyncio

from ..models import PullRequestInfo
from ..slot_lease import parked
from . import local_rebase, polling
from .context import RebaseContext, Step5Outcome, _record_rebase, _set_tracker_state


async def _run_local_path(
    *,
    ctx: RebaseContext,
    pr_info: PullRequestInfo,
    owner: str,
    repo: str,
    local_reason: str,
) -> None:
    """Local-rebase path.  Always succeeds from the caller's POV.

    Whether the underlying ``git`` workflow succeeds or fails, we
    never fall back to REST ``update-branch`` (doing so would
    defeat the whole point of the local path).

    Auto-merge enablement and ``_rebased_prs`` marking are linked:

    - If auto-merge gets enabled, mark ``_rebased_prs`` so Step
      5.5 skips this PR; auto-merge will handle the bounded wait
      server-side and Step 6's skip gate routes to
      ``AUTO_MERGE_PENDING``.
    - If auto-merge cannot be enabled (repo doesn't allow it,
      branch protection blocks it, etc.), do **not** mark
      ``_rebased_prs`` so Step 5.5 still runs its bounded poll
      loop. Without this, GitHub may still be recomputing
      mergeability after the force-push when Step 6 fires and a
      manual merge attempt would 405 transiently. Letting Step
      5.5 wait gives GitHub time to settle before Step 6 acts.
    """
    ctx.log.debug(
        "Local rebase: %s [%s]",
        pr_info.html_url,
        local_reason,
    )
    try:
        local_rebase_ok = await local_rebase.local_rebase_pr(
            pr_info=pr_info,
            owner=owner,
            repo=repo,
            token=ctx.token,
            log=ctx.log,
            host=ctx.host,
        )
    except Exception as exc:
        ctx.log.debug(
            "Local rebase raised unexpectedly for %s: %s",
            pr_info.html_url,
            exc,
        )
        local_rebase_ok = False

    # Try to enable auto-merge. We capture the return value so we
    # can decide whether Step 5.5 should also run (see comment
    # below).
    try:
        auto_merge_ok = await ctx.enable_auto_merge(pr_info, owner, repo)
    except Exception as exc:
        ctx.log.debug(
            "Could not enable auto-merge after local rebase for %s: %s",
            pr_info.html_url,
            exc,
        )
        auto_merge_ok = False

    # Only mark ``_rebased_prs`` when auto-merge is active.
    # Otherwise leave it unset so Step 5.5 still runs its bounded
    # poll loop — GitHub may still be recomputing mergeability
    # after the force-push, and a manual merge attempt in Step 6
    # without that wait would 405 transiently.  When auto-merge
    # *is* active, Step 6's skip gate routes the PR to
    # ``AUTO_MERGE_PENDING`` directly, so the Step 5.5 wait would
    # only double the merge_timeout.
    if auto_merge_ok:
        ctx.rebased_prs.add(f"{owner}/{repo}#{pr_info.number}")

    # Either way the rebase attempt is over, so clear the "rebasing"
    # state ``perform_step5_rebase`` set before dispatch.  Leaving it
    # set on the failure branch would strand the PR displaying as
    # "Rebasing" for the rest of the run, since this path defers to
    # auto-merge rather than reaching a terminal outcome that would
    # clear it.
    _set_tracker_state(ctx, pr_info, None)
    if local_rebase_ok:
        ctx.log.debug("Rebased (local): %s", pr_info.html_url)
        # The cumulative "Rebased" total is what keeps a record of the
        # rebase from here on.
        _record_rebase(ctx)
    else:
        ctx.log.debug(
            "Local rebase failed; deferring to auto-merge: %s",
            pr_info.html_url,
        )


async def _run_dependabot_macro_path(
    *,
    ctx: RebaseContext,
    pr_info: PullRequestInfo,
    owner: str,
    repo: str,
    local_reason: str,
) -> bool:
    """Request a rebase via the ``@dependabot rebase`` macro.

    Used instead of the local-rebase path for dependabot PRs: the bot
    rebases the branch onto the current base and force-pushes a
    commit signed with its own key, preserving the ``Verified`` badge
    that signature-requiring branch protection demands.

    The rebase completes asynchronously (dependabot typically takes
    one to a few minutes), so this path never waits for it.  It mirrors
    the local path's auto-merge contract instead:

    - Auto-merge armed → mark ``_rebased_prs`` so Step 5.5 skips the
      PR and Step 6's skip gate routes it to ``AUTO_MERGE_PENDING``;
      GitHub merges server-side once the rebase lands and checks pass.
    - Auto-merge unavailable → leave the PR unmarked so Step 5.5
      still runs its bounded wait for the rebase + checks.

    Returns True when the macro was posted (or was already pending);
    False when it could not be requested — the caller then falls back
    to the local-rebase path.
    """
    if ctx.request_dependabot_rebase is None:
        return False

    ctx.log.debug(
        "Dependabot rebase macro: %s [%s]",
        pr_info.html_url,
        local_reason,
    )
    try:
        requested = await ctx.request_dependabot_rebase(pr_info, owner, repo)
    except Exception as exc:
        ctx.log.debug(
            "Dependabot rebase request raised for %s: %s",
            pr_info.html_url,
            exc,
        )
        requested = False
    if not requested:
        return False

    try:
        auto_merge_ok = await ctx.enable_auto_merge(pr_info, owner, repo)
    except Exception as exc:
        ctx.log.debug(
            "Could not enable auto-merge after dependabot rebase request for %s: %s",
            pr_info.html_url,
            exc,
        )
        auto_merge_ok = False

    # Same marking rule as the local path: only skip Step 5.5 when
    # auto-merge is armed to finish the job server-side.
    if auto_merge_ok:
        ctx.rebased_prs.add(f"{owner}/{repo}#{pr_info.number}")

    ctx.log.debug("Rebase requested (dependabot macro): %s", pr_info.html_url)
    # No counting here: ``ctx.request_dependabot_rebase`` owns the
    # cumulative totals and records them only when it actually posts
    # the macro (its duplicate guard returns True without posting when
    # an earlier run already requested the rebase).  Counting again
    # here would double the posted case and invent the guarded one.
    _set_tracker_state(ctx, pr_info, None)
    return True


async def _run_rest_path(
    *,
    ctx: RebaseContext,
    pr_info: PullRequestInfo,
    owner: str,
    repo: str,
) -> Step5Outcome:
    """Legacy REST ``update-branch`` path with post-rebase polling.

    Uses the GitHub REST API to bring the PR up to date, enables
    auto-merge so the PR merges even if we time out waiting for
    status checks, then polls until checks complete or
    ``merge_timeout`` elapses.  Updates ``pr_info`` in place with
    the post-rebase state.

    Returns a :class:`Step5Outcome` whose ``failed`` field is True
    when ``update_branch`` (or the polling apparatus) raises an
    exception — the caller should mark the merge as ``FAILED`` in
    that case.
    """
    if ctx.github_client is None:
        return Step5Outcome(failed=True, error_message="GitHub client not initialized")
    client = ctx.github_client

    try:
        await client.update_branch(owner, repo, pr_info.number)
        _record_rebase(ctx)

        # Enable auto-merge so the PR merges even if we time out
        # waiting for status checks.
        auto_merge_ok = await ctx.enable_auto_merge(pr_info, owner, repo)
        if auto_merge_ok:
            ctx.log.debug(
                "Auto-merge enabled after rebase for %s/%s#%s",
                owner,
                repo,
                pr_info.number,
            )

        # Wait briefly for GitHub to start processing the update.
        # The full recheck interval (default 10s) is unnecessary here:
        # ``_poll_post_rebase`` polls at that cadence anyway and
        # tolerates the transient ``null``/``behind`` states GitHub
        # reports while recomputing, so a short head start just gets
        # the first data point sooner.  The settle sleep and the poll
        # are both waits on GitHub-side processing, so the worker's
        # concurrency slot is released for their duration
        # (``parked()`` — see ``slot_lease.py``).
        ctx.log.debug("Waiting for rebase to process: %s", pr_info.html_url)
        _set_tracker_state(ctx, pr_info, "waiting")
        async with parked():
            await asyncio.sleep(min(2.0, ctx.merge_recheck_interval))

            (
                updated_mergeable,
                updated_mergeable_state,
            ) = await polling._poll_post_rebase(
                ctx=ctx,
                pr_info=pr_info,
                owner=owner,
                repo=repo,
                auto_merge_ok=auto_merge_ok,
            )

        # Update our PR info with the latest state.  Preserve the
        # previous non-None values when the refresh returns
        # ``null`` (GitHub is still computing).  The Step 6
        # auto-merge skip gate accepts both ``True`` and ``None``
        # (it excludes only the explicit ``False`` case), so a
        # transient null no longer blocks the auto-merge path on
        # its own.  We still preserve the prior known ``True`` so
        # downstream logging and any future tightening of that
        # predicate get an accurate state to work with.  The same
        # rationale applies to ``mergeable_state``: GitHub returns
        # ``null`` while still computing, and the post-rebase
        # reporting / Step 5.5 logic branches on this value (e.g.
        # "clean" vs "blocked" vs "behind"); a transient ``None``
        # would otherwise be classified as the catch-all "other
        # state" branch.
        if updated_mergeable is not None:
            pr_info.mergeable = updated_mergeable
        if updated_mergeable_state is not None:
            pr_info.mergeable_state = updated_mergeable_state

        # Mark this PR as having gone through the Step 5 rebase
        # + poll path.  Step 5.5 will consult ``_rebased_prs`` to
        # avoid doubling the merge_timeout when the rebase exits
        # in ``blocked`` or ``behind`` state.
        ctx.rebased_prs.add(f"{owner}/{repo}#{pr_info.number}")

        _set_tracker_state(ctx, pr_info, None)
        polling._log_post_rebase_status(ctx=ctx, pr_info=pr_info)
        return Step5Outcome()

    except Exception as exc:
        ctx.log.warning("Rebase failed for %s: %s", pr_info.html_url, exc)
        return Step5Outcome(failed=True, error_message=f"Failed to rebase PR: {exc}")
