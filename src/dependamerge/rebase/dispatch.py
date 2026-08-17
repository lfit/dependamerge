# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Step 5 of the merge flow: the top-level rebase dispatcher.

:func:`perform_step5_rebase` is the only function in this package the
merge manager calls.  It asks
:func:`~dependamerge.rebase.decide.should_use_local_rebase` which path
to take and then hands off to one of the helpers in
:mod:`~dependamerge.rebase.paths`.
"""

from __future__ import annotations

from ..bot_identity import is_dependabot
from ..models import PullRequestInfo
from .context import RebaseContext, Step5Outcome, _set_tracker_state
from .decide import BaseRef, should_use_local_rebase
from .paths import _run_dependabot_macro_path, _run_local_path, _run_rest_path


async def perform_step5_rebase(
    *,
    ctx: RebaseContext,
    pr_info: PullRequestInfo,
    owner: str,
    repo: str,
) -> Step5Outcome:
    """Run Step 5 of the merge flow: bring the PR up to date with its base.

    Dispatches between the local-git path (signature-preserving)
    and the legacy REST ``update-branch`` path based on
    :func:`should_use_local_rebase`.  When the local path is
    selected, REST ``update-branch`` is **never** called — even on
    local-rebase failure — so we never destroy a verified
    signature.  In the failure case we mark the PR as having been
    through Step 5 (so Step 5.5 doesn't double the configured
    ``merge_timeout``) and let auto-merge take over server-side.

    Returns a :class:`Step5Outcome`.  ``failed=True`` indicates the
    caller should set ``MergeStatus.FAILED`` and bail; the legacy
    REST path is the only path that can produce this outcome (a
    raised exception during ``update_branch`` or the polling loop).
    """
    if ctx.preview_mode:
        # NOTE: In preview mode, we should NOT print here as it
        # breaks single-line reporting.  The preview output
        # should only be a single line per PR in the evaluation
        # section.
        return Step5Outcome()

    ctx.log.debug("Rebasing %s [behind base branch]", pr_info.html_url)
    _set_tracker_state(ctx, pr_info, "rebasing")

    use_local, local_reason = await should_use_local_rebase(
        github_client=ctx.github_client,
        pr_info=pr_info,
        base=BaseRef(
            owner=owner,
            repo=repo,
            branch=pr_info.base_branch or "main",
        ),
        rebase_local=ctx.rebase_local,
        log=ctx.log,
    )

    if use_local:
        # For dependabot PRs, prefer the ``@dependabot rebase`` comment
        # macro over a local clone + rebase + sign: dependabot
        # force-pushes a freshly signed rebase itself, which satisfies
        # the very signature requirement that routed us to the local
        # path — without touching the operator's signing key (a local
        # rebase on a signature-requiring repo can trigger interactive
        # prompts, e.g. a YubiKey PIN).  Falls back to the local path
        # when the macro cannot be posted.
        if is_dependabot(pr_info.author) and ctx.request_dependabot_rebase is not None:
            handled = await _run_dependabot_macro_path(
                ctx=ctx,
                pr_info=pr_info,
                owner=owner,
                repo=repo,
                local_reason=local_reason,
            )
            if handled:
                return Step5Outcome()
        await _run_local_path(
            ctx=ctx,
            pr_info=pr_info,
            owner=owner,
            repo=repo,
            local_reason=local_reason,
        )
        return Step5Outcome()

    return await _run_rest_path(
        ctx=ctx,
        pr_info=pr_info,
        owner=owner,
        repo=repo,
    )
