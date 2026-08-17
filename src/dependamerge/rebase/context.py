# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Dependencies and results shared by every rebase path.

:class:`RebaseContext` is the bundle the Step 5 dispatcher and its
helper paths receive in place of a full ``AsyncMergeManager``
reference; :class:`Step5Outcome` is what the dispatcher hands back.

The two tracker shims (:func:`_set_tracker_state` and
:func:`_record_rebase`) live here because they are pure operations on
a :class:`RebaseContext`, shared by the dispatcher, the local path,
the dependabot macro path and the REST path alike.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console

from ..models import PullRequestInfo

if TYPE_CHECKING:
    from ..github_async import GitHubAsync


@dataclass
class RebaseContext:
    """Bundle of dependencies the rebase orchestrator needs.

    Passed in lieu of a full ``AsyncMergeManager`` reference so the
    rebase logic stays decoupled from manager internals and can be
    tested without standing up the whole merge pipeline.
    """

    github_client: GitHubAsync | None
    token: str
    rebase_local: bool
    preview_mode: bool
    merge_recheck_interval: float
    merge_poll_max_attempts: int
    log: logging.Logger
    console: Console
    # Mutable set on the manager that records PR keys (``owner/repo#N``)
    # which have already been through Step 5.  Step 5.5 consults this
    # to avoid doubling the configured ``merge_timeout``.  We keep the
    # raw set reference rather than a callback so the existing
    # invariant (Step 5 always adds, Step 5.5 always reads) stays
    # obvious at the call site.
    rebased_prs: set[str]
    # Async callable equivalent to ``manager._enable_auto_merge_for_pr``.
    # Passed in to avoid a circular import.
    enable_auto_merge: Callable[[PullRequestInfo, str, str], Awaitable[bool]]
    # Optional callback (``manager._track_pr_state``) that moves the PR
    # between transitory states on the Rich progress tracker
    # ("rebasing", "waiting", or ``None`` to clear).
    # Default ``None`` keeps existing test constructions working and
    # makes the tracker strictly optional for isolated use.
    track_pr_state: Callable[[PullRequestInfo, str | None], None] | None = None
    # Optional callback (``manager._record_rebase``) that increments the
    # tracker's cumulative "Rebased" total.  Called once per rebase this
    # module performs itself — the local force-push path and the REST
    # ``update-branch`` path.  The ``@dependabot rebase`` macro path is
    # *not* counted here: ``request_dependabot_rebase`` owns the
    # accounting for the macro (including its duplicate guard, which
    # counts nothing), and counting again would double it.
    record_rebase: Callable[[], None] | None = None
    # Optional async callable equivalent to
    # ``manager._request_dependabot_rebase``: posts ``@dependabot
    # rebase`` on the PR (idempotent — the manager implementation
    # skips duplicate comments) and returns True when the rebase is
    # considered requested.  When provided, dependabot PRs that would
    # otherwise take the local-rebase path use the macro instead:
    # dependabot force-pushes a freshly *signed* rebase, so there is
    # no need to clone + rebase + sign locally (which can require
    # interactive key access, e.g. a YubiKey PIN prompt).  Default
    # ``None`` preserves the local path for isolated use and existing
    # tests.
    request_dependabot_rebase: (
        Callable[[PullRequestInfo, str, str], Awaitable[bool]] | None
    ) = None


@dataclass
class Step5Outcome:
    """Result of :func:`perform_step5_rebase`.

    ``failed`` indicates the caller should mark the PR as ``FAILED``
    and bail out of the merge attempt.  ``error_message`` is the
    user-visible reason in that case.
    """

    failed: bool = False
    error_message: str | None = None


def _set_tracker_state(
    ctx: RebaseContext,
    pr_info: PullRequestInfo,
    state: str | None,
) -> None:
    """Move the PR between transitory progress-tracker states.

    No-op when the context was constructed without a
    ``track_pr_state`` callback (isolated tests, preview mode).
    """
    if ctx.track_pr_state is not None:
        ctx.track_pr_state(pr_info, state)


def _record_rebase(ctx: RebaseContext) -> None:
    """Count one rebase operation on the progress tracker.

    No-op when the context was constructed without a ``record_rebase``
    callback (isolated tests).  Preview runs never reach here at all:
    :func:`perform_step5_rebase` returns before dispatching to any
    rebase path.
    """
    if ctx.record_rebase is not None:
        ctx.record_rebase()
