# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Timing constants and the mergeable-state display table.

Every polling loop in the package takes its cadence from here, so the
timings can be reasoned about — and substituted in tests — in one
place.  They are re-exported by the package, and the methods that read
them do so through the package rather than binding them at import
time, so a substitution is seen wherever it is made.
"""

from __future__ import annotations

# Centralised timing constants for all async merge operations.
#
# Every polling loop in this module (post-rebase status checks,
# pre-commit.ci re-runs, @dependabot recreate, recreated-PR readiness)
# derives its iteration count from these two values so that the timeout
# is consistent and easy to adjust from a single place or via the
# ``--merge-timeout`` CLI flag.
DEFAULT_MERGE_TIMEOUT: float = 300.0  # seconds (5 minutes)
DEFAULT_MERGE_RECHECK_INTERVAL: float = 10.0  # seconds between polls

# After a sibling PR merges (or a concurrent dependamerge run lands a
# change), GitHub recomputes a PR's mergeability asynchronously and
# briefly reports ``mergeable=null`` / ``mergeable_state="unknown"`` —
# typically for a few seconds.  Before dispatching a merge in a
# repo-scoped batch we re-read the PR and, when GitHub is still
# computing, poll up to this many seconds for a concrete value so the
# merge decision is made against fresh state rather than the
# (possibly stale) fetch-time snapshot.
MERGEABILITY_REFRESH_TIMEOUT_SECONDS: float = 10.0

# First-poll delay for the auto-merge wait loop.  The loop's steady
# cadence is ``DEFAULT_MERGE_RECHECK_INTERVAL`` (10s), but the *first*
# refresh happens after this much shorter delay: when auto-merge fires
# the moment it is armed (checks were already green and approval was
# the only blocker) a full-interval first sleep would discover the
# merge ~8 seconds late — per PR, serialized per repository in striped
# runs.  One extra lightweight GET is a fair trade for that.
MERGE_WAIT_FIRST_POLL_SECONDS: float = 2.0

# When GitHub answers a merge dispatch with 405 "Merge already in
# progress" it has accepted a merge (usually auto-merge armed earlier in
# this run) and is completing it asynchronously.  Completion is normally
# a matter of seconds, but the previous handling --- a 3s then 6s
# backoff before giving up --- routinely expired first and reported a
# failure for a PR that merged moments later.  These bound a short watch
# for completion instead.
MERGE_IN_PROGRESS_TIMEOUT_SECONDS: float = 60.0
MERGE_IN_PROGRESS_POLL_SECONDS: float = 5.0

# Pause between the two observations required before concluding that a
# required workflow will never be dispatched.  A single snapshot cannot
# tell "never dispatched" from "dispatched moments ago and not yet
# visible", and that mistake reports a terminal failure on a PR that
# would have merged.  The condition being detected persists for hours,
# so a few seconds costs nothing against the five-minute wait it saves.
UNDISPATCHED_CONFIRM_DELAY_SECONDS: float = 10.0
# Room reserved for the *requests* that follow the pause: a re-read of
# the PR head, then the workflow-run lookup, which retries and
# paginates.  Budgeting the pause alone would let the confirmation
# start a request the caller's deadline had already expired on, holding
# a worker slot past the run's ceiling.
UNDISPATCHED_CONFIRM_LOOKUP_SECONDS: float = 5.0
# As with ``MERGE_WAIT_FIRST_POLL_SECONDS``, the first poll comes early:
# GitHub is already completing the merge, so it often lands within a
# second or two and a full-interval first sleep would report it late.
MERGE_IN_PROGRESS_FIRST_POLL_SECONDS: float = 1.0

# Required verification checks (DCO, lint, build, etc.) normally
# start reporting within a few seconds.  When a *required* check has
# been pending for longer than this on a PR that itself was created
# / last updated more than this many seconds ago, the check is
# treated as stuck.  Used by ``_detect_stuck_required_check`` to
# decide whether to ask dependabot to recreate the PR.
STUCK_CHECK_THRESHOLD_SECONDS: float = 60.0

# pre-commit.ci normally reports back within a few minutes.  A
# ``pre-commit.ci - pr`` status stuck in ``pending`` for longer than
# this is treated as a hung run that needs a fresh ``pre-commit.ci
# run`` trigger.  Kept deliberately generous so a slow-but-normal run
# is never interrupted; used by ``_trigger_stale_precommit_ci``.
PRECOMMIT_CI_STUCK_PENDING_SECONDS: float = 300.0

# Icon and Rich style for each GitHub mergeable_state, used when
# rendering PR status. States not listed fall back to a neutral
# "investigating" icon with no style.
_MERGEABILITY_ICON_AND_STYLE: dict[str | None, tuple[str, str | None]] = {
    "dirty": ("🛑", "red"),
    "behind": ("⚠️", "yellow"),
    "clean": ("✅", "green"),
    "draft": ("📝", "blue"),
}
