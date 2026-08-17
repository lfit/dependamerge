<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 The Linux Foundation -->

# Merge Orchestration Engine — Design

Package: **removed**. This document survives for the analysis in
[Motivation](#motivation), which still describes the system accurately
and drove the changes listed below.

Status: **superseded and deleted.** The `src/dependamerge/engine/`
package was never imported by `src/` — it existed alongside the
production path rather than replacing it. Two useful semantics came out
of it, and both now live elsewhere:

- **Waiting holds no concurrency slot** — ported to `slot_lease.py`,
  which the merge path uses.
- **Centralised polling of parked work** — superseded by
  `pr_poller.py`. The engine's `Reconciler` polled one GET per waiting
  PR per tick (its own docstring said so), which is precisely the cost
  the batched poller removed; wiring it in would have preserved the
  problem rather than fixing it.

Rather than carry the package, its tests and the `ladder`/`model`
abstractions as a third option nobody was choosing between, this change
deletes them. See `docs/BULK_RUN_PERFORMANCE_AUDIT.md` §3 (P2.7).

## Motivation

The current orchestration (`merge_prs_parallel` → `_merge_single_pr`)
grew a series of inline waiting loops, each added for a specific
scenario. Three structural problems fell out of that growth, all
observed in production bulk-merge runs:

### 1. Waiting pins concurrency slots

Every waiting loop (`_wait_for_auto_merge`, the conflict-recovery
poll, the post-rebase REST poll, the recreate poll, the
recreated-checks wait, the pre-commit.ci re-trigger wait) runs inside
the worker task that holds one of the N global concurrency slots. A PR
that needs a `@dependabot rebase` occupies a slot for the full wait
(rebase turnaround is routinely 3–5 minutes; CI adds more).

Worked example from a real run: 41 repositories all needed a rebase to
re-run a required org check. With 10 slots and a five-minute wait
each, the batch drains in ceil(41/10) × ~5 min ≈ **20–25 minutes of
idle waiting**, while runnable PRs in other repositories queue
behind parked ones. With parking (waiting holds no slot), the same
batch issues all 41 rebases in one scheduling pass and total
wall time collapses to ~one rebase+CI latency.

### 2. Budgets are per-loop, not per-run

A single loop — `_wait_for_auto_merge` — honours `--max-wait` (the run
deadline) and `--no-wait`. The pre-commit.ci poll, the recreate poll,
the recreated-checks wait, and the post-rebase poll each
independently burn up to `--merge-timeout` (default 300 s) — a single
unlucky PR can stack ~5 of these sequentially. So the run deadline
does not bound the run.

### 3. Scattered, incomplete recovery routing

Recovery decisions live in four places (`_merge_single_pr` Steps
0.5/5/5.5/6, `_handle_merge_conflict`, `_report_merge_failure`, the
rebase module), each with its own entry conditions. Gaps fall through
the cracks; the motivating incident: an org-required workflow-audit
check failed against **stale branch content** (the base branch
already carried the fix). The legacy code classified "completed failing
required check" as terminal and reported failure for 88 of 92 PRs —
when a rebase would have re-run the check against current base and
allowed every one of them to merge.

Related accounting bugs (fixed by construction here): two failure
paths return a `MergeResult` without ever calling the progress
tracker, so the ticker over-counts in-flight PRs; and
`_merge_pr_with_retry` sleeps while holding the per-repo dispatch
lock.

## What became of the design

The sections that followed --- architecture, migration plan and testing ---
described `scheduler.py`, `reconciler.py`, `ladder.py` and `model.py`, none of
which exist any longer. They told contributors to wire up `Engine` and
`LadderInput` and pointed at `tests/engine/`, so keeping them would send
readers towards deleted APIs. Git history holds the full text if the detail
is ever wanted.

Where each idea ended up:

| Idea | Now lives in |
| --- | --- |
| Waiting holds no concurrency slot | `slot_lease.py` |
| Centralised polling of parked work | `pr_poller.py`, batched into one query |
| Per-repository lane serialisation | `merge_manager._run_striped` |
| Recovery ladder | still distributed across `merge_manager`; see #380 |

The recovery ladder is the one piece with no home yet. Issue #380 tracks the
failure modes it targets.
