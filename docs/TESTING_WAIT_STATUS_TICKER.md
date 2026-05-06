<!--
SPDX-FileCopyrightText: 2026 The Linux Foundation
SPDX-License-Identifier: Apache-2.0
-->

# Testing the wait-status ticker (countdown UI)

This document captures options for **manually verifying** the
single-line Rich countdown introduced in PR #270:

```text
⏳ Waiting for N PRs to complete checks [Ns]
```

The ticker fires when a specific set of conditions line up at once,
which makes ad-hoc verification fiddly. This doc lays out four
approaches — ranked by effort vs. fidelity — so we can pick one when
we want to confirm the UX in a real terminal.

## What we're trying to verify

The ticker (`AsyncMergeManager._wait_status_ticker()`) updates
the progress display when **all** of the following are true:

1. `self._waiting_prs` has at least one entry. Step 5.5 is the sole
   producer; it adds entries when a PR meets every condition of
   `should_wait` (`mergeable_state` in
   `("blocked", "behind", "unstable")`, not in `_rebased_prs`,
   `force_level != "all"`, not in preview mode, and — for
   `blocked` PRs — `analyze_block_reason()` indicates pending
   required checks). The current code does **not** filter on the
   `mergeable` boolean: GitHub returns `mergeable=False` transiently
   while computing or when a non-required check failed, so Step 5.5
   accepts `True`, `False`, and `None` to give auto-merge a chance
   to rescue the PR.
2. The manager has a `MergeProgressTracker` attached.
3. The tracker's `rich_available` is `True`. If it isn't, the ticker
   delegates to a 15-second plain-console fallback
   (`_wait_status_ticker_plain`).
4. The displayed message changes between samples (debounced via
   `last_message != message`), so the seconds counter has
   to tick down.

The interesting visual transitions are:

- The seconds counter animating in place (`[40s]` → `[39s]` → …).
- The `N` count decreasing as individual workers finish.
- The line clearing when the last waiter exits.

## Option A — synthetic real PRs in `lfreleng-actions/test-python-project`

Set up two or more PRs that legitimately end up in the wait loop, and
run `dependamerge` against them with a short `--merge-timeout` so the
countdown is observable.

Sketch:

1. Configure branch protection on `main` to require a status check
   that takes a fixed long duration (e.g. a small GitHub Actions
   workflow that runs `sleep 45`). Mark it "required".
2. Open two PRs from short-lived branches.
3. Land an unrelated commit on `main` so both PRs go `behind`.
4. Run:

   ```bash
   dependamerge merge --no-confirm --merge-timeout 60 \
     https://github.com/lfreleng-actions/test-python-project/pull/<n>
   ```

5. Watch the terminal. The Rich Live single-line update should show
   `⏳ Waiting for 2 PRs to complete checks [Ns]` ticking down,
   then dropping to `1 PR` before clearing.

### Pros (Option A)

- End-to-end validation against real GitHub plumbing (auto-merge
  enable, `_waiting_prs` registration, ticker render).
- Exercises the same code path users hit in production.

### Cons (Option A)

- Requires branch-protection rules with a long-running check.
- Slow iteration loop: each test run takes minutes, and you have
  little control over GitHub's mergeability-computation timing.
- Pollutes a real repo with synthetic PRs.
- If GitHub resolves the rebase to `clean` faster than the slow
  check fires, Step 5.5 may exit before the wait loop runs and
  the ticker may not remain on screen long enough to observe.
- Hard to reproduce edge cases (e.g. all PRs finishing
  simultaneously, mid-run cancellation, `rich_available=False`).

## Option B — production CLI flag (e.g. `--test-wait-secs N`)

Add a hidden CLI flag (or environment variable) that injects synthetic
entries into `self._waiting_prs` for a configurable duration without
touching real GitHub state.

### Pros (Option B)

- Simple to invoke in any environment.

### Cons (Option B)

- Couples test-specific code to the production CLI surface.
- Risk of users discovering the flag and relying on it.
- Doesn't fit naturally with the rest of the merge flow (we'd need
  to short-circuit half the work to inject the entries).

**Recommendation: avoid.** The benefits don't justify the production
complexity.

## Option C — standalone demo script in `scripts/`

A small, standalone script that:

1. Builds a `MergeProgressTracker`, starts it.
2. Builds an `AsyncMergeManager` with a dummy token (no API calls
   happen — we call the ticker method directly).
3. Wires the tracker into the manager.
4. Populates `_waiting_prs` directly with three staggered deadlines.
5. Spawns `_wait_status_ticker()` as a background task.
6. Removes entries one by one over the run, simulating PRs
   completing.
7. Cancels the ticker after removing the last entry.

Invocation:

```bash
uv run python scripts/demo_wait_ticker.py            # Rich mode (~40s)
uv run python scripts/demo_wait_ticker.py --plain    # Plain fallback (~50s)
```

Expected Rich output (single line, animating in place):

```text
🔬 Demo in demo-org (0/3 PRs, 0%)
   ⏳ Waiting for 3 PRs to complete checks [40s]
   ⏱️  Elapsed: 1s
```

…ticking through `[39s]`, `[38s]`, …, then dropping to
`2 PRs` / `1 PR` as the staggered deadlines expire, then clearing.

Plain-fallback output (one line every 15 s):

```text
⏳ Waiting for 3 PRs to complete checks [40s remaining]
⏳ Waiting for 2 PRs to complete checks [25s remaining]
⏳ Waiting for 1 PR to complete checks [10s remaining]
```

### Pros (Option C)

- Zero production code changes.
- Exercises the actual `_wait_status_ticker()` and
  `_wait_status_ticker_plain()` code paths, not stand-ins.
- Reproducible and self-contained; runs in seconds.
- `--plain` flag exercises the non-Rich fallback explicitly.
- Straightforward to adapt for new edge cases (cancellation,
  single PR, rapid additions, etc.).

### Cons (Option C)

- Doesn't exercise the upstream Step 5.5 registration logic
  (auto-merge enable, REST refresh polling, head_sha capture,
  closed-state handling). That coverage already lives in
  `tests/test_auto_merge.py::TestStep5_5EnablesAutoMergeAndTimesOut`
  and `TestStep5_5HandlesMergeableNone`.

**Recommendation: this is the lowest-friction way to confirm the
visual behaviour.**

## Option D — same as C but exposed as a hidden CLI command

Expose the demo as `dependamerge demo-ticker` so it's discoverable
via `dependamerge --help`.

### Pros (Option D)

- Slightly more discoverable than a script in `scripts/`.

### Cons (Option D)

- Adds CLI surface for what is essentially an internal one-off
  developer tool.

**Recommendation: skip in favour of Option C** unless we accumulate
more internal demos and want a `demo` subcommand group.

## Recommendation

Use **Option C** as `scripts/demo_wait_ticker.py` (committed
under `scripts/`, intentionally not packaged) with a `--plain` flag
that also exercises the non-Rich path.

Avoid Option A as the primary verification mechanism; the timing
dependencies on real CI infrastructure make it brittle. Keep it in
mind for cases where a regression seems plausible and we need an
end-to-end sanity check against a real GitHub run.

### Proposed minimal layout

```text
scripts/
  demo_wait_ticker.py     # standalone demo, executable via `uv run python …`
docs/
  TESTING_WAIT_STATUS_TICKER.md   # this document
```

The demo script should:

- Import `AsyncMergeManager` and `MergeProgressTracker` from the
  installed package (no monkey-patching of production code).
- Use the same `_waiting_prs` and `_wait_status_ticker` symbols the
  production code uses.
- Accept `--plain` to force the plain-console fallback by passing
  `progress_tracker=None` to the manager OR by toggling
  `tracker.rich_available = False`.
- Print clear "started" / "finished" markers so the demo output is
  unambiguous in CI logs if anyone ever runs it there.

PR #270 ships both the bug fix and this verification harness
(Option C demo at `scripts/demo_wait_ticker.py` plus this guide).
