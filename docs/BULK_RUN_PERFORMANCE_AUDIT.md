<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 The Linux Foundation -->

# Bulk Run Performance Audit — 503-PR `lfreleng-actions` run

Status: **Analysis**. No code changes proposed here are implemented yet.

Subject run: `dependamerge merge --no-confirm https://github.com/lfreleng-actions`,
503 automation PRs across 116 repositories, elapsed **1h 21m 20s**, reported
**469 merged / 34 failed**. Observed symptom: the first ~300 merges completed
at speed, after which throughput collapsed.

This document records what the code does today, what the evidence says
actually happened, and how a persistent local record of run history could
change the tool's behaviour.

---

## 0. Status as of v0.11.0

Every P0 and P1 item has shipped, along with six of the seven P2 items. The
seventh was recommended in error and needed no work. The measurements in §2
are **left as originally recorded**: they are the observations that motivated
the work, and rewriting them would destroy the baseline any future run is
compared against.

<!-- markdownlint-disable MD013 -->

| Group                         | Shipped | Note                                              |
| ----------------------------- | ------- | ------------------------------------------------- |
| P0 — stop the cliff           | 7 of 7  | v0.10.2, v0.11.0                                  |
| P1 — cut API volume           | 4 of 4  | v0.10.3, v0.11.0; item 4 by a different mechanism |
| P2 — correctness of reporting | 6 of 7  | item 5 recommended in error                       |
| §4 — persistent record        | 0       | not started                                       |

<!-- markdownlint-enable MD013 -->

Per-item status is marked inline in §3.

**The combined effect has not yet been measured.** Five pull requests across
three releases rest on a single data point. A repeat bulk run remains the
highest-value next step, and belongs here as a new section comparing before
and after rather than as edits to §2.

---

## 1. Executive summary

Three independent problems compound:

1. **Polling cost scales with the number of *waiting* PRs, and the REST
   budget cannot pay for it.** Each parked PR issues 6 API calls per minute
   forever. The REST budget is 83 calls per minute. Beyond ~14 simultaneously
   parked PRs, polling alone consumes 100% of the budget.
2. **The adaptive throttle is a one-way ratchet with a dead recovery path.**
   Once the remaining REST budget drops below 10%, concurrency and RPS halve
   on *every subsequent successful response* until they hit the floor
   (2 concurrent / 1.0 rps) and **never recover for the lifetime of the
   process**. This is the mechanism behind the sudden, permanent cliff.
3. **Failure reporting inverts the truth for most of the set.**
   Of the 34 reported failures, **21 (62%) had merged**, most within two
   minutes of being reported failed. The tool arms auto-merge, gives up
   waiting, reports `FAILED`, and GitHub completes the merge moments later.

Only 12 of the 34 reported failures were real. Of those, 9 are GitHub
infrastructure failures (ruleset-injected workflows queued but never executed)
that this tool cannot address, and 3 are Dependabot title/commit-subject
mismatches now tracked as lfreleng-actions/dependamerge#405 (§2.7).

---

## 2. Evidence

### 2.1 Polling cost is O(parked PRs)

Measured during a live run while 4 PRs were parked in `_wait_for_auto_merge`,
sampling `X-RateLimit-Remaining` from response headers:

```text
16:28:13 remaining 3624 delta 6 over 20s
16:28:34 remaining 3615 delta 8 over 20s
16:28:55 remaining 3606 delta 8 over 20s
16:29:16 remaining 3599 delta 6 over 20s
16:29:37 remaining 3590 delta 8 over 20s
16:29:58 remaining 3581 delta 8 over 20s
```

≈ 8 calls per 20 s for 4 parked PRs = **6 calls/min/parked PR**, exactly the
`DEFAULT_MERGE_RECHECK_INTERVAL = 10.0` cadence
(`merge_manager.py:56`). The poll is an unbatched
`GET /repos/{o}/{r}/pulls/{n}` per PR per tick
(`merge_manager.py:5107`).

Against the authenticated REST budget of 5000/hr = **83 calls/min**:

| Parked PRs | Polling calls/min | % of REST budget |
| ---------- | ----------------- | ---------------- |
| 5          | 30                | 36%              |
| 14         | 84                | **100%**         |
| 40         | 240               | 289%             |
| 60         | 360               | 434%             |

`slot_lease.parked()` deliberately releases the *work* slot while waiting so
that waiting never starves runnable work — but it does **not** bound how many
PRs may poll concurrently. The number of simultaneous pollers is unbounded by
`--concurrency`. The design that fixed slot starvation converted it into
budget starvation.

> Measurement note: the `GET /rate_limit` endpoint returns a stale, cached
> value and is unusable for this measurement — it read `3856` while response
> headers on the same token read `3643`. Use response headers.

### 2.2 The throttle ratchet

`github_async.py:617-655`, executed after every successful response:

```python
if limit > 0:
    remaining_ratio = remaining / max(1, limit)
    should_throttle = remaining_ratio < 0.1 or error_rate > 0.1
    if should_throttle:
        throttle_factor = 0.3 if error_rate > 0.2 else 0.5
        new_concurrency = max(2, int(self._max_concurrency * throttle_factor))
        ...
        new_rps = max(1.0, self._current_rps * throttle_factor)
        ...
else:
    # Gradually increase limits when healthy  <-- UNREACHABLE
```

Three defects:

1. **The recovery branch is dead code.** It lives in the `else:` of
   `if limit > 0:`. `limit` is parsed from `X-RateLimit-Limit` with a default
   of `"60"` (`github_async.py:467`) and GitHub always sends `5000`.
   `limit > 0` is always true, so ramp-up **never executes**.
   The comment at `github_async.py:243-246` states the intent ("adaptive
   tuning ramps concurrency back up to *this* value") — the intent is not
   realised.
2. **The error-rate signal is inert.** `_get_recent_error_rate`
   (`github_async.py:2638-2654`) returns
   `len(recent) / (len(recent) * _ESTIMATED_REQUESTS_PER_ERROR)` with
   `_ESTIMATED_REQUESTS_PER_ERROR = 10` (`:200`) — that is **exactly 0.1**
   whenever any error exists, and 0.0 otherwise. Both tests
   (`> 0.1`, `> 0.2`) are always false. Errors can never trigger throttling,
   and `throttle_factor` is always `0.5`.
3. **Live semaphore replacement.** `self.semaphore = asyncio.Semaphore(n)`
   (`:635`) swaps the object while tasks hold permits from the previous
   instance. In-flight tasks release the old object; the new one starts fully
   unclaimed, so the cap is transiently violated by up to the old concurrency.

The single live trigger is `remaining_ratio < 0.1`, i.e. **fewer than 500
REST calls remaining**. Descent is `20→10→5→2` concurrency and
`8.0→4.0→2.0→1.0` rps, permanent for the process.

At an observed active-phase burn of roughly 200-500 calls/minute, a 503-PR run
crosses `remaining < 500` after roughly 20 minutes — which matches the
reported "first 300 were very quick, then it slowed down considerably".

**The two problems reinforce each other**: more parked PRs → faster budget
burn → ratchet trips → 1 rps → each PR takes longer → more PRs parked
simultaneously → more polling. That feedback loop, not any single constant,
is the "exponential" behaviour.

### 2.3 Sticky adaptive delay

`_apply_retry_after_throttling` (`github_async.py:2656-2675`) sets
`self._adaptive_delay` (up to 5.0 s), which is then slept **before every
subsequent successful request** (`:613-615`). Its decay logic lives *inside
the same function*, so it only decays when another `Retry-After` arrives.
A single `Retry-After: 60` therefore adds a permanent serial delay to every
request for the rest of the run. At ~10 calls per PR that is +50 s per PR.

On a secondary rate limit the code sleeps the advised delay
(`:533`) **and then** raises `SecondaryRateLimitError`, which tenacity retries
with its own `wait_random_exponential(multiplier=0.5, max=10.0)` (`:485`).
The sleeps stack rather than being taken as a maximum.

### 2.4 Uncoordinated clients share one real budget

`AsyncMergeManager.__init__` builds a `GitHubAsync` (`merge_manager.py:371`)
*and* a `GitHubService` (`:374`), which builds a **second** `GitHubAsync`
(`github_service.py:126`). Neither is passed `max_concurrency` or
`requests_per_second`, so each defaults to 20/8.0 — a combined ceiling of
**40 concurrent / 16 rps**, not the `concurrency=20, rps=8.0` the UI reports.
Separately, `GitHubClient` constructs a **fresh `GitHubAsync` per call** at
seven sites (`github_client.py:78,190,210,247,264,344,384`), each with a new
limiter and zero throttle memory.

GraphQL also flows through the same `_request` (`github_async.py:721`) and so
updates the same `remaining`/`limit` pair, despite GraphQL having a separate
5000-*point* budget. A GraphQL response ratchets down the REST path and vice
versa. No query requests `rateLimit { cost remaining }`.

### 2.5 Per-repo waits stack serially

The striped scheduler runs one serial worker per repository, so a repo's PRs
are processed strictly in sequence. Observed in the re-run:

```text
16:21:07  ⏳ Waiting: workflows-template/pull/29 [required workflows still running]
16:26:15  ❌ Failed:  workflows-template/pull/29
16:26:28  ⏳ Waiting: workflows-template/pull/30 [required workflows still running]
```

PR #29 burned the full 300 s and failed; PR #30 then began its **own** fresh
300 s wait for the identical, already-known-hopeless reason. A repo with four
such PRs burns 20 minutes serially to learn the same fact four times. Nothing
propagates the first PR's outcome to its siblings — not even within a single
run, let alone across runs.

Compounding this, the per-loop budgets do not roll up: only
`_wait_for_auto_merge` honours `--max-wait`. The pre-commit.ci poll
(`merge_manager.py:3520`), the dependabot-recreate poll (`:3939`), the
recreated-checks wait (`:4091`) and the post-rebase poll (`rebase.py:953`)
each independently burn up to `--merge-timeout` (300 s).
`docs/MERGE_ENGINE_DESIGN.md` §2 already documents this; it remains true.

### 2.6 Reported failures were mostly not failures

Re-checking all 34 reported failures against the API:

| Reported reason                      | Count | Actually merged |
| ------------------------------------ | ----: | --------------: |
| Required workflows not satisfied     | 13    | 6               |
| Approve failed — HTTP 500            | 6     | 4               |
| Merge already in progress            | 5     | 4               |
| Required status checks not satisfied | 5     | 5               |
| Required workflows failed            | 4     | 2               |
| Merge conflicts                      | 1     | 1               |
| **Total**                            | 34    | **21 (62%)**    |

Most merged between 15:16:46 and 15:17:50 — a tight cluster in the run's final
minutes, i.e. auto-merge firing seconds *after* the tool had already declared
failure and moved on. The tool never re-verifies terminal state before
reporting.

Two specific gaps:

- **HTTP 500 on approve is not retried.** `_is_retryable_status` is
  `(429, 502, 503, 504)` (`github_async.py:114`); 500 is excluded and
  `approve_pull_request` has no local retry. GitHub's
  `POST /pulls/{n}/reviews` endpoint returns transient 500s — yet
  4 of the 6 affected PRs merged anyway, meaning the review was
  created despite the 500.
- **"Merge already in progress" is treated as terminal** after
  `3.0*(attempt+1)` s backoff × 2 retries (`merge_manager.py:4564`). GitHub
  needs 10-30 s. 4 of 5 subsequently merged.

### 2.7 The genuine residual failures

A verbose re-run (`--verbose --no-progress`, log at
`/tmp/dependamerge-runs/run2.log`) found only 13 PRs still open, merged 1, and
failed 12 in **11 minutes** (16:20:40 → 16:31:38). Every one of those 12 is
still open, so this time the failures are real. They collapse into exactly
**two** root causes:

<!-- markdownlint-disable MD060 -->

| Failing required check                                        | PRs |
| ------------------------------------------------------------- | --: |
| `AI Slop Scan 🧹` + `Zizmor Scan 🌈` — queued, never executed | 9   |
| `Semantic Pull Request 🛠️` — title/commit-subject mismatch    | 3   |

<!-- markdownlint-enable MD060 -->

**Cause A — required workflows queued but never executed. Out of scope.**
For `verify-release-schema-action#106` and `workflows-template#29` the head SHA
carries only two check runs:

```text
DCO      completed success
CodeQL   completed neutral
```

`AI Slop Scan 🧹` and `Zizmor Scan 🌈` are injected org-wide by a repository
ruleset, with the workflow code hosted in the special `.github` repository —
so the consuming repositories contain no local workflow files for them. Under
load, GitHub queues these runs and never executes them. This is a GitHub
infrastructure failure, **not** a repository misconfiguration and **not**
something this tooling can repair. No fix is proposed.

The only tooling-side observation worth recording is the cost: the tool waited
the full 300 s per affected PR, serially per repository, to rediscover a
condition that cannot change (§2.5).

**Cause B — Dependabot title / commit-subject mismatch.** These are legitimate
check failures. Dependabot writes a PR title that differs from the single
commit's subject, and the org check runs
`validateSingleCommitMatchesPrTitle`:

<!-- markdownlint-disable MD013 -->

| PR                                 | PR title                                                                             | Commit subject                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| `tag-validate-action#283`          | `Chore: Bump cryptography from 49.0.0 to 50.0.0 in the uv group across 1 directory`  | `Chore: Bump cryptography in the uv group across 1 directory`   |
| `github-security-report-action#96` | `CI(deps): Bump github-security-report from 0.8.0 to 0.10.0 in /.github/runtime-pin` | `CI(deps): Bump github-security-report in /.github/runtime-pin` |
| `sigul-sign-docker#175`            | `CI(actions): Bump lfit/…/reuse-openssf-scorecard.yaml from 0.9.1 to 0.10.1`         | `CI(actions): Bump lfit/…/reuse-openssf-scorecard.yaml`         |

<!-- markdownlint-enable MD013 -->

The upstream reusable workflow already relaxes the exact match to a **prefix**
match when the commit subject is a leading substring of the title — which
covers Dependabot dropping a trailing `from X to Y`. The first two cases above
are **mid-string elision** (the version range is removed from the middle), so
the relaxation does not apply and the check fails correctly.

`sigul-sign-docker#175` is covered by the relaxation and its check did pass —
but the workflow sets `concurrency.cancel-in-progress: true`, leaving a
`cancelled` duplicate run alongside the `success`, and ruleset evaluation is
blocked by the `cancelled` one.

Tracked as **lfreleng-actions/dependamerge#405** — align the PR title with the
commit subject, then let the workflow's `edited` trigger re-run the check.

So of the 34 originally reported failures: 21 merged on their own, 9 are GitHub
infrastructure failures outside this tool's control, and 3 are actionable via
issue #405.

### 2.8 Dead code

> **Resolved in v0.11.0.** The package was deleted. `slot_lease.py` already
> carried the parking semantic, and `pr_poller.py` supersedes the reconciler,
> so nothing remained worth wiring. See §3 P2.7.

`src/dependamerge/engine/` (`scheduler.py`, `reconciler.py`, `ladder.py`,
`model.py`) is **not imported anywhere in `src/`** — only by `tests/engine/`.
`slot_lease.py` ported one semantic (parking) out of it into the legacy path.
The reconciler — the component designed to centralise and batch exactly the
polling that is now the dominant cost — is unwired. Notably its own docstring
(`reconciler.py:24-25`) concedes it is *also* one-GET-per-waiting-PR, so
wiring it as-is would not fix §2.1 without adding batching.

---

## 3. Recommended fixes, in priority order

### P0 — stop the cliff (small, surgical, high impact)

All seven shipped in v0.10.2 and v0.11.0.

1. ✅ **Fix the ratchet.** Restructure `github_async.py:617-655` to
   `if should_throttle: ... else: ramp_up()` so recovery is reachable. Ramp up
   on sustained healthy responses toward `_base_max_concurrency`/`_base_rps`.
2. ✅ **Delete or fix `_get_recent_error_rate`.** As written it is a constant.
   Either track total requests (a simple counter alongside errors) or remove
   the error term and drive throttling from budget headroom alone.
3. ✅ **Do not replace live semaphores.** Use a resizable limiter that adjusts by
   acquiring/releasing spare permits, so the cap is never transiently violated.
4. ✅ **Decay `_adaptive_delay` on read**, not only inside
   `_apply_retry_after_throttling`; cap its total contribution per run.
5. ✅ **Do not stack sleeps.** On a secondary rate limit, either sleep locally
   *or* delegate to tenacity — not both.
6. ✅ **Separate REST and GraphQL budget accounting.** Key the throttle state on
   `X-RateLimit-Resource` (already returned; observed as `core`), and add
   `rateLimit { cost remaining resetAt }` to GraphQL queries.
7. ✅ **Share one `GitHubAsync`.** Inject the manager's client into
   `GitHubService` and `GitHubClient` so there is exactly one limiter, one
   semaphore, and one throttle state per run.

### P1 — cut API volume (the real scaling fix)

All four shipped; item 4 by a different mechanism than proposed.

1. ✅ **Batch parked-PR polling.** Replace N× `GET /pulls/{n}` per tick with a
   single aliased GraphQL query covering all parked PRs
   (`pr0: repository(...){pullRequest(number:){...}} pr1: ...`). This turns
   O(parked) into O(1) per tick and is the single largest lever available:
   60 parked PRs go from 360 calls/min to ~6.
2. ✅ **Adaptive first-poll delay.** Do not poll from t=0 at 10 s intervals when
   the repo's checks historically take 4 minutes. Schedule the first poll at
   `p50_check_seconds × 0.8`. *Shipped as an in-run measurement rather than a
   stored one, so it needs no persistence and does not pre-commit §4's design.*
3. ✅ **Cache `analyze_block_reason`.** It costs ≥5 calls
   and is invoked from three sites per PR. *Shipped with a short TTL rather
   than the per-`(repo, head_sha)` key proposed here: the reason changes as
   checks complete while the head SHA does not, so a run-lifetime memo would
   answer "still blocked" forever and break the wait-and-retry paths.*
4. ⚠️ **Propagate a repo's first outcome to its siblings within a run.** If PR
   #29 waited out 300 s for a required check that never started, do not make
   #30, #31 and #32 repeat the same discovery. *The **outcome** ships but not
   the **mechanism**: propagation proved unsafe. Absence of a workflow run is
   a fact about one commit, not a repository — a workflow missing from #29's
   head says nothing about #30's, which may have been pushed later and
   dispatched fine. Reusing the finding would skip a wait that could have
   succeeded and report a failure instead. Each sibling therefore detects the
   condition itself, in seconds rather than the 300 s this item set out to
   save, so no PR repeats the expensive discovery.*

### P2 — correctness of reporting

Six of seven shipped; item 5 was recommended in error (see below).

1. ✅ **Re-verify before reporting failure.** Before emitting `FAILED`, re-read
   the PR once. This alone would have converted 21 of 34 "failures" into
   successes or `AUTO_MERGE_PENDING`.
2. ✅ **Retry 500 on approve**, guarded by a re-read of existing reviews so a
   duplicate approval is never created.
3. ✅ **Treat "Merge already in progress" as park-and-verify**, not terminal.
4. ✅ **Bound the wait for required checks that have not started.** *Also
   closes the tooling half of #380 Category C.* Where a
   required check has no check-run on the head SHA, the cause is usually a
   GitHub infrastructure problem — ruleset-injected workflows queued but never
   executed — which no amount of waiting resolves. That cause is out of scope
   for this tool, but the tool need not spend 300 s per sibling PR
   rediscovering it: report it distinctly from "check failed" and stop waiting.
5. ⚠️ **Resolve duplicate check runs by latest attempt.** *Recommended in
   error — already implemented before this audit was written.*
   `check_runs.py` landed on 2026-07-31, a fortnight before this document,
   and `latest_check_run_per_name` already collapses each name to its latest
   run; `github_async`, `github_service` and `merge_manager` all consume it.
   The observed symptom was real but its cause was elsewhere:
   `sigul-sign-docker#175` was blocked by **GitHub's own ruleset evaluation**
   picking the `cancelled` run, not by this tool's scoring, which had it
   right. There was never a tooling-side fix to make. Tracked upstream as
   lfreleng-actions/.github#171.
6. ✅ **Align Dependabot PR titles with commit subjects.** Tracked separately as
   lfreleng-actions/dependamerge#405.
7. ✅ **Wire or delete `src/dependamerge/engine/`.** Deleted. Carrying an
   unused scheduler plus its test suite is a maintenance tax and a trap for
   future readers.

---

## 4. Proposal: a persistent local record

### 4.1 Why

Every expensive decision the tool makes today is made with **zero memory**:
how long this repo's checks take, whether this repo's required
workflows ever fire, how many API calls a run of this size costs, which error
classes resolve on retry. Each run rediscovers this, at full price, and
then throws it away. Worse, it rediscovers some of it *repeatedly within a
single run* (§2.5).

A local record earns its place solely by **changing behaviour**. The
sections below pair each stored fact with the decision it improves.

### 4.2 Store

SQLite via the stdlib `sqlite3` module — no new dependency — at
`${XDG_STATE_HOME:-~/.local/state}/dependamerge/history.db`, WAL mode, schema
version in `PRAGMA user_version`. Local only; **never** store tokens.
Provide `--no-history` to opt out, plus `dependamerge history prune|export`.

```sql
runs(id, started_at, finished_at, target, mode, tool_version,
     concurrency, rps, max_wait, merge_timeout,
     total, merged, failed, skipped, api_calls_rest, api_calls_graphql)

pr_attempts(id, run_id, repo, pr_number, author, head_sha,
            outcome, error_class, error_detail, attempts, duration_s,
            rebased, retriggered, verified_final_state, phase_timings_json)

check_observations(repo, check_name, required, head_sha,
                   observed_at, first_seen_s, completed_s, conclusion)

budget_samples(run_id, at, resource, remaining, limit,
               effective_concurrency, effective_rps)
```

Plus a derived, cheaply-recomputed `repo_profile` view: per-repo p50/p90 check
completion, required-check inventory, rebase-required rate, failure rate by
class, and last-success timestamp.

### 4.3 How each stored fact changes behaviour

<!-- markdownlint-disable MD013 -->

| Stored fact                         | Decision it changes                                                                                        | Expected effect                                                                                                                         |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| p50/p90 check completion per repo   | First-poll delay and park deadline (`p90 × 1.5`, clamped) instead of a flat 300 s / 10 s                   | Removes most of the ~22 wasted GETs per PR; stops fast repos burning 300 s and stops slow repos failing spuriously                      |
| Required-check inventory per repo   | Distinguish "required check queued but never executed" from "check failed", and stop waiting on the former | Correctly attributes the 9 `AI Slop Scan`/`Zizmor Scan` PRs to GitHub infrastructure instead of burning 300 s each                      |
| Per-repo expected latency           | Schedule repos longest-first (LPT) rather than first-seen                                                  | Shortens makespan; the tail stops being decided by scheduling luck                                                                      |
| Calls-per-PR by outcome class       | Pre-flight admission control: compare `n_prs × p90_calls_per_pr` against live `remaining`                  | Warn "this run needs ~7,400 calls but only 4,300 remain" *before* starting, or auto-lower concurrency — instead of a silent 4× slowdown |
| `budget_samples` timeline           | Post-hoc diagnosis of exactly the ratchet in §2.2                                                          | Would have made this audit a five-minute `dependamerge history show`                                                                    |
| Error class → eventual outcome      | Learn that `500 on reviews` and `Merge already in progress` resolve ~80% of the time                       | Justifies the retry-policy changes in P2 with data, and can drive them automatically                                                    |
| Repeated failure signature per repo | `--skip-known-blocked`, and a "these repos need attention" report                                          | Stops a bulk run spending 20 minutes per repo re-deriving a config problem                                                              |
| Historical merge duration per repo  | Regression alerting on CI time                                                                             | Useful releng signal independent of merging                                                                                             |

<!-- markdownlint-enable MD013 -->

### 4.4 Sequencing

The record is **P2, not P0**. It makes the tool smarter; it does not fix the
cliff. Ship P0 and P1 first — those are bounded, surgical changes to
`github_async.py` and the polling path, and they address the reported symptom
directly. Then add the record, initially write-only (`runs`, `pr_attempts`,
`budget_samples`) plus a `dependamerge history` reporting command. Once real
data exists, the adaptive read-paths in §4.3 can be switched on, each behind a
flag, so a bad heuristic cannot degrade runs unnoticed.

---

## 5. Suggested immediate operational workaround

Until P0 lands, split large runs so no single process crosses the
ratchet, and keep the parked-PR population under ~14:

```sh
dependamerge merge --no-confirm --max-wait 0 https://github.com/lfreleng-actions
```

`--max-wait 0` is fire-and-forget: it arms auto-merge and returns without
waiting. It eliminates the polling load entirely and lets GitHub do the
waiting for free. Re-run later to sweep up anything auto-merge did not
complete. Given §2.6 — 62% of "failures" merged on their own within two
minutes — this should produce a *better* outcome than the current
blocking behaviour, in a fraction of the wall-clock time.
