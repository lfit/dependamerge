# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Batched pull-request state polling.

Every PR waiting on an external event (auto-merge, CI, a rebase) polls
GitHub for its own state on a fixed cadence.  Done per PR over REST that
costs one request per parked PR per tick --- measured at **6 API calls
per minute per parked PR** during the 503-PR run analysed in
``docs/BULK_RUN_PERFORMANCE_AUDIT.md``.

Against an authenticated REST budget of 5000/hr (83 calls/min) that
ceiling arrives fast:

===========  ==================  =================
Parked PRs   Polling calls/min   % of REST budget
===========  ==================  =================
5            30                  36%
14           84                  100%
40           240                 289%
===========  ==================  =================

Beyond roughly fourteen simultaneously parked PRs, polling alone consumes
the entire budget asking "are you done yet?", which then trips the
client's adaptive throttle and slows everything further --- the feedback
loop behind the reported collapse after ~300 merges.

This module removes the per-PR cost.  Concurrent reads arriving close
together are coalesced into a single aliased GraphQL query, turning
O(parked) requests per tick into O(1):

.. code-block:: text

    60 parked PRs, REST:     360 calls/min
    60 parked PRs, batched:  ~6 calls/min

Requests for the *same* PR share one in-flight result, so duplicate
readers cost nothing extra.

The returned mapping is shaped like the REST ``GET /pulls/{n}`` payload
--- ``mergeable``, ``mergeable_state``, ``state``, ``merged``,
``merged_at``, ``head.sha`` --- so existing call sites need no rewriting
and continue to work with helpers such as ``_merged_from_payload``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

__all__ = ["PullRequestStatePoller", "PRKey"]

PRKey = tuple[str, str, int]

# Fields mirroring the REST payload keys the wait loops consult.
_FRAGMENT = """
fragment PRState on PullRequest {
  number
  state
  merged
  mergedAt
  mergeable
  mergeStateStatus
  headRefOid
}
"""

# How long to hold a batch open for further arrivals.  Long enough for
# the parked PRs woken by one tick to land in the same query, short
# enough to be imperceptible against a 10 s poll interval.
DEFAULT_WINDOW_SECONDS = 0.05

# Aliased lookups per query.  GitHub charges a single-object lookup ~1
# point, so the cap is about query size and blast radius on failure
# rather than rate-limit cost.
DEFAULT_MAX_BATCH = 50

_MERGEABLE = {"MERGEABLE": True, "CONFLICTING": False, "UNKNOWN": None}

_MERGE_STATE = {
    "CLEAN": "clean",
    "DIRTY": "dirty",
    "BLOCKED": "blocked",
    "BEHIND": "behind",
    "DRAFT": "draft",
    "UNSTABLE": "unstable",
    "UNKNOWN": "unknown",
    # ``has_hooks`` is a REST ``mergeable_state`` value too, so this is the
    # faithful mapping.  Do not "simplify" it to ``blocked``: GitHub's schema
    # defines HAS_HOOKS as "Mergeable with passing commit status and
    # pre-receive hooks", i.e. a *mergeable* state closer to ``clean``.
    # Reporting it as blocked would make the wait loops keep waiting on a PR
    # that is ready to merge.
    "HAS_HOOKS": "has_hooks",
    # ``DRAFT`` above is not currently in GitHub's MergeStateStatus enum
    # (drafts surface via ``isDraft``), but REST does report a ``draft``
    # mergeable_state, so the mapping is kept for parity and forward safety.
}


class _Client(Protocol):
    """The subset of ``GitHubAsync`` this module needs."""

    async def graphql(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...

    async def get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]: ...


def _to_rest_shape(node: dict[str, Any]) -> dict[str, Any]:
    """Convert a GraphQL ``PullRequest`` node to the REST payload shape.

    GraphQL reports ``state`` as ``OPEN``/``CLOSED``/``MERGED`` whereas
    REST only ever says ``open``/``closed`` and carries merged-ness
    separately, so ``MERGED`` is normalised to ``closed`` with
    ``merged`` set.  Callers already treat those as independent.
    """
    raw_state = (node.get("state") or "").upper()
    merged = node.get("merged")
    state: str | None
    if raw_state == "MERGED":
        state = "closed"
        if merged is None:
            merged = True
    elif raw_state == "CLOSED":
        state = "closed"
    elif raw_state == "OPEN":
        state = "open"
    else:
        state = raw_state.lower() or None

    merge_state = _MERGE_STATE.get((node.get("mergeStateStatus") or "").upper())

    payload: dict[str, Any] = {
        "number": node.get("number"),
        "state": state,
        "merged": merged,
        # Always present (``null`` when unmerged), so ``_merged_from_payload``
        # can fall back to it rather than degrading to "unknown".
        "merged_at": node.get("mergedAt"),
        "mergeable": _MERGEABLE.get((node.get("mergeable") or "").upper()),
        "mergeable_state": merge_state,
    }
    head_sha = node.get("headRefOid")
    if head_sha:
        payload["head"] = {"sha": head_sha}
    return payload


class PullRequestStatePoller:
    """Coalesces concurrent PR state reads into batched GraphQL queries.

    Usage mirrors a plain fetch; the batching is invisible to callers::

        payload = await poller.fetch(owner, repo, number)

    Thread-safety is not a concern --- this is single-event-loop code ---
    but re-entrancy is: ``fetch`` may be called from arbitrarily many
    parked tasks at once, and every waiter must be resolved exactly once
    even when the underlying query fails.
    """

    def __init__(
        self,
        client: _Client,
        *,
        window: float = DEFAULT_WINDOW_SECONDS,
        max_batch: int = DEFAULT_MAX_BATCH,
        log: logging.Logger | None = None,
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self._client = client
        self._window = max(0.0, window)
        self._max_batch = max_batch
        self.log = log or logging.getLogger(__name__)
        self._pending: dict[PRKey, list[asyncio.Future[dict[str, Any] | None]]] = {}
        self._flush_task: asyncio.Task[None] | None = None
        # Observability: lets a run report how much batching actually saved.
        self.requests_served = 0
        self.queries_issued = 0
        self.rest_fallback_calls = 0

    @property
    def calls_saved(self) -> int:
        """Requests batching removed, versus one call per read.

        Counts every API call actually made --- batched GraphQL queries
        *and* any per-PR REST reads the fallback had to issue --- so a
        run that degraded to the fallback reports the savings it really
        achieved rather than the savings it intended.
        """
        issued = self.queries_issued + self.rest_fallback_calls
        return max(0, self.requests_served - issued)

    async def fetch(self, owner: str, repo: str, number: int) -> dict[str, Any] | None:
        """Return PR state in REST payload shape, or ``None`` if absent.

        ``None`` also results when the batched query failed *and* the
        per-PR REST fallback failed for this particular PR while
        succeeding for others in the same batch --- callers already treat
        a missing payload as "no news, poll again".

        Raises only when both the batched query and the whole fallback
        failed, propagating the original query error so callers keep
        their existing error handling.
        """
        key: PRKey = (owner, repo, number)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any] | None] = loop.create_future()
        self.requests_served += 1
        # A second reader of the same PR joins the in-flight result
        # rather than adding another lookup to the query.
        self._pending.setdefault(key, []).append(future)
        self._ensure_flush_scheduled()
        return await future

    def _ensure_flush_scheduled(self) -> None:
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(
                self._flush_soon(), name="pr-poller-flush"
            )

    async def _flush_soon(self) -> None:
        # Hold the batch open briefly so near-simultaneous arrivals join it.
        if self._window:
            await asyncio.sleep(self._window)
        while self._pending:
            batch = list(self._pending.keys())[: self._max_batch]
            waiters = {k: self._pending.pop(k) for k in batch}
            try:
                await self._run_batch(waiters)
            except asyncio.CancelledError:
                self._fail_all(waiters, asyncio.CancelledError())
                raise
            except Exception as exc:  # pragma: no cover - defensive
                # ``_run_batch`` resolves its own waiters; reaching here
                # means an unexpected failure outside that handling, and
                # leaving futures pending would hang every parked PR.
                self.log.debug("PR poller batch failed unexpectedly: %s", exc)
                self._fail_all(waiters, exc)

    async def _run_batch(
        self, waiters: dict[PRKey, list[asyncio.Future[dict[str, Any] | None]]]
    ) -> None:
        keys = list(waiters)
        results: dict[PRKey, dict[str, Any] | None]
        try:
            results = await self._query(keys)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.debug(
                "Batched PR query failed for %d PR(s); falling back to REST: %s",
                len(keys),
                exc,
            )
            fallback = await self._rest_fallback(keys)
            if fallback is None:
                self._fail_all(waiters, exc)
                return
            results = fallback
        for key, futures in waiters.items():
            payload = results.get(key)
            for fut in futures:
                if not fut.done():
                    fut.set_result(payload)

    async def _query(self, keys: list[PRKey]) -> dict[PRKey, dict[str, Any] | None]:
        decls: list[str] = []
        selections: list[str] = []
        variables: dict[str, Any] = {}
        for i, (owner, repo, number) in enumerate(keys):
            decls.append(f"$o{i}: String!, $r{i}: String!, $n{i}: Int!")
            selections.append(
                f"  p{i}: repository(owner: $o{i}, name: $r{i}) "
                f"{{ pullRequest(number: $n{i}) {{ ...PRState }} }}"
            )
            variables[f"o{i}"] = owner
            variables[f"r{i}"] = repo
            variables[f"n{i}"] = number
        query = (
            "query BatchedPRState("
            + ", ".join(decls)
            + ") {\n"
            + "\n".join(selections)
            + "\n}\n"
            + _FRAGMENT
        )
        self.queries_issued += 1
        data = await self._client.graphql(query, variables)
        out: dict[PRKey, dict[str, Any] | None] = {}
        for i, key in enumerate(keys):
            repo_node = (data or {}).get(f"p{i}") or {}
            pr_node = (
                repo_node.get("pullRequest") if isinstance(repo_node, dict) else None
            )
            out[key] = _to_rest_shape(pr_node) if isinstance(pr_node, dict) else None
        return out

    async def _rest_fallback(
        self, keys: list[PRKey]
    ) -> dict[PRKey, dict[str, Any] | None] | None:
        """Per-PR REST reads when the batched query fails.

        Costs exactly what the unbatched implementation cost, so a
        GraphQL outage degrades throughput rather than blinding every
        wait loop until it times out.  Returns ``None`` when the
        fallback itself fails, letting the caller surface the original
        error.
        """
        try:
            self.rest_fallback_calls += len(keys)
            payloads = await asyncio.gather(
                *(
                    self._client.get(f"/repos/{owner}/{repo}/pulls/{number}")
                    for owner, repo, number in keys
                ),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return None
        out: dict[PRKey, dict[str, Any] | None] = {}
        recovered = False
        for key, payload in zip(keys, payloads, strict=True):
            if isinstance(payload, dict):
                out[key] = payload
                recovered = True
            else:
                out[key] = None
        return out if recovered else None

    @staticmethod
    def _fail_all(
        waiters: dict[PRKey, list[asyncio.Future[dict[str, Any] | None]]],
        exc: BaseException,
    ) -> None:
        for futures in waiters.values():
            for fut in futures:
                if not fut.done():
                    fut.set_exception(exc)
