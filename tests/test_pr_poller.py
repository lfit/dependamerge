# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for batched pull-request state polling.

The unbatched implementation cost one REST request per parked PR per
tick --- 6 calls/min each, so ~14 parked PRs consumed the entire REST
budget (see ``docs/BULK_RUN_PERFORMANCE_AUDIT.md`` §2.1).  These tests
pin the batching, the REST-compatible output shape, and --- most
importantly --- that every waiter is resolved exactly once even when the
underlying query fails, since a lost future would hang a parked PR until
its deadline.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dependamerge.pr_poller import PullRequestStatePoller, _to_rest_shape


def _node(**over: Any) -> dict[str, Any]:
    node = {
        "number": 1,
        "state": "OPEN",
        "merged": False,
        "mergedAt": None,
        "mergeable": "MERGEABLE",
        "mergeStateStatus": "CLEAN",
        "headRefOid": "a" * 40,
    }
    node.update(over)
    return node


def _reply(keys: list[tuple[str, str, int]], **by_number: Any) -> dict[str, Any]:
    """Build a GraphQL reply shaped like the aliased batch query's result."""
    out: dict[str, Any] = {}
    for i, (_o, _r, number) in enumerate(keys):
        node = by_number.get(f"n{number}", _node(number=number))
        out[f"p{i}"] = {"pullRequest": node}
    return out


# --------------------------------------------------------------------------
# Output shape
# --------------------------------------------------------------------------


class TestRestShape:
    def test_open_clean_pr(self) -> None:
        p = _to_rest_shape(_node())
        assert p["state"] == "open"
        assert p["merged"] is False
        assert p["mergeable"] is True
        assert p["mergeable_state"] == "clean"
        assert p["head"]["sha"] == "a" * 40

    def test_graphql_merged_state_maps_to_closed(self) -> None:
        """REST reports state/merged independently; GraphQL folds them."""
        p = _to_rest_shape(
            _node(state="MERGED", merged=True, mergedAt="2026-08-13T15:17:37Z")
        )
        assert p["state"] == "closed"
        assert p["merged"] is True
        assert p["merged_at"] == "2026-08-13T15:17:37Z"

    def test_merged_inferred_when_boolean_absent(self) -> None:
        p = _to_rest_shape({"state": "MERGED"})
        assert p["merged"] is True

    def test_closed_unmerged(self) -> None:
        p = _to_rest_shape(_node(state="CLOSED", merged=False))
        assert p["state"] == "closed"
        assert p["merged"] is False

    @pytest.mark.parametrize(
        ("enum", "expected"),
        [("MERGEABLE", True), ("CONFLICTING", False), ("UNKNOWN", None)],
    )
    def test_mergeable_enum(self, enum: str, expected: bool | None) -> None:
        assert _to_rest_shape(_node(mergeable=enum))["mergeable"] is expected

    @pytest.mark.parametrize(
        ("enum", "expected"),
        [
            ("CLEAN", "clean"),
            ("DIRTY", "dirty"),
            ("BLOCKED", "blocked"),
            ("BEHIND", "behind"),
            ("UNSTABLE", "unstable"),
            ("DRAFT", "draft"),
        ],
    )
    def test_merge_state_status(self, enum: str, expected: str) -> None:
        assert (
            _to_rest_shape(_node(mergeStateStatus=enum))["mergeable_state"] == expected
        )

    def test_merged_at_key_always_present(self) -> None:
        """``_merged_from_payload`` falls back to it, so it must not vanish."""
        assert "merged_at" in _to_rest_shape(_node())

    def test_has_hooks_is_not_reported_as_blocked(self) -> None:
        """HAS_HOOKS is a *mergeable* state, per GitHub's schema.

        Its description is "Mergeable with passing commit status and
        pre-receive hooks".  Mapping it to ``blocked`` would keep the
        wait loops waiting on a PR that is ready to merge, so this pins
        the faithful REST mapping against a plausible-looking
        "simplification".
        """
        assert (
            _to_rest_shape(_node(mergeStateStatus="HAS_HOOKS"))["mergeable_state"]
            == "has_hooks"
        )

    def test_missing_head_omits_key(self) -> None:
        assert "head" not in _to_rest_shape(_node(headRefOid=None))


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


class TestBatching:
    @pytest.mark.asyncio
    async def test_concurrent_reads_share_one_query(self) -> None:
        """The whole point: O(parked) requests become O(1)."""
        client = AsyncMock()
        keys = [("o", "r", n) for n in range(1, 21)]
        client.graphql = AsyncMock(return_value=_reply(keys))
        poller = PullRequestStatePoller(client, window=0.01)

        results = await asyncio.gather(*(poller.fetch(o, r, n) for o, r, n in keys))

        assert len(results) == 20
        assert client.graphql.await_count == 1
        assert poller.requests_served == 20
        assert poller.queries_issued == 1
        assert poller.calls_saved == 19

    @pytest.mark.asyncio
    async def test_each_pr_gets_its_own_result(self) -> None:
        client = AsyncMock()
        keys = [("o", "r", 1), ("o", "r", 2), ("o2", "r2", 3)]
        client.graphql = AsyncMock(
            return_value=_reply(
                keys,
                n1=_node(number=1, mergeStateStatus="CLEAN"),
                n2=_node(number=2, mergeStateStatus="BLOCKED"),
                n3=_node(number=3, state="MERGED", merged=True),
            )
        )
        poller = PullRequestStatePoller(client, window=0.01)

        a, b, c = await asyncio.gather(
            poller.fetch("o", "r", 1),
            poller.fetch("o", "r", 2),
            poller.fetch("o2", "r2", 3),
        )

        assert a is not None and b is not None and c is not None
        assert a["mergeable_state"] == "clean"
        assert b["mergeable_state"] == "blocked"
        assert c["merged"] is True

    @pytest.mark.asyncio
    async def test_duplicate_readers_share_one_lookup(self) -> None:
        client = AsyncMock()
        keys = [("o", "r", 1)]
        client.graphql = AsyncMock(return_value=_reply(keys))
        poller = PullRequestStatePoller(client, window=0.01)

        a, b, c = await asyncio.gather(
            poller.fetch("o", "r", 1),
            poller.fetch("o", "r", 1),
            poller.fetch("o", "r", 1),
        )

        assert client.graphql.await_count == 1
        # One aliased lookup, not three.
        variables = client.graphql.await_args.args[1]
        assert "n1" not in variables
        assert a == b == c

    @pytest.mark.asyncio
    async def test_batch_is_capped(self) -> None:
        client = AsyncMock()

        async def _graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
            count = sum(1 for k in variables if k.startswith("n"))
            return {f"p{i}": {"pullRequest": _node(number=i)} for i in range(count)}

        client.graphql = AsyncMock(side_effect=_graphql)
        poller = PullRequestStatePoller(client, window=0.01, max_batch=5)

        await asyncio.gather(*(poller.fetch("o", "r", n) for n in range(12)))

        assert client.graphql.await_count == 3  # 5 + 5 + 2

    @pytest.mark.asyncio
    async def test_query_uses_variables_not_interpolation(self) -> None:
        """Owner/repo go in variables, so no quoting or injection concern."""
        client = AsyncMock()
        keys = [("my-org", "my-repo", 7)]
        client.graphql = AsyncMock(return_value=_reply(keys))
        poller = PullRequestStatePoller(client, window=0.01)

        await poller.fetch("my-org", "my-repo", 7)

        query, variables = client.graphql.await_args.args
        assert "my-org" not in query
        assert variables["o0"] == "my-org"
        assert variables["r0"] == "my-repo"
        assert variables["n0"] == 7

    @pytest.mark.asyncio
    async def test_absent_pr_yields_none(self) -> None:
        client = AsyncMock()
        client.graphql = AsyncMock(return_value={"p0": {"pullRequest": None}})
        poller = PullRequestStatePoller(client, window=0.01)

        assert await poller.fetch("o", "r", 404) is None

    @pytest.mark.asyncio
    async def test_sequential_reads_issue_separate_queries(self) -> None:
        client = AsyncMock()
        client.graphql = AsyncMock(
            side_effect=lambda q, v: {"p0": {"pullRequest": _node()}}
        )
        poller = PullRequestStatePoller(client, window=0.001)

        await poller.fetch("o", "r", 1)
        await poller.fetch("o", "r", 1)

        assert client.graphql.await_count == 2


# --------------------------------------------------------------------------
# Failure handling — a lost future hangs a parked PR
# --------------------------------------------------------------------------


class TestFailureHandling:
    @pytest.mark.asyncio
    async def test_graphql_failure_falls_back_to_rest(self) -> None:
        """A GraphQL outage must degrade throughput, not blind the waits."""
        client = AsyncMock()
        client.graphql = AsyncMock(side_effect=RuntimeError("graphql down"))
        client.get = AsyncMock(
            return_value={"state": "open", "merged": False, "merged_at": None}
        )
        poller = PullRequestStatePoller(client, window=0.01)

        a, b = await asyncio.gather(
            poller.fetch("o", "r", 1), poller.fetch("o", "r", 2)
        )

        assert a is not None and b is not None
        assert a["state"] == "open"
        assert b["state"] == "open"
        assert client.get.await_count == 2

    @pytest.mark.asyncio
    async def test_savings_metric_accounts_for_rest_fallback(self) -> None:
        """A degraded run must not report savings it did not achieve."""
        client = AsyncMock()
        client.graphql = AsyncMock(side_effect=RuntimeError("graphql down"))
        client.get = AsyncMock(return_value={"state": "open"})
        poller = PullRequestStatePoller(client, window=0.01)

        await asyncio.gather(*(poller.fetch("o", "r", n) for n in range(4)))

        # 4 reads served by 1 failed query + 4 REST reads: no saving.
        assert poller.requests_served == 4
        assert poller.rest_fallback_calls == 4
        assert poller.calls_saved == 0

    @pytest.mark.asyncio
    async def test_total_failure_propagates_original_error(self) -> None:
        client = AsyncMock()
        client.graphql = AsyncMock(side_effect=RuntimeError("graphql down"))
        client.get = AsyncMock(side_effect=RuntimeError("rest down too"))
        poller = PullRequestStatePoller(client, window=0.01)

        with pytest.raises(RuntimeError, match="graphql down"):
            await poller.fetch("o", "r", 1)

    @pytest.mark.asyncio
    async def test_every_waiter_resolved_on_failure(self) -> None:
        """No future may be left pending; a parked PR would hang."""
        client = AsyncMock()
        client.graphql = AsyncMock(side_effect=RuntimeError("boom"))
        client.get = AsyncMock(side_effect=RuntimeError("boom"))
        poller = PullRequestStatePoller(client, window=0.01)

        results = await asyncio.gather(
            *(poller.fetch("o", "r", n) for n in range(10)),
            return_exceptions=True,
        )

        assert len(results) == 10
        assert all(isinstance(r, RuntimeError) for r in results)
        assert poller._pending == {}

    @pytest.mark.asyncio
    async def test_partial_rest_fallback_still_serves_recovered_prs(self) -> None:
        client = AsyncMock()
        client.graphql = AsyncMock(side_effect=RuntimeError("graphql down"))
        client.get = AsyncMock(
            side_effect=[{"state": "open"}, RuntimeError("one failed")]
        )
        poller = PullRequestStatePoller(client, window=0.01)

        results = await asyncio.gather(
            poller.fetch("o", "r", 1),
            poller.fetch("o", "r", 2),
            return_exceptions=True,
        )

        assert results[0] == {"state": "open"}
        assert results[1] is None

    @pytest.mark.asyncio
    async def test_pending_map_is_drained(self) -> None:
        client = AsyncMock()
        keys = [("o", "r", n) for n in range(5)]
        client.graphql = AsyncMock(return_value=_reply(keys))
        poller = PullRequestStatePoller(client, window=0.01)

        await asyncio.gather(*(poller.fetch("o", "r", n) for n in range(5)))

        assert poller._pending == {}

    def test_rejects_nonsense_batch_size(self) -> None:
        with pytest.raises(ValueError):
            PullRequestStatePoller(AsyncMock(), max_batch=0)


# --------------------------------------------------------------------------
# Wiring into the merge manager
# --------------------------------------------------------------------------


class TestMergeManagerWiring:
    @pytest.mark.asyncio
    async def test_uses_poller_when_configured(self) -> None:
        from tests.conftest import make_merge_manager

        mgr, client = make_merge_manager()
        keys = [("o", "r", 1)]
        client.graphql = AsyncMock(return_value=_reply(keys))
        mgr._pr_poller = PullRequestStatePoller(client, window=0.001)

        out = await mgr._fetch_pr_state("o", "r", 1)

        assert isinstance(out, dict)
        assert out["mergeable_state"] == "clean"
        client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_rest_without_poller(self) -> None:
        """Keeps a manager usable outside its async context manager."""
        from tests.conftest import make_merge_manager

        mgr, client = make_merge_manager()
        client.get = AsyncMock(return_value={"state": "open"})
        assert mgr._pr_poller is None

        out = await mgr._fetch_pr_state("o", "r", 1)

        assert out == {"state": "open"}
        client.get.assert_awaited_once_with("/repos/o/r/pulls/1")

    @pytest.mark.asyncio
    async def test_exit_clears_the_poller(self) -> None:
        """The poller holds the client, so it must not outlive it.

        Leaving it in place after ``__aexit__`` would route later reads
        into a closed client rather than the direct-read fallback that
        ``_fetch_pr_state`` documents.
        """
        from tests.conftest import make_merge_manager

        mgr, client = make_merge_manager()
        mgr._pr_poller = PullRequestStatePoller(client, window=0.001)
        mgr._github_service = None

        await mgr.__aexit__(None, None, None)

        assert mgr._pr_poller is None
        assert mgr._github_client is None
