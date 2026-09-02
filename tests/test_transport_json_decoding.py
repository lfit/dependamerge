# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Tests for decoding a response body into JSON at the transport boundary.

A live run aborted with a bare ``Expecting value: line 1 column 1
(char 0)`` --- a ``JSONDecodeError`` escaping from the transport with
no status, URL or body to act on, abandoning a 21-pull-request merge.
It arrived during an upstream wobble, two thirds of requests failing
with 502, when an intermediary answered 200 with something that was
not JSON.

These tests cover the three things that went wrong: a body that cannot
be JSON was parsed anyway, the failure was not treated as transient so
nothing retried, and the report named nothing useful.
"""

from __future__ import annotations

from typing import cast

import httpx
import pytest
import tenacity

from dependamerge.github_async import GitHubAsync
from dependamerge.github_async._errors import RetryableError


def _client(handler) -> GitHubAsync:
    """Build a client whose transport is driven by ``handler``."""
    api = GitHubAsync(token="t")
    api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return api


def _responder(status: int, body: bytes, content_type: str | None):
    headers = {"content-type": content_type} if content_type else {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body, headers=headers)

    return handler


def _retrying() -> tenacity.AsyncRetrying:
    """The retry controller shared by every ``_request_json`` call.

    Tenacity attaches this to the *function* at decoration time, so the
    typing stubs do not know about it.
    """
    return cast(tenacity.AsyncRetrying, GitHubAsync._request_json.retry)  # type: ignore[attr-defined]


@pytest.fixture
def single_attempt(monkeypatch):
    """Disable retrying for tests that assert the failure itself.

    The controller belongs to the *function*, so it is shared by every
    instance and every test in the process.  Assigning to it directly
    leaks into unrelated tests --- which it did, until two retry tests
    started failing for no reason of their own.  ``monkeypatch``
    restores it afterwards.
    """
    monkeypatch.setattr(_retrying(), "stop", tenacity.stop_after_attempt(1))


@pytest.fixture
def no_backoff(monkeypatch):
    """Keep the attempt count, drop the waiting between attempts."""
    monkeypatch.setattr(_retrying(), "wait", tenacity.wait_none())


class TestBodiesThatCannotBeJson:
    """A body that cannot be JSON is never handed to the parser."""

    @pytest.mark.asyncio
    async def test_a_bodiless_status_yields_an_empty_mapping(self):
        # 204 is defined to carry no body, so there is nothing to parse
        # and nothing wrong.  Three of the six original call sites
        # omitted this check and raised JSONDecodeError instead.
        api = _client(_responder(204, b"", None))
        try:
            assert await api.get("/repos/o/r") == {}
        finally:
            await api._client.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("body", "content_type", "expected"),
        [
            (b"", "application/json", "empty body"),
            (b"   \n ", "application/json", "empty body"),
            (b"<html>502 Bad Gateway</html>", "text/html", "content-type"),
            (b"not json at all", "application/json", "unparsable"),
        ],
    )
    async def test_the_failure_names_what_went_wrong(
        self, single_attempt, body, content_type, expected
    ):
        api = _client(_responder(200, body, content_type))
        try:
            with pytest.raises(RetryableError) as excinfo:
                await api.get("/repos/o/r")
        finally:
            await api._client.aclose()

        message = str(excinfo.value)
        assert expected in message
        # The original report was "Expecting value: line 1 column 1
        # (char 0)" and nothing else.  Method, URL and status are the
        # minimum needed to act on it.
        assert "GET" in message
        assert "/repos/o/r" in message
        assert "200" in message

    @pytest.mark.asyncio
    async def test_an_unexpected_body_is_quoted_back(self, single_attempt):
        api = _client(_responder(200, b"<html>Gateway Timeout</html>", "text/html"))
        try:
            with pytest.raises(RetryableError, match="Gateway Timeout"):
                await api.get("/repos/o/r")
        finally:
            await api._client.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content_type",
        [
            "application/json",
            "application/json; charset=utf-8",
            "application/vnd.github+json",
            "application/vnd.github.v3+json; charset=utf-8",
            None,
        ],
    )
    async def test_json_content_types_are_accepted(self, content_type):
        # The complement.  GitHub answers with the vendor media type, so
        # a naive "application/json" equality test would reject every
        # real response.  An absent header is allowed too: the body then
        # settles it, and refusing outright would reject responses that
        # parse perfectly well.
        api = _client(_responder(200, b'{"ok": true}', content_type))
        try:
            assert await api.get("/repos/o/r") == {"ok": True}
        finally:
            await api._client.aclose()


class TestABadBodyIsTransient:
    """A 2xx that is not JSON retries, rather than ending the run."""

    @pytest.mark.asyncio
    async def test_it_recovers_when_a_later_attempt_is_valid(self):
        # The shape actually observed: an intermediary answers for the
        # API during upstream trouble, then stops.  ``JSONDecodeError``
        # matched no retry predicate, so this aborted the whole run
        # instead of retrying.
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] <= 2:
                return httpx.Response(
                    200,
                    content=b"<html>502</html>",
                    headers={"content-type": "text/html"},
                )
            return httpx.Response(
                200,
                content=b'{"recovered": true}',
                headers={"content-type": "application/json"},
            )

        api = _client(handler)
        try:
            assert await api.get("/repos/o/r") == {"recovered": True}
        finally:
            await api._client.aclose()

        assert attempts["n"] == 3

    @pytest.mark.asyncio
    async def test_a_persistently_bad_endpoint_still_fails(self, no_backoff):
        # Retrying a transient fault must not become retrying forever.
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(
                200, content=b"", headers={"content-type": "application/json"}
            )

        api = _client(handler)
        try:
            with pytest.raises(RetryableError):
                await api.get("/repos/o/r")
        finally:
            await api._client.aclose()

        assert attempts["n"] == 6


class TestEveryVerbDecodesTheSameWay:
    """No verb keeps its own copy of this handling.

    Six call sites each decided it independently, and three omitted the
    bodiless-status check.  Routing them through one function is what
    stops them drifting apart again.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("verb", ["get", "post", "put", "patch"])
    async def test_a_non_json_body_is_refused_by_every_verb(self, single_attempt, verb):
        api = _client(_responder(200, b"<html>oops</html>", "text/html"))
        try:
            with pytest.raises(RetryableError, match="content-type"):
                await getattr(api, verb)("/repos/o/r")
        finally:
            await api._client.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("verb", ["post", "put", "patch"])
    async def test_a_bodiless_status_is_accepted_by_every_verb(self, verb):
        api = _client(_responder(204, b"", None))
        try:
            assert await getattr(api, verb)("/repos/o/r") == {}
        finally:
            await api._client.aclose()

    @pytest.mark.asyncio
    async def test_graphql_refuses_a_non_json_body(self, single_attempt):
        api = _client(_responder(200, b"<html>oops</html>", "text/html"))
        try:
            with pytest.raises(RetryableError, match="content-type"):
                await api.graphql("query { viewer { login } }")
        finally:
            await api._client.aclose()

    @pytest.mark.asyncio
    async def test_pagination_refuses_a_non_json_body(self, single_attempt):
        api = _client(_responder(200, b"<html>oops</html>", "text/html"))
        try:
            with pytest.raises(RetryableError, match="content-type"):
                async for _ in api.get_paginated("/repos/o/r/pulls"):
                    pass
        finally:
            await api._client.aclose()

    @pytest.mark.asyncio
    async def test_pagination_still_reads_the_link_header(self):
        # ``_request_json`` returns the response as well as the body
        # precisely so this keeps working.
        pages = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            pages["n"] += 1
            headers = {"content-type": "application/json"}
            if pages["n"] == 1:
                headers["Link"] = '<https://api.github.com/next>; rel="next"'
            return httpx.Response(200, content=b'[{"number": 1}]', headers=headers)

        api = _client(handler)
        try:
            seen = [page async for page in api.get_paginated("/repos/o/r/pulls")]
        finally:
            await api._client.aclose()

        assert len(seen) == 2
        assert pages["n"] == 2
