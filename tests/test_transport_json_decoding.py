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


class TestValidJsonThatIsNotUsable:
    """Parsing is not the same as being a usable payload.

    ``null``, a bare string and a number are all valid JSON, so a cast
    to ``dict | list`` would satisfy the type checker and then fail in
    a caller far from here.  The shape is checked where it is known.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("payload", "described"),
        [(b"null", "NoneType"), (b'"hello"', "str"), (b"42", "int")],
    )
    async def test_a_non_container_payload_is_refused(
        self, single_attempt, payload, described
    ):
        api = _client(_responder(200, payload, "application/json"))
        try:
            with pytest.raises(RetryableError) as excinfo:
                await api.get("/repos/o/r")
        finally:
            await api._client.aclose()

        message = str(excinfo.value)
        assert "neither an object nor an array" in message
        assert described in message

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", [b"{}", b"[]", b'[{"a": 1}]'])
    async def test_containers_are_accepted(self, payload):
        # The complement.  Only the top level is checked --- verifying
        # every element of a hundred-item page would cost more than it
        # is worth, and callers already inspect what they use.
        api = _client(_responder(200, payload, "application/json"))
        try:
            assert await api.get("/repos/o/r") is not None
        finally:
            await api._client.aclose()


class TestEveryFailureCarriesTheSameContext:
    """Each rejection names method, URL, status and content-type.

    The original report was ``Expecting value: line 1 column 1 (char
    0)`` and nothing else, so partial context in some branches would
    leave the same gap for whichever case an operator happens to hit.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("body", "content_type"),
        [
            (b"", "application/json"),
            (b"<html>oops</html>", "text/html"),
            (b"not json", "application/json"),
            (b"null", "application/json"),
        ],
    )
    async def test_the_context_is_complete(self, single_attempt, body, content_type):
        api = _client(_responder(200, body, content_type))
        try:
            with pytest.raises(RetryableError) as excinfo:
                await api.get("/repos/o/r")
        finally:
            await api._client.aclose()

        message = str(excinfo.value)
        for expected in ("GET", "/repos/o/r", "200", content_type):
            assert expected in message, f"{expected!r} missing from {message!r}"


class TestABadBodyIsNotCountedAsSuccess:
    """The throttler must see a malformed body as a failure.

    It infers load from tracked errors against tracked requests, so
    recording a malformed body as a success keeps the error rate
    looking healthy during exactly the upstream trouble that produces
    malformed bodies --- and it then declines to back off.
    """

    @pytest.mark.asyncio
    async def test_a_decode_failure_is_tracked_as_an_error(self, no_backoff):
        api = _client(_responder(200, b"", "application/json"))
        try:
            with pytest.raises(RetryableError):
                await api.get("/repos/o/r")
        finally:
            await api._client.aclose()

        assert len(api._error_history) == 6

    @pytest.mark.asyncio
    async def test_a_good_body_is_still_counted_as_success(self):
        api = _client(_responder(200, b'{"ok": true}', "application/json"))
        try:
            await api.get("/repos/o/r")
        finally:
            await api._client.aclose()

        assert not api._error_history


class TestRetryBoundsDoNotMultiply:
    """GraphQL retries its own faults, not the transport's again.

    Sharing ``RetryableError`` between the two layers meant a decode
    failure was retried six times inside the transport and then five
    more by the GraphQL loop, so a persistently bad response made up to
    thirty requests instead of the six intended.
    """

    @pytest.mark.asyncio
    async def test_a_bad_body_is_not_retried_twice(self, no_backoff):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(
                200, content=b"<html>x</html>", headers={"content-type": "text/html"}
            )

        api = _client(handler)
        try:
            with pytest.raises(RetryableError):
                await api.graphql("query { viewer { login } }")
        finally:
            await api._client.aclose()

        assert attempts["n"] == 6

    @pytest.mark.asyncio
    async def test_a_transient_graphql_fault_still_retries(self):
        # The complement: the GraphQL loop must keep retrying the faults
        # it owns, which are reported in a 200 payload.
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                # Must match the transient heuristic in ``_errors``,
                # which looks for wording like "rate limit" rather than
                # an error type code.
                body = b'{"errors": [{"message": "API rate limit exceeded"}]}'
            else:
                body = b'{"data": {"ok": true}}'
            return httpx.Response(
                200, content=body, headers={"content-type": "application/json"}
            )

        api = _client(handler)
        try:
            assert await api.graphql("query { viewer { login } }") == {"ok": True}
        finally:
            await api._client.aclose()

        assert attempts["n"] == 2

    @pytest.mark.asyncio
    async def test_a_payload_without_data_retries(self):
        # Every branch the loop describes as transient has to raise the
        # type the loop retries on.  This one kept raising the transport
        # exception, so it bypassed the loop and failed after a single
        # request while still logging "retrying".
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            body = b"{}" if attempts["n"] == 1 else b'{"data": {"ok": true}}'
            return httpx.Response(
                200, content=body, headers={"content-type": "application/json"}
            )

        api = _client(handler)
        try:
            assert await api.graphql("query { viewer { login } }") == {"ok": True}
        finally:
            await api._client.aclose()

        assert attempts["n"] == 2


class TestObjectEndpointsRefuseAnArray:
    """``post``/``put``/``patch`` promise an object, so they check.

    They previously declared ``dict`` and silenced the checker with an
    ignore comment, which is the same unchecked promise this module
    exists to prevent --- one layer further out.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("verb", ["post", "put", "patch"])
    async def test_an_array_is_refused(self, single_attempt, verb):
        api = _client(_responder(200, b"[]", "application/json"))
        try:
            with pytest.raises(RetryableError) as excinfo:
                await getattr(api, verb)("/repos/o/r")
        finally:
            await api._client.aclose()

        message = str(excinfo.value)
        assert "array where an object was expected" in message
        assert verb.upper() in message

    @pytest.mark.asyncio
    @pytest.mark.parametrize("verb", ["post", "put", "patch"])
    async def test_an_object_is_returned(self, verb):
        api = _client(_responder(200, b'{"ok": true}', "application/json"))
        try:
            assert await getattr(api, verb)("/repos/o/r") == {"ok": True}
        finally:
            await api._client.aclose()

    @pytest.mark.asyncio
    async def test_get_still_accepts_an_array(self):
        # The complement: collection endpoints legitimately return one,
        # so the narrowing must not reach ``get``.
        api = _client(_responder(200, b'[{"number": 1}]', "application/json"))
        try:
            assert await api.get("/repos/o/r/pulls") == [{"number": 1}]
        finally:
            await api._client.aclose()


class TestTheSnippetDoesNotDecodeEverything:
    """Quoting the body must not cost the size of the body.

    A large HTML error page is exactly what this is most likely to be
    looking at, and it arrives when the server is already struggling.
    """

    @pytest.mark.asyncio
    async def test_a_large_body_is_truncated(self, single_attempt):
        body = b"<html>" + b"x" * 500_000 + b"</html>"
        api = _client(_responder(200, body, "text/html"))
        try:
            with pytest.raises(RetryableError) as excinfo:
                await api.get("/repos/o/r")
        finally:
            await api._client.aclose()

        # The quoted portion is bounded, whatever the body's size.
        assert len(str(excinfo.value)) < 500

    @pytest.mark.asyncio
    async def test_an_undecodable_prefix_does_not_break_the_message(
        self, single_attempt
    ):
        # The prefix may end mid-character, and a diagnostic must never
        # fail on the input it is describing.
        api = _client(_responder(200, b"\xff\xfe" + b"\xc3" * 400, "text/html"))
        try:
            with pytest.raises(RetryableError):
                await api.get("/repos/o/r")
        finally:
            await api._client.aclose()


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
