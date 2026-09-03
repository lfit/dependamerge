# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
HTTP transport for the async GitHub client.

The single retrying request path plus the REST verb helpers, the
GraphQL call and the paginated iterator built on top of it.  Rate-limit
header parsing and the sleep-until-reset wait live here too, because
they are only meaningful to a request in flight.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import (
    Any,
    TypeVar,
    cast,
)

import httpx
from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# ``_now`` stays an attribute of the package rather than a name bound
# here: it was a module-level attribute of ``dependamerge.github_async``
# before the split, and callers substitute it there.
import dependamerge.github_async as _pkg

from ._base import _GitHubAsyncBase
from ._errors import (
    _TENACITY_MAX_BACKOFF,
    GraphQLError,
    RetryableError,
    SecondaryRateLimitError,
    _is_retryable_status,
    _is_transient_graphql_error,
)
from ._json_body import decode_json_body, require_json_object
from ._responses import (
    _finish_successful_request,
    _handle_forbidden,
    _handle_retryable_status,
)
from ._throttling import _maybe_await

_F = TypeVar("_F", bound=Callable[..., Any])


class _GraphQLRetry(RetryableError):
    """A GraphQL-level fault worth another attempt.

    Distinct from a plain :class:`RetryableError` so the GraphQL loop
    can retry its own faults without also re-retrying transport and
    decode faults, which ``_request_json`` has already exhausted ---
    sharing one type multiplied the two bounds.

    It *subclasses* ``RetryableError`` rather than standing apart
    because that type is exported from the package: when these retries
    are exhausted the exception reaches callers, and before this split
    it reached them as ``RetryableError``.  Subclassing keeps that
    contract while still letting the loop select on the narrower type.

    The direction that matters is one-way: a transport failure is not a
    ``_GraphQLRetry``, so the loop still cannot catch one, and the
    six-attempt bound holds.
    """


def _transport_retry() -> Callable[[_F], _F]:
    """The retry policy every transport entry point shares.

    Defined once rather than repeated per method.  ``_request`` and
    ``_request_json`` differ only in whether they decode, and they are
    required to behave identically otherwise; copying the decorator
    would let a later change to the attempt count, the backoff or the
    exception list apply to one and silently not the other.
    """
    return cast(
        "Callable[[_F], _F]",
        retry(
            reraise=True,
            stop=stop_after_attempt(6),
            wait=wait_random_exponential(multiplier=0.5, max=_TENACITY_MAX_BACKOFF),
            retry=retry_if_exception_type(
                (
                    httpx.TransportError,
                    httpx.ReadTimeout,
                    RetryableError,
                    SecondaryRateLimitError,
                )
            ),
        ),
    )


class _TransportMixin(_GitHubAsyncBase):
    """REST/GraphQL request plumbing mixed into ``GitHubAsync``."""

    def _parse_rate_limit_headers(
        self, r: httpx.Response
    ) -> tuple[int, int, float | None]:
        """
        Parse GitHub rate limit headers.

        Returns:
            (remaining, limit, reset_epoch)
        """
        remaining = int(r.headers.get("X-RateLimit-Remaining", "1"))
        limit = int(r.headers.get("X-RateLimit-Limit", "60"))
        reset = r.headers.get("X-RateLimit-Reset")
        reset_epoch = float(reset) if reset else None
        return remaining, limit, reset_epoch

    async def _sleep_until(self, reset_epoch: float) -> None:
        now = _pkg._now()
        delay = max(0.0, reset_epoch - now)
        if delay > 0:
            await _maybe_await(self.on_rate_limited, reset_epoch)
            try:
                await asyncio.sleep(delay)
            finally:
                await _maybe_await(self.on_rate_limit_cleared)

    async def _request_once(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Perform one attempt, with rate-limit and transient handling.

        Un-retried on purpose.  Both :meth:`_request` and
        :meth:`_request_json` wrap this with the *same* retry policy, so
        decoding can happen inside the retried scope without nesting two
        policies or duplicating one.
        """
        async with self.semaphore:
            async with self.limiter:
                r = await self._client.request(method, url, **kwargs)

        # 401 should not be retried (bad credentials)
        if r.status_code == 401:
            r.raise_for_status()

        # Primary rate limit: examine headers and body
        if r.status_code == 403:
            await _handle_forbidden(self, r)

        # Retryable transient statuses
        if _is_retryable_status(r.status_code):
            await _handle_retryable_status(self, r)

        # All other errors -> raise.  This covers 4xx and 5xx, and also
        # 3xx, since redirects are not followed --- so nothing below
        # this line is looking at a failed response.
        r.raise_for_status()
        return r

    @_transport_retry()
    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """
        Low-level request with concurrency limit, RPS limit, and retry handling.
        Handles primary/secondary rate limits and transient statuses.
        """
        r = await self._request_once(method, url, **kwargs)
        await _finish_successful_request(self, r)
        return r

    @_transport_retry()
    async def _request_json(
        self, method: str, url: str, *, require_object: bool = False, **kwargs
    ) -> tuple[httpx.Response, dict[str, Any] | list[Any]]:
        """Perform a request and decode its JSON body, retrying both.

        Decoding sits *inside* the retried scope deliberately.  A 2xx
        carrying an empty or non-JSON body is a transient symptom of
        upstream trouble, so it should be retried like any other, rather
        than escaping as a decode failure that matches no predicate.

        ``require_object`` belongs here for the same reason, rather than
        at the calling verb.  Applied afterwards it sat outside the
        retried scope and after success had been recorded, so an array
        where an object was documented raised once, was never retried,
        and was still counted as a healthy request.

        Success is recorded *after* every check, not before.  The
        throttler infers load from the ratio of tracked errors to
        tracked requests, so counting a malformed body as a success
        would keep the error rate looking healthy during precisely the
        upstream trouble that produces malformed bodies --- and it would
        then decline to back off.

        Returns the response alongside the body, because a caller may
        still need the headers --- pagination reads ``Link``.
        """
        r = await self._request_once(method, url, **kwargs)
        try:
            body = decode_json_body(r, method, url)
            if require_object:
                body = require_json_object(body, r, method, url)
        except RetryableError:
            self._track_error("transient_error")
            raise
        await _finish_successful_request(self, r)
        return r, body

    async def _fetch_json(
        self, method: str, url: str, **kwargs
    ) -> tuple[httpx.Response, dict[str, Any] | list[Any]]:
        """Restore the type the retry decorator erases.

        Tenacity's decorator is untyped, so awaiting :meth:`_request_json`
        yields ``Any`` and that leaks into every caller --- which is how
        the unchecked ``.json()`` calls went unnoticed in the first place.
        Narrowing once here keeps all six verbs honest without six casts.
        """
        return cast(
            "tuple[httpx.Response, dict[str, Any] | list[Any]]",
            await self._request_json(method, url, **kwargs),
        )

    async def _fetch_object(self, method: str, url: str, **kwargs) -> dict[str, Any]:
        """Fetch a body the endpoint documents as an object.

        The narrowing happens inside the retried scope, so the cast here
        follows a check rather than replacing one.
        """
        _, body = await self._fetch_json(method, url, require_object=True, **kwargs)
        return cast("dict[str, Any]", body)

    async def get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[Any]:
        _, body = await self._fetch_json("GET", f"{self.api_url}{path}", params=params)
        return body

    async def post(
        self, path: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._fetch_object("POST", f"{self.api_url}{path}", json=json)

    async def put(
        self, path: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._fetch_object("PUT", f"{self.api_url}{path}", json=json)

    async def patch(
        self, path: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._fetch_object("PATCH", f"{self.api_url}{path}", json=json)

    async def graphql(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute a GraphQL query with retry for transient GraphQL errors.

        Note: HTTP-level issues are handled by _request's retry. Here we add
        retry for 200 OK responses that include GraphQL-level transient errors.
        """
        payload = {"query": query, "variables": variables or {}}

        # Retries only on *GraphQL-level* faults --- a 200 whose payload
        # reports a transient error.  Transport and decode faults are
        # already retried inside ``_fetch_json`` and must not be caught
        # again here: sharing ``RetryableError`` between the two layers
        # multiplied the bounds, so a persistently malformed response
        # made up to thirty requests rather than the six intended.
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(5),
            wait=wait_random_exponential(multiplier=0.5, max=10.0),
            retry=retry_if_exception_type(_GraphQLRetry),
        ):
            with attempt:
                _, data = await self._fetch_json("POST", self.graphql_url, json=payload)
                if not isinstance(data, dict):
                    self.log.debug("GraphQL response was not an object; retrying")
                    raise _GraphQLRetry("Malformed GraphQL response")
                if "errors" in data and data["errors"]:
                    # Retry on transient errors, otherwise raise
                    if _is_transient_graphql_error(data["errors"]):
                        self.log.debug(
                            "Transient GraphQL error encountered; retrying: %s",
                            data["errors"],
                        )
                        raise _GraphQLRetry("Transient GraphQL error")
                    # Non-transient; raise detailed error
                    raise GraphQLError(json.dumps(data["errors"]))
                if "data" not in data:
                    # Unexpected shape; treat as transient.  This is a
                    # GraphQL-level fault, so it must raise the type the
                    # loop below retries on --- raising ``RetryableError``
                    # here bypassed the loop entirely and failed after a
                    # single request, while still logging "retrying".
                    self.log.debug("GraphQL response missing 'data'; retrying")
                    raise _GraphQLRetry("Malformed GraphQL response")
                return data["data"]  # type: ignore[no-any-return]

        # Should not be reached due to reraise=True; keep mypy happy
        raise GraphQLError("GraphQL request failed after retries")

    async def get_paginated(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        per_page: int = 100,
        max_pages: int | None = None,
    ) -> AsyncIterator[dict[str, Any] | list[Any]]:
        """
        Iterate through a paginated REST collection.

        Yields each page as the API returned it.  That is a *list* for
        collection endpoints and a *dict* for the ones that wrap their
        items in an object (``/actions/runs``, for instance), so callers
        check the shape before using it.  The annotation said ``dict``
        only, which was never true; the type is now honest about both.
        """
        page = 1
        while True:
            q = dict(params or {})
            q.update({"per_page": per_page, "page": page})
            r, data = await self._fetch_json("GET", f"{self.api_url}{path}", params=q)
            if not data:
                return
            yield data
            page += 1
            if max_pages and page > max_pages:
                return
            # Stop when Link header doesn't include 'rel="next"'
            link = r.headers.get("Link", "")
            if 'rel="next"' not in link:
                return
