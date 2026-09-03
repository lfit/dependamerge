# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Turning a successful response body into JSON, in one place.

Six call sites used to decide this for themselves, and three of them
omitted the empty-body check, so a body that could not be JSON was
parsed anyway and the resulting ``JSONDecodeError`` escaped the
transport with no status, URL or body to act on --- abandoning a whole
multi-repository run over one bad response.

Kept apart from :mod:`_transport` so the decision lives somewhere
nameable rather than being repeated per verb.
"""

from __future__ import annotations

from typing import Any, cast

import httpx

from ._errors import RetryableError

#: Status codes defined to carry no body, so there is nothing to parse.
_BODILESS_STATUSES = frozenset({204, 205})

#: How much of an unexpected body to quote back.  Enough to recognise an
#: HTML error page or a proxy notice; short enough for one log line.
_BODY_SNIPPET_LIMIT = 200


def _snippet(r: httpx.Response) -> str:
    """Quote the start of a body without decoding all of it.

    Slicing ``r.text`` would decode the whole payload first, which is
    wasteful for the large HTML error pages this is most likely to be
    looking at --- and those arrive when the server is already
    struggling.  Decoding only the prefix keeps the cost bounded.

    ``errors="replace"`` because the prefix may end mid-character, and
    a diagnostic message must never fail on the input it describes.
    """
    prefix = (r.content or b"")[:_BODY_SNIPPET_LIMIT]
    return " ".join(prefix.decode("utf-8", errors="replace").split())


def _looks_like_json(content_type: str) -> bool:
    """Report whether a content-type declares a JSON body.

    Accepts ``application/json`` and the ``+json`` structured-suffix
    family (``application/vnd.github+json``), ignoring parameters such
    as ``; charset=utf-8``.  An absent content-type is treated as JSON,
    because the body itself then settles it and refusing outright would
    reject responses that parse perfectly well.
    """
    media_type = content_type.split(";", 1)[0].strip().lower()
    if not media_type:
        return True
    return media_type == "application/json" or media_type.endswith("+json")


def decode_json_body(
    r: httpx.Response, method: str, url: str
) -> dict[str, Any] | list[Any]:
    """Decode a successful response body, or say why it could not be.

    The single place the transport turns bytes into JSON.  Every verb
    routes through here so content-type handling, empty-body handling
    and error wording cannot drift apart --- previously six call sites
    each decided this for themselves, and three of them omitted the
    204 check entirely.

    Only 2xx responses reach this function: ``raise_for_status`` has
    already rejected 4xx and 5xx, and rejects 3xx too because redirects
    are not followed.  So the question here is never "did the request
    fail" but "is this body parseable".

    A body that cannot be JSON is reported as :class:`RetryableError`,
    not as a decode failure.  On a 2xx this almost always means an
    intermediary answered in place of the API --- an HTML error page or
    an empty body during upstream trouble --- which is transient, and
    the surrounding retry policy is exactly the machinery for it.  A
    bare ``JSONDecodeError`` matched no retry predicate, so one bad body
    abandoned an entire multi-repository run.

    Args:
        r: The successful response.
        method: HTTP method, for the error message.
        url: Request URL, for the error message.

    Returns:
        The decoded body, or an empty dict for a status defined to
        carry none.

    Raises:
        RetryableError: The body is absent or is not JSON.
    """
    if r.status_code in _BODILESS_STATUSES:
        # Defined to carry no body; an empty string is the correct
        # answer here rather than a parse failure.
        return {}

    content_type = r.headers.get("content-type", "")
    body = r.content or b""
    if not body.strip():
        raise RetryableError(
            f"{method} {url} returned {r.status_code} with content-type "
            f"{content_type!r} and an empty body; expected JSON"
        )
    if not _looks_like_json(content_type):
        snippet = _snippet(r)
        raise RetryableError(
            f"{method} {url} returned {r.status_code} with content-type "
            f"{content_type!r}, expected JSON. Body began: {snippet!r}"
        )
    try:
        # ``cast`` rather than an ignore comment: silencing the warning
        # would leave the value typed ``Any``, which then leaks through
        # every caller and defeats the checking this function exists to
        # make possible.  The cast is honest only because the shape is
        # checked immediately below.
        decoded = cast("object", r.json())
    except ValueError as exc:
        # Declared JSON but is not.  Same conclusion as a wrong
        # content-type, and worth quoting the body for the same reason.
        snippet = _snippet(r)
        raise RetryableError(
            f"{method} {url} returned {r.status_code} with content-type "
            f"{content_type!r} and an unparsable JSON body ({exc}). "
            f"Body began: {snippet!r}"
        ) from exc

    # Valid JSON is not necessarily a *usable* payload: ``null``, a bare
    # string and a number all parse.  Returning one would satisfy the
    # type checker through the cast while failing in a caller far from
    # here, so the container shape is checked where it is known.
    #
    # Only the top level is checked.  Verifying every element of a
    # hundred-item page would cost more than it is worth, and callers
    # that care already inspect the items they use.
    if not isinstance(decoded, dict | list):
        snippet = _snippet(r)
        raise RetryableError(
            f"{method} {url} returned {r.status_code} with content-type "
            f"{content_type!r} and JSON that is neither an object nor an "
            f"array ({type(decoded).__name__}). Body began: {snippet!r}"
        )
    return cast("dict[str, Any] | list[Any]", decoded)


def require_json_object(
    body: dict[str, Any] | list[Any], r: httpx.Response, method: str, url: str
) -> dict[str, Any]:
    """Narrow a decoded body to an object, or say why it is not one.

    For callers whose endpoint documents an object.  Declaring
    ``dict`` and silencing the checker would restate the mistake this
    module exists to prevent: an unchecked promise about a shape.

    Treated as transient for the same reason a non-container body is.
    An array where the API documents an object means something other
    than the API answered, which a later attempt may not repeat.

    Args:
        body: The decoded body.
        r: The response it came from, for the status.
        method: HTTP method, for the error message.
        url: Request URL, for the error message.

    Returns:
        The body, as an object.

    Raises:
        RetryableError: The body is an array.
    """
    if isinstance(body, dict):
        return body
    content_type = r.headers.get("content-type", "")
    raise RetryableError(
        f"{method} {url} returned {r.status_code} with content-type "
        f"{content_type!r} and a JSON array where an object was expected"
    )
