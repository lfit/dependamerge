# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Parsing for Gerrit topic search URLs.

Supported formats (see :func:`parse_gerrit_topic_url` for the full
contract):
    https://gerrit.example.org/q/topic:some-topic
    https://gerrit.onap.org/r/q/topic:some-topic
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

from .models import ChangeSource, ParsedGerritTopicUrl, UrlParseError
from .redaction import redact_target
from .shorthand import looks_like_host, normalize_target

_SCHEME_RE = re.compile(r"\A[A-Za-z][A-Za-z0-9+.-]*://")


def _names_a_host(value: str) -> bool:
    """Report whether a target names a server of its own.

    A scheme settles it.  Without one, the first path segment does:
    ``gerrit.example.org/q/topic:x`` is a scheme-less URL, whereas
    ``q/topic:x`` is owner shorthand that would resolve against the
    GitHub default host.

    Args:
        value: The target as the operator typed it.

    Returns:
        True when the target carries its own host.
    """
    raw = value.strip()
    if _SCHEME_RE.match(raw):
        return True
    # ``//host/path`` names an authority without a scheme, and
    # ``normalize_target`` recognises it as a web URL, so refusing it
    # here made this parser disagree with every other one about the
    # same input.
    if raw.startswith("//"):
        return True
    return looks_like_host(raw.split("/", 1)[0])


# aislop-ignore-file ai-slop/hardcoded-url -- This module parses and builds
# GitHub/Gerrit URLs, so URL literals here are the subject matter, not
# stray configuration: example URLs in error/usage messages and
# docstrings, plus the canonical https://api.github.com endpoints for
# GitHub.  Enterprise hosts are always derived from the caller's input.


def parse_gerrit_topic_url(url: str) -> ParsedGerritTopicUrl:
    """
    Parse a Gerrit topic search URL.

    Supported formats (the optional base path, e.g. "r", precedes /q/):
        https://gerrit.example.org/q/topic:some-topic
        https://gerrit.onap.org/r/q/topic:some-topic
        https://gerrit.example.org/#/q/topic:some-topic  (legacy UI)

    Additional search operators in the query (separated by '+' or
    whitespace) are tolerated; only the ``topic:`` term is extracted.
    Quoted topics (``topic:"some topic"``) and percent-encoded
    characters are handled.

    Args:
        url: The URL to parse.

    Returns:
        A ParsedGerritTopicUrl with the host, base path, and topic.

    Raises:
        UrlParseError: If the URL is not a Gerrit topic search URL.
    """
    original_url = url.strip()
    if not original_url:
        raise UrlParseError("URL cannot be empty")

    # Expand git remote forms and a missing scheme into an absolute
    # URL.  Centralised so every parser understands the same set of
    # abbreviations.
    #
    # Owner shorthand is excluded, because it is a *GitHub* convenience
    # and resolves against the GitHub default host.  Letting it through
    # manufactured a Gerrit target on the wrong server: ``q/topic:x``
    # expanded to ``https://github.com/q/topic:x``, whose path this
    # parser then accepted, so ``merge`` dispatched a Gerrit topic run
    # against github.com.  A Gerrit search has to name its own host.
    if not _names_a_host(original_url):
        raise UrlParseError(
            f"Not a Gerrit search URL (no host): {redact_target(original_url)}. "
            "A topic search must name the Gerrit server, as in "
            "gerrit.example.org/q/topic:release."
        )
    normalized = normalize_target(original_url)

    try:
        parsed = urlparse(normalized)
    except Exception as exc:
        raise UrlParseError(f"Invalid URL format: {exc}") from exc

    if not parsed.hostname:
        raise UrlParseError("URL must include a hostname")

    host = parsed.hostname.lower()

    # The legacy UI keeps the query in the fragment (/#/q/...); the
    # PolyGerrit UI keeps it in the path (/q/...).
    path = unquote(parsed.path).rstrip("/")
    fragment = unquote(parsed.fragment).rstrip("/")
    if fragment.startswith("/q/"):
        query_expr = fragment[len("/q/") :]
        base_segments = [s for s in path.split("/") if s]
    else:
        segments = [s for s in path.split("/") if s]
        if "q" not in segments:
            raise UrlParseError(
                f"Not a Gerrit search URL (no /q/ segment): {redact_target(original_url)}"
            )
        q_index = segments.index("q")
        base_segments = segments[:q_index]
        query_expr = "/".join(segments[q_index + 1 :])

    if not query_expr:
        raise UrlParseError(
            f"Gerrit search URL contains no query expression: {redact_target(original_url)}"
        )

    # Gerrit search URLs separate terms with '+' (rendered as space).
    match = re.search(
        r'(?:^|[+\s])topic:(?:"([^"]+)"|([^+\s"]+))',
        query_expr,
    )
    if not match:
        raise UrlParseError(
            "Only topic searches are supported for Gerrit query URLs. "
            f"Expected: https://{host}/q/topic:some-topic "
            f"(got query: {query_expr})"
        )

    topic = (match.group(1) or match.group(2) or "").strip()
    if not topic:
        raise UrlParseError("Gerrit topic cannot be empty")

    base_path = "/".join(base_segments) if base_segments else None

    return ParsedGerritTopicUrl(
        source=ChangeSource.GERRIT,
        host=host,
        base_path=base_path,
        topic=topic,
        original_url=original_url,
    )
