# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Discovery of the HTTP base path a Gerrit server is served under.

Some Gerrit deployments sit behind a reverse-proxy prefix such as
``/infra/``.  This module probes a host for that prefix, caches the
answer for the process lifetime, and guards the probes with a circuit
breaker so an unreachable host is not hammered on every lookup.

Discovery always fails open: any network problem yields an empty base
path rather than an exception.
"""

from __future__ import annotations

import logging
import socket
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger("dependamerge.gerrit.urls")

# Module-level cache for discovered base paths
_BASE_PATH_CACHE: dict[str, str] = {}

# Circuit breaker state: tracks hosts that have failed recently
# Maps host -> (failure_count, last_failure_timestamp)
_CIRCUIT_BREAKER: dict[str, tuple[int, float]] = {}

# Circuit breaker configuration
_CIRCUIT_BREAKER_THRESHOLD = 3  # Number of failures before opening circuit
_CIRCUIT_BREAKER_RESET_SECONDS = 300.0  # Time before resetting failure count


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """HTTP handler that captures redirects instead of following them."""

    def http_error_301(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any
    ) -> Any:
        return fp

    def http_error_302(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any
    ) -> Any:
        return fp

    def http_error_303(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any
    ) -> Any:
        return fp

    def http_error_307(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any
    ) -> Any:
        return fp

    def http_error_308(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any
    ) -> Any:
        return fp


def _check_circuit_breaker(host: str) -> bool:
    """
    Check if the circuit breaker is open for a host.

    Returns:
        True if the circuit is open (should skip network requests),
        False if the circuit is closed (OK to make requests).
    """
    if host not in _CIRCUIT_BREAKER:
        return False

    failure_count, last_failure_time = _CIRCUIT_BREAKER[host]
    now = time.monotonic()

    # Check if enough time has passed to reset the circuit
    if now - last_failure_time >= _CIRCUIT_BREAKER_RESET_SECONDS:
        del _CIRCUIT_BREAKER[host]
        log.debug("Circuit breaker reset for host: %s", host)
        return False

    # Circuit is open if we've exceeded the failure threshold
    return failure_count >= _CIRCUIT_BREAKER_THRESHOLD


def _record_circuit_breaker_failure(host: str) -> None:
    """Record a network failure for circuit breaker tracking."""
    now = time.monotonic()

    if host in _CIRCUIT_BREAKER:
        failure_count, last_failure_time = _CIRCUIT_BREAKER[host]
        # Reset count if it's been a while since the last failure
        if now - last_failure_time >= _CIRCUIT_BREAKER_RESET_SECONDS:
            failure_count = 0
        _CIRCUIT_BREAKER[host] = (failure_count + 1, now)
    else:
        _CIRCUIT_BREAKER[host] = (1, now)

    new_count = _CIRCUIT_BREAKER[host][0]
    if new_count >= _CIRCUIT_BREAKER_THRESHOLD:
        log.warning(
            "Circuit breaker opened for host %s after %d failures",
            host,
            new_count,
        )


def _reset_circuit_breaker(host: str) -> None:
    """Reset the circuit breaker for a host after a successful request."""
    if host in _CIRCUIT_BREAKER:
        del _CIRCUIT_BREAKER[host]
        log.debug("Circuit breaker reset after success for host: %s", host)


def _record_discovered_base_path(
    host: str, location: str, known_endpoints: set[str]
) -> str:
    """Cache and log the base path implied by a redirect Location header."""
    base_path = _extract_base_path(host, location, known_endpoints)
    _BASE_PATH_CACHE[host] = base_path
    log.debug("Discovered base path for %s: %r", host, base_path)
    return base_path


def _probe_base_path(
    opener: Any, url: str, host: str, probe: str, timeout: float, endpoints: set[str]
) -> tuple[str | None, bool]:
    """
    Issue a single discovery probe and interpret the response.

    Returns:
        A ``(base_path, network_failure)`` pair.  ``base_path`` is the
        discovered base path (possibly ``""`` for "no base path") when
        the probe was conclusive, or None when discovery should carry on
        with the next probe.  ``network_failure`` is True when the probe
        failed at the network level.
    """
    try:
        resp = opener.open(url, timeout=timeout)
        code = getattr(resp, "getcode", lambda: 0)() or getattr(resp, "status", 0)

        # Successful connection - reset circuit breaker
        _reset_circuit_breaker(host)

        # 200 OK means no base path needed
        if code == 200:
            log.debug("Discovered base path for %s: (none)", host)
            _BASE_PATH_CACHE[host] = ""
            return "", False

        if code in (301, 302, 303, 307, 308):
            headers = getattr(resp, "headers", {}) or {}
            location = headers.get("Location") or headers.get("location") or ""
            if location:
                return _record_discovered_base_path(host, location, endpoints), False

    except urllib.error.HTTPError as http_err:
        # HTTPError also contains response info - this is not a
        # network failure, just an HTTP error response
        _reset_circuit_breaker(host)

        code = http_err.code
        if code in (301, 302, 303, 307, 308):
            location = (
                http_err.headers.get("Location")
                or http_err.headers.get("location")
                or ""
            )
            if location:
                return _record_discovered_base_path(host, location, endpoints), False

    except urllib.error.URLError as url_err:
        # URLError covers DNS failures, connection refused, etc.
        reason = getattr(url_err, "reason", str(url_err))

        if isinstance(reason, socket.gaierror):
            log.debug("DNS resolution failed for %s: %s", host, reason)
        elif isinstance(reason, socket.timeout):
            log.debug("Connection timeout for %s%s", host, probe)
        elif isinstance(reason, ConnectionRefusedError):
            log.debug("Connection refused for %s%s", host, probe)
        else:
            log.debug("Network error for %s%s: %s", host, probe, reason)

        _record_circuit_breaker_failure(host)
        return None, True

    except OSError as os_err:
        # OSError covers low-level network issues
        log.debug("OS-level network error for %s%s: %s", host, probe, os_err)
        _record_circuit_breaker_failure(host)
        return None, True

    except Exception as exc:
        # Catch-all for unexpected errors
        log.debug(
            "Unexpected error during base path probe for %s%s: %s", host, probe, exc
        )

    return None, False


def _run_base_path_probes(
    host: str, timeout: float, max_total_time: float
) -> str | None:
    """
    Probe *host* until a probe is conclusive or the budget is spent.

    Returns:
        The discovered base path, or None if no probe was conclusive.
    """
    start_time = time.monotonic()

    # Known Gerrit endpoints that should exist
    known_endpoints = {
        "changes",
        "accounts",
        "dashboard",
        "c",
        "q",
        "admin",
        "login",
        "settings",
        "plugins",
        "Documentation",
    }

    opener = urllib.request.build_opener(_NoRedirect)
    opener.addheaders = [("User-Agent", "dependamerge/gerrit-urls")]

    # Probe endpoints that typically redirect to the base path
    probes = ["/dashboard/self", "/"]

    # Track consecutive network failures for circuit breaker
    network_failures = 0

    for scheme in ("https", "http"):
        for probe in probes:
            # Check if we've exceeded the total time budget
            elapsed = time.monotonic() - start_time
            if elapsed >= max_total_time:
                log.debug("Discovery timeout for %s after %.1fs", host, elapsed)
                break

            url = f"{scheme}://{host}{probe}"

            base_path, network_failure = _probe_base_path(
                opener, url, host, probe, timeout, known_endpoints
            )
            if network_failure:
                network_failures += 1
            if base_path is not None:
                return base_path

    # Log if all probes failed due to network issues
    if network_failures > 0:
        log.warning(
            "Base path discovery failed for %s due to %d network error(s), "
            "defaulting to no base path",
            host,
            network_failures,
        )

    return None


def discover_base_path(
    host: str, timeout: float = 5.0, max_total_time: float = 15.0
) -> str:
    """
    Discover the HTTP base path for a Gerrit host.

    This function probes the Gerrit server to detect if it uses a base path
    (like "/infra/") by checking for redirects from common endpoints.

    The discovery result is cached for the process lifetime.

    Network Resilience:
        - Uses a circuit breaker pattern to avoid repeated requests to
          hosts that are experiencing network issues.
        - Implements an overall timeout for the entire discovery process.
        - Gracefully handles DNS failures, connection timeouts, and
          other network errors.

    Args:
        host: The Gerrit hostname (without scheme).
        timeout: Connection timeout in seconds for individual requests.
        max_total_time: Maximum total time for the entire discovery process.

    Returns:
        The base path (e.g., "infra") or empty string if none.
        Returns empty string on network failures (fails open).
    """
    if not host:
        return ""

    cached = _BASE_PATH_CACHE.get(host)
    if cached is not None:
        return cached

    # Check circuit breaker - if open, fail fast with default
    if _check_circuit_breaker(host):
        log.debug("Circuit breaker open for %s, skipping discovery", host)
        return ""

    base_path = _run_base_path_probes(host, timeout, max_total_time)
    if base_path is not None:
        return base_path

    # Default to no base path (fail open)
    _BASE_PATH_CACHE[host] = ""
    log.debug("No base path discovered for %s", host)
    return ""


def _extract_base_path(host: str, location: str, known_endpoints: set[str]) -> str:
    """Extract the base path from a redirect Location header."""
    parsed = urlparse(location)

    path = parsed.path if parsed.scheme or parsed.netloc else location

    # Split into segments
    segments = [s for s in path.split("/") if s]

    if not segments:
        return ""

    # The first segment is the base path if it's not a known endpoint
    first = segments[0]
    if first not in known_endpoints:
        return first

    return ""
