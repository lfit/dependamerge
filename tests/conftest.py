# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""Shared pytest fixtures for dependamerge tests.

Typed Mock Client Pattern
=========================

Problem
-------
``AsyncMergeManager`` (and ``AsyncCloseManager``) declare their internal HTTP
client as an optional type::

    self._github_client: GitHubAsync | None = None

The client is only populated inside ``__aenter__()`` (the async context
manager).  In tests we routinely bypass the context manager and inject an
``AsyncMock`` directly::

    mgr = AsyncMergeManager(token="t")
    mgr._github_client = AsyncMock()
    mgr._github_client.get = AsyncMock(return_value=...)  # ← warning!

Because the *declared* type is ``GitHubAsync | None``, basedpyright cannot
prove the value is non-``None`` after assignment and flags every subsequent
attribute access as ``reportOptionalMemberAccess``.

Solution
--------
The ``make_merge_manager`` helper (and any similar helpers in individual test
modules) returns a **tuple** ``(manager, client)`` where ``client`` is typed
as ``AsyncMock`` — a concrete, non-optional reference to the same object
stored in ``manager._github_client``.  All subsequent mock configuration
should go through the ``client`` variable::

    mgr, client = make_merge_manager(token="t")
    client.get = AsyncMock(return_value=...)          # ✓ no warning
    client.post_issue_comment = AsyncMock()            # ✓ no warning
    client.post_issue_comment.assert_called_once()     # ✓ no warning

This eliminates basedpyright ``reportOptionalMemberAccess`` warnings without
changing any production code or adding ``assert ... is not None`` boilerplate
to every test.

Guidelines for New Tests
------------------------
1. **Always** use the ``make_merge_manager`` helper (or a module-local
   wrapper around it) when you need an ``AsyncMergeManager`` with a mocked
   GitHub client outside of ``async with``.

2. Hold on to the returned ``client`` variable and use it — *not*
   ``mgr._github_client`` — for all mock setup and assertions.

3. If a test intentionally sets ``_github_client = None`` to exercise the
   "no client" code path, do that *after* unpacking the tuple::

       mgr, _client = make_merge_manager()
       mgr._github_client = None   # intentional for this test

4. If you use ``async with AsyncMergeManager(...) as mgr:`` (which calls
   ``__aenter__`` and sets the real client), you can safely replace the
   client inside the block because basedpyright already narrowed the type.
   You do **not** need this helper in that case.

5. If instead you patch ``dependamerge.merge_manager.GitHubAsync`` and
   drive the CLI end to end, use ``make_github_async_mock`` rather than
   hand-rolling the ``__aenter__`` plumbing.  It mirrors the real
   client's synchronous methods *and* its dict-returning coroutines,
   both of which a blanket ``AsyncMock`` gets wrong.

See Also
--------
- ``tests/test_dependabot_recreate.py`` — ``_make_manager()`` wraps this
  helper with module-specific defaults.
- ``tests/test_precommit_ci_trigger.py`` — same pattern.
- ``tests/test_github2gerrit_detector.py`` — same pattern for
  ``_make_mgr_with_no_gitreview``.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dependamerge.merge_manager import AsyncMergeManager
from dependamerge.url_parser import set_github_host

_RUN_INTEGRATION_ENV = "DEPENDAMERGE_RUN_INTEGRATION"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-integration`` opt-in flag.

    Live integration tests (see ``tests/integration``) hit real GitHub /
    Gerrit servers.  They already self-skip when credentials are absent,
    but they must not run as part of the ordinary unit-test suite even
    when a token happens to be present in the environment, because they
    are slow and network-dependent.  They run only when explicitly
    requested via ``--run-integration`` or the
    ``DEPENDAMERGE_RUN_INTEGRATION`` environment variable.
    """
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run live GitHub/Gerrit integration tests (marked 'integration').",
    )


def _integration_enabled(config: pytest.Config) -> bool:
    if config.getoption("--run-integration"):
        return True
    return os.environ.get(_RUN_INTEGRATION_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``integration`` tests unless explicitly opted in."""
    if _integration_enabled(config):
        return
    skip_integration = pytest.mark.skip(
        reason="integration tests disabled (pass --run-integration or set "
        f"{_RUN_INTEGRATION_ENV}=1)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def _reset_github_host_override(monkeypatch):
    """Detach every test from ambient GitHub host configuration.

    The resolved host comes from a process-wide override *and* three
    environment variables.  Clearing only the override leaves the rest:
    a developer who legitimately has ``GH_HOST`` set for their
    Enterprise installation would run the otherwise-dotcom suite
    against that host and see failures nobody else can reproduce.

    The same hazard as the ambient git configuration in
    ``tests/test_local_repo.py``, and worth the same treatment.  Tests
    that want a host set do so explicitly, after this has run.
    """
    for name in (
        "DEPENDAMERGE_GITHUB_HOST",
        "DEPENDAMERGE_GITHUB_HOSTS",
        "GH_HOST",
    ):
        monkeypatch.delenv(name, raising=False)
    set_github_host(None)
    yield
    set_github_host(None)


def make_merge_manager(**overrides: Any) -> tuple[AsyncMergeManager, AsyncMock]:
    """Build an ``AsyncMergeManager`` with a pre-injected ``AsyncMock`` client.

    Returns a ``(manager, client)`` tuple.  The ``client`` reference is typed
    as ``AsyncMock`` (never ``None``), so attribute access on it will not
    trigger basedpyright ``reportOptionalMemberAccess`` warnings.

    All keyword arguments are forwarded to ``AsyncMergeManager.__init__``.
    A ``token`` default of ``"test-token"`` is provided if not supplied.

    Usage::

        mgr, client = make_merge_manager(preview_mode=True)
        client.get = AsyncMock(return_value={...})
        result = await mgr._some_method(pr)
        client.get.assert_called_once()

    Parameters
    ----------
    **overrides:
        Keyword arguments forwarded to ``AsyncMergeManager()``.

    Returns
    -------
    tuple[AsyncMergeManager, AsyncMock]
        The manager instance and its mock GitHub client.
    """
    defaults: dict[str, Any] = {"token": "test-token"}
    defaults.update(overrides)
    mgr = AsyncMergeManager(**defaults)
    client = AsyncMock()
    # Mirror the real client's shape for the handful of *synchronous*
    # methods on it.  A blanket ``AsyncMock`` turns these into
    # coroutines that production code never awaits, producing
    # "never awaited" warnings that obscure genuine ones.
    client.invalidate_block_reason = MagicMock()
    client.clear_block_reasons = MagicMock()
    mgr._github_client = client
    return mgr, client


def make_github_async_mock() -> AsyncMock:
    """Build a faithful stand-in for the ``GitHubAsync`` class's instance.

    For tests that patch ``dependamerge.merge_manager.GitHubAsync`` and
    drive the CLI end to end, rather than injecting a client directly as
    :func:`make_merge_manager` does.  Assign the result to the patched
    class's ``return_value``.

    A *single* mock is returned, and its ``__aenter__`` yields that same
    object, because the real ``GitHubAsync.__aenter__`` returns ``self``.
    Splitting it into a separate "instance" and "client" pair --- the
    shape this replaced --- is the trap: ``AsyncMergeManager`` keeps the
    object it constructed and calls ``__aenter__`` only for its side
    effect, then shares that object with ``GitHubService``.  Configuring
    the ``__aenter__`` result therefore left the object actually in use
    bare, and the divergence was invisible because both are mocks.

    Two further details keep the double honest where a blanket
    ``AsyncMock`` silently diverges:

    *Synchronous methods.*  ``invalidate_block_reason`` and
    ``clear_block_reasons`` do not return awaitables, and production
    code calls them without ``await``.  Mocked as coroutine functions
    they leak "never awaited" warnings.

    *Dict-returning coroutines.*  ``graphql`` is declared to return
    ``dict[str, Any]``, but awaiting a bare ``AsyncMock`` yields another
    ``AsyncMock``.  Production code then calls ``.get(...)`` on it,
    which manufactures a coroutine nobody awaits.  That warning is
    reported against whichever line happens to be running when the
    garbage collector reclaims the object, so it points nowhere near the
    cause --- see #420 for how long that misdirection held.  Returning a
    real empty dict keeps the "no repository data" branch honest;
    override it when a test cares about the payload.

    Usage::

        with patch("dependamerge.merge_manager.GitHubAsync") as cls:
            client = make_github_async_mock()
            cls.return_value = client
            client.merge_pull_request = AsyncMock(return_value=True)

    Returns
    -------
    AsyncMock
        The client, which is also its own async-context-manager result.
    """
    client = AsyncMock()
    client.invalidate_block_reason = MagicMock()
    client.clear_block_reasons = MagicMock()
    client.graphql = AsyncMock(return_value={})
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client
