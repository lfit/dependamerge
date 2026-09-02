# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Shared attribute and method declarations for the ``GitHubAsync`` mixins.

``GitHubAsync`` is too large for one reviewable module, so its methods
live in ``_XxxMixin`` classes that are mixed back together in
``dependamerge.github_async._client``.  Each mixin reads state that
``GitHubAsync.__init__`` establishes and calls methods implemented by
its siblings; this base declares both, and declares nothing at runtime,
so the real implementations always win.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

import httpx
from aiolimiter import AsyncLimiter

if TYPE_CHECKING:
    from ._errors import PermissionError
    from ._throttling import _Budget, _ResizableSemaphore


class _GitHubAsyncBase:
    """Declarations shared by the ``GitHubAsync`` mixins.

    ``GitHubAsync`` is assembled from mixins that each hold one area of
    the client.  They necessarily read state established by
    ``GitHubAsync.__init__`` and call methods that live in sibling
    mixins.  This base declares both so that each mixin type-checks in
    isolation; it defines nothing at runtime beyond the class itself.
    """

    # Established by GitHubAsync.__init__.
    token: str | None
    api_url: str
    graphql_url: str
    semaphore: _ResizableSemaphore
    limiter: AsyncLimiter
    log: logging.Logger
    on_rate_limited: Callable[[float], None | Awaitable[None]] | None
    on_rate_limit_cleared: Callable[[], None | Awaitable[None]] | None
    on_metrics: Callable[[int, float], None | Awaitable[None]] | None
    _client: httpx.AsyncClient
    _timeout: float
    _max_concurrency: int
    _base_max_concurrency: int
    _base_rps: float
    _current_rps: float
    _error_history: list[tuple[float, str]]
    _request_history: list[float]
    _error_window: int
    _last_retry_after: float | None
    _adaptive_delay: float
    _last_adaptive_update: float | None
    _budgets: dict[str, _Budget]
    _healthy_streak: int
    _authenticated_user_login: str | None
    _default_branch_cache: dict[str, str | None]
    _required_checks_cache: dict[str, list[dict[str, Any]]]
    _branch_protection_cache: dict[str, dict[str, Any]]
    _requires_signatures_cache: dict[str, bool]
    _requires_strict_checks_cache: dict[str, bool]
    _block_reason_cache: dict[tuple[str, str, int, str, str | None], tuple[float, str]]
    _token_scopes: set[str] | None
    _token_scopes_fetched: bool

    # Implemented by sibling mixins; declared here (type-checking only,
    # so nothing is defined at runtime) for cross-mixin calls.
    if TYPE_CHECKING:

        async def _request(self, method: str, url: str, **kwargs) -> httpx.Response: ...

        async def get(
            self, path: str, params: dict[str, Any] | None = None
        ) -> dict[str, Any] | list[dict[str, Any]]: ...

        async def post(
            self, path: str, json: dict[str, Any] | None = None
        ) -> dict[str, Any]: ...

        async def put(
            self, path: str, json: dict[str, Any] | None = None
        ) -> dict[str, Any]: ...

        async def patch(
            self, path: str, json: dict[str, Any] | None = None
        ) -> dict[str, Any]: ...

        async def graphql(
            self, query: str, variables: dict[str, Any] | None = None
        ) -> dict[str, Any]: ...

        def get_paginated(
            self,
            path: str,
            *,
            params: dict[str, Any] | None = None,
            per_page: int = 100,
            max_pages: int | None = None,
        ) -> AsyncIterator[dict[str, Any] | list[dict[str, Any]]]: ...

        def _track_error(self, error_type: str) -> None: ...

        def _track_request(self) -> None: ...

        def _record_budget(self, r: httpx.Response) -> None: ...

        def _headroom(self) -> float | None: ...

        def _tune(self, headroom: float | None) -> None: ...

        def _current_adaptive_delay(self) -> float: ...

        def _apply_retry_after_throttling(self, retry_after_seconds: float) -> None: ...

        def _parse_permission_error(
            self, error: Exception, operation: str, owner: str = "", repo: str = ""
        ) -> PermissionError | None: ...

        async def check_workflow_scope(self) -> bool | None: ...

        async def get_authenticated_user_login(self) -> str | None: ...

        def invalidate_block_reason(
            self, owner: str, repo: str, number: int
        ) -> None: ...

        @staticmethod
        def _ruleset_applies_to_branch(
            conditions: dict[str, Any],
            branch: str,
            default_branch: str | None = None,
        ) -> bool: ...

        async def get_required_status_checks(
            self, owner: str, repo: str, branch: str
        ) -> list[dict[str, Any]]: ...

        async def get_branch_protection(
            self, owner: str, repo: str, branch: str
        ) -> dict[str, Any]: ...

        async def _resolve_default_branch(
            self, owner: str, repo: str
        ) -> str | None: ...

        async def _detect_branch_protection_kind(
            self, owner: str, repo: str, branch: str
        ) -> str: ...
