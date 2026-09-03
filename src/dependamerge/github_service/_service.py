# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""The concrete ``GitHubService`` class.

Construction, the shared-client callback plumbing and the rate-limit
callbacks.  Its remaining methods live in the sibling ``_Xxx`` mixin
modules and are assembled here.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from ..github_async import GitHubAsync
from ..url_parser import default_github_host, derive_api_urls
from ._constants import (
    DEFAULT_COMMENTS_PAGE_SIZE,
    DEFAULT_CONTEXTS_PAGE_SIZE,
    DEFAULT_FILES_PAGE_SIZE,
    DEFAULT_PRS_PAGE_SIZE,
)
from ._fetch import _FetchMixin
from ._helpers import _chain_callbacks, _unchain_callback
from ._merge_policy import _MergePolicyMixin
from ._pr_extract import _PullRequestExtractMixin
from ._pr_nodes import _PullRequestNodeMixin
from ._repositories import _RepositoriesMixin
from ._scan import _ScanMixin
from ._similar import _SimilarPullRequestsMixin
from ._status import _StatusMixin
from ._status_stats import _StatusStatsMixin


class GitHubService(
    _RepositoriesMixin,
    _ScanMixin,
    _PullRequestNodeMixin,
    _PullRequestExtractMixin,
    _SimilarPullRequestsMixin,
    _FetchMixin,
    _MergePolicyMixin,
    _StatusMixin,
    _StatusStatsMixin,
):
    """
    Asynchronous service orchestrating GraphQL paging and mapping results
    into the project's existing Pydantic models. Designed to be used by a thin
    adapter so the rest of the codebase can keep a stable interface.

    This service:
      - Paginates organization repositories and their open PRs via GraphQL
      - Extracts status rollups, file changes, and Copilot comments
      - Detects common unmergeable reasons
      - Provides helpers to convert GraphQL PR nodes to PullRequestInfo
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        host: str | None = None,
        progress_tracker: Any | None = None,
        max_repo_tasks: int = 8,
        max_page_tasks: int = 16,
        debug_matching: bool = False,
        client: GitHubAsync | None = None,
    ) -> None:
        """
        Args:
            token: GitHub token; if None, reads from env GITHUB_TOKEN.
            host: The GitHub host to address.  Ignored when ``client``
                is supplied, since that client already carries its own
                base URLs.
            progress_tracker: Optional ProgressTracker-compatible instance.
            max_repo_tasks: Max concurrent repository scans to schedule at once.
            debug_matching: Enable detailed debugging output for PR matching.
            client: An existing client to share.  Rate limiting,
                concurrency and adaptive throttling are per-instance, so a
                second client doubles the effective ceiling against a
                budget that is shared server-side and keeps each half
                blind to the pressure the other is causing.  Callers that
                already hold a client should pass it; the service then
                does not own its lifecycle and will not close it.
        """
        self._owns_api = client is None
        self._callbacks_attached = False
        if client is None:
            # Resolved only when this service builds its own transport.
            # Doing it unconditionally makes a shared-client caller ---
            # whose endpoints are already fixed --- fail on an
            # unrelated GitHub host misconfiguration it never uses.
            self.host = (host or default_github_host()).strip().lower()
            api_url, graphql_url = derive_api_urls(self.host)
            self._api = GitHubAsync(
                token=token,
                api_url=api_url,
                graphql_url=graphql_url,
                on_rate_limited=self._on_rate_limited,
                on_rate_limit_cleared=self._on_rate_limit_cleared,
                on_metrics=self._on_metrics,
            )
        else:
            # The shared client already carries its base URLs, so the
            # ``host`` argument is documented as ignored here.
            self.host = (host or "").strip().lower()
            self._api = client
        if client is not None:
            # A shared client arrives with whatever callbacks its owner
            # registered, and this service's own must still fire: without
            # ``_on_rate_limited`` the ``_rate_limited`` flag never sets,
            # and the GraphQL paging below silently stops shrinking its
            # page sizes under rate-limit pressure.  Chain rather than
            # replace so the owner's callbacks keep working too.
            self._attach_callbacks(client)
        self._progress = progress_tracker
        self._max_repo_tasks = max_repo_tasks
        self._max_page_tasks = max_page_tasks
        self._repo_semaphore = asyncio.Semaphore(self._max_repo_tasks)
        self._page_semaphore = asyncio.Semaphore(self._max_page_tasks)
        # Rate limit awareness
        self._rate_limited = False
        self._debug_matching = debug_matching
        # Cache for branch protection settings to avoid repeated API calls
        self._branch_protection_cache: dict[str, dict[str, Any] | None] = {}
        # Cache for resolved owner account type (organization vs user),
        # keyed by owner login.  Value is a ``(root_key, query)`` tuple so
        # repeated repository-pagination pages do not re-probe.
        self._owner_root_cache: dict[str, tuple[str, str]] = {}
        self.log = logging.getLogger("dependamerge.github_service")

    def _attach_callbacks(self, client: GitHubAsync) -> None:
        """Add this service's rate-limit callbacks to a shared client."""
        client.on_rate_limited = _chain_callbacks(
            client.on_rate_limited, self._on_rate_limited
        )
        client.on_rate_limit_cleared = _chain_callbacks(
            client.on_rate_limit_cleared, self._on_rate_limit_cleared
        )
        client.on_metrics = _chain_callbacks(client.on_metrics, self._on_metrics)
        self._callbacks_attached = True

    def _detach_callbacks(self) -> None:
        """Take this service's callbacks back off a borrowed client.

        Leaving them attached would keep a closed service alive and
        receiving events, and attaching a replacement service to the same
        client would stack a second copy, duplicating every rate-limit
        and progress update.

        Only this service's own links are removed.  Anything registered
        afterwards --- including a second service sharing the client ---
        keeps working.
        """
        if not self._callbacks_attached:
            return
        self._api.on_rate_limited = _unchain_callback(
            self._api.on_rate_limited, self._on_rate_limited
        )
        self._api.on_rate_limit_cleared = _unchain_callback(
            self._api.on_rate_limit_cleared, self._on_rate_limit_cleared
        )
        self._api.on_metrics = _unchain_callback(self._api.on_metrics, self._on_metrics)
        self._callbacks_attached = False

    async def close(self) -> None:
        # Only close what this service created: a shared client outlives
        # it and is closed by whoever owns it.  Its callbacks, however,
        # belong to this service and must come off either way.
        self._detach_callbacks()
        if self._owns_api:
            await self._api.aclose()

    async def _on_rate_limited(self, reset_epoch: float) -> None:
        # Mark rate-limited and report current tuning metrics
        self._rate_limited = True
        if self._progress:
            try:
                reset_time = datetime.fromtimestamp(reset_epoch)
                self._progress.set_rate_limited(reset_time)
                # Report current tuning metrics for visibility
                self._progress.update_operation(
                    f"Tuning: prs={DEFAULT_PRS_PAGE_SIZE} files={DEFAULT_FILES_PAGE_SIZE} comments={DEFAULT_COMMENTS_PAGE_SIZE} contexts={DEFAULT_CONTEXTS_PAGE_SIZE}"
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Progress display is best-effort; ignore UI errors.
                self.log.debug(
                    f"Progress update failed (rate-limited): {exc}",
                    exc_info=True,
                )

    async def _on_rate_limit_cleared(self) -> None:
        # Clear rate-limited flag and report current tuning metrics
        self._rate_limited = False
        if not self._progress:
            return
        try:
            self._progress.clear_rate_limited()
            self._progress.update_operation(
                f"Tuning: prs={DEFAULT_PRS_PAGE_SIZE} files={DEFAULT_FILES_PAGE_SIZE} comments={DEFAULT_COMMENTS_PAGE_SIZE} contexts={DEFAULT_CONTEXTS_PAGE_SIZE}"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Progress display is best-effort; ignore UI errors.
            self.log.debug(
                f"Progress update failed (rate-limit cleared): {exc}",
                exc_info=True,
            )

    async def _on_metrics(self, concurrency: int, rps: float) -> None:
        """Receive current concurrency and RPS from the async client and push to progress display."""
        if not self._progress:
            return
        try:
            # Round RPS to a single decimal for display, actual value passed through
            self._progress.update_metrics(concurrency, rps)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Metrics are best-effort; ignore UI errors
            self.log.debug(f"Progress metrics update failed: {exc}", exc_info=True)
