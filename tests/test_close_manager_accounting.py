# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Close-command accounting and permission diagnostics.

Two defects lived on the close manager's failure paths, and both went
unnoticed because nothing asserted on the length of ``_results`` or on
the permission message.

1. Every skipped PR was recorded **twice**.  The early-return branches
   appended to ``_results`` and then returned from inside a ``try``
   whose ``finally`` appended the same object again --- and ``return``
   inside ``try`` runs ``finally`` first.  ``get_summary`` and
   ``get_results`` both read that list, so a run skipping ten
   already-closed PRs reported twenty outcomes.

2. A permission denial never reached the handler written for it.  The
   retry loop's inner ``except Exception`` is broader than the dedicated
   ``except GitHubPermissionError`` that follows it, so it caught the
   denial first --- retrying pointlessly, since a token cannot gain
   scopes between attempts, and discarding the guidance that names the
   missing scope.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dependamerge.close_manager import AsyncCloseManager, CloseStatus
from dependamerge.github_async import PermissionError as GitHubPermissionError
from dependamerge.models import PullRequestInfo

REPO = "lfreleng-actions/some-repo"


def _pr(
    *,
    state: str = "open",
    mergeable_state: str = "clean",
    repository_full_name: str = REPO,
) -> PullRequestInfo:
    return PullRequestInfo(
        number=1,
        title="t",
        body=None,
        author="dependabot[bot]",
        head_sha="a" * 40,
        base_branch="main",
        head_branch="x",
        state=state,
        mergeable=True,
        mergeable_state=mergeable_state,
        behind_by=None,
        files_changed=[],
        repository_full_name=repository_full_name,
        html_url=f"https://github.com/{repository_full_name}/pull/1",
        reviews=[],
        review_comments=[],
    )


def _mgr(**overrides) -> tuple[AsyncCloseManager, MagicMock]:
    """Build a manager with a typed reference to its stubbed console.

    Returns ``(manager, console)``. The console reference is typed as
    ``MagicMock`` --- the declared attribute is a real ``Console``, so
    reading ``print.call_args_list`` off it would not type-check ---
    mirroring the ``(manager, client)`` pattern in ``tests/conftest.py``.
    """
    mgr = AsyncCloseManager(token="test-token", **overrides)
    console = MagicMock()
    # Keep the console quiet and out of the assertions.
    mgr._console = console
    return mgr, console


class TestEachOutcomeIsRecordedOnce:
    """``_results`` backs the reported counts, so duplicates inflate them."""

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"state": "closed"}, CloseStatus.SKIPPED),
            ({"mergeable_state": "draft"}, CloseStatus.SKIPPED),
            ({"repository_full_name": "malformed"}, CloseStatus.FAILED),
        ],
        ids=["already-closed", "draft", "malformed-repo"],
    )
    @pytest.mark.asyncio
    async def test_a_skipped_pr_is_recorded_once(self, kwargs, expected) -> None:
        mgr, console = _mgr()

        result = await mgr._close_single_pr(_pr(**kwargs))

        assert len(mgr._results) == 1
        assert mgr._results[0] is result
        assert result.status is expected
        assert mgr.get_summary()["total"] == 1

    @pytest.mark.asyncio
    async def test_a_closed_pr_is_recorded_once(self) -> None:
        """The path that already appended exactly once must be unchanged."""
        mgr, console = _mgr()
        client = AsyncMock()
        client.close_pull_request = AsyncMock(return_value=None)
        mgr._github_client = client

        result = await mgr._close_single_pr(_pr())

        assert len(mgr._results) == 1
        assert result.status is CloseStatus.CLOSED

    @pytest.mark.asyncio
    async def test_a_preview_is_recorded_once(self) -> None:
        mgr, console = _mgr(preview_mode=True)

        result = await mgr._close_single_pr(_pr())

        assert len(mgr._results) == 1
        assert result.status is CloseStatus.CLOSED


class TestPermissionErrorsReachTheirHandler:
    """A denial is actionable only if the scope-specific guidance survives."""

    @pytest.mark.asyncio
    async def test_a_denial_is_not_retried(self) -> None:
        """Retrying cannot help: the token gains no scopes between attempts."""
        mgr, console = _mgr(max_retries=3)
        client = AsyncMock()
        client.close_pull_request = AsyncMock(
            side_effect=GitHubPermissionError(
                operation="close_pull_request",
                message="Token lacks the required scope",
            )
        )
        mgr._github_client = client

        result = await mgr._close_single_pr(_pr())

        assert client.close_pull_request.await_count == 1
        assert result.status is CloseStatus.FAILED

    @pytest.mark.asyncio
    async def test_a_denial_keeps_its_guidance(self) -> None:
        """The dedicated handler runs, so the operation is named for the user."""
        mgr, console = _mgr()
        client = AsyncMock()
        client.close_pull_request = AsyncMock(
            side_effect=GitHubPermissionError(
                operation="close_pull_request",
                message="Token lacks the required scope",
            )
        )
        mgr._github_client = client

        result = await mgr._close_single_pr(_pr())

        printed = " ".join(
            str(call.args[0]) for call in console.print.call_args_list if call.args
        )
        assert "Token Permission Issue" in printed
        assert result.error and "scope" in result.error.lower()

    @pytest.mark.asyncio
    async def test_an_ordinary_error_is_still_retried(self) -> None:
        """Narrowing must not disturb the retry behaviour it sits beside."""
        mgr, console = _mgr(max_retries=2)
        client = AsyncMock()
        client.close_pull_request = AsyncMock(side_effect=RuntimeError("flaky"))
        mgr._github_client = client

        import dependamerge.close_manager as mod

        original = mod.asyncio.sleep
        mod.asyncio.sleep = AsyncMock(return_value=None)  # type: ignore[assignment]
        try:
            result = await mgr._close_single_pr(_pr())
        finally:
            mod.asyncio.sleep = original  # type: ignore[assignment]

        assert client.close_pull_request.await_count == 2
        assert result.status is CloseStatus.FAILED
        assert len(mgr._results) == 1
