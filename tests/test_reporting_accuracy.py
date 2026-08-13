# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for reporting accuracy on the merge path.

The 503-PR run analysed in ``docs/BULK_RUN_PERFORMANCE_AUDIT.md``
reported 34 failures, of which **21 had actually merged** --- most
within two minutes of being reported.  Three distinct causes are
covered here:

- a terminal outcome was never re-checked before being reported;
- ``POST .../reviews`` returned a transient 500 that was not retried,
  even though the review had frequently been created;
- ``405 Merge already in progress`` was treated as terminal after a
  backoff far shorter than GitHub takes to finish the merge.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from dependamerge.github_async import GitHubAsync
from dependamerge.merge_manager import (
    MergeResult,
    MergeStatus,
    _merge_already_in_progress,
    _merged_from_payload,
)
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

REPO = "lfreleng-actions/example-action"
OWNER, NAME = REPO.split("/")


def _make_pr(number: int = 101, state: str = "open") -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
        title="Chore: Bump something",
        body="bump",
        author="dependabot[bot]",
        head_sha="c0ffee11" * 5,
        base_branch="main",
        head_branch="dependabot/x",
        state=state,
        mergeable=True,
        mergeable_state="clean",
        behind_by=None,
        files_changed=[],
        repository_full_name=REPO,
        html_url=f"https://github.com/{REPO}/pull/{number}",
        reviews=[],
        review_comments=[],
    )


# --------------------------------------------------------------------------
# Re-verifying a reported failure
# --------------------------------------------------------------------------


class TestConfirmFailure:
    @pytest.mark.asyncio
    async def test_merged_pr_is_corrected_to_merged(self) -> None:
        """The headline case: 21 of 34 reported failures had merged."""
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(
            return_value={
                "merged": True,
                "merged_at": "2026-08-13T15:17:37Z",
                "state": "closed",
            }
        )
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, error="required checks not satisfied"
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.status == MergeStatus.MERGED
        assert out.error is None
        # The stale reason is preserved as context, not as an error.
        assert out.warning is not None
        assert "required checks not satisfied" in out.warning

    @pytest.mark.asyncio
    async def test_genuinely_open_failure_is_left_alone(self) -> None:
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(return_value={"merged": False, "state": "open"})
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="boom")

        out = await mgr._confirm_failure(pr, result)

        assert out.status == MergeStatus.FAILED
        assert out.error == "boom"

    @pytest.mark.asyncio
    async def test_closed_unmerged_pr_becomes_closed(self) -> None:
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(
            return_value={"merged": False, "merged_at": None, "state": "closed"}
        )
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="boom")

        out = await mgr._confirm_failure(pr, result)

        assert out.status == MergeStatus.CLOSED

    @pytest.mark.asyncio
    async def test_closed_with_unknown_merged_state_stays_failed(self) -> None:
        """Unknown must never be read as "did not merge".

        A trimmed payload carrying neither ``merged`` nor ``merged_at``
        cannot tell us the PR failed to merge, so reclassifying it as
        CLOSED would assert something the data never said.
        """
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(return_value={"state": "closed"})
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="boom")

        out = await mgr._confirm_failure(pr, result)

        assert out.status == MergeStatus.FAILED
        assert out.error == "boom"

    @pytest.mark.asyncio
    async def test_non_failed_results_are_not_rechecked(self) -> None:
        """Verification must cost nothing for the overwhelmingly common path."""
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock()
        result = MergeResult(pr_info=pr, status=MergeStatus.MERGED)

        out = await mgr._confirm_failure(pr, result)

        assert out.status == MergeStatus.MERGED
        client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_preview_mode_does_not_call_the_api(self) -> None:
        mgr, client = make_merge_manager(preview_mode=True)
        pr = _make_pr()
        client.get = AsyncMock()
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="boom")

        out = await mgr._confirm_failure(pr, result)

        assert out.status == MergeStatus.FAILED
        client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_permission_failed_repo_is_not_rechecked(self) -> None:
        """One bad repo must not cost an API call per remaining PR."""
        mgr, client = make_merge_manager()
        pr = _make_pr()
        mgr._permission_failed_repos.add(REPO)
        client.get = AsyncMock()
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, error="token lacks permissions"
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.status == MergeStatus.FAILED
        client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_api_error_leaves_result_untouched(self) -> None:
        """Verification must never turn a reportable failure into a crash."""
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(side_effect=RuntimeError("network gone"))
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="boom")

        out = await mgr._confirm_failure(pr, result)

        assert out.status == MergeStatus.FAILED
        assert out.error == "boom"

    @pytest.mark.asyncio
    async def test_wrapper_applies_verification_to_early_returns(self) -> None:
        """The wrapper exists because the impl has many early returns."""
        mgr, client = make_merge_manager()
        pr = _make_pr()
        mgr._merge_single_pr_impl = AsyncMock(  # type: ignore[method-assign]
            return_value=MergeResult(
                pr_info=pr, status=MergeStatus.FAILED, error="stale"
            )
        )
        client.get = AsyncMock(return_value={"merged": True, "state": "closed"})

        out = await mgr._merge_single_pr(pr)

        assert out.status == MergeStatus.MERGED


# --------------------------------------------------------------------------
# "Merge already in progress"
# --------------------------------------------------------------------------


class TestMergeAlreadyInProgressPredicate:
    @pytest.mark.parametrize(
        "msg",
        [
            "Merge already in progress",
            "405 Method Not Allowed ... Merge already in progress",
            "merge already in progress.",
        ],
    )
    def test_matches(self, msg: str) -> None:
        assert _merge_already_in_progress(msg)

    @pytest.mark.parametrize(
        "msg",
        [
            "Base branch was modified",
            "Pull Request is not mergeable",
            "Required status checks not satisfied",
            "",
        ],
    )
    def test_does_not_match(self, msg: str) -> None:
        assert not _merge_already_in_progress(msg)


class TestAwaitInProgressMerge:
    @pytest.mark.asyncio
    async def test_returns_true_once_merged(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_POLL_SECONDS", 0.001
        )
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_FIRST_POLL_SECONDS", 0.001
        )
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(
            side_effect=[
                {"merged": False, "state": "open"},
                {"merged": True, "state": "closed"},
            ]
        )

        assert await mgr._await_in_progress_merge(
            OWNER, NAME, pr, f"{REPO}#{pr.number}"
        )
        assert pr.state == "closed"

    @pytest.mark.asyncio
    async def test_returns_false_when_closed_unmerged(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_POLL_SECONDS", 0.001
        )
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_FIRST_POLL_SECONDS", 0.001
        )
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(
            return_value={"merged": False, "merged_at": None, "state": "closed"}
        )

        assert not await mgr._await_in_progress_merge(
            OWNER, NAME, pr, f"{REPO}#{pr.number}"
        )

    @pytest.mark.asyncio
    async def test_unknown_merged_state_keeps_polling(self, monkeypatch) -> None:
        """An ambiguous payload must not end the watch early.

        The merge may still be completing; giving up here would
        reinstate exactly the false failure this wait exists to prevent.
        """
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_POLL_SECONDS", 0.001
        )
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_FIRST_POLL_SECONDS", 0.001
        )
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(
            side_effect=[
                {"state": "closed"},  # unknown merged-ness
                {"state": "closed", "merged_at": "2026-08-13T15:17:37Z"},
            ]
        )

        assert await mgr._await_in_progress_merge(
            OWNER, NAME, pr, f"{REPO}#{pr.number}"
        )
        assert client.get.await_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_at_the_timeout(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_POLL_SECONDS", 0.001
        )
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_FIRST_POLL_SECONDS", 0.001
        )
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_TIMEOUT_SECONDS", 0.02
        )
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(
            return_value={"merged": False, "merged_at": None, "state": "open"}
        )

        assert not await mgr._await_in_progress_merge(
            OWNER, NAME, pr, f"{REPO}#{pr.number}"
        )

    @pytest.mark.asyncio
    async def test_always_polls_at_least_once(self, monkeypatch) -> None:
        """An already-elapsed deadline must still get one confirmation.

        The run-wide ``max_wait`` clamps this deadline, so it can already
        be in the past on entry.  Returning without a single GET would
        report the false failure this watch exists to prevent.
        """
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_FIRST_POLL_SECONDS", 0.001
        )
        mgr, client = make_merge_manager()
        mgr._run_deadline = asyncio.get_running_loop().time() - 5.0
        pr = _make_pr()
        client.get = AsyncMock(
            return_value={"state": "closed", "merged_at": "2026-08-13T15:17:37Z"}
        )

        assert await mgr._await_in_progress_merge(
            OWNER, NAME, pr, f"{REPO}#{pr.number}"
        )
        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_expired_deadline_polls_exactly_once(self, monkeypatch) -> None:
        """ "At least once" must not become "twice" when no time remains."""
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_FIRST_POLL_SECONDS", 0.001
        )
        mgr, client = make_merge_manager()
        mgr._run_deadline = asyncio.get_running_loop().time() - 5.0
        pr = _make_pr()
        # Still open: the loop has no definitive answer and would spin
        # again if the deadline check were skipped on the first pass.
        client.get = AsyncMock(
            return_value={"merged": False, "merged_at": None, "state": "open"}
        )

        assert not await mgr._await_in_progress_merge(
            OWNER, NAME, pr, f"{REPO}#{pr.number}"
        )
        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_no_wait_mode_returns_immediately(self) -> None:
        mgr, client = make_merge_manager()
        mgr._no_wait = True
        pr = _make_pr()
        client.get = AsyncMock()

        assert not await mgr._await_in_progress_merge(
            OWNER, NAME, pr, f"{REPO}#{pr.number}"
        )
        client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_waiting_registration_is_always_cleaned_up(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_POLL_SECONDS", 0.001
        )
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_FIRST_POLL_SECONDS", 0.001
        )
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(return_value={"merged": True, "state": "closed"})

        await mgr._await_in_progress_merge(OWNER, NAME, pr, f"{REPO}#{pr.number}")

        assert mgr._waiting_prs == {}


# --------------------------------------------------------------------------
# Merged-ness derivation
# --------------------------------------------------------------------------


class TestMergedFromPayload:
    def test_prefers_explicit_boolean(self) -> None:
        assert _merged_from_payload({"merged": True, "merged_at": None}) is True
        assert _merged_from_payload({"merged": False, "merged_at": "x"}) is False

    def test_falls_back_to_merged_at(self) -> None:
        """A trimmed payload may omit the boolean but keep the timestamp."""
        assert _merged_from_payload({"merged_at": "2026-08-13T15:17:37Z"}) is True
        assert _merged_from_payload({"merged_at": None}) is False

    def test_unknown_when_neither_usable(self) -> None:
        assert _merged_from_payload({}) is None
        assert _merged_from_payload({"state": "closed"}) is None

    def test_unknown_on_unexpected_merged_at_type(self) -> None:
        assert _merged_from_payload({"merged_at": 12345}) is None


class TestConfirmFailureUsesSharedDerivation:
    @pytest.mark.asyncio
    async def test_detects_merge_from_merged_at_alone(self) -> None:
        """A payload without the ``merged`` bool must not read as 'not merged'."""
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(
            return_value={"state": "closed", "merged_at": "2026-08-13T15:17:37Z"}
        )
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="stale")

        out = await mgr._confirm_failure(pr, result)

        assert out.status == MergeStatus.MERGED

    @pytest.mark.asyncio
    async def test_ambiguous_payload_does_not_invent_a_merge(self) -> None:
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(return_value={"state": "open"})
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="stale")

        out = await mgr._confirm_failure(pr, result)

        assert out.status == MergeStatus.FAILED

    @pytest.mark.asyncio
    async def test_in_progress_wait_detects_merge_from_merged_at(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_POLL_SECONDS", 0.001
        )
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_FIRST_POLL_SECONDS", 0.001
        )
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(
            return_value={"state": "closed", "merged_at": "2026-08-13T15:17:37Z"}
        )

        assert await mgr._await_in_progress_merge(
            OWNER, NAME, pr, f"{REPO}#{pr.number}"
        )


class TestCancellationPropagates:
    @pytest.mark.asyncio
    async def test_confirm_failure_does_not_swallow_cancellation(self) -> None:
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(side_effect=asyncio.CancelledError())
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="boom")

        with pytest.raises(asyncio.CancelledError):
            await mgr._confirm_failure(pr, result)

    @pytest.mark.asyncio
    async def test_in_progress_wait_does_not_swallow_cancellation(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_POLL_SECONDS", 0.001
        )
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_FIRST_POLL_SECONDS", 0.001
        )
        mgr, client = make_merge_manager()
        pr = _make_pr()
        client.get = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await mgr._await_in_progress_merge(OWNER, NAME, pr, f"{REPO}#{pr.number}")
        # The waiting registry must still be cleaned up on cancellation.
        assert mgr._waiting_prs == {}


# --------------------------------------------------------------------------
# Approve retry on transient 500
# --------------------------------------------------------------------------


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.github.com/x")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


class TestApproveRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_time_without_extra_calls(self) -> None:
        c = GitHubAsync(token="t")
        c.post = AsyncMock(return_value={})  # type: ignore[method-assign]
        c.get = AsyncMock()  # type: ignore[method-assign]

        await c.approve_pull_request("o", "r", 1, "lgtm")

        c.post.assert_awaited_once()
        c.get.assert_not_called()
        await c.aclose()

    @pytest.mark.asyncio
    async def test_retries_transient_500(self, monkeypatch) -> None:
        """Regression: 500 is absent from ``_is_retryable_status``."""
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        c = GitHubAsync(token="t")
        c.post = AsyncMock(side_effect=[_http_error(500), {}])  # type: ignore[method-assign]
        c._has_own_approval = AsyncMock(return_value=False)  # type: ignore[method-assign]

        await c.approve_pull_request("o", "r", 1, "lgtm")

        assert c.post.await_count == 2
        await c.aclose()

    @pytest.mark.asyncio
    async def test_does_not_duplicate_an_approval_that_landed(
        self, monkeypatch
    ) -> None:
        """A 500 does not mean the review was not created."""
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        c = GitHubAsync(token="t")
        c.post = AsyncMock(side_effect=_http_error(500))  # type: ignore[method-assign]
        c._has_own_approval = AsyncMock(return_value=True)  # type: ignore[method-assign]

        await c.approve_pull_request("o", "r", 1, "lgtm")

        # One attempt only; the retry saw the approval already present.
        assert c.post.await_count == 1
        await c.aclose()

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_attempts(self, monkeypatch) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        c = GitHubAsync(token="t")
        c.post = AsyncMock(side_effect=_http_error(500))  # type: ignore[method-assign]
        c._has_own_approval = AsyncMock(return_value=False)  # type: ignore[method-assign]

        with pytest.raises(httpx.HTTPStatusError):
            await c.approve_pull_request("o", "r", 1, "lgtm")
        await c.aclose()

    @pytest.mark.asyncio
    async def test_does_not_retry_client_errors(self) -> None:
        """422 and friends are deterministic; retrying them wastes budget."""
        c = GitHubAsync(token="t")
        c.post = AsyncMock(side_effect=_http_error(422))  # type: ignore[method-assign]
        c._has_own_approval = AsyncMock(return_value=False)  # type: ignore[method-assign]

        with pytest.raises(httpx.HTTPStatusError):
            await c.approve_pull_request("o", "r", 1, "lgtm")
        assert c.post.await_count == 1
        await c.aclose()

    @pytest.mark.parametrize("status", [429, 502, 503, 504])
    @pytest.mark.asyncio
    async def test_does_not_double_retry_inner_statuses(self, status: int) -> None:
        """Statuses ``_request`` already retries must not be retried again.

        ``_request`` gives 429/502/503/504 six tenacity attempts. Retrying
        them out here as well would nest the loops --- up to 18 requests
        and two sets of backoff sleeps for one approval.
        """
        c = GitHubAsync(token="t")
        c.post = AsyncMock(side_effect=_http_error(status))  # type: ignore[method-assign]
        c._has_own_approval = AsyncMock(return_value=False)  # type: ignore[method-assign]

        with pytest.raises(httpx.HTTPStatusError):
            await c.approve_pull_request("o", "r", 1, "lgtm")
        assert c.post.await_count == 1
        await c.aclose()

    @pytest.mark.asyncio
    async def test_final_check_rescues_a_landed_approval(self, monkeypatch) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        c = GitHubAsync(token="t")
        c.post = AsyncMock(side_effect=_http_error(500))  # type: ignore[method-assign]
        # Not present on the pre-attempt checks, but present at the end:
        # the last POST created it despite reporting failure.
        c._has_own_approval = AsyncMock(side_effect=[False, False, True])  # type: ignore[method-assign]

        await c.approve_pull_request("o", "r", 1, "lgtm")
        await c.aclose()


class TestHasOwnApproval:
    @staticmethod
    def _pages(*pages: list[dict]) -> object:
        async def _gen(*args: object, **kwargs: object):
            for page in pages:
                yield page

        return _gen

    @pytest.mark.asyncio
    async def test_true_when_our_approval_present(self) -> None:
        c = GitHubAsync(token="t")
        c.get_authenticated_user_login = AsyncMock(return_value="me")  # type: ignore[method-assign]
        c.get_paginated = self._pages(  # type: ignore[method-assign,assignment]
            [
                {"state": "COMMENTED", "user": {"login": "me"}},
                {"state": "APPROVED", "user": {"login": "someone-else"}},
                {"state": "APPROVED", "user": {"login": "me"}},
            ]
        )
        assert await c._has_own_approval("o", "r", 1)
        await c.aclose()

    @pytest.mark.asyncio
    async def test_finds_approval_beyond_the_first_page(self) -> None:
        """A busy PR must not defeat duplicate suppression."""
        c = GitHubAsync(token="t")
        c.get_authenticated_user_login = AsyncMock(return_value="me")  # type: ignore[method-assign]
        c.get_paginated = self._pages(  # type: ignore[method-assign,assignment]
            [{"state": "COMMENTED", "user": {"login": "bot"}}] * 100,
            [{"state": "APPROVED", "user": {"login": "me"}}],
        )
        assert await c._has_own_approval("o", "r", 1)
        await c.aclose()

    @pytest.mark.asyncio
    async def test_false_when_only_others_approved(self) -> None:
        c = GitHubAsync(token="t")
        c.get_authenticated_user_login = AsyncMock(return_value="me")  # type: ignore[method-assign]
        c.get_paginated = self._pages(  # type: ignore[method-assign,assignment]
            [{"state": "APPROVED", "user": {"login": "other"}}]
        )
        assert not await c._has_own_approval("o", "r", 1)
        await c.aclose()

    @pytest.mark.asyncio
    async def test_false_on_api_error(self) -> None:
        c = GitHubAsync(token="t")
        c.get_authenticated_user_login = AsyncMock(side_effect=RuntimeError("nope"))  # type: ignore[method-assign]
        assert not await c._has_own_approval("o", "r", 1)
        await c.aclose()

    @pytest.mark.asyncio
    async def test_false_when_authenticated_login_unknown(self) -> None:
        """An unknown login must never match a null reviewer.

        ``get_authenticated_user_login`` returns ``None`` on failure
        rather than raising.  A review with ``user: null`` would then
        compare ``None == None`` and report an approval that does not
        exist, stopping the retry and reporting success having approved
        nothing.
        """
        c = GitHubAsync(token="t")
        c.get_authenticated_user_login = AsyncMock(return_value=None)  # type: ignore[method-assign]
        c.get_paginated = self._pages(  # type: ignore[method-assign,assignment]
            [{"state": "APPROVED", "user": None}]
        )
        assert not await c._has_own_approval("o", "r", 1)
        await c.aclose()

    @pytest.mark.asyncio
    async def test_null_reviewer_never_matches(self) -> None:
        c = GitHubAsync(token="t")
        c.get_authenticated_user_login = AsyncMock(return_value="me")  # type: ignore[method-assign]
        c.get_paginated = self._pages(  # type: ignore[method-assign,assignment]
            [
                {"state": "APPROVED", "user": None},
                {"state": "APPROVED", "user": {"login": None}},
            ]
        )
        assert not await c._has_own_approval("o", "r", 1)
        await c.aclose()
