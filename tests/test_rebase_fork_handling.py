# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""The local-rebase path's fork handling and post-rebase poll.

Both defects sit behind ``--rebase-local``, and the first fails into a
``debug`` log and a ``False`` return, so the tool reported "local rebase
not possible" rather than an error. ``test_local_rebase_signing.py`` and
``test_merge_state_tracking.py`` substitute ``local_rebase_pr``
wholesale, so neither exercised the real workspace preparation.

**#426** --- two defects that compound:

1. Git's shallow state is **repository-wide, not per-remote**, so the
   first ``--unshallow`` completes the repository and a second one is
   fatal (``exit 128``). Every fork pull request therefore failed
   workspace preparation.
2. ``is_fork`` is populated from the head repository's own fork flag,
   which says whether *that repository* is a fork of something --- not
   whether this pull request crosses repositories. Given precedence over
   the two checks that do answer that question, a same-repository PR
   opened inside a forked repository was classified as cross-repository
   and routed into defect 1.

**#437** --- the post-``update_branch`` poll treated ``"unknown"`` and
``""`` as terminal, though every other refresh in the codebase treats
them, with ``None``, as "GitHub is still computing". This poll runs
immediately after ``update_branch``, which is exactly when a recompute
is most likely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from dependamerge.git_ops import GitError
from dependamerge.models import PullRequestInfo
from dependamerge.rebase.local_plan import _build_rebase_plan, _RebasePlan
from dependamerge.rebase.local_workspace import _unshallow_remotes
from dependamerge.rebase.polling import _poll_should_continue

LOG = logging.getLogger("test")


def _git_error(message: str, *, returncode: int = 128) -> GitError:
    """Build a ``GitError`` the way ``git_ops`` does.

    The real constructor takes the command context as keyword-only
    arguments, so tests must supply them rather than raising a bare
    ``GitError(message)``.
    """
    return GitError(
        message,
        args=["git", "fetch"],
        returncode=returncode,
        stdout="",
        stderr=message,
    )


def _pr(
    *,
    is_fork: bool | None = None,
    head_full: str = "lfreleng-actions/some-repo",
    base_full: str = "lfreleng-actions/some-repo",
) -> PullRequestInfo:
    pr = PullRequestInfo(
        number=1,
        title="t",
        body=None,
        author="dependabot[bot]",
        head_sha="a" * 40,
        base_branch="main",
        head_branch="topic",
        state="open",
        mergeable=True,
        mergeable_state="behind",
        behind_by=1,
        files_changed=[],
        repository_full_name=base_full,
        html_url=f"https://github.com/{base_full}/pull/1",
        reviews=[],
        review_comments=[],
    )
    pr.is_fork = is_fork
    pr.head_repo_full_name = head_full
    pr.base_repo_full_name = base_full
    pr.head_repo_clone_url = f"https://github.com/{head_full}.git"
    pr.base_repo_clone_url = f"https://github.com/{base_full}.git"
    return pr


def _plan(is_fork: bool) -> _RebasePlan:
    return _RebasePlan(
        origin_url="https://github.com/fork/some-repo.git",
        upstream_url="https://github.com/lfreleng-actions/some-repo.git",
        head_branch="topic",
        base_branch="main",
        head_full="fork/some-repo",
        base_full="lfreleng-actions/some-repo",
        is_fork=is_fork,
        html_url="https://github.com/lfreleng-actions/some-repo/pull/1",
    )


class _FetchSpy:
    """Records every ``fetch`` call, and reproduces git's own refusal.

    Git exits 128 with *"--unshallow on a complete repository does not
    make sense"* once the repository is complete, verified against a
    real shallow clone. Modelling that here is what makes the test fail
    against the unfixed code rather than merely observing arguments.

    ``completes_repo`` models the divergent case: an ``--unshallow`` of
    one remote does not necessarily complete the *repository*, because
    it only fetches history reachable from that remote. Verified with
    genuinely diverged remotes, where ``origin --unshallow`` leaves the
    repository shallow and a plain ``upstream`` fetch deepens nothing.
    """

    def __init__(self, completes_repo: bool = True) -> None:
        self.calls: list[tuple[str, bool]] = []
        self._complete = False
        self._completes_repo = completes_repo

    def __call__(self, remote: str, *_args: Any, **kwargs: Any) -> None:
        unshallow = bool(kwargs.get("unshallow", False))
        self.calls.append((remote, unshallow))
        if unshallow:
            if self._complete:
                raise _git_error(
                    "fatal: --unshallow on a complete repository does not make sense"
                )
            # Origin's unshallow completes the repository only when the
            # remotes have not diverged past the base fetch depth.
            if remote != "origin" or self._completes_repo:
                self._complete = True

    @property
    def is_shallow(self) -> bool:
        return not self._complete

    @property
    def remotes(self) -> list[str]:
        return [remote for remote, _ in self.calls]


class TestUnshallowHappensOnce:
    """Deepen the repository, without ever unshallowing a complete one.

    Two facts pull in opposite directions, and the fix has to satisfy
    both. Git's shallow state is repository-wide, so a second
    ``--unshallow`` on a complete repository is fatal (exit 128). But
    ``--unshallow`` only completes history reachable from the remote
    fetched, so ``origin --unshallow`` does **not** necessarily complete
    a repository whose ``upstream/<base>`` was fetched at ``depth=50``
    and has since advanced further than that.

    Verified against genuinely diverged remotes: after
    ``origin --unshallow`` the repository is still shallow, a plain
    ``upstream`` fetch deepens nothing, and ``merge-base`` is empty ---
    so the rebase could not proceed.
    """

    def _patch(self, monkeypatch, spy: _FetchSpy) -> None:
        monkeypatch.setattr("dependamerge.rebase.local_workspace.fetch", spy)
        monkeypatch.setattr(
            "dependamerge.rebase.local_workspace._is_shallow",
            lambda **_kw: spy.is_shallow,
        )

    @pytest.mark.asyncio
    async def test_a_fork_pr_can_be_prepared(self, monkeypatch) -> None:
        """The original defect: this failed for *every* fork pull request."""
        spy = _FetchSpy(completes_repo=True)
        self._patch(monkeypatch, spy)

        ok = _unshallow_remotes(
            plan=_plan(is_fork=True),
            workspace=Path("/tmp/ws"),
            token="t",
            log=LOG,
        )

        assert ok is True
        assert spy.remotes == ["origin", "upstream"]
        # Origin completed the repository, so upstream must NOT be
        # unshallowed again --- that is the fatal case.
        assert [u for _, u in spy.calls] == [True, False]

    @pytest.mark.asyncio
    async def test_a_diverged_fork_deepens_upstream_too(self, monkeypatch) -> None:
        """Origin cannot always supply upstream's missing ancestors.

        When the base advanced beyond the depth-50 fetch since the fork
        point, the repository stays shallow after ``origin --unshallow``
        and a plain fetch leaves the rebase without a merge base.
        """
        spy = _FetchSpy(completes_repo=False)
        self._patch(monkeypatch, spy)

        ok = _unshallow_remotes(
            plan=_plan(is_fork=True),
            workspace=Path("/tmp/ws"),
            token="t",
            log=LOG,
        )

        assert ok is True
        assert spy.remotes == ["origin", "upstream"]
        # Still shallow after origin, so upstream *is* unshallowed.
        assert [u for _, u in spy.calls] == [True, True]

    @pytest.mark.asyncio
    async def test_a_same_repo_pr_touches_only_origin(self, monkeypatch) -> None:
        spy = _FetchSpy()
        self._patch(monkeypatch, spy)

        ok = _unshallow_remotes(
            plan=_plan(is_fork=False),
            workspace=Path("/tmp/ws"),
            token="t",
            log=LOG,
        )

        assert ok is True
        assert spy.remotes == ["origin"]

    @pytest.mark.asyncio
    async def test_a_genuine_fetch_failure_is_still_reported(self, monkeypatch) -> None:
        """The tolerant return is preserved for real failures."""

        def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise _git_error("network unreachable")

        monkeypatch.setattr("dependamerge.rebase.local_workspace.fetch", _boom)

        assert (
            _unshallow_remotes(
                plan=_plan(is_fork=True),
                workspace=Path("/tmp/ws"),
                token="t",
                log=LOG,
            )
            is False
        )


class TestForkMeansCrossRepository:
    """``is_fork`` answers a related question, not this one."""

    def test_same_repo_pr_inside_a_fork_is_not_cross_repository(self) -> None:
        """The defect: ``isFork`` is true, but head and base are one repo.

        Not hypothetical for this project --- ``modeseven-lfreleng-actions``
        is itself a fork, so branch-to-branch PRs within it hit this.
        """
        plan = _build_rebase_plan(
            pr_info=_pr(
                is_fork=True,
                head_full="modeseven-lfreleng-actions/dependamerge",
                base_full="modeseven-lfreleng-actions/dependamerge",
            ),
            owner="modeseven-lfreleng-actions",
            repo="dependamerge",
            log=LOG,
            host="github.com",
        )

        assert plan is not None
        assert plan.is_fork is False

    def test_a_genuine_cross_repo_pr_is_a_fork(self) -> None:
        plan = _build_rebase_plan(
            pr_info=_pr(
                is_fork=True,
                head_full="a-contributor/dependamerge",
                base_full="lfreleng-actions/dependamerge",
            ),
            owner="lfreleng-actions",
            repo="dependamerge",
            log=LOG,
            host="github.com",
        )

        assert plan is not None
        assert plan.is_fork is True

    def test_identities_win_over_a_false_flag(self) -> None:
        """Head and base differ, so it is cross-repository regardless."""
        plan = _build_rebase_plan(
            pr_info=_pr(
                is_fork=False,
                head_full="a-contributor/dependamerge",
                base_full="lfreleng-actions/dependamerge",
            ),
            owner="lfreleng-actions",
            repo="dependamerge",
            log=LOG,
            host="github.com",
        )

        assert plan is not None
        assert plan.is_fork is True

    def test_equal_full_names_beat_differing_url_spellings(self) -> None:
        """An identity source must be decisive both ways, not just for "fork".

        The same repository can be spelled two ways --- ``https://`` and
        ``git@`` --- so a URL mismatch is not evidence of a different
        repository when the full names already say otherwise.
        """
        pr = _pr(
            is_fork=None,
            head_full="lfreleng-actions/dependamerge",
            base_full="lfreleng-actions/dependamerge",
        )
        pr.head_repo_clone_url = "git@github.com:lfreleng-actions/dependamerge.git"
        pr.base_repo_clone_url = "https://github.com/lfreleng-actions/dependamerge.git"

        plan = _build_rebase_plan(
            pr_info=pr,
            owner="lfreleng-actions",
            repo="dependamerge",
            log=LOG,
            host="github.com",
        )

        assert plan is not None
        assert plan.is_fork is False

    def test_the_flag_is_never_consulted(self) -> None:
        """Identities are always comparable, so the flag is not needed.

        ``_build_rebase_plan`` fails closed unless the head has a name
        or a URL, ``base_full`` falls back to ``owner/repo``, and both
        clone URLs are synthesised from the names. A pair is therefore
        always available --- so a ``True`` flag cannot override an
        equal-identity verdict, whichever pair is used.
        """
        pr = _pr(
            is_fork=True,
            head_full="modeseven-lfreleng-actions/dependamerge",
            base_full="modeseven-lfreleng-actions/dependamerge",
        )
        pr.head_repo_full_name = None
        pr.base_repo_full_name = None

        plan = _build_rebase_plan(
            pr_info=pr,
            owner="modeseven-lfreleng-actions",
            repo="dependamerge",
            log=LOG,
            host="github.com",
        )

        assert plan is not None
        assert plan.is_fork is False

    def test_a_pr_without_head_identity_is_refused(self) -> None:
        """Fail closed rather than guess and push to the wrong remote."""
        pr = _pr(is_fork=True)
        pr.head_repo_full_name = None
        pr.head_repo_clone_url = None

        assert (
            _build_rebase_plan(
                pr_info=pr,
                owner="lfreleng-actions",
                repo="some-repo",
                log=LOG,
                host="github.com",
            )
            is None
        )


class TestTheRebasePollWaitsOutAComputingState:
    """``null``, ``""`` and ``"unknown"`` all mean "still computing"."""

    def _ctx(self) -> Any:
        ctx = MagicMock()
        ctx.merge_poll_max_attempts = 5
        ctx.log = LOG
        return ctx

    @pytest.mark.parametrize("state", [None, "", "unknown"])
    def test_a_computing_state_keeps_polling(self, state) -> None:
        assert (
            _poll_should_continue(
                ctx=self._ctx(),
                pr_info=_pr(),
                attempt=0,
                mergeable_state=state,
                auto_merge_ok=False,
            )
            is True
        )

    @pytest.mark.parametrize("state", [None, "", "unknown"])
    def test_a_computing_state_still_stops_on_the_last_attempt(self, state) -> None:
        """The budget still bounds the wait; only the classification changed."""
        assert (
            _poll_should_continue(
                ctx=self._ctx(),
                pr_info=_pr(),
                attempt=4,
                mergeable_state=state,
                auto_merge_ok=False,
            )
            is False
        )

    @pytest.mark.parametrize("state", ["clean", "dirty", "draft", "unstable"])
    def test_a_concrete_state_still_ends_the_poll(self, state) -> None:
        assert (
            _poll_should_continue(
                ctx=self._ctx(),
                pr_info=_pr(),
                attempt=0,
                mergeable_state=state,
                auto_merge_ok=False,
            )
            is False
        )
