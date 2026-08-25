# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Resolving a pull request into a concrete local-rebase plan.

:func:`_build_rebase_plan` answers the questions the workspace phase
cannot: which remote is ``origin``, which is ``upstream``, is this a
fork, and which branches are involved.  It fails closed — returning
None rather than guessing — whenever the head repository's identity is
ambiguous, because guessing wrong means force-pushing to somebody
else's repository.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..models import PullRequestInfo


@dataclass(frozen=True)
class _RebasePlan:
    """Everything the workspace phase needs, resolved from the PR.

    Built by :func:`_build_rebase_plan`.  ``html_url`` travels along
    because every log line the workspace phase emits identifies the PR
    by it.
    """

    origin_url: str
    upstream_url: str
    head_branch: str
    base_branch: str
    head_full: str
    base_full: str
    is_fork: bool
    html_url: str


def _build_rebase_plan(
    *,
    pr_info: PullRequestInfo,
    owner: str,
    repo: str,
    log: logging.Logger,
) -> _RebasePlan | None:
    """Resolve remotes, branches and fork status, or None to give up.

    Returning None means the caller must not proceed: either the head
    repository's identity is ambiguous, or the PR has no head branch.
    """
    # We need the head/base clone URLs. They are populated for PRs
    # surfaced by recent versions of the find-similar / merge flows;
    # if missing we synthesise them from the repository names.
    head_clone_url = pr_info.head_repo_clone_url
    base_clone_url = pr_info.base_repo_clone_url
    head_full = pr_info.head_repo_full_name
    base_full = pr_info.base_repo_full_name or f"{owner}/{repo}"

    # Fail closed when head repo identity is ambiguous. Without a
    # confirmed ``head_repo_full_name`` or ``head_repo_clone_url``
    # we cannot tell whether the PR is from a fork. Synthesising a
    # clone URL from the base repo would push to the wrong remote
    # for fork PRs (creating or overwriting a branch on the base
    # repo). The caller falls through to the auto-merge path on
    # False, which is always safe.
    if not head_full and not head_clone_url:
        log.debug(
            "Local rebase: PR %s/%s#%s missing head_repo identity "
            "(head_repo_full_name and head_repo_clone_url are both unset); "
            "failing closed to avoid pushing to the wrong remote.",
            owner,
            repo,
            pr_info.number,
        )
        return None

    # Both fields populated, or one of them — synthesise the
    # missing URL from the known full_name. Safe because
    # ``head_full`` is now confirmed to refer to the head repo,
    # not the base.
    if not head_clone_url:
        head_clone_url = f"https://github.com/{head_full}.git"
    if not base_clone_url:
        base_clone_url = f"https://github.com/{base_full}.git"

    # Decide whether the PR is from a fork *before* we collapse
    # ``head_full`` to ``base_full`` for clone-URL fallback.
    # Treating fork PRs as non-fork would cause us to fetch the
    # base branch from ``origin`` (the fork) instead of
    # ``upstream`` (the canonical base repo) — either failing
    # to fetch (the fork doesn't have the latest base commits)
    # or, worse, fetching stale state and pushing back to the
    # wrong remote.
    #
    # What this needs to know is whether head and base are
    # *different repositories*.  ``pr_info.is_fork`` does not answer
    # that: it is populated from the head repository's own fork flag
    # (``headRepository.isFork`` / ``head.repo.fork``), which says
    # whether that repository is itself a fork of something --- not
    # whether this pull request crosses repositories.  A
    # same-repository PR opened inside a forked repository sets it
    # true, and would then be given a redundant ``upstream`` remote.
    #
    # So consult identities, making the first available pair
    # **decisive in both directions**.  Answering only "different
    # means fork" leaves two gaps: equal full names with differently
    # spelled clone URLs would be called a fork by the URL test, and
    # equal clone URLs with no full names would fall through to the
    # flag --- both misclassifying a same-repository PR.
    #
    # One of these pairs is always available: the function fails closed
    # above unless the head has a name or a URL, ``base_full`` falls
    # back to ``owner/repo``, and both clone URLs are synthesised from
    # the names just above.  ``pr_info.is_fork`` is therefore never
    # consulted, which is the point --- it answers a different
    # question, and this no longer depends on it at all.
    if head_full and base_full:
        is_fork = head_full != base_full
    else:
        is_fork = head_clone_url != base_clone_url

    # Now safe to collapse ``head_full`` for the clone-URL
    # synthesis fallback. ``is_fork`` has already been computed
    # above and won't be misled by this assignment.
    head_full = head_full or base_full

    head_branch = pr_info.head_branch
    base_branch = pr_info.base_branch or "main"
    if not head_branch:
        log.debug(
            "Local rebase: PR %s/%s#%s missing head_branch",
            owner,
            repo,
            pr_info.number,
        )
        return None

    return _RebasePlan(
        origin_url=head_clone_url,
        upstream_url=base_clone_url,
        head_branch=head_branch,
        base_branch=base_branch,
        head_full=head_full,
        base_full=base_full,
        is_fork=is_fork,
        html_url=pr_info.html_url,
    )
