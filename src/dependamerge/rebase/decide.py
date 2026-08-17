# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""The gate that chooses between the local-git and REST rebase paths.

:func:`should_use_local_rebase` is consulted once per PR by
:func:`~dependamerge.rebase.perform_step5_rebase`.  It answers a single
question — would REST ``update-branch`` destroy something we care about
(a verified commit signature)? — and never performs any rebase itself.

The repository coordinates the gate consults branch protection for are
carried by :class:`BaseRef` rather than three loose strings, so the
signature stays inside the project's six-parameter budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..models import PullRequestInfo

if TYPE_CHECKING:
    from ..github_async import GitHubAsync


def authed_clone_url(clone_url: str, token: str) -> str:
    """Deprecated: return ``clone_url`` unchanged.

    Tokens are no longer embedded in clone URLs. Doing so leaked the
    secret into process listings (``ps`` / ``/proc/<pid>/cmdline``)
    during the git invocation and persisted it on disk in the
    workspace's ``.git/config`` (``remote.origin.url``). Authentication
    is now performed via the ``token`` argument of the
    :mod:`dependamerge.git_ops` helpers, which supply the secret through
    a GIT_ASKPASS helper reading from the process environment.

    This shim is retained so third-party callers fail safe (no secret
    injection) rather than break; it will be removed in a future
    release.
    """
    return clone_url


@dataclass(frozen=True)
class BaseRef:
    """The repository and branch a pull request would be rebased onto.

    ``owner``/``repo`` identify the *base* repository (the one whose
    branch protection is consulted), and ``branch`` is the base branch
    itself.  Grouped into one value because they are always read
    together and are meaningless apart.
    """

    owner: str
    repo: str
    branch: str


async def should_use_local_rebase(
    *,
    github_client: GitHubAsync | None,
    pr_info: PullRequestInfo,
    base: BaseRef,
    rebase_local: bool,
    log: logging.Logger,
) -> tuple[bool, str]:
    """Decide whether Step 5 should rebase locally instead of via REST.

    Returns ``(use_local, reason)``.  ``reason`` is a short
    human-readable string suitable for debug logging or a
    user-visible note when ``use_local`` is True.

    The gate activates when ``rebase_local`` is True AND either:

    - the PR is from ``pre-commit-ci[bot]`` (always — that bot has
      no comment macro for recreating a PR with a re-signed
      commit), OR
    - the base branch requires verified signatures AND the current
      PR head commit is itself verified (so REST update-branch
      *would* break verification).

    Strict ``is True`` comparison is used on the
    ``requires_commit_signatures`` return so ``AsyncMock`` defaults
    don't accidentally route real PRs into the local-rebase path
    in tests that haven't explicitly set ``return_value``.  If the
    requirement check raises, we fail safely to the REST path.
    """
    if not rebase_local:
        return False, "--no-rebase-local set"

    # Always use local rebase for pre-commit.ci PRs. The bot has
    # no comment macro to recover from a verification break, so
    # we treat it as opt-in regardless of branch protection.
    if pr_info.author == "pre-commit-ci[bot]":
        return True, "pre-commit-ci[bot] has no recreate/rebase macro"

    if github_client is None:
        return False, "no GitHub client"

    # Branch-protection signature requirement (classic + rulesets)
    try:
        requires_signatures = await github_client.requires_commit_signatures(
            base.owner, base.repo, base.branch
        )
    except Exception as exc:
        log.debug(
            "Could not determine signature requirement for %s/%s:%s: %s",
            base.owner,
            base.repo,
            base.branch,
            exc,
        )
        return False, "signature requirement check failed"

    # Strict ``is True`` rather than truthy check: ``AsyncMock``
    # default returns evaluate as truthy, and we explicitly do
    # not want to enter the network-touching local-rebase path
    # in test mocks that haven't been set up to handle it.
    if requires_signatures is not True:
        return False, "base branch does not require signatures"

    # Base does require verified signatures. Check whether the
    # current PR head is itself verified — if it isn't, REST
    # update-branch can't make things worse, so we don't need
    # the local-rebase machinery.
    try:
        all_verified, _unverified = await github_client.check_pr_commit_signatures(
            base.owner, base.repo, pr_info.number
        )
    except Exception as exc:
        log.debug(
            "Could not check PR commit signatures for %s/%s#%s: %s",
            base.owner,
            base.repo,
            pr_info.number,
            exc,
        )
        # Fail closed: if we can't confirm the PR head is
        # verified, route to the REST path. The opposite
        # (assuming verification and using the local path)
        # would mean transient API failures could trigger
        # network-touching local clones, and would conflict
        # with the documented gate ("base requires signatures
        # AND PR head is verified"). When verification isn't
        # established, REST update-branch can't make things
        # any worse than they already are.
        return False, "signature check failed"

    if all_verified:
        return True, "base requires signatures and PR head is verified"
    return False, "PR head is not currently verified"
