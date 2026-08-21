# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Merge ordering and the grouped PR listing.

Within a repository PRs drain oldest-first; owner-wide, repositories with
the most PRs come first so the striped scheduler always has work spread
across repositories.
"""

from ..models import PullRequestInfo
from ._app import console


def _repo_merge_order(
    prs: list[PullRequestInfo],
) -> list[PullRequestInfo]:
    """Order a single repository's PRs for oldest-first merging.

    Sorts ascending by PR number, i.e. in the order the automation raised
    them (oldest first).  Merging the oldest PR first minimises the rebase
    churn imposed on the newer siblings: each merge advances the base
    branch, so a newer sibling merged ahead of an older one would leave
    the older PR ``behind`` and trigger an avoidable rebase + CI wait.

    This is the single-repository analogue of the within-repository key
    used owner-wide by :func:`_owner_merge_order`, keeping both schemes'
    intra-repository sequencing identical.
    """
    return sorted(prs, key=lambda p: p.number)


def _owner_merge_order(
    prs: list[PullRequestInfo],
) -> list[PullRequestInfo]:
    """Order owner-wide PRs for striped merging.

    Sorts so that:

    - Repositories with the most in-scope PRs come first.  These take the
      longest to drain (each merge can make the next sibling ``behind`` /
      ``dirty`` and trigger a rebase + CI wait), so starting them earliest
      gives them the most wall-clock head start under the striped
      scheduler's concurrent per-repository workers.
    - Within a repository, PRs ascend by number, i.e. in the order the
      automation raised them (oldest first).  Merging the oldest first
      minimises the rebase churn imposed on the newer siblings.

    Ties between equally-sized repositories break on repository name so
    the order (and the grouped listing derived from it) is deterministic.
    """
    counts: dict[str, int] = {}
    for pr in prs:
        counts[pr.repository_full_name] = counts.get(pr.repository_full_name, 0) + 1
    return sorted(
        prs,
        key=lambda p: (
            -counts[p.repository_full_name],
            p.repository_full_name,
            p.number,
        ),
    )


def _print_prs_grouped_by_repo(
    prs: list[PullRequestInfo],
) -> None:
    """Print a PR list grouped by repository for owner-wide readability.

    Emits a header per repository followed by its PRs indented beneath,
    so a large owner-wide list stays scannable.  Repositories are listed
    in the order they first appear in ``prs`` (the caller passes them in
    merge order via :func:`_owner_merge_order`, so the listing mirrors the
    sequence in which they will be merged); PRs within a repository are
    shown in the supplied order.
    """
    by_repo: dict[str, list[PullRequestInfo]] = {}
    for pr in prs:
        by_repo.setdefault(pr.repository_full_name, []).append(pr)

    for repo, repo_prs in by_repo.items():
        console.print(f"\n📁 {repo} ({len(repo_prs)} PR(s))")
        for pr in repo_prs:
            console.print(f"  #{pr.number} {pr.title} (by {pr.author})")
