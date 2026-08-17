# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The cross-organization similar-PR search.

``find_similar_prs`` walks an owner's repositories; the per-repository
sweep, the candidate filter and the ``--debug`` comparison dump live in
helpers so each stays reviewable.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from ..models import ComparisonResult, PullRequestInfo
from ._base import _GitHubServiceBase


class _SimilarPullRequestsMixin(_GitHubServiceBase):
    """Similar-PR discovery for ``GitHubService``."""

    async def find_similar_prs(
        self,
        org: str,
        source_pr: PullRequestInfo,
        comparator,
        *,
        only_automation: bool,
    ) -> list[tuple[PullRequestInfo, ComparisonResult]]:
        """
        Find PRs across an organization that are similar to the provided source PR.

        This integrates progress updates:
        - Updates total repositories
        - Starts/completes repository sections
        - Increments PR analysis count per PR
        - Tracks similar PRs found

        Args:
            org: Organization login.
            source_pr: The PR to compare against.
            comparator: Provides compare_pull_requests(source, target) -> ComparisonResult.
            only_automation: If True, restrict candidates to automation PRs; otherwise, same author as source.

        Returns:
            List of (PullRequestInfo, ComparisonResult) tuples for similar PRs.
        """
        results: list[tuple[PullRequestInfo, ComparisonResult]] = []

        # Repo total is set automatically by _iter_org_repositories
        # on the first GraphQL page via totalCount.
        async for repo in self._iter_org_repositories_with_open_prs(org):
            repo_full_name = repo.get("nameWithOwner") or ""
            if not repo_full_name or "/" not in repo_full_name:
                if self._progress:
                    self._progress.add_error()
                continue

            if self._progress:
                self._progress.start_repository(repo_full_name)
                self._progress.update_operation(
                    f"Getting open PRs from {repo_full_name}"
                )

            matching_prs_in_repo = await self._find_similar_prs_in_repo(
                repo_full_name,
                source_pr,
                comparator,
                only_automation=only_automation,
            )

            results.extend(matching_prs_in_repo)

            if self._progress:
                self._progress.complete_repository(len(matching_prs_in_repo))

        return results

    async def _find_similar_prs_in_repo(
        self,
        repo_full_name: str,
        source_pr: PullRequestInfo,
        comparator,
        *,
        only_automation: bool,
    ) -> list[tuple[PullRequestInfo, ComparisonResult]]:
        """Compare every open PR of one repository against *source_pr*."""
        owner_n, name_n = repo_full_name.split("/", 1)
        first_nodes, page_info = await self._fetch_repo_prs_first_page(owner_n, name_n)
        prs = list(first_nodes)
        has_next = bool(page_info.get("hasNextPage"))
        end_cursor = page_info.get("endCursor") or None

        # Include additional pages if present
        if has_next:
            async for pr_node in self._iter_repo_open_prs_pages(
                owner_n, name_n, end_cursor
            ):
                prs.append(pr_node)

        matching_prs_in_repo: list[tuple[PullRequestInfo, ComparisonResult]] = []

        for pr_node in prs:
            target_pr = self.to_pull_request_info(repo_full_name, pr_node)

            if not self._is_similar_pr_candidate(
                source_pr, target_pr, only_automation=only_automation
            ):
                continue

            if self._progress:
                self._progress.analyze_pr(target_pr.number, repo_full_name)

            comparison: ComparisonResult = comparator.compare_pull_requests(
                source_pr, target_pr, only_automation
            )

            if self._debug_matching:
                self._print_comparison_debug(
                    repo_full_name, source_pr, target_pr, comparator, comparison
                )

            if comparison.is_similar:
                matching_prs_in_repo.append((target_pr, comparison))
                if self._progress:
                    # We can reuse 'found_similar_pr' if using MergeProgressTracker,
                    # otherwise this call will be a no-op for ProgressTracker.
                    try:
                        self._progress.found_similar_pr()  # type: ignore[attr-defined]
                    except Exception:
                        # No-op when the tracker lacks this method or
                        # the display update fails; counting is
                        # cosmetic only.
                        pass

        return matching_prs_in_repo

    @staticmethod
    def _is_similar_pr_candidate(
        source_pr: PullRequestInfo,
        target_pr: PullRequestInfo,
        *,
        only_automation: bool,
    ) -> bool:
        """Whether *target_pr* is worth comparing against *source_pr*."""
        # Skip the source PR itself
        if (
            target_pr.number == source_pr.number
            and target_pr.repository_full_name == source_pr.repository_full_name
        ):
            return False

        # Candidate filtering
        if only_automation:
            return any(
                bot in (target_pr.author or "").lower()
                for bot in [
                    "dependabot",
                    "renovate",
                    "pre-commit",
                    "github-actions",
                    "bot",
                ]
            )
        return (target_pr.author or "") == (source_pr.author or "")

    @staticmethod
    def _print_comparison_debug(
        repo_full_name: str,
        source_pr: PullRequestInfo,
        target_pr: PullRequestInfo,
        comparator,
        comparison: ComparisonResult,
    ) -> None:
        """Print the per-comparison score breakdown for ``--debug`` runs."""
        from rich.console import Console

        debug_console = Console()
        debug_console.print(
            f"\n🔍 [bold]Comparing {repo_full_name}#{target_pr.number}[/bold]"
        )
        debug_console.print(f"   Title: {target_pr.title}")
        debug_console.print(f"   Author: {target_pr.author}")

        # Show individual scores
        title_score = comparator._compare_titles(source_pr.title, target_pr.title)
        body_score = comparator._compare_bodies(source_pr.body, target_pr.body)
        files_score = comparator._compare_file_changes(
            source_pr.files_changed, target_pr.files_changed
        )
        author_score = (
            1.0
            if comparator._normalize_author(source_pr.author)
            == comparator._normalize_author(target_pr.author)
            else 0.0
        )

        debug_console.print(f"   📝 Title score: {title_score:.3f}")
        debug_console.print(f"   📄 Body score: {body_score:.3f}")
        debug_console.print(f"   📁 Files score: {files_score:.3f}")
        debug_console.print(f"   👤 Author score: {author_score:.3f}")
        debug_console.print(
            f"   🎯 Overall: {comparison.confidence_score:.3f} (threshold: 0.8)"
        )

        if comparison.is_similar:
            debug_console.print(
                f"   ✅ [green]SIMILAR[/green] - {', '.join(comparison.reasons)}"
            )
            return

        debug_console.print("   ❌ [red]NOT SIMILAR[/red]")

        # Show why it failed
        if title_score == 0:
            source_pkg = comparator._extract_package_name(source_pr.title)
            target_pkg = comparator._extract_package_name(target_pr.title)
            debug_console.print(f"      📦 Source package: '{source_pkg}'")
            debug_console.print(f"      📦 Target package: '{target_pkg}'")

        if body_score < 0.6:
            if target_pr.body is None:
                debug_console.print("      ⚠️ Target PR has no body")
            elif source_pr.body is None:
                debug_console.print("      ⚠️ Source PR has no body")
            else:
                debug_console.print(
                    f"      📄 Body comparison failed (score: {body_score:.3f})"
                )
