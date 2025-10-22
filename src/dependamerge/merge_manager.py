# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from rich.console import Console

from .copilot_handler import CopilotCommentHandler
from .github_async import GitHubAsync
from .github_service import GitHubService
from .models import ComparisonResult, PullRequestInfo
from .progress_tracker import MergeProgressTracker


class MergeStatus(Enum):
    """Status of a PR merge operation."""

    PENDING = "pending"
    APPROVING = "approving"
    APPROVED = "approved"
    MERGING = "merging"
    MERGED = "merged"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass
class MergeResult:
    """Result of a PR merge operation."""

    pr_info: PullRequestInfo
    status: MergeStatus
    error: str | None = None
    attempts: int = 0
    duration: float = 0.0


class AsyncMergeManager:
    """
    Manages parallel approval and merging of pull requests.

    This class handles:
    - Concurrent approval of PRs
    - Concurrent merging with retry logic
    - Progress tracking and error handling
    - Rate limit-aware processing
    """

    def __init__(
        self,
        token: str,
        merge_method: str = "merge",
        max_retries: int = 2,
        concurrency: int = 5,
        fix_out_of_date: bool = False,
        progress_tracker: MergeProgressTracker | None = None,
        dry_run: bool = False,
        dismiss_copilot: bool = False,
        force_level: str = "none",
    ):
        self.token = token
        self.default_merge_method = merge_method
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.fix_out_of_date = fix_out_of_date
        self.progress_tracker = progress_tracker
        self.dry_run = dry_run
        self.dismiss_copilot = dismiss_copilot
        self.force_level = force_level
        self.log = logging.getLogger(__name__)

        # Track merge operations
        self._merge_semaphore = asyncio.Semaphore(concurrency)
        self._results: list[MergeResult] = []
        self._github_client: GitHubAsync | None = None
        self._github_service: GitHubService | None = None
        self._copilot_handler: CopilotCommentHandler | None = None
        self._console = Console()

        # Track merge methods per repository
        self._pr_merge_methods: dict[str, str] = {}

    def _log_and_print(self, message: str, style: str | None = None) -> None:
        """Log message and also print to stdout for CLI visibility."""
        self.log.info(message)
        if style:
            self._console.print(message, style=style)
        else:
            print(message)

    def _get_mergeability_icon_and_style(
        self, mergeable_state: str | None
    ) -> tuple[str, str | None]:
        """Get appropriate icon and style for mergeable state."""
        if mergeable_state == "dirty":
            return "🛑", "red"
        elif mergeable_state == "behind":
            return "⚠️", "yellow"
        elif mergeable_state == "clean":
            return "✅", "green"
        elif mergeable_state == "draft":
            return "📝", "blue"
        else:
            return "🔍", None

    async def __aenter__(self):
        """Async context manager entry."""
        self._github_client = GitHubAsync(token=self.token)
        await self._github_client.__aenter__()

        # Initialize GitHubService for branch protection detection
        self._github_service = GitHubService(token=self.token)

        # Initialize Copilot handler if dismissal is enabled
        if self.dismiss_copilot:
            self._copilot_handler = CopilotCommentHandler(
                self._github_client, dry_run=self.dry_run, debug=True
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._github_service:
            await self._github_service.close()
        if self._github_client:
            await self._github_client.__aexit__(exc_type, exc_val, exc_tb)

    async def merge_prs_parallel(
        self,
        pr_list: list[tuple[PullRequestInfo, ComparisonResult | None]],
    ) -> list[MergeResult]:
        """
        Merge multiple PRs in parallel.

        Args:
            pr_list: List of (PullRequestInfo, ComparisonResult) tuples

        Returns:
            List of MergeResult objects with operation results
        """
        if not pr_list:
            return []

        if self.dry_run:
            self.log.info(f"🔍 DRY RUN: Would merge {len(pr_list)} PRs")
        else:
            self.log.info(f"Starting parallel merge of {len(pr_list)} PRs")

        # Create tasks for all PRs
        tasks = []
        for pr_info, _comparison in pr_list:
            task = asyncio.create_task(
                self._merge_single_pr_with_semaphore(pr_info),
                name=f"merge-{pr_info.repository_full_name}#{pr_info.number}",
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        final_results: list[MergeResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pr_info = pr_list[i][0]
                error_result = MergeResult(
                    pr_info=pr_info, status=MergeStatus.FAILED, error=str(result)
                )
                final_results.append(error_result)
                self.log.error(
                    f"Unexpected error merging PR {pr_info.repository_full_name}#{pr_info.number}: {result}"
                )
            else:
                # result is guaranteed to be MergeResult here since it's not an Exception
                final_results.append(cast(MergeResult, result))

        self._results = final_results
        return final_results

    async def _merge_single_pr_with_semaphore(
        self, pr_info: PullRequestInfo
    ) -> MergeResult:
        """Merge a single PR with concurrency control."""
        async with self._merge_semaphore:
            return await self._merge_single_pr(pr_info)

    async def _merge_single_pr(self, pr_info: PullRequestInfo) -> MergeResult:
        """
        Merge a single pull request with retry logic.

        Args:
            pr_info: Pull request information

        Returns:
            MergeResult with operation status and details
        """
        start_time = time.time()
        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)

        # Early determination of merge method based on repository settings
        merge_method = await self._get_merge_method_for_repo(repo_owner, repo_name)

        # Store the determined merge method for this PR
        self._pr_merge_methods[f"{repo_owner}/{repo_name}"] = merge_method

        result = MergeResult(pr_info=pr_info, status=MergeStatus.PENDING)

        try:
            # Check if PR is closed before processing
            if pr_info.state != "open":
                result.status = MergeStatus.FAILED
                result.error = "PR is already closed"
                self._console.print(f"🛑 Failed: {pr_info.html_url} \\[already closed]")
                return result

            if not self._is_pr_mergeable(pr_info):
                # Get detailed status for a more informative skip message
                # Use async method to avoid event loop conflicts
                repo_owner, repo_name = pr_info.repository_full_name.split("/")

                # Check if blocked to get more detailed status
                if pr_info.mergeable_state == "blocked" and self._github_client:
                    try:
                        detailed_status = (
                            await self._github_client.analyze_block_reason(
                                repo_owner, repo_name, pr_info.number, pr_info.head_sha
                            )
                        )
                    except Exception:
                        detailed_status = f"Blocked (state: {pr_info.mergeable_state})"
                else:
                    # For non-blocked states, provide basic status
                    if pr_info.mergeable_state == "dirty":
                        detailed_status = "Merge conflicts"
                    elif pr_info.mergeable_state == "behind":
                        detailed_status = "Rebase required (out of date)"
                    elif pr_info.mergeable_state == "draft":
                        detailed_status = "Draft PR"
                    else:
                        detailed_status = (
                            f"Not mergeable (state: {pr_info.mergeable_state})"
                        )

                # Use the detailed status as the skip reason, with fallback
                if detailed_status and detailed_status != "Status unclear":
                    skip_reason = detailed_status.lower()
                else:
                    # Fallback to basic mapping if detailed status is unclear
                    if pr_info.mergeable_state == "dirty":
                        skip_reason = "merge conflicts"
                    elif pr_info.mergeable_state == "behind":
                        skip_reason = "behind"
                    elif pr_info.mergeable_state == "blocked":
                        if pr_info.mergeable is True:
                            skip_reason = "blocked, requires review"
                        else:
                            skip_reason = "blocked by failing checks"
                    elif pr_info.mergeable_state == "unstable":
                        skip_reason = "unstable"
                    elif pr_info.mergeable is False:
                        skip_reason = "not mergeable"
                    else:
                        skip_reason = "unknown"

                # Determine if this is truly blocked (unmergeable) or just skipped
                if pr_info.mergeable_state == "dirty" or (
                    pr_info.mergeable_state == "behind" and pr_info.mergeable is False
                ):
                    result.status = MergeStatus.BLOCKED
                    icon = "🛑"
                    self._console.print(
                        f"{icon} Blocked: {pr_info.html_url} \\[{skip_reason}]"
                    )
                else:
                    result.status = MergeStatus.SKIPPED
                    icon = "⏭️"
                    self._console.print(
                        f"{icon} Skipped: {pr_info.html_url} \\[{skip_reason}]"
                    )

                result.error = f"PR is not mergeable (state: {pr_info.mergeable_state}, mergeable: {pr_info.mergeable})"

                # For the result error (used in CLI output), use the detailed status if it's more informative
                if detailed_status and detailed_status != "Status unclear":
                    result.error = detailed_status

                return result

            # Check for blocking reviews (changes requested)
            if self._has_blocking_reviews(pr_info):
                # Only skip if not forcing with 'all' level
                if self.force_level != "all":
                    result.status = MergeStatus.SKIPPED
                    result.error = "PR has reviews requesting changes - will not override human feedback"
                    self._console.print(
                        f"⏭️ Skipped: {pr_info.html_url} \\[has reviews requesting changes]"
                    )
                    self.log.info(
                        f"⏭️  Skipping {pr_info.repository_full_name}#{pr_info.number}: {result.error}"
                    )
                    return result
                else:
                    # Only log during dry-run evaluation to avoid duplicate messages
                    if self.dry_run:
                        self.log.warning(
                            f"⚠️  Overriding blocking reviews for {pr_info.repository_full_name}#{pr_info.number} (--force=all)"
                        )

            # Step 1: Check merge requirements (including branch protection)
            can_merge, merge_check_reason = await self._check_merge_requirements(
                pr_info
            )

            if not can_merge:
                result.status = MergeStatus.SKIPPED
                result.error = f"Merge requirements not met: {merge_check_reason}"
                self._console.print(
                    f"⏭️ Skipped: {pr_info.html_url} \\[{merge_check_reason.lower()}]"
                )
                self.log.info(
                    f"⏭️  Skipping {pr_info.repository_full_name}#{pr_info.number}: {result.error}"
                )
                return result

            # Step 2: Dismiss Copilot comments if enabled
            copilot_processing_successful = True
            if self.dismiss_copilot and self._copilot_handler:
                # Analyze what types of reviews we have
                self._copilot_handler.analyze_copilot_review_dismissibility(pr_info)

                try:
                    (
                        processed_count,
                        total_count,
                    ) = await self._copilot_handler.dismiss_copilot_comments_for_pr(
                        pr_info
                    )
                    if total_count > 0:
                        # Silent processing in background
                        pass
                except Exception as e:
                    self.log.warning(
                        f"⚠️  Failed to process Copilot items for PR {pr_info.number}: {e}"
                    )
                    copilot_processing_successful = False

            # Step 3: Only approve if Copilot processing was successful
            if not copilot_processing_successful:
                result.status = MergeStatus.FAILED
                result.error = "Copilot review processing incomplete - not approving to avoid pollution"
                self._console.print(
                    f"❌ Failed: {pr_info.html_url} \\[copilot processing incomplete]"
                )
                return result

            result.status = MergeStatus.APPROVING

            if self.progress_tracker:
                self.progress_tracker.update_operation(
                    f"Approving PR {pr_info.number} in {pr_info.repository_full_name}"
                )

            if not self.dry_run:
                approval_added = await self._approve_pr(
                    repo_owner, repo_name, pr_info.number
                )
                if not approval_added:
                    # Already approved, no need to log additional approval
                    pass
            result.status = MergeStatus.APPROVED

            # Step 5: Handle rebase if needed before merge
            if pr_info.mergeable_state == "behind" and self.fix_out_of_date:
                if self.dry_run:
                    # NOTE: In dry-run mode, we should NOT print here as it breaks single-line reporting
                    # The dry-run output should only be a single line per PR in the evaluation section
                    pass
                else:
                    self.log.info(
                        f"PR {pr_info.repository_full_name}#{pr_info.number} is behind - rebasing before merge"
                    )
                    self._console.print(
                        f"🔄 Rebasing {pr_info.html_url} (behind base branch)"
                    )

                    try:
                        if self._github_client is None:
                            raise RuntimeError("GitHub client not initialized")
                        await self._github_client.update_branch(
                            repo_owner, repo_name, pr_info.number
                        )

                        # Wait for GitHub to process the update and run checks
                        self._console.print(
                            "⏳ Waiting for checks to complete after rebase..."
                        )
                        await asyncio.sleep(5.0)

                        # Re-check PR status after rebase with extended retry logic
                        # Poll for status checks to complete after rebase (max 2 minutes)
                        # Initialize variables before the loop
                        updated_mergeable: bool | None = pr_info.mergeable
                        updated_mergeable_state: str | None = pr_info.mergeable_state

                        # Wait up to 2 minutes for rebase to complete and PR state to stabilize
                        for check_attempt in range(24):
                            updated_pr_data = await self._github_client.get(
                                f"/repos/{repo_owner}/{repo_name}/pulls/{pr_info.number}"
                            )

                            # Extract mergeable state from API response
                            # The GitHub PR API returns a dict, but we need to check type for MyPy
                            if isinstance(updated_pr_data, dict):
                                updated_mergeable = updated_pr_data.get("mergeable")
                                updated_mergeable_state = updated_pr_data.get(
                                    "mergeable_state"
                                )
                            else:
                                # This should not happen for a single PR request, but handle for type safety
                                updated_mergeable = None
                                updated_mergeable_state = None

                            # Wait for status checks to complete - not just for "behind" to clear
                            if updated_mergeable_state == "clean":
                                # Perfect - PR is ready to merge
                                break
                            elif updated_mergeable_state == "behind":
                                # Still processing rebase
                                if check_attempt < 23:
                                    self.log.debug(
                                        f"PR still processing rebase, waiting... (attempt {check_attempt + 1}/24)"
                                    )
                                    await asyncio.sleep(5.0)
                            elif updated_mergeable_state == "blocked":
                                # Status checks are running - need to wait longer
                                if check_attempt < 23:
                                    self.log.debug(
                                        f"PR status checks running after rebase, waiting... (attempt {check_attempt + 1}/24)"
                                    )
                                    await asyncio.sleep(5.0)
                                else:
                                    # Timeout waiting for status checks
                                    self.log.warning(
                                        f"Timeout waiting for status checks to complete for PR {pr_info.repository_full_name}#{pr_info.number}. Proceeding with merge attempt."
                                    )
                                    break
                            else:
                                # Other states (dirty, unstable, etc.) - exit early
                                break

                        # Update our PR info with the latest state (variables are guaranteed to be set by loop initialization)
                        pr_info.mergeable = updated_mergeable
                        pr_info.mergeable_state = updated_mergeable_state

                        self.log.info(
                            f"After rebase: PR {pr_info.repository_full_name}#{pr_info.number} state is {pr_info.mergeable_state}"
                        )

                        # Report post-rebase status
                        if pr_info.mergeable_state == "clean":
                            self._console.print(
                                f"✅ Rebase completed successfully for {pr_info.html_url}"
                            )
                        elif pr_info.mergeable_state == "behind":
                            self._console.print(
                                f"⚠️  PR {pr_info.html_url} still behind after rebase - may need manual intervention"
                            )
                        elif pr_info.mergeable_state == "blocked":
                            self._console.print(
                                f"⏳ PR {pr_info.html_url} rebased, waiting for status checks..."
                            )
                        else:
                            self._console.print(
                                f"ℹ️  PR {pr_info.html_url} rebased, current state: {pr_info.mergeable_state}"
                            )

                    except Exception as e:
                        result.status = MergeStatus.FAILED
                        result.error = f"Failed to rebase PR: {e}"

                        if self.progress_tracker:
                            self.progress_tracker.merge_failure()
                        self._console.print(
                            f"❌ Failed: {pr_info.html_url} \\[rebase error: {e}]"
                        )
                        return result

            # Step 6: Attempt merge
            result.status = MergeStatus.MERGING
            if self.dry_run:
                # IMPORTANT: Dry-run output must be SINGLE LINE per PR for clean evaluation display
                # Each PR should have exactly one line of output under "🔍 Dependamerge Evaluation"

                # In dry-run, simulate what would happen based on current PR state
                if pr_info.mergeable_state == "behind" and not self.fix_out_of_date:
                    result.status = MergeStatus.SKIPPED
                    result.error = "PR is behind base branch and --no-fix option is set"
                    self._console.print(
                        f"⏭️ Skipped: {pr_info.html_url} \\[behind, rebase disabled]"
                    )
                elif pr_info.mergeable_state == "behind" and self.fix_out_of_date:
                    # For behind PRs with fix enabled, show warning with rebase info
                    result.status = MergeStatus.MERGED  # Would succeed after rebase
                    result.error = "behind base branch"
                    if self.progress_tracker:
                        self.progress_tracker.merge_success()
                    self._console.print(
                        f"⚠️  Rebase/merge: {pr_info.html_url} \\[behind base branch]"
                    )
                elif pr_info.mergeable_state == "dirty":
                    result.status = MergeStatus.BLOCKED
                    result.error = "PR has merge conflicts"
                    self._console.print(
                        f"🛑 Blocked: {pr_info.html_url} \\[merge conflicts]"
                    )
                elif (
                    pr_info.mergeable is False and pr_info.mergeable_state == "blocked"
                ):
                    result.status = MergeStatus.BLOCKED
                    result.error = "PR blocked by failing checks"
                    self._console.print(
                        f"🛑 Blocked: {pr_info.html_url} \\[blocked by failing checks]"
                    )
                else:
                    # Simulate successful merge in dry run
                    result.status = MergeStatus.MERGED
                    if self.progress_tracker:
                        self.progress_tracker.merge_success()
                    # Single line summary for successful dry-run
                    self._console.print(f"☑️ Approve/merge: {pr_info.html_url}")
            else:
                if self.progress_tracker:
                    self.progress_tracker.update_operation(
                        f"Merging PR {pr_info.number} in {pr_info.repository_full_name}"
                    )

                merged = await self._merge_pr_with_retry(pr_info, repo_owner, repo_name)

                if merged:
                    result.status = MergeStatus.MERGED
                    if self.progress_tracker:
                        self.progress_tracker.merge_success()
                    # Single line summary for successful merge
                    self._console.print(f"✅ Success: {pr_info.html_url}")
                    self.log.info(
                        f"✅ Merge successful for PR {pr_info.repository_full_name}#{pr_info.number}"
                    )
                else:
                    result.status = MergeStatus.FAILED
                    result.error = "Failed to merge after all retry attempts"
                    if self.progress_tracker:
                        self.progress_tracker.merge_failure()
                    # Single line summary for failed merge with reason
                    failure_reason = self._get_failure_summary(pr_info)
                    self._console.print(
                        f"❌ Failed: {pr_info.html_url} \\[{failure_reason}]"
                    )

        except Exception as e:
            result.status = MergeStatus.FAILED
            result.error = str(e)
            if self.progress_tracker:
                self.progress_tracker.merge_failure()

            # Provide clean single-line error messages instead of dumping HTTP details
            error_str = str(e)
            if "Failed to approve PR" in error_str and "422" in error_str:
                reason = "cannot approve"
                if "Unprocessable Entity" in error_str:
                    reason = "approval restrictions"
                self._console.print(f"❌ Failed: {pr_info.html_url} [{reason}]")
            else:
                # For other errors, use the existing detailed logging but clean console output
                self._console.print(f"❌ Failed: {pr_info.html_url} [processing error]")

            self.log.error(
                f"Error processing PR {pr_info.repository_full_name}#{pr_info.number}: {e}"
            )

        finally:
            result.duration = time.time() - start_time

        return result

    def _is_pr_mergeable(self, pr_info: PullRequestInfo) -> bool:
        """
        Check if a PR is mergeable.

        Args:
            pr_info: Pull request information

        Returns:
            True if the PR can be merged, False otherwise
        """
        # Handle different types of blocks intelligently - matches original logic
        if pr_info.mergeable_state == "blocked" and pr_info.mergeable is True:
            # This is blocked by branch protection but tool can handle it (approval, etc.)
            return True
        elif pr_info.mergeable_state == "blocked" and pr_info.mergeable is False:
            # Blocked by failing checks - we can try merging anyway
            return True
        elif pr_info.mergeable is False:
            # Only skip if mergeable is explicitly False (not None/UNKNOWN)
            # Use appropriate icon based on state - only truly unmergeable PRs get blocked
            if pr_info.mergeable_state == "dirty" or (
                pr_info.mergeable_state == "behind" and pr_info.mergeable is False
            ):
                icon = "🛑"
                action = "Blocking"
            else:
                icon = "⏭️"
                action = "Skipping"

            # Just log internally, don't show verbose messages
            skip_msg = f"{icon}  {action} unmergeable PR {pr_info.number} in {pr_info.repository_full_name} (mergeable: {pr_info.mergeable}, state: {pr_info.mergeable_state})"
            self.log.info(skip_msg)
            return False
        elif pr_info.mergeable is None:
            # Handle UNKNOWN mergeable state - treat as potentially mergeable
            # GitHub is still calculating mergeable state, but we can attempt merge
            self.log.info(
                f"ℹ️ PR {pr_info.number} in {pr_info.repository_full_name} has unknown mergeable state - treating as potentially mergeable"
            )
            return True

        # All other cases are considered mergeable
        self.log.info(
            f"✅ PR {pr_info.number} in {pr_info.repository_full_name} is considered mergeable (mergeable: {pr_info.mergeable}, state: {pr_info.mergeable_state})"
        )
        return True

    def _has_blocking_reviews(self, pr_info: PullRequestInfo) -> bool:
        """
        Check if a PR has reviews that would block automatic approval.

        Args:
            pr_info: Pull request information

        Returns:
            True if there are blocking reviews (changes requested), False otherwise
        """
        for review in pr_info.reviews:
            if review.state == "CHANGES_REQUESTED":
                self.log.info(
                    f"⚠️  PR {pr_info.number} has changes requested by {review.user} - will not override human feedback"
                )
                return True
        return False

    async def _check_merge_requirements(
        self, pr_info: PullRequestInfo
    ) -> tuple[bool, str]:
        """
        Check if a PR meets all requirements for merging, including branch protection rules.

        Args:
            pr_info: Pull request information

        Returns:
            Tuple of (can_merge: bool, reason: str)
        """
        if not self._github_client:
            return False, "GitHub client not initialized"

        repo_owner, repo_name = pr_info.repository_full_name.split("/")

        try:
            # Check branch protection rules
            base_branch = pr_info.base_branch or "main"
            protection_rules = await self._github_client.get_branch_protection(
                repo_owner, repo_name, base_branch
            )

            if protection_rules:
                # Check required reviews
                required_reviews = protection_rules.get(
                    "required_pull_request_reviews", {}
                )
                if required_reviews:
                    require_code_owner = required_reviews.get(
                        "require_code_owner_reviews", False
                    )

                    # If code owner reviews are required, our automated approval might not be sufficient
                    if require_code_owner:
                        # Check if user wants to bypass code owner checks
                        if self.force_level in [
                            "code-owners",
                            "protection-rules",
                            "all",
                        ]:
                            # Only log during dry-run evaluation to avoid duplicate messages
                            if self.dry_run:
                                self.log.warning(
                                    f"⚠️  Bypassing code owner review requirement for {repo_owner}/{repo_name}#{pr_info.number} (--force={self.force_level})"
                                )
                        else:
                            return (
                                False,
                                "code owner reviews are required - cannot auto-approve",
                            )

        except Exception:
            # Don't fail the merge attempt if we can't check protection rules
            pass

        # Test merge capability during dry-run by attempting a test merge
        if self.dry_run:
            try:
                # Use pre-determined merge method for this repository
                cache_key = f"{repo_owner}/{repo_name}"
                merge_method = self._pr_merge_methods.get(
                    cache_key, self.default_merge_method
                )

                # Attempt a test merge to detect hidden branch protection rules
                test_result = await self._test_merge_capability(
                    repo_owner, repo_name, pr_info.number, merge_method
                )
                if not test_result[0]:
                    # Check if we should bypass protection rules
                    if self.force_level in ["protection-rules", "all"]:
                        # Only log during dry-run evaluation to avoid duplicate messages
                        if self.dry_run:
                            self.log.warning(
                                f"⚠️  Bypassing branch protection check for {repo_owner}/{repo_name}#{pr_info.number}: {test_result[1]} (--force={self.force_level})"
                            )
                    else:
                        return False, test_result[1]

            except Exception as e:
                # If we can't test merge, continue with other checks
                self.log.debug(
                    f"Could not test merge capability for {repo_owner}/{repo_name}#{pr_info.number}: {e}"
                )

        # Additional checks based on PR state
        if pr_info.mergeable_state == "blocked":
            # Check if Copilot comments might be the blocker
            if self.dismiss_copilot and self._copilot_handler:
                has_copilot_comments = (
                    self._copilot_handler.has_blocking_copilot_comments(pr_info)
                )
                if has_copilot_comments:
                    return (
                        True,
                        "PR blocked but has Copilot comments that will be dismissed",
                    )

            # For blocked PRs, if mergeable is True, it just needs approval - we can handle that
            if pr_info.mergeable is True:
                return True, "PR ready for approval and merge"
            else:
                # If mergeable is False and state is blocked, it's blocked by failing checks
                if self.force_level == "all":
                    # Only log during dry-run evaluation to avoid duplicate messages
                    if self.dry_run:
                        self.log.warning(
                            f"⚠️  Bypassing failing status checks for {repo_owner}/{repo_name}#{pr_info.number} (--force=all)"
                        )
                    return True, "PR blocked but forcing merge attempt (--force=all)"
                else:
                    return False, "blocked by failing status checks"
        elif pr_info.mergeable_state == "behind":
            if not self.fix_out_of_date:
                if self.force_level == "all":
                    self.log.warning(
                        f"⚠️  Attempting merge despite being behind for {repo_owner}/{repo_name}#{pr_info.number} (--force=all)"
                    )
                    return True, "PR behind but forcing merge attempt (--force=all)"
                else:
                    return False, "PR is behind base branch and --no-fix option is set"
            else:
                return True, "PR is behind - will rebase before merge"
        elif pr_info.mergeable_state == "dirty":
            if self.force_level == "all":
                self.log.warning(
                    f"⚠️  Attempting merge despite conflicts for {repo_owner}/{repo_name}#{pr_info.number} (--force=all)"
                )
                return True, "PR has conflicts but forcing merge attempt (--force=all)"
            else:
                return (False, "merge conflicts")

        return True, "All merge requirements appear to be met"

    async def _approve_pr(self, owner: str, repo: str, pr_number: int) -> bool:
        """
        Approve a pull request if not already approved by the current user or sufficiently approved.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            True if approval was added, False if already approved/sufficient

        Raises:
            Exception: If approval fails
        """
        if not self._github_client:
            raise RuntimeError("GitHub client not initialized")

        try:
            # Check if current user has already approved this PR
            pr_data = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_number}"
            )

            if isinstance(pr_data, dict):
                # Get current user login
                user_data = await self._github_client.get("/user")
                current_user = (
                    user_data.get("login") if isinstance(user_data, dict) else None
                )

                if current_user:
                    # Check existing reviews
                    reviews_data = await self._github_client.get(
                        f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
                    )

                    if isinstance(reviews_data, list):
                        # Look for existing approval by current user
                        for review in reviews_data:
                            if (
                                review.get("user", {}).get("login") == current_user
                                and review.get("state") == "APPROVED"
                            ):
                                self._log_and_print(
                                    f"⏩ Skipping approval on {owner}/{repo}#{pr_number} - already approved by {current_user}",
                                    "blue",
                                )
                                return False

                        # Check if PR already has sufficient approvals from others
                        approved_reviews = [
                            review
                            for review in reviews_data
                            if review.get("state") == "APPROVED"
                            and review.get("user", {}).get("login") != current_user
                        ]

                        if (
                            approved_reviews
                            and pr_data.get("mergeable_state") == "clean"
                        ):
                            self._log_and_print(
                                f"⏩ Skipping approval on {owner}/{repo}#{pr_number} - already has {len(approved_reviews)} approval(s)",
                                "blue",
                            )
                            return False

            await self._github_client.approve_pull_request(
                owner, repo, pr_number, "Auto-approved by dependamerge"
            )
            return True
        except Exception as e:
            # Handle 422 errors specifically
            error_str = str(e)
            if "422" in error_str and "Unprocessable Entity" in error_str:
                # This usually means the PR can't be approved (e.g., already approved by user, or other restrictions)
                self._log_and_print(
                    f"⏩ Skipping approval on {owner}/{repo}#{pr_number} - cannot approve (already approved or restricted)",
                    "blue",
                )
                return False
            else:
                raise RuntimeError(
                    f"Failed to approve PR {owner}/{repo}#{pr_number}: {e}"
                ) from e

    async def _merge_pr_with_retry(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """
        Merge a PR with retry logic for transient failures.

        Args:
            pr_info: Pull request information
            owner: Repository owner
            repo: Repository name

        Returns:
            True if merged successfully, False otherwise
        """
        if not self._github_client:
            raise RuntimeError("GitHub client not initialized")

        for attempt in range(self.max_retries + 1):
            try:
                # Check if PR has already been closed/merged before attempting
                if attempt > 0:
                    # Re-fetch PR state to check if it was merged by a previous attempt
                    # or by external processes
                    try:
                        current_pr_data = await self._github_client.get(
                            f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
                        )
                        if isinstance(current_pr_data, dict):
                            current_state = current_pr_data.get("state")
                            current_merged = current_pr_data.get("merged", False)

                            if current_state == "closed" and current_merged:
                                self.log.info(
                                    f"✅ PR {owner}/{repo}#{pr_info.number} was already merged, skipping retry"
                                )
                                return True
                            elif current_state == "closed" and not current_merged:
                                self.log.info(
                                    f"⚠️ PR {owner}/{repo}#{pr_info.number} was closed without merging, aborting retry"
                                )
                                # This will be caught by the outer merge logic and formatted consistently
                                return False
                    except Exception as state_check_error:
                        self.log.debug(
                            f"Failed to check PR state before retry {attempt + 1}: {state_check_error}"
                        )

                # Use pre-determined merge method for this repository
                cache_key = f"{owner}/{repo}"
                merge_method = self._pr_merge_methods.get(
                    cache_key, self.default_merge_method
                )

                # Attempt the merge
                merged = await self._github_client.merge_pull_request(
                    owner, repo, pr_info.number, merge_method
                )

                if merged:
                    self.log.info(
                        f"✅ Merge API returned success for PR {owner}/{repo}#{pr_info.number}"
                    )
                    return True

                # Merge failed, check if we can fix it
                self.log.warning(
                    f"⚠️ Merge API returned false for PR {owner}/{repo}#{pr_info.number} (attempt {attempt + 1})"
                )
                if attempt < self.max_retries:
                    should_retry = await self._handle_merge_failure(
                        pr_info, owner, repo
                    )
                    if should_retry:
                        self.log.info(
                            f"Retrying merge for PR {owner}/{repo}#{pr_info.number} (attempt {attempt + 2})"
                        )
                        continue
                    else:
                        self.log.info(
                            f"Not retrying PR {owner}/{repo}#{pr_info.number} - no fixable issues found"
                        )
                        break

            except Exception as e:
                error_msg = str(e)

                # Enhanced error handling with specific status code checks
                if "405" in error_msg and "Method Not Allowed" in error_msg:
                    # Don't log here - will be handled in failure summary
                    if "behind" in error_msg.lower() and self.fix_out_of_date:
                        # Allow retry for behind PRs
                        pass
                    elif pr_info.mergeable_state == "blocked":
                        # Don't retry immediately - status checks need more time
                        break
                    else:
                        # Don't retry 405 errors unless they're "behind" issues
                        break
                elif "403" in error_msg and "Forbidden" in error_msg:
                    break
                elif "422" in error_msg:
                    break
                else:
                    # Only log for debugging purposes
                    self.log.debug(
                        f"Merge attempt {attempt + 1} failed for PR {owner}/{repo}#{pr_info.number}: {e}"
                    )

                if attempt >= self.max_retries:
                    break

                # Don't retry certain error types that are unlikely to be transient
                # Exception: Allow retry for 405 errors on "behind" PRs if fix_out_of_date is enabled
                if ("405" in error_msg and "behind" not in error_msg.lower()) or (
                    "422" in error_msg and "not mergeable" in error_msg.lower()
                ):
                    self.log.info(
                        f"Not retrying PR {owner}/{repo}#{pr_info.number} due to permanent error condition"
                    )
                    break
                elif (
                    "405" in error_msg
                    and "behind" in error_msg.lower()
                    and not self.fix_out_of_date
                ):
                    self.log.info(
                        f"Not retrying PR {owner}/{repo}#{pr_info.number} - behind base branch but --no-fix is set"
                    )
                    break

                # Wait a bit before retrying
                await asyncio.sleep(1.0)

        return False

    def _get_failure_summary(self, pr_info: PullRequestInfo) -> str:
        """
        Generate a detailed failure summary based on PR state.

        Args:
            pr_info: Pull request information

        Returns:
            Detailed description of why the merge failed
        """
        if pr_info.mergeable_state == "behind":
            return "behind base branch"
        elif pr_info.mergeable_state == "blocked":
            # Use detailed block analysis for blocked PRs
            try:
                from .github_client import GitHubClient

                client = GitHubClient(token=self.token)
                detailed_reason = client._analyze_block_reason(pr_info)
                # Convert the detailed reason to a more concise format for console output
                if detailed_reason.startswith("Blocked by failing check:"):
                    check_name = detailed_reason.replace(
                        "Blocked by failing check: ", ""
                    )
                    return f"failing check: {check_name}"
                elif (
                    detailed_reason.startswith("Blocked by")
                    and "failing checks" in detailed_reason
                ):
                    return detailed_reason.replace("Blocked by ", "").lower()
                elif "Human reviewer requested changes" in detailed_reason:
                    return "human reviewer requested changes"
                elif "Copilot" in detailed_reason:
                    return detailed_reason.replace("Blocked by ", "").lower()
                elif "branch protection" in detailed_reason.lower():
                    return "branch protection rules prevent merge"
                else:
                    return detailed_reason.replace("Blocked by ", "").lower()
            except Exception as e:
                self.log.debug(f"Failed to get detailed block reason: {e}")
                # Fallback to generic message
                pass

            # Fallback logic when detailed analysis fails
            if pr_info.mergeable is True:
                return "branch protection rules prevent merge"
            else:
                return "blocked by failing status checks"
        elif pr_info.mergeable_state == "dirty":
            return "merge conflicts"
        elif pr_info.mergeable_state == "draft":
            return "draft PR"
        elif pr_info.mergeable is False:
            return "cannot update protected ref - organization or branch protection rules prevent merge"
        elif pr_info.mergeable_state == "unknown":
            # For unknown state, try to get more details using the GitHub client
            try:
                from .github_client import GitHubClient

                client = GitHubClient(token=self.token)
                detailed_reason = client._analyze_block_reason(pr_info)
                if "failing check" in detailed_reason.lower():
                    if detailed_reason.startswith("Blocked by failing check:"):
                        check_name = detailed_reason.replace(
                            "Blocked by failing check: ", ""
                        )
                        return f"failing check: {check_name}"
                    else:
                        return detailed_reason.replace("Blocked by ", "").lower()
                else:
                    return detailed_reason.replace("Blocked by ", "").lower()
            except Exception as e:
                self.log.debug(f"Failed to analyze unknown state: {e}")
                return "status checks pending or failed"
        else:
            return f"merge failed: {pr_info.mergeable_state}"

    async def _get_merge_method_for_repo(self, owner: str, repo: str) -> str:
        """
        Get the appropriate merge method for a specific repository based on branch protection settings.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Merge method to use: "merge", "squash", or "rebase"
        """
        if not self._github_service:
            self.log.warning("GitHubService not available, using default merge method")
            return self.default_merge_method

        try:
            # Get branch protection settings for main branch
            protection_settings = (
                await self._github_service.get_branch_protection_settings(
                    owner, repo, "main"
                )
            )

            # Determine appropriate merge method
            merge_method = self._github_service.determine_merge_method(
                protection_settings, self.default_merge_method
            )

            if merge_method != self.default_merge_method:
                self.log.info(
                    f"Repository {owner}/{repo} requires '{merge_method}' merge method "
                    f"(protection: requiresLinearHistory={protection_settings and protection_settings.get('requiresLinearHistory', False)})"
                )

            return merge_method

        except Exception as e:
            self.log.warning(
                f"Failed to determine merge method for {owner}/{repo}, using default '{self.default_merge_method}': {e}"
            )
            return self.default_merge_method

    async def _handle_merge_failure(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """
        Handle a merge failure and determine if we should retry.

        Args:
            pr_info: Pull request information
            owner: Repository owner
            repo: Repository name

        Returns:
            True if we should retry, False otherwise
        """
        if not self._github_client:
            return False

        # Check if the branch is out of date and we can fix it
        if self.fix_out_of_date and pr_info.mergeable_state == "behind":
            try:
                self.log.info(
                    f"PR {owner}/{repo}#{pr_info.number} is behind - updating branch"
                )
                await self._github_client.update_branch(owner, repo, pr_info.number)
                # Wait a moment for GitHub to process the update
                await asyncio.sleep(2.0)
                return True
            except Exception as e:
                self.log.error(
                    f"Failed to update branch for PR {owner}/{repo}#{pr_info.number}: {e}"
                )

        # For other failure types, don't retry
        return False

    async def _test_merge_capability(
        self, owner: str, repo: str, pr_number: int, merge_method: str
    ) -> tuple[bool, str]:
        """
        Test if a PR can be merged by attempting a dry-run merge operation.

        This helps detect branch protection rules that aren't visible through the API,
        such as organization-level restrictions.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            merge_method: Merge method to test

        Returns:
            Tuple of (can_merge: bool, reason: str)
        """
        if not self._github_client:
            return False, "GitHub client not initialized"

        try:
            # Check organization-level restrictions that may not be visible in branch protection
            try:
                org_data = await self._github_client.get(f"/orgs/{owner}")
                if isinstance(org_data, dict):
                    # Some organizations have additional restrictions
                    web_commit_signoff = org_data.get(
                        "web_commit_signoff_required", False
                    )
                    if web_commit_signoff:
                        self.log.debug(f"Organization {owner} requires commit signoff")
            except Exception as org_check_error:
                # Organization check failed, continue with other checks
                self.log.debug(
                    f"Could not check organization settings: {org_check_error}"
                )

            # Note: Removed DCO signoff check as web_commit_signoff_required only affects
            # web-based commits, not PR merges. DCO enforcement for PRs is handled by
            # status checks/apps, not repository settings.

            # Check the PR's merge status through the API
            pr_data = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_number}"
            )

            if isinstance(pr_data, dict):
                mergeable_state = pr_data.get("mergeable_state", "unknown")
                mergeable = pr_data.get("mergeable")

                self.log.debug(
                    f"PR {owner}/{repo}#{pr_number} REST API status: mergeable={mergeable}, mergeable_state={mergeable_state}"
                )

                # Check for specific blocking conditions that indicate protection rules
                if mergeable_state == "blocked" and mergeable is False:
                    return False, "branch protection rules prevent merge"
                elif mergeable_state == "dirty":
                    return False, "merge conflicts"
                elif mergeable_state == "behind":
                    if not self.fix_out_of_date:
                        return (
                            False,
                            "PR is behind base branch and --no-fix option is set",
                        )
                    # Otherwise it's fixable
                elif mergeable is False and mergeable_state in ["unknown", "blocked"]:
                    # This often indicates hidden branch protection rules
                    return (
                        False,
                        "cannot update protected ref - organization or branch protection rules prevent merge",
                    )

            return True, "merge capability test passed"

        except Exception as e:
            error_msg = str(e)
            self.log.debug(
                f"Exception in _test_merge_capability for {owner}/{repo}#{pr_number}: {error_msg}"
            )

            # Look for specific DCO-related errors in the GitHub API response
            # DCO errors typically come as 422 validation errors with specific messages
            is_dco_error = False
            if "422" in error_msg and (
                "commit signoff required" in error_msg.lower()
                or "commits must have verified signatures" in error_msg.lower()
                or (
                    "dco" in error_msg.lower()
                    and ("required" in error_msg.lower() or "sign" in error_msg.lower())
                )
            ):
                is_dco_error = True
            elif "commit signoff required" in error_msg.lower():
                # Catch DCO errors that don't include status codes
                is_dco_error = True

            if is_dco_error:
                # This error comes from GitHub API, not our code - but these PRs are actually mergeable
                # The DCO requirement doesn't apply to API merges, only web-based commits
                self.log.info(
                    f"Ignoring DCO-related error for {owner}/{repo}#{pr_number} - API merges are allowed"
                )
                return True, "DCO enforcement not applicable to API merges"

            if (
                "protected ref" in error_msg.lower()
                or "cannot update" in error_msg.lower()
            ):
                return (
                    False,
                    "cannot update protected ref - organization or branch protection rules prevent merge",
                )
            elif "403" in error_msg:
                return (
                    False,
                    "insufficient permissions or branch protection rules prevent merge",
                )
            elif "405" in error_msg:
                return False, "merge method not allowed by repository settings"
            else:
                # Unknown error during test - assume it's mergeable
                self.log.debug(f"Test merge capability failed with unknown error: {e}")
                return True, "test merge capability failed - assuming mergeable"

    def get_results_summary(self) -> dict[str, Any]:
        """
        Get a summary of merge results.

        Returns:
            Dictionary with merge statistics
        """
        if not self._results:
            return {
                "total": 0,
                "merged": 0,
                "failed": 0,
                "skipped": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
            }

        total = len(self._results)
        merged = sum(1 for r in self._results if r.status == MergeStatus.MERGED)
        failed = sum(1 for r in self._results if r.status == MergeStatus.FAILED)
        skipped = sum(1 for r in self._results if r.status == MergeStatus.SKIPPED)

        success_rate = (merged / total) * 100 if total > 0 else 0.0
        average_duration = (
            sum(r.duration for r in self._results) / total if total > 0 else 0.0
        )

        return {
            "total": total,
            "merged": merged,
            "failed": failed,
            "skipped": skipped,
            "success_rate": success_rate,
            "average_duration": average_duration,
            "results": self._results,
        }

    def get_failed_prs(self) -> list[MergeResult]:
        """
        Get list of failed merge results.

        Returns:
            List of MergeResult objects that failed
        """
        return [r for r in self._results if r.status == MergeStatus.FAILED]

    def get_successful_prs(self) -> list[MergeResult]:
        """
        Get list of successful merge results.

        Returns:
            List of MergeResult objects that were merged successfully
        """
        return [r for r in self._results if r.status == MergeStatus.MERGED]
