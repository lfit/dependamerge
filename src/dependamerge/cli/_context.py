# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The state shared by every step of a GitHub merge run.

``_MergeContext`` carries the command's options plus the objects and
findings each step contributes, so the steps exchange one value rather
than a growing parameter list.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
import dependamerge.cli as _pkg

from ..models import ComparisonResult, PullRequestInfo
from ..progress_tracker import MergeProgressTracker
from ._app import DEFAULT_MAX_WAIT


@dataclass
class _MergeContext:
    """Shared state threaded through the merge sub-routines."""

    # CLI parameters
    pr_url: str
    no_confirm: bool
    similarity_threshold: float
    merge_method: str
    token: str | None
    override: str | None
    no_fix: bool
    merge_timeout: float
    show_progress: bool
    debug_matching: bool
    dismiss_copilot: bool
    force: str
    verbose: bool
    no_netrc: bool
    netrc_file: Path | None
    netrc_optional: bool
    github2gerrit_mode: str
    include_human_prs: bool = False
    # Optional behaviour flag; defaulted so a context can be built
    # without every construction site opting in explicitly.
    fix_semantic_title: bool = True
    rebase_local: bool = True
    # Dry-run: perform the full analysis and preview but never merge,
    # approve, rebase, or close anything.  Because no write occurs, the
    # write-permission pre-flight check is skipped so the command can run
    # under a read-only token (e.g. in CI).  Implies preview-only and
    # suppresses the interactive confirmation prompt.
    dry_run: bool = False
    # Owner-wide global wait ceiling (seconds).  Default 900 (15 min);
    # 0 = fire-and-forget (arm auto-merge, report pending, never block).
    # Applies to owner/user-wide runs; ignored for single-PR and
    # single-repository merges.
    max_wait: float = DEFAULT_MAX_WAIT

    # Derived / mutable state
    github_client: _pkg.GitHubClient | None = None
    # The GitHub host this run addresses, taken from the parsed target.
    # Empty until the target is parsed, at which point every client and
    # service built from this context inherits it --- a GitHub
    # Enterprise Server install serves its API from different base URLs,
    # so a run that loses the host silently addresses github.com.
    host: str = ""
    owner: str = ""
    repo_name: str = ""
    pr_number: int = 0
    comparator: _pkg.PRComparator | None = None
    # Worker count for the owner-wide striped merge, derived from the
    # number of distinct repositories in scope.  Carried on the context so
    # the preview/confirm handoff does not have to thread it through as a
    # parameter of its own.
    merge_concurrency: int = 1
    source_pr: PullRequestInfo | None = None
    progress_tracker: MergeProgressTracker | None = None
    all_similar_prs: list[tuple[PullRequestInfo, ComparisonResult]] = field(
        default_factory=list
    )
