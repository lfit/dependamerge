<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2025 The Linux Foundation -->

# Gerrit Support Implementation Plan

This document outlines the implementation plan for adding Gerrit support to
the `dependamerge` tool. The goal: enable bulk review and submission of
similar Gerrit changes, mirroring the existing GitHub PR functionality.

## Overview

The `dependamerge` tool supports bulk merging of similar GitHub pull
requests. This enhancement extends that capability to Gerrit Code Review
servers, allowing users to:

1. Parse and check Gerrit change URLs
2. Fetch change details from Gerrit REST API
3. Scan all open changes on a Gerrit server
4. Match similar changes using the existing comparison engine
5. Approve (+2 Code-Review) and submit matched changes

## Target URL Format

Gerrit URLs follow this pattern:

```text
https://gerrit.linuxfoundation.org/infra/c/releng/gerrit_to_platform/+/74080
       |__________________________|_____|___________________________|  |_____|
                host              base_path       project              change#
```

Example invocation:

```sh
dependamerge merge --no-confirm \
    https://gerrit.linuxfoundation.org/infra/c/releng/gerrit_to_platform/+/74080
```

---

## Phase 1: URL Detection and Parsing

**Goal:** Create a unified URL parser that distinguishes GitHub from Gerrit
URLs and extracts the necessary components.

### Phase 1 Implementation Steps

1. Create `src/dependamerge/url_parser.py` module
2. Add `ChangeSource` enum with values `GITHUB` and `GERRIT`
3. Add `ParsedUrl` dataclass with fields:
   - `source: ChangeSource`
   - `host: str`
   - `base_path: str | None` (Gerrit-specific)
   - `project: str` (Gerrit) or `owner/repo` (GitHub)
   - `change_number: int`
4. Add `parse_change_url(url: str) -> ParsedUrl` function
5. Add detection heuristics:
   - Contains `github.com` or `github` in host → GitHub
   - Contains `/pull/` in path → GitHub
   - Contains `/c/` and `/+/` in path → Gerrit
   - Contains `gerrit` in host → Gerrit

### Phase 1 Tests

- `tests/test_url_parser.py`
  - Test GitHub URL parsing (standard, enterprise)
  - Test Gerrit URL parsing (with/without base path)
  - Test invalid URL handling
  - Test edge cases (trailing slashes, query params)

### Phase 1 Dependencies

- None (pure Python parsing)

---

## Phase 2: Gerrit REST Client Module

**Goal:** Import and adapt Gerrit REST client code from `github2gerrit-action`
for use in `dependamerge`.

### Source Files (from github2gerrit-action)

The following modules provide proven Gerrit REST functionality:

- `gerrit_rest.py` - REST client with retry/timeout handling
- `gerrit_urls.py` - URL builder with base path discovery
- `gerrit_query.py` - Change query and GerritChange dataclass

### Phase 2 Implementation Steps

1. Create `src/dependamerge/gerrit/` package
2. Copy and adapt `gerrit_rest.py` → `gerrit/client.py`
   - Rename `GerritRestClient` for clarity
   - Simplify retry framework (reuse existing `tenacity` dependency)
   - Remove github2gerrit-specific logging
3. Copy and adapt `gerrit_urls.py` → `gerrit/urls.py`
   - Keep `GerritUrlBuilder` and `create_gerrit_url_builder`
   - Keep base path discovery logic
4. Create `gerrit/__init__.py` with public exports
5. Add `pygerrit2>=2.0.15` dependency to `pyproject.toml`

### Authentication Support

Add dual authentication sources:

1. **Environment variables:**
   - `GERRIT_USERNAME` or `GERRIT_HTTP_USER`
   - `GERRIT_PASSWORD` or `GERRIT_HTTP_PASSWORD`

2. **CLI arguments:**
   - `--gerrit-username TEXT`
   - `--gerrit-password TEXT`

### Phase 2 Tests

- `tests/test_gerrit_client.py`
  - Test REST client initialization
  - Test XSSI guard stripping
  - Test retry behavior (mock HTTP errors)
  - Test authentication header construction

---

## Phase 3: Gerrit Data Models

**Goal:** Create Pydantic models for Gerrit changes that parallel the existing
GitHub `PullRequestInfo` model.

### Phase 3 Implementation Steps

1. Create `src/dependamerge/gerrit/models.py`
2. Add models:

```python
class GerritFileChange(BaseModel):
    """Represents a file change in a Gerrit change."""
    filename: str
    status: str  # ADDED, MODIFIED, DELETED, RENAMED
    lines_inserted: int
    lines_deleted: int


class GerritChangeInfo(BaseModel):
    """Represents a Gerrit change (parallels PullRequestInfo)."""
    number: int
    change_id: str  # I-prefixed Change-Id
    subject: str  # First line of commit message
    message: str | None  # Full commit message
    owner: str  # Author username
    project: str
    branch: str
    status: str  # NEW, MERGED, ABANDONED
    submittable: bool
    mergeable: bool | None
    files_changed: list[GerritFileChange]
    current_revision: str
    url: str
    created: str
    updated: str


class GerritComparisonResult(BaseModel):
    """Result of comparing two Gerrit changes."""
    is_similar: bool
    confidence_score: float
    reasons: list[str]
```

1. Add factory method `GerritChangeInfo.from_api_response(data: dict)` to
   parse Gerrit REST API responses

### Phase 3 Tests

- `tests/test_gerrit_models.py`
  - Test model construction
  - Test API response parsing
  - Test field checking

---

## Phase 4: Gerrit Service Layer

**Goal:** Create a service class that queries Gerrit for changes and handles
server-wide enumeration.

### Phase 4 Implementation Steps

1. Create `src/dependamerge/gerrit/service.py`
2. Add `GerritService` class:

```python
class GerritService:
    """Service for querying and operating on Gerrit changes."""

    def __init__(
        self,
        host: str,
        base_path: str | None = None,
        username: str | None = None,
        password: str | None = None,
        progress_tracker: Any | None = None,
    ) -> None: ...

    async def get_change_info(
        self, change_number: int
    ) -> GerritChangeInfo: ...

    async def get_open_changes(
        self, project: str | None = None,
        limit: int = 500,
    ) -> list[GerritChangeInfo]: ...

    async def get_all_projects(self) -> list[str]: ...

    async def find_similar_changes(
        self,
        source_change: GerritChangeInfo,
        comparator: Any,
        only_automation: bool = True,
    ) -> list[tuple[GerritChangeInfo, GerritComparisonResult]]: ...
```

1. Add Gerrit REST API queries:
   - `GET /changes/{id}?o=CURRENT_REVISION&o=CURRENT_FILES&o=DETAILED_LABELS`
   - `GET /changes/?q=status:open&o=CURRENT_REVISION&o=CURRENT_FILES`
   - `GET /projects/` for project enumeration

2. Add pagination support for large result sets

### Phase 4 Tests

- `tests/test_gerrit_service.py`
  - Test change info fetching (mock responses)
  - Test open changes enumeration
  - Test project listing
  - Test pagination handling

---

## Phase 5: Change Comparator Adaptation

**Goal:** Extend the existing `PRComparator` to work with Gerrit changes.

### Phase 5 Implementation Steps

1. Update `src/dependamerge/pr_comparator.py`:
   - Make comparison methods accept both `PullRequestInfo` and
     `GerritChangeInfo` types
   - Create type alias `ChangeInfo = PullRequestInfo | GerritChangeInfo`
   - Update `_is_automation_pr()` to detect Gerrit automation patterns

2. Create `src/dependamerge/gerrit/comparator.py` with Gerrit-specific logic:
   - Detect automation authors (dependabot, pre-commit-ci patterns)
   - Compare subjects (parallels PR titles)
   - Compare commit message bodies
   - Compare file changes

3. Map Gerrit fields to comparison:

| GitHub Field  | Gerrit Field  | Comparison Method   |
|---------------|---------------|---------------------|
| title         | subject       | `_compare_titles()` |
| body          | message       | `_compare_bodies()` |
| author        | owner         | Author match        |
| files_changed | files_changed | `_compare_files()`  |

1. Automation detection patterns for Gerrit:
   - `dependabot/` in project or topic
   - `pre-commit-ci` in commit message
   - Known bot usernames in owner field

### Phase 5 Tests

- `tests/test_gerrit_comparator.py`
  - Test subject comparison
  - Test file change comparison
  - Test automation detection
  - Test cross-platform comparison (if needed)

---

## Phase 6: Gerrit Submit Operations

**Goal:** Add approval (+2 Code-Review) and submit operations for
Gerrit changes.

### Phase 6 Implementation Steps

1. Create `src/dependamerge/gerrit/submit_manager.py`
2. Add `GerritSubmitManager` class:

```python
class SubmitStatus(Enum):
    PENDING = "pending"
    REVIEWING = "reviewing"
    REVIEWED = "reviewed"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    FAILED = "failed"
    BLOCKED = "blocked"


class SubmitResult:
    change_info: GerritChangeInfo
    status: SubmitStatus
    error: str | None
    duration: float


class GerritSubmitManager:
    """Manages parallel approval and submission of Gerrit changes."""

    async def submit_changes_parallel(
        self,
        changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
    ) -> list[SubmitResult]: ...

    async def _review_change(
        self, change_number: int, labels: dict[str, int]
    ) -> bool: ...

    async def _submit_change(
        self, change_number: int
    ) -> bool: ...
```

1. Add Gerrit REST API calls:
   - `POST /changes/{id}/revisions/current/review` with
     `{"labels": {"Code-Review": 2}}`
   - `POST /changes/{id}/submit`

2. Handle submit requirements:
   - Check `submittable` field before attempting submit
   - Handle "Needs Verified" label if the project uses it
   - Report blocking conditions in output

3. Add retry logic for transient failures

### Phase 6 Tests

- `tests/test_gerrit_submit_manager.py`
  - Test review operation (mock API)
  - Test submit operation (mock API)
  - Test parallel submission
  - Test error handling (not submittable, auth failures)

---

## Phase 7: CLI Integration

**Goal:** Update the `merge` command to support both GitHub and Gerrit URLs.

### Phase 7 Implementation Steps

1. Update `src/dependamerge/cli.py`:

```python
@app.command()
def merge(
    change_url: str = typer.Argument(
        ..., help="GitHub PR or Gerrit change URL"
    ),
    # Existing GitHub options...
    gerrit_username: str | None = typer.Option(
        None,
        "--gerrit-username",
        envvar="GERRIT_USERNAME",
        help="Gerrit HTTP username",
    ),
    gerrit_password: str | None = typer.Option(
        None,
        "--gerrit-password",
        envvar="GERRIT_PASSWORD",
        help="Gerrit HTTP password",
    ),
    ...
) -> None:
```

1. Add URL type detection at command entry:

```python
parsed = parse_change_url(change_url)
if parsed.source == ChangeSource.GERRIT:
    _merge_gerrit_changes(parsed, ...)
else:
    _merge_github_prs(parsed, ...)
```

1. Create `_merge_gerrit_changes()` function with parallel structure to
   existing GitHub flow

2. Update help text and argument descriptions

3. Ensure consistent output formatting between platforms

### CLI Help Changes

Update the help text for the `merge` command:

```text
Usage: dependamerge merge [OPTIONS] CHANGE_URL

  Bulk approve/merge changes across a GitHub organization or Gerrit server.

Arguments:
  CHANGE_URL  GitHub PR URL or Gerrit change URL  [required]
```

### Phase 7 Tests

- `tests/test_cli_gerrit.py`
  - Test Gerrit URL detection in merge command
  - Test credential handling (CLI args, env vars)
  - Test dry-run mode for Gerrit

---

## Phase 8: Comprehensive Unit Tests

**Goal:** Ensure robust test coverage for all new functionality.

### Test Files Structure

```text
tests/
├── test_url_parser.py          # URL detection and parsing
├── test_gerrit_client.py       # REST client operations
├── test_gerrit_models.py       # Model construction/parsing
├── test_gerrit_service.py      # Service layer operations
├── test_gerrit_comparator.py   # Change comparison logic
├── test_gerrit_submit_manager.py  # Submit operations
├── test_cli_gerrit.py          # CLI Gerrit integration
└── conftest.py                 # Shared fixtures
```

### Test Fixtures (conftest.py additions)

```python
@pytest.fixture
def mock_gerrit_change() -> dict:
    """Sample Gerrit change API response."""
    return {
        "_number": 74080,
        "change_id": "I1234567890abcdef...",
        "subject": "Chore: Bump actions/checkout from 4.1.0 to 4.2.0",
        "status": "NEW",
        # ... full response structure
    }


@pytest.fixture
def gerrit_client(monkeypatch) -> GerritRestClient:
    """Mocked Gerrit REST client."""
    # ... setup mocked HTTP responses
```

### Coverage Targets

- At least 80% line coverage for new modules
- 100% coverage for URL parsing logic
- 100% coverage for model parsing logic

---

## Phase 9: Integration Testing and Final Checks

**Goal:** Ensure the implementation works end-to-end and passes all linting.

### Phase 9 Checks

1. **Run pre-commit hooks:**

   ```sh
   pre-commit run -a
   ```

2. **Check type hints:**
   - All new code must pass `mypy` strict mode
   - All new code must pass `basedpyright`
   - Use explicit type annotations on all functions

3. **Line wrapping:**
   - All code wrapped at 80 characters
   - All documentation wrapped at 80 characters
   - Configure `ruff` line-length appropriately

4. **write-good compliance:**
   - Avoid passive voice in documentation
   - Avoid weasel words
   - Use active, direct language

5. **REUSE compliance:**
   - Add license headers to all new files
   - Update `REUSE.toml` if needed

### Manual Testing Checklist

- [ ] Parse Gerrit URL formats
- [ ] Connect to test Gerrit server
- [ ] Fetch change details
- [ ] List open changes
- [ ] Compare changes for similarity
- [ ] Review a change (+2)
- [ ] Submit a change
- [ ] Handle authentication errors
- [ ] Handle network errors with retry
- [ ] Preview mode shows correct output

---

## New Python Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing deps ...
    "pygerrit2>=2.0.15",  # Gerrit REST API client
]
```

No new development dependencies needed.

---

## File Structure Summary

New files to create:

```text
src/dependamerge/
├── url_parser.py              # URL detection and parsing
├── gerrit/
│   ├── __init__.py            # Package exports
│   ├── client.py              # REST client (from github2gerrit)
│   ├── urls.py                # URL builder (from github2gerrit)
│   ├── models.py              # Pydantic models
│   ├── service.py             # Service layer
│   ├── comparator.py          # Change comparison
│   └── submit_manager.py      # Submit operations

tests/
├── test_url_parser.py
├── test_gerrit_client.py
├── test_gerrit_models.py
├── test_gerrit_service.py
├── test_gerrit_comparator.py
├── test_gerrit_submit_manager.py
└── test_cli_gerrit.py
```

Files to update:

```text
src/dependamerge/
├── cli.py                     # Add Gerrit support
├── pr_comparator.py           # Generalize for both platforms

pyproject.toml                 # Add pygerrit2 dependency
```

---

## Implementation Timeline

| Phase     | Description               | Estimated Effort |
|-----------|---------------------------|------------------|
| 1         | URL Detection and Parsing | 2-3 hours        |
| 2         | Gerrit REST Client        | 3-4 hours        |
| 3         | Gerrit Data Models        | 2-3 hours        |
| 4         | Gerrit Service Layer      | 4-5 hours        |
| 5         | Change Comparator         | 2-3 hours        |
| 6         | Submit Operations         | 4-5 hours        |
| 7         | CLI Integration           | 3-4 hours        |
| 8         | Unit Tests                | 4-6 hours        |
| 9         | Integration/Final Checks  | 2-3 hours        |
| **Total** |                           | **26-36 hours**  |

---

## Risk Mitigation

### Potential Risks

1. **Gerrit API variations:** Different Gerrit versions may have API
   differences
   - *Mitigation:* Test against two or three Gerrit instances; use defensive
     parsing

2. **Authentication complexity:** Some Gerrit servers use SSO or special
   auth
   - *Mitigation:* Support standard HTTP Basic Auth; document limitations

3. **Submit requirements:** Gerrit projects may have complex submit rules
   - *Mitigation:* Check `submittable` field; report requirements in output

4. **Rate limiting:** Large Gerrit servers may rate limit API calls
   - *Mitigation:* Use exponential backoff; reuse retry framework

### Fallback Strategy

If pygerrit2 proves problematic, the implementation can fall back to direct
HTTP requests using `httpx` (already a dependency), as demonstrated in the
`github2gerrit-action` codebase.

---

## References

### Source Repositories

- [github2gerrit-action](https://github.com/lfreleng-actions/github2gerrit-action)
  - `gerrit_rest.py` - REST client implementation
  - `gerrit_urls.py` - URL construction utilities
  - `gerrit_query.py` - Change query utilities

- [tag-validate-action](https://github.com/lfreleng-actions/tag-validate-action)
  - `gerrit_keys.py` - Gerrit authentication patterns

### Gerrit API Documentation

- [Gerrit REST API](https://gerrit-review.googlesource.com/Documentation/rest-api.html)
- [Changes Endpoint](https://gerrit-review.googlesource.com/Documentation/rest-api-changes.html)
- [Set Review](https://gerrit-review.googlesource.com/Documentation/rest-api-changes.html#set-review)
- [Submit Change](https://gerrit-review.googlesource.com/Documentation/rest-api-changes.html#submit-change)

### Environment Variables

| Variable               | Description                   | Auth Use |
|------------------------|-------------------------------|----------|
| `GERRIT_USERNAME`      | Gerrit HTTP username          | Yes      |
| `GERRIT_PASSWORD`      | Gerrit HTTP password          | Yes      |
| `GERRIT_HTTP_USER`     | Alternative username variable | Yes      |
| `GERRIT_HTTP_PASSWORD` | Alternative password variable | Yes      |

Note: Authentication enables review and submit operations.
