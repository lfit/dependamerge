<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2025 The Linux Foundation
-->

# Dependamerge

Find blocked pull requests in GitHub organizations and automatically merge
similar pull requests across GitHub organizations, supporting both automation
tools (like Dependabot, pre-commit.ci, Renovate) and regular GitHub users.

## Overview

Dependamerge provides two main functions:

1. **Finding Blocked PRs**: Check entire GitHub organizations to identify
   pull requests with conflicts, failing checks, or other blocking issues
2. **Automated Merging**: Analyze a source pull request and find similar pull
   requests across all repositories in the same GitHub organization, then
   automatically approve and merge the matching PRs

This saves time on routine dependency updates, maintenance tasks, and
coordinated changes across all repositories while providing visibility into
unmergeable PRs that need attention.

**Works with any pull request** regardless of author, automation tool, or origin.

## Features

### Finding Blocked PRs

- **Comprehensive PR Analysis**: Checks all repositories in a GitHub
  organization for unmergeable pull requests
- **Blocking Reason Detection**: Identifies specific reasons preventing PR
  merges (conflicts, failing checks, blocked reviews)
- **Copilot Integration**: Counts unresolved GitHub Copilot feedback comments
  (column shown when present)
- **Smart Filtering**: Excludes standard code review requirements, focuses on
  technical blocking issues
- **Detailed Reporting**: Provides comprehensive tables and summaries of
  problematic PRs
- **Real-time Progress**: Live progress display shows checking status and
  current operations

### Automated Merging

- **Universal PR Support**: Works with any pull request regardless of author
  or automation tool
- **Smart Matching**: Uses content similarity algorithms to match related PRs
  across repositories
- **Bulk Operations**: Approve and merge related similar PRs with a single command
- **Security Features**: SHA-based authentication for non-automation PRs
  ensures authorized bulk merges
- **Dry Run Mode**: Preview what changes will apply without modifications

### General Features

- **Rich CLI Output**: Beautiful terminal output with progress indicators and tables
- **Real-time Progress**: Live progress updates for both checking and merge operations
- **Output Formats**: Support for table and JSON output formats
- **Error Handling**: Graceful handling of API rate limits and repository
  access issues

## Supported Pull Requests

- Any pull request from any author
- Manual pull requests from developers
- Automation tool pull requests (Dependabot, Renovate, etc.)
- Bot-generated pull requests
- Coordinated changes across repositories

## Installation

```bash
# Install from source
git clone <repository-url>
cd dependamerge
pip install -e .

# Or install dependencies directly
pip install typer requests PyGithub rich pydantic
```

## Authentication

### Quick Start

**You need a GitHub personal access token with these permissions:**

- **Organization access**: Read organization repositories
- **Repository access**: Read pull requests, contents, metadata, and checks
- **Pull request management**: Write access to approve and merge pull requests

### Token Types

GitHub offers two types of personal access tokens: **fine-grained**
(recommended) and **classic**.

### Fine-Grained Personal Access Tokens (Recommended)

Fine-grained tokens provide more precise control and better security.
Create one at:
GitHub Settings → Developer settings → Personal access tokens → Fine-grained tokens

#### Required Repository Permissions

Your token needs these permissions on **all repositories** in the target organization:

- **Pull requests**: Read and Write
  - Read: View PR details, status, files, commits, and reviews
    (for finding blocked PRs and similarity matching)
  - Write: Approve and merge pull requests (for automated bulk operations)
- **Contents**: Read
  - Read repository files and commit information (for analyzing file changes in PRs)
- **Metadata**: Read
  - Basic repository information and access (required for all GitHub API operations)
- **Checks**: Read
  - View status checks and workflow results
    (for identifying blocked PRs due to failing checks)
- **Actions**: Read *(if using GitHub Actions for CI/CD)*
  - Read workflow run status and check results
    (for comprehensive status analysis)

#### Required Account Permissions

- **Organization permissions** → **Members**: Read
  - List repositories in the organization (required for organization-wide scanning)

#### Repository Access

Choose one of:

- **Selected repositories**: Grant access to specific repositories you want to
manage
- **All repositories**: Grant access to all repositories in organizations
you're a member of

*Note: The tool scans entire organizations, so it needs access to all
repositories you want to include in the analysis.*

### Classic Personal Access Tokens (Legacy)

If you prefer classic tokens, create one at:
GitHub Settings → Developer settings → Personal access tokens →
Tokens (classic)

Required scopes:

- `repo` (Full control of private repositories)
- `public_repo` (Access public repositories)
- `read:org` (Read organization membership and repositories)

### Setting Your Token

Set the token as an environment variable:

```bash
export GITHUB_TOKEN=your_token_here
```

Or pass it directly to the command using `--token`.

### Verifying Your Token

Before running operations, test your token permissions:

```bash
# Test basic access - should list organization repositories
dependamerge blocked myorganization --dry-run

# Test with a small operation first
dependamerge merge https://github.com/myorg/testrepo/pull/123 --dry-run
```

If these commands work without permission errors, your token is properly configured.

### Security Best Practices

- Use fine-grained tokens when possible for better security
- Set appropriate expiry dates (GitHub recommends 90 days or less)
- Store tokens securely (environment variables, secret managers)
- Never commit tokens to version control
- Rotate tokens periodically
- Use the fewest required permissions for your use case

## Usage

### Finding Blocked PRs (New Feature)

Find blocked pull requests in an entire GitHub organization:

```bash
# Basic organization check for blocked PRs
dependamerge blocked myorganization

# Check with JSON output
dependamerge blocked myorganization --format json

# Disable real-time progress display
dependamerge blocked myorganization --no-progress
```

The blocked command will:

- Analyze all repositories in the organization
- Identify PRs with technical blocking issues
- Report blocking reasons (merge conflicts, failing workflows, etc.)
- Count unresolved GitHub Copilot feedback comments (displayed when present)
- Exclude standard code review requirements from blocking reasons

### Basic Pull Request Merging

For any pull request from any author:

```bash
dependamerge merge https://github.com/lfreleng-actions/python-project-name-action/pull/22
```

### Optional Security Validation

For extra security, you can use the --override flag with SHA-based validation:

```bash
dependamerge merge https://github.com/owner/repo/pull/123 \
  --override a1b2c3d4e5f6g7h8
```

The SHA hash derives from:

- The PR author's GitHub username
- The first line of the commit message
- This provides an extra layer of validation for sensitive operations

### Basic Merge Usage

```bash
dependamerge merge \
  https://github.com/lfreleng-actions/python-project-name-action/pull/22
```

### Dry Run (Preview Mode)

```bash
dependamerge merge https://github.com/owner/repo/pull/123 --dry-run
```

### Custom Merge Options

```bash
dependamerge merge https://github.com/owner/repo/pull/123 \
  --threshold 0.9 \
  --merge-method squash \
  --fix \
  --no-progress \
  --token your_github_token
```

### Command Options

#### Blocked Command Options

- `--format TEXT`: Output format - table or json (default: table)

- `--progress/--no-progress`: Show real-time progress updates (default: progress)
- `--token TEXT`: GitHub token (alternative to GITHUB_TOKEN env var)

#### Merge Command Options

- `--dry-run`: Show what changes will apply without making them
- `--threshold FLOAT`: Similarity threshold for matching PRs (0.0-1.0,
  default: 0.8)
- `--merge-method TEXT`: Merge method - merge, squash, or rebase (default: merge)
- `--fix`: Automatically fix out-of-date branches before merging
- `--progress/--no-progress`: Show real-time progress updates (default: progress)
- `--token TEXT`: GitHub token (alternative to GITHUB_TOKEN env var)
- `--override TEXT`: SHA hash for extra security validation

## How It Works

### Pull Request Processing

1. **Parse Source PR**: Analyzes the provided pull request URL and extracts metadata
2. **Organization Check**: Lists all repositories in the same GitHub organization
3. **PR Discovery**: Finds all open pull requests in each repository
4. **Content Matching**: Compares PRs using different similarity metrics:
   - Title similarity (normalized to remove version numbers)
   - File change patterns
   - Author matching
5. **Optional Validation**: If `--override` provided, validates SHA for extra security
6. **Approval & Merge**: For matching PRs above the threshold:
   - Adds an approval review
   - Merges the pull request
7. **Source PR Merge**: Merges the original source PR that served as the baseline

## Similarity Matching

The tool uses different algorithms to determine if PRs are similar:

### Title Normalization

- Removes version numbers (e.g., "1.2.3", "v2.0.0")
- Removes commit hashes
- Removes dates
- Normalizes whitespace

### File Change Analysis

- Compares changed filenames using Jaccard similarity
- Accounts for path normalization
- Ignores version-specific filename differences

### Confidence Scoring

Combines different factors:

- Title similarity score
- File change similarity score
- Author matching (same automation tool)

## Examples

### Example: Finding Blocked PRs

```bash
# Check organization for blocked PRs
dependamerge blocked myorganization

# Get detailed JSON output
dependamerge blocked myorganization --format json > unmergeable_prs.json

# Check without progress display
dependamerge blocked myorganization --no-progress
```

### Example: Automated Merging

#### Dependency Update PR

```bash
# Merge a dependency update across all repos
dependamerge merge https://github.com/myorg/repo1/pull/45
```

#### Documentation Update PR

```bash
# Merge documentation updates
dependamerge merge https://github.com/myorg/repo1/pull/12 --threshold 0.85
```

#### Feature PR with Security Validation

```bash
# Merge with optional security validation
dependamerge merge https://github.com/myorg/repo1/pull/89 \
  --override f1a2b3c4d5e6f7g8
```

#### Dry Run with Fix Option

```bash
# See what changes will apply and automatically fix out-of-date branches
dependamerge merge https://github.com/myorg/repo1/pull/78 \
  --dry-run --fix --threshold 0.9 --progress
```

## Safety Features

### For All PRs

- **Mergeable Check**: Verifies PRs are in a mergeable state before attempting merge
- **Auto-Fix**: Automatically update out-of-date branches when using `--fix` option
- **Detailed Status**: Shows specific reasons preventing PR merges (conflicts,
  blocked by checks, etc.)
- **Similarity Threshold**: Configurable confidence threshold prevents incorrect
  matches
- **Dry Run Mode**: Always test with `--dry-run` first
- **Detailed Logging**: Shows which PRs match and why they match

### Security for All PRs

- **SHA-Based Validation**: Provides unique SHA hash for security
- **Author Isolation**: When using SHA validation, processes PRs from the same
  author as source PR
- **Commit Binding**: SHA changes if commit message changes, preventing replay
  attacks
- **Cross-Author Protection**: When enabled, one author's SHA cannot work for
  another author's PRs

## Enhanced URL Support

The tool now supports GitHub PR URLs with path segments:

```bash
# These URL formats now work:
dependamerge https://github.com/owner/repo/pull/123
dependamerge https://github.com/owner/repo/pull/123/
dependamerge https://github.com/owner/repo/pull/123/files
dependamerge https://github.com/owner/repo/pull/123/commits
dependamerge https://github.com/owner/repo/pull/123/files/diff
```

This enhancement allows you to copy URLs directly from GitHub's PR pages
without worrying about the specific tab you're viewing.

## Development

### Setup Development Environment

```bash
git clone <repository-url>
cd dependamerge
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black src tests

# Lint code
flake8 src tests

# Type checking
mypy src
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Troubleshooting

### Common Issues

#### Authentication Error

```text
Error: GitHub token needed
```

Solution: Set `GITHUB_TOKEN` environment variable or use `--token` flag.

#### Permission Errors

**Organization Access Error:**

```text
Failed to fetch organization repositories
```

Solutions:

- **Fine-grained tokens**: Ensure "Members: Read" permission under Organization permissions
- **Classic tokens**: Ensure your token has `read:org` scope
- Verify you're a member of the organization or have appropriate access

**Repository Access Error:**

```text
403 Forbidden when accessing repository
```

Solutions:

- **Fine-grained tokens**: Add the repository to your token's
  "Selected repositories" or use "All repositories"
- **Classic tokens**: Ensure your token has `repo` scope for private repos
  or `public_repo` for public repos
- Verify the repository exists and you have access to it

**Pull Request Operation Error:**

```text
403 Forbidden when approving/merging pull request
```

Solutions:

- **Fine-grained tokens**: Enable "Pull requests: Write" permission
- **Classic tokens**: Ensure your token has `repo` scope
- Verify you have push/admin access to the target repository
- Check if branch protection rules require specific reviewers

**GraphQL Permission Error:**

```text
Resource not accessible by integration
```

Solutions:

- Verify you have granted all required repository permissions
- For fine-grained tokens, ensure the token has access to all repositories
  in the organization
- Some GraphQL operations require higher permissions than REST API equivalents

#### No Similar PRs Found

- Check that other repositories have open automation PRs
- Try lowering the similarity threshold with `--threshold 0.7`
- Use `--dry-run` to see detailed matching information

#### Merge Failures

- Ensure PRs are in mergeable state (no conflicts)
- Check that you have write permissions to the target repositories
- Verify the repository settings permit the merge method

### Getting Help

- Check the command help: `dependamerge --help`
- Get specific command help:
    `dependamerge blocked --help` or `dependamerge merge --help`
- Enable verbose output with environment variables
- Review the similarity scoring in dry-run mode for merge operations
- Use JSON output format for programmatic processing of blocked PR results

## GitHub API Operations

For transparency, here are the specific GitHub API operations that dependamerge
performs with your token:

### Organization Scanning Operations

**GraphQL Queries:**

- List all repositories in an organization
- Retrieve open pull requests across all repositories
- Fetch PR metadata (title, author, state, files changed, status checks)
- Get commit information and check run results

**REST API Calls:**

- `GET /orgs/{org}/repos` - List organization repositories (paginated)

### Pull Request Analysis Operations

**REST API Calls:**

- `GET /repos/{owner}/{repo}/pulls/{number}` - Get detailed PR information
- `GET /repos/{owner}/{repo}/pulls/{number}/files` - Get files changed in PR
- `GET /repos/{owner}/{repo}/pulls/{number}/commits` - Get PR commit history
- `GET /repos/{owner}/{repo}/pulls/{number}/reviews` - Get existing PR reviews
- `GET /repos/{owner}/{repo}/commits/{sha}/check-runs` - Get status check results

### Pull Request Management Operations

**REST API Calls:**

- `POST /repos/{owner}/{repo}/pulls/{number}/reviews` - Approve pull requests
- `PUT /repos/{owner}/{repo}/pulls/{number}/merge` - Merge pull requests
- `PUT /repos/{owner}/{repo}/pulls/{number}/update-branch` - Update out-of-date branches

### Data Usage

The tool:

- **Reads** organization and repository information
- **Reads** pull request details, files, commits, and status
- **Writes** pull request approvals and merges
- **Does not** access repository code content beyond PR diffs
- **Does not** change repository settings or configurations
- **Does not** access user personal information beyond public profile data

### Rate Limiting

The tool implements:

- Automatic rate limit detection and backoff
- Concurrent request limiting to respect GitHub's API limits
- Progress tracking to show operation status during long-running scans

## Security Considerations

- Store GitHub tokens securely (environment variables, not in code)
- Use tokens with minimal required permissions
- Rotate access tokens periodically
- Review PR changes in dry-run mode first
- Be cautious with low similarity thresholds
