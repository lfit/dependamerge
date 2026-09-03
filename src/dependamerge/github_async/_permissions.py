# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Permission-failure classification and token scope discovery.

The operation/permission requirement table, the translation of an HTTP
failure into a :class:`PermissionError` carrying token-type guidance,
and the OAuth scope lookup that backs the workflow-scope check.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from urllib.parse import urlparse

from ._base import _GitHubAsyncBase
from ._errors import PermissionError

# Permission requirements mapping for operations
OPERATION_PERMISSIONS = {
    "list_repos": {
        "classic": "read:org scope",
        "fine_grained": "Organization members: Read access",
        "description": "List organization repositories",
    },
    "approve": {
        "classic": "repo scope",
        "fine_grained": "Pull requests: Read and write",
        "description": "Approve pull requests",
    },
    "merge": {
        "classic": "repo scope",
        "fine_grained": "Contents: Read and write",
        "description": "Merge pull requests",
    },
    "merge_workflow": {
        "classic": "workflow scope (in addition to repo)",
        "fine_grained": "Workflows: Read and write",
        "description": "Merge pull requests that modify GitHub Actions workflows",
    },
    "update_branch": {
        "classic": "repo scope",
        "fine_grained": "Contents: Read and write, Pull requests: Read and write",
        "description": "Update/rebase pull request branches",
    },
    "close": {
        "classic": "repo scope",
        "fine_grained": "Pull requests: Read and write",
        "description": "Close pull requests",
    },
    "branch_protection": {
        "classic": "repo scope",
        "fine_grained": "Administration: Read access",
        "description": "Read branch protection rules",
    },
    "checks": {
        "classic": "repo scope (or workflow for actions)",
        "fine_grained": "Actions: Read access, Workflows: Read access",
        "description": "Read status checks and workflow runs",
    },
}


def web_host_for(api_url: str) -> str:
    """Return the web host whose settings pages govern an API endpoint.

    Token guidance points at settings pages and at ``gh auth refresh``.
    Both are per-installation, so an Enterprise operator told to visit
    github.com is being sent to a site that knows nothing about their
    credentials.

    Args:
        api_url: The REST base URL the client addresses.

    Returns:
        The hostname to build settings URLs from.
    """
    # ``hostname`` rather than ``netloc``: the authority includes any
    # userinfo, so a caller-supplied
    # ``https://user:TOKEN@ghe.example.com/api/v3`` put that credential
    # into the settings URL and the ``gh -h`` argument this builds ---
    # both of which are printed to the terminal.  ``hostname`` excludes
    # userinfo by construction, and lowercases for free.
    host = (urlparse(api_url or "").hostname or "").lower()
    if not host:
        return "github.com"
    # Dotcom splits its API onto api.github.com; everywhere else the
    # API and the web UI share a host.
    if host == "api.github.com":
        return "github.com"
    return host


def _unauthorized_permission_error(
    operation: str, host: str = "github.com"
) -> PermissionError:
    """Build the 401 (expired/invalid token) permission error."""
    return PermissionError(
        operation=operation,
        message="Token authentication failed - token may be expired or invalid",
        token_type_guidance={
            "classic": f"Regenerate your token at: https://{host}/settings/tokens",
            "fine_grained": f"Check token expiration at: https://{host}/settings/personal-access-tokens",
            "fix": f"Run: gh auth refresh -h {host}",
        },
    )


def _forbidden_permission_error(
    error: Exception,
    error_str: str,
    operation: str,
    owner: str,
    repo: str,
    host: str = "github.com",
) -> PermissionError:
    """Classify a 403 (forbidden) failure into a permission error."""
    # Try to get more detailed error info from response
    response_text = ""
    response = getattr(error, "response", None)
    if response is not None:
        try:
            response_text = str(getattr(response, "text", "")).lower()
        except AttributeError:
            # Response object exposes no readable body; fall
            # back to the empty default and keep classifying.
            pass

    error_lower = error_str.lower()

    # Check for specific permission scenarios

    # 1. Workflow scope (already handled but included for completeness)
    if (
        "refusing to allow" in response_text
        and "workflow" in response_text
        and operation == "merge"
    ):
        perms = OPERATION_PERMISSIONS.get("merge_workflow", {})
        return PermissionError(
            operation="merge_workflow",
            message=f"Missing workflow permissions to merge PR in {owner}/{repo} that modifies GitHub Actions workflows",
            token_type_guidance={
                "classic": f"Add scope: {perms.get('classic', 'workflow')}",
                "fine_grained": f"Enable: {perms.get('fine_grained', 'Workflows: Read and write')}",
                "fix": f"Run: gh auth refresh -h {host} -s workflow",
            },
        )

    # 2. Fine-grained token repository scope
    if "resource not accessible" in response_text or "not in scope" in error_lower:
        return PermissionError(
            operation=operation,
            message=f"Repository {owner}/{repo} is not accessible with this token",
            token_type_guidance={
                "classic": "Token should have 'repo' scope for private repositories, or 'public_repo' for public repositories",
                "fine_grained": f"Add {owner}/{repo} to the token's repository access list at: https://{host}/settings/tokens",
                "fix": f"Edit your fine-grained token and add '{owner}/{repo}' to repository access",
            },
        )

    # 3. Operation-specific permission errors
    perms = OPERATION_PERMISSIONS.get(operation, {})
    if perms:
        location = f" in {owner}/{repo}" if owner and repo else ""
        return PermissionError(
            operation=operation,
            message=f"Insufficient permissions to {perms.get('description', operation)}{location}",
            token_type_guidance={
                "classic": f"Required scope: {perms.get('classic', 'repo')}",
                "fine_grained": f"Required permission: {perms.get('fine_grained', 'unknown')}",
                "fix": f"Update your token permissions at: https://{host}/settings/tokens",
            },
        )

    # 4. Generic 403
    return PermissionError(
        operation=operation,
        message=f"Permission denied for {operation} operation{' in ' + owner + '/' + repo if owner and repo else ''}",
        token_type_guidance={
            "classic": "Ensure token has 'repo' scope for full repository access",
            "fine_grained": "Check that token has appropriate permissions and repository access",
            "fix": f"Review and update token permissions at: https://{host}/settings/tokens",
        },
    )


def _approve_unprocessable_permission_error(
    error_str: str, operation: str
) -> PermissionError | None:
    """Classify a 422 raised by an approval attempt."""
    if "review cannot be requested from pull request author" in error_str.lower():
        return PermissionError(
            operation=operation,
            message="Cannot approve your own pull request",
            token_type_guidance={
                "classic": "GitHub does not allow self-approval of pull requests",
                "fine_grained": "GitHub does not allow self-approval of pull requests",
                "fix": "Request review from another team member",
            },
        )
    elif "unprocessable entity" in error_str.lower():
        return PermissionError(
            operation=operation,
            message="Pull request approval failed - repository may have approval restrictions",
            token_type_guidance={
                "classic": "Check repository settings for review requirements",
                "fine_grained": "Check repository settings for review requirements",
                "fix": "Contact repository administrator to review branch protection rules",
            },
        )
    return None


class _PermissionsMixin(_GitHubAsyncBase):
    """Permission-error parsing and scope lookups for ``GitHubAsync``."""

    def _parse_permission_error(
        self, error: Exception, operation: str, owner: str = "", repo: str = ""
    ) -> PermissionError | None:
        """Parse HTTP error to determine if it's a permission issue.

        Args:
            error: The exception that was raised
            operation: The operation being performed (e.g., 'approve', 'merge')
            owner: Repository owner (for context in error messages)
            repo: Repository name (for context in error messages)

        Returns:
            PermissionError if this is a permission issue, None otherwise
        """
        error_str = str(error)
        # Guidance points at settings pages and ``gh auth refresh``, both
        # of which are per-installation.  Derive them from the endpoint
        # this client actually addresses.
        host = web_host_for(self.api_url)

        # Check for 401 (unauthorized/expired token)
        if "401" in error_str or "Unauthorized" in error_str:
            return _unauthorized_permission_error(operation, host)

        # Check for 403 (forbidden/permission denied)
        if "403" in error_str or "Forbidden" in error_str:
            return _forbidden_permission_error(
                error, error_str, operation, owner, repo, host
            )

        # Check for 422 (unprocessable entity - often approval restrictions)
        if "422" in error_str and operation == "approve":
            return _approve_unprocessable_permission_error(error_str, operation)

        # Not a permission error we recognize
        return None

    async def get_token_scopes(self) -> set[str] | None:
        """Return the OAuth scopes granted to a classic personal access token.

        Classic PATs advertise their granted scopes in the
        ``X-OAuth-Scopes`` response header on every authenticated request.
        Fine-grained PATs and GitHub App installation tokens do **not** send
        this header — their permission model is per-resource and cannot be
        introspected this way.

        Returns:
            A ``set`` of scope strings for a classic PAT (possibly empty if
            the token was created with no scopes selected), or ``None`` when
            the token type does not expose scopes (fine-grained PAT / app
            token) or the lookup could not be performed.  Callers MUST treat
            ``None`` as "undeterminable", never as "no scopes granted".
        """
        if self._token_scopes_fetched:
            return self._token_scopes

        try:
            # Any authenticated REST endpoint echoes the header.
            # ``/rate_limit`` is the cheapest and is itself exempt from the
            # primary rate limit, so it never consumes quota.
            r = await self._request("GET", f"{self.api_url}/rate_limit")
        except Exception as e:
            # A transient probe failure must NOT be cached as
            # "undeterminable": doing so would let a one-off network error
            # suppress accurate scope diagnosis for the rest of the run
            # (a classic PAT that has ``workflow`` could still be reported
            # as missing it).  Leave the cache unset so a later call can
            # retry and produce an accurate result.
            self.log.debug("Could not determine token scopes: %s", e)
            return None

        raw = r.headers.get("X-OAuth-Scopes")
        if raw is None:
            # Header absent on a successful probe → fine-grained / app
            # token.  The scope set is genuinely undeterminable; cache it.
            self._token_scopes = None
        else:
            # Header present (possibly empty for a scope-less classic PAT).
            self._token_scopes = {s.strip() for s in raw.split(",") if s.strip()}
        self._token_scopes_fetched = True
        return self._token_scopes

    async def check_workflow_scope(self) -> bool | None:
        """Determine whether the token may update GitHub Actions workflows.

        Merging a PR that touches ``.github/workflows/**`` requires the
        classic ``workflow`` scope (or, for fine-grained PATs, the
        ``Workflows: Read and write`` permission).

        Returns:
            ``True``  — classic PAT that carries the ``workflow`` scope.
            ``False`` — classic PAT that is missing the ``workflow`` scope.
            ``None``  — the token type cannot be introspected (fine-grained
            PAT / app token).  The requirement cannot be verified up-front;
            callers should defer to merge-time error handling.
        """
        scopes = await self.get_token_scopes()
        if scopes is None:
            return None
        return "workflow" in scopes
