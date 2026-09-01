# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Up-front token capability probes for the async GitHub client.

Resolves the authenticated login, reports whether that account may
bypass branch protection on a repository, and runs the pre-flight
permission report used before a bulk operation starts.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import (
    Any,
)

from ._base import _GitHubAsyncBase
from ._permissions import OPERATION_PERMISSIONS, web_host_for


async def _check_repo_write_permission(
    api: _GitHubAsyncBase,
    result: dict[str, Any],
    operation: str,
    owner: str,
    repo: str,
) -> None:
    """Verify write access to a specific repository."""
    # Use the collaborator permission endpoint to verify
    # the token has write access to this specific repo.
    #
    # The previous approach (GET /repos/{owner}/{repo} and
    # inspecting permissions.push) is unreliable for
    # fine-grained PATs: GitHub returns the *user's*
    # org-level permissions regardless of token scope,
    # producing false positives when the token is scoped
    # to a different org.
    #
    # The collaborator endpoint correctly returns 403
    # ("Resource not accessible by personal access token")
    # when the token doesn't cover the target repo.

    # Resolve authenticated username (cached after first call)
    if api._authenticated_user_login is None:
        user_data = await api.get("/user")
        if isinstance(user_data, dict):
            api._authenticated_user_login = user_data.get("login")

    username = api._authenticated_user_login
    if not username:
        result["error"] = "Could not determine authenticated user"
    else:
        collab_data = await api.get(
            f"/repos/{owner}/{repo}/collaborators/{username}/permission"
        )
        if isinstance(collab_data, dict):
            perm_level = collab_data.get("permission", "none")
            # write, maintain, or admin is required for approve/merge/close/update
            if perm_level in ("write", "maintain", "admin"):
                result["has_permission"] = True
            else:
                result["error"] = (
                    f"Token has '{perm_level}' access to "
                    f"{owner}/{repo} — write, maintain, or admin is required"
                )
                perms = OPERATION_PERMISSIONS.get(operation, {})
                result["guidance"] = {
                    "classic": perms.get("classic"),
                    "fine_grained": perms.get("fine_grained"),
                }
        else:
            result["error"] = "Could not determine collaborator permissions"


async def _check_branch_protection_permission(
    api: _GitHubAsyncBase, result: dict[str, Any], owner: str, repo: str
) -> None:
    """Probe branch protection to verify Administration: Read."""
    # Verify Administration: Read permission by probing
    # the branch protection endpoint.  A token with this
    # permission receives either 200 (rules exist) or
    # 404 "Branch not protected"; without it GitHub
    # returns 403 "Resource not accessible".
    #
    # The repo metadata fetch is separated from the
    # branch-protection probe so that a 404 from
    # GET /repos/{owner}/{repo} (repo doesn't exist or
    # token can't see it) is NOT silently treated as
    # success.
    default_branch = "main"
    try:
        repo_data = await api.get(f"/repos/{owner}/{repo}")
        if isinstance(repo_data, dict):
            default_branch = repo_data.get("default_branch", "main")
    except Exception:
        # Repo metadata fetch failed — token may lack
        # access.  Let the error propagate to the outer
        # handler which will surface it as a permission
        # error.  Do NOT fall through to treat this as
        # success.
        raise

    try:
        await api.get(f"/repos/{owner}/{repo}/branches/{default_branch}/protection")
        result["has_permission"] = True
    except Exception as e:
        if "404" in str(e):
            # 404 = branch exists but has no protection
            # rules — the token still has the permission.
            result["has_permission"] = True
        else:
            raise


async def _check_merge_workflow_permission(
    api: _GitHubAsyncBase, result: dict[str, Any]
) -> None:
    """Verify the token may merge PRs that modify workflow files."""
    # This is only checkable for classic PATs, which advertise their
    # scopes via the ``X-OAuth-Scopes`` header.  Fine-grained PATs and
    # app tokens do not expose scopes, so the check returns ``None`` and
    # we pass it through here — the requirement cannot be verified
    # up-front for those token types and is instead surfaced (with
    # accurate guidance) by the merge-time handler if it actually bites.
    has_workflow = await api.check_workflow_scope()
    if has_workflow is False:
        perms = OPERATION_PERMISSIONS.get("merge_workflow", {})
        result["error"] = (
            "Token is missing the 'workflow' scope, which is "
            "required to merge pull requests that modify "
            "GitHub Actions workflow files "
            "(.github/workflows/**)"
        )
        result["guidance"] = {
            "classic": perms.get("classic"),
            "fine_grained": perms.get("fine_grained"),
            # Derived from the endpoint in use: ``gh auth refresh``
            # takes the host whose credentials need changing, and
            # naming dotcom sends an Enterprise operator to modify an
            # unrelated token.
            "fix": (f"Run: gh auth refresh -h {web_host_for(api.api_url)} -s workflow"),
        }
    else:
        # ``True`` (scope present) or ``None``
        # (undeterminable token type) — do not block.
        result["has_permission"] = True


class _PermissionChecksMixin(_GitHubAsyncBase):
    """Token and account capability probes for ``GitHubAsync``."""

    async def get_authenticated_user_login(self) -> str | None:
        """Return the authenticated user's login, cached for the session.

        The login never changes for a given token, so the ``/user``
        round-trip is paid at most once per client instance.  Returns
        ``None`` when the lookup fails (callers should degrade
        gracefully); failures are not cached so a transient error can
        recover on the next call.
        """
        if self._authenticated_user_login is None:
            try:
                user_data = await self.get("/user")
            except Exception as e:
                self.log.debug("Could not resolve authenticated user: %s", e)
                return None
            if isinstance(user_data, dict):
                login = user_data.get("login")
                if isinstance(login, str) and login:
                    self._authenticated_user_login = login
        return self._authenticated_user_login

    async def check_user_can_bypass_protection(
        self, owner: str, repo: str, force_level: str = "code-owners"
    ) -> tuple[bool, str]:
        """
        Check if the authenticated user has permissions to bypass branch protection.

        Args:
            owner: Repository owner
            repo: Repository name
            force_level: The force level being used ("code-owners", "protection-rules", "all")

        Returns:
            Tuple of (can_bypass: bool, reason: str)
        """
        try:
            repo_data = await self.get(f"/repos/{owner}/{repo}")
            if not isinstance(repo_data, dict):
                return False, "Could not fetch repository information"

            permissions = repo_data.get("permissions", {})
            self.log.debug(
                f"Repository permissions for {owner}/{repo}: admin={permissions.get('admin')}, push={permissions.get('push')}, pull={permissions.get('pull')}"
            )

            # Check if user has admin permissions (which includes bypass)
            if permissions.get("admin"):
                self.log.debug(f"User has admin permissions for {owner}/{repo}")
                return True, "User has admin permissions"

            # Try to get more detailed permission info from user's repository membership
            try:
                # For organization repos, check if user has bypass permissions
                # This requires checking the user's role/permissions
                # Use cached login to avoid repeated /user calls
                if self._authenticated_user_login is None:
                    user_data = await self.get("/user")
                    if isinstance(user_data, dict):
                        self._authenticated_user_login = user_data.get("login")

                username = self._authenticated_user_login
                if username:
                    collab_data = await self.get(
                        f"/repos/{owner}/{repo}/collaborators/{username}/permission"
                    )
                    if isinstance(collab_data, dict):
                        permission_level = collab_data.get("permission")
                        # admin permission can bypass
                        if permission_level == "admin":
                            return True, "User has admin collaborator permissions"
            except Exception as e:
                # If we can't check detailed permissions, continue with basic check
                self.log.debug(
                    f"Could not check detailed collaborator permissions: {e}"
                )

            # If we have push permissions but not admin
            if permissions.get("push"):
                # All force levels require admin permissions to actually bypass branch protection
                # at the GitHub API level. Push permissions alone are not sufficient.
                self.log.debug(
                    f"User has push permissions for {owner}/{repo} but not admin (required to bypass branch protection at GitHub API level)"
                )
                return (
                    False,
                    "User has push permissions but not admin/bypass permissions (admin required to bypass branch protection)",
                )

            self.log.debug(
                f"User does not have sufficient permissions for {owner}/{repo}"
            )
            return False, "User does not have bypass permissions"

        except Exception as e:
            # If we can't determine permissions, return conservative result
            self.log.debug(f"Could not check bypass permissions: {e}")
            return False, f"Could not verify permissions: {str(e)}"

    async def check_token_permissions(
        self, operations: list[str], owner: str = "", repo: str = ""
    ) -> dict[str, dict[str, Any]]:
        """Pre-flight check for token permissions.

        Tests whether the token has the necessary permissions for the specified
        operations without actually performing them. This allows failing fast
        with clear error messages before attempting bulk operations.

        Args:
            operations: List of operations to check (e.g., ['approve', 'merge', 'close'])
            owner: Repository owner (required for repository-specific checks)
            repo: Repository name (required for repository-specific checks)

        Returns:
            Dictionary mapping operation names to check results:
            {
                'operation_name': {
                    'has_permission': bool,
                    'error': str | None,
                    'guidance': dict | None
                }
            }

        Example:
            >>> results = await client.check_token_permissions(['approve', 'merge'], 'owner', 'repo')
            >>> if not results['approve']['has_permission']:
            ...     print(results['approve']['error'])
        """
        results: dict[str, dict[str, Any]] = {}

        for operation in operations:
            result: dict[str, Any] = {
                "has_permission": False,
                "error": None,
                "guidance": None,
            }

            try:
                # Perform a lightweight check for each operation
                if (
                    operation in ("approve", "merge", "close", "update_branch")
                    and owner
                    and repo
                ):
                    await _check_repo_write_permission(
                        self, result, operation, owner, repo
                    )

                elif operation == "branch_protection" and owner and repo:
                    await _check_branch_protection_permission(self, result, owner, repo)

                elif operation == "list_repos":
                    if owner:
                        await self.get(f"/orgs/{owner}/repos?per_page=1")
                    result["has_permission"] = True

                elif operation == "merge_workflow":
                    await _check_merge_workflow_permission(self, result)

                else:
                    result["error"] = f"Unknown operation: {operation}"

            except Exception as e:
                perm_error = self._parse_permission_error(e, operation, owner, repo)
                if perm_error:
                    result["has_permission"] = False
                    result["error"] = str(perm_error)
                    result["guidance"] = perm_error.token_type_guidance
                else:
                    # Unexpected error - be conservative
                    result["has_permission"] = False
                    result["error"] = f"Could not verify permissions: {str(e)}"

            results[operation] = result

        return results
