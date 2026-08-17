# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Discovery of .netrc files and resolution of Gerrit credentials.

Locates a .netrc file using the standard search order, checks its
permissions, and resolves Gerrit credentials from the available sources
(CLI arguments, .netrc, environment variables) in priority order.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

from .models import (
    CredentialSource,
    GerritCredentials,
    NetrcCredentials,
    NetrcParseError,
)
from .parser import NetrcParser, _normalize_host_for_netrc_lookup

log = logging.getLogger("dependamerge.netrc")


def find_netrc_file(
    search_local: bool = True,
    explicit_path: Path | None = None,
) -> Path | None:
    """
    Find a .netrc file using standard search order.

    Search order:
    1. Explicit path (if provided)
    2. Local directory .netrc (if search_local=True)
    3. ~/.netrc
    4. ~/_netrc (Windows fallback)

    Args:
        search_local: Whether to search current directory first.
        explicit_path: Explicit path to a netrc file.

    Returns:
        Path to found netrc file, or None if not found.
    """
    if explicit_path is not None:
        if explicit_path.is_file():
            log.debug("Using explicit netrc file: %s", explicit_path)
            return explicit_path
        log.warning("Explicit netrc file not found: %s", explicit_path)
        return None

    candidates: list[Path] = []

    # Local directory
    if search_local:
        candidates.append(Path.cwd() / ".netrc")

    # Home directory
    home = Path.home()
    candidates.append(home / ".netrc")

    # Windows fallback
    if os.name == "nt":
        candidates.append(home / "_netrc")

    for candidate in candidates:
        if candidate.is_file():
            log.debug("Found netrc file: %s", candidate)
            return candidate

    log.debug("No netrc file found in search paths")
    return None


def check_netrc_permissions(path: Path) -> bool:
    """
    Check if netrc file has secure permissions.

    Warns if the file is readable by others (Unix only).

    Args:
        path: Path to the netrc file.

    Returns:
        True if permissions are secure, False otherwise.
    """
    if os.name == "nt":
        # Windows doesn't have the same permission model
        return True

    try:
        mode = path.stat().st_mode
    except OSError as e:
        log.warning("Could not check permissions for %s: %s", path, e)
        return True

    # Check if group or others have read permission
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        log.warning(
            "Netrc file %s has insecure permissions. Consider running: chmod 600 %s",
            path,
            path,
        )
        return False
    return True


def load_netrc(
    path: Path | None = None,
    search_local: bool = True,
) -> NetrcParser | None:
    """
    Load and parse a netrc file.

    Args:
        path: Explicit path to netrc file (optional).
        search_local: Search current directory for .netrc.

    Returns:
        NetrcParser instance, or None if no file found.

    Raises:
        NetrcParseError: If the file exists but cannot be parsed.
    """
    netrc_path = find_netrc_file(
        search_local=search_local,
        explicit_path=path,
    )

    if netrc_path is None:
        return None

    check_netrc_permissions(netrc_path)

    try:
        content = netrc_path.read_text(encoding="utf-8")
    except OSError:
        log.exception("Could not read netrc file %s", netrc_path)
        return None

    try:
        return NetrcParser(content)
    except NetrcParseError:
        log.exception("Could not parse netrc file %s", netrc_path)
        raise


def get_credentials_for_host(
    host: str,
    netrc_file: Path | None = None,
    search_local: bool = True,
    use_netrc: bool = True,
    netrc_optional: bool = True,
) -> NetrcCredentials | None:
    """
    Get credentials for a Gerrit host from .netrc file.

    This is the main entry point for credential lookup. It handles
    the full workflow of finding, parsing, and querying the netrc file.

    Args:
        host: Gerrit server hostname (e.g., 'gerrit.onap.org').
        netrc_file: Explicit path to netrc file (optional).
        search_local: Search current directory for .netrc.
        use_netrc: Whether to use netrc at all (--no-netrc sets False).
        netrc_optional: If True, don't fail if netrc not found.

    Returns:
        NetrcCredentials if found, None otherwise.

    Raises:
        NetrcParseError: If netrc file exists but cannot be parsed.
        FileNotFoundError: If netrc_optional=False and no file found.
    """
    if not use_netrc:
        log.debug("Netrc lookup disabled")
        return None

    # Normalize host - remove scheme, path, and port if present
    normalized_host = _normalize_host_for_netrc_lookup(host)

    # Find the netrc file path first so we can include it in log messages
    netrc_path = find_netrc_file(
        search_local=search_local,
        explicit_path=netrc_file,
    )

    if netrc_path is None:
        if not netrc_optional:
            msg = "No .netrc file found and netrc is required"
            raise FileNotFoundError(msg)
        return None

    netrc = load_netrc(
        path=netrc_path,
        search_local=False,  # Already found the path
    )

    if netrc is None:
        # load_netrc returns None if file couldn't be read
        return None

    credentials = netrc.get_credentials(normalized_host)
    if credentials:
        # SECURITY: Do not log credential values (usernames, passwords).
        # Log the credential source and host, not the credentials themselves.
        # See CodeQL rule py/clear-text-logging-sensitive-data.
        log.debug(
            "Found netrc credentials for host %s in %s",
            normalized_host,
            netrc_path,
        )
    else:
        log.warning(
            "No netrc credentials found for %s in %s",
            normalized_host,
            netrc_path,
        )

    return credentials


def resolve_gerrit_credentials(
    host: str,
    *,
    explicit_username: str | None = None,
    explicit_password: str | None = None,
    use_netrc: bool = True,
    netrc_file: Path | None = None,
    env_username_var: str = "GERRIT_USERNAME",
    env_password_var: str = "GERRIT_PASSWORD",
    fallback_env_username_var: str | None = "GERRIT_HTTP_USER",
    fallback_env_password_var: str | None = "GERRIT_HTTP_PASSWORD",
) -> GerritCredentials | None:
    """
    Resolve Gerrit credentials from multiple sources with defined priority.

    This is the canonical function for resolving Gerrit credentials.
    It returns a single GerritCredentials object that contains both
    the credentials and metadata about their source.

    Priority order:
    1. Explicit CLI arguments (explicit_username/explicit_password)
    2. .netrc file (if use_netrc=True)
    3. Primary environment variables (env_username_var/env_password_var)
    4. Fallback environment variables (if provided)

    Args:
        host: Gerrit server hostname for netrc lookup.
        explicit_username: Username from CLI argument (highest priority).
        explicit_password: Password from CLI argument (highest priority).
        use_netrc: Whether to try .netrc for credentials.
        netrc_file: Explicit path to a .netrc file.
        env_username_var: Primary environment variable for username.
        env_password_var: Primary environment variable for password.
        fallback_env_username_var: Fallback environment variable for username.
        fallback_env_password_var: Fallback environment variable for password.

    Returns:
        GerritCredentials with resolved credentials and source info,
        or None if no credentials found.
    """
    # 1. Check explicit CLI arguments first
    if explicit_username and explicit_password:
        log.debug("Using credentials from CLI arguments")
        return GerritCredentials(
            username=explicit_username.strip(),
            password=explicit_password.strip(),
            source=CredentialSource.CLI_ARGUMENT,
            source_detail="--username/--password",
        )

    # 2. Try .netrc file
    if use_netrc:
        netrc_path = find_netrc_file(
            search_local=True,
            explicit_path=netrc_file,
        )

        if netrc_path is not None:
            netrc = load_netrc(path=netrc_path, search_local=False)
            if netrc is not None:
                # Normalize host for lookup
                normalized_host = _normalize_host_for_netrc_lookup(host)

                netrc_creds = netrc.get_credentials(normalized_host)
                if netrc_creds:
                    # SECURITY: Do not log credential values.
                    # See CodeQL rule py/clear-text-logging-sensitive-data.
                    log.debug(
                        "Using credentials from .netrc file %s for host %s",
                        netrc_path,
                        normalized_host,
                    )
                    return GerritCredentials(
                        username=netrc_creds.login,
                        password=netrc_creds.password,
                        source=CredentialSource.NETRC,
                        source_detail=str(netrc_path),
                    )
                else:
                    log.warning(
                        "No netrc credentials found for %s in %s",
                        normalized_host,
                        netrc_path,
                    )

    # 3. Try primary environment variables
    env_user = os.getenv(env_username_var, "").strip()
    env_pass = os.getenv(env_password_var, "").strip()

    if env_user and env_pass:
        # SECURITY: Break CodeQL taint path — log a fixed string
        # describing the credential source, not parameter values.
        # See CodeQL rule py/clear-text-logging-sensitive-data.
        log.debug("Resolved Gerrit credentials from environment variables")
        return GerritCredentials(
            username=env_user,
            password=env_pass,
            source=CredentialSource.ENVIRONMENT,
            source_detail=f"{env_username_var}/{env_password_var}",
        )

    # 4. Try fallback environment variables
    if fallback_env_username_var and fallback_env_password_var:
        fallback_user = os.getenv(fallback_env_username_var, "").strip()
        fallback_pass = os.getenv(fallback_env_password_var, "").strip()

        if fallback_user and fallback_pass:
            # SECURITY: Break CodeQL taint path — log a fixed string.
            # See CodeQL rule py/clear-text-logging-sensitive-data.
            log.debug("Resolved Gerrit credentials from fallback environment variables")
            return GerritCredentials(
                username=fallback_user,
                password=fallback_pass,
                source=CredentialSource.ENVIRONMENT,
                source_detail=f"{fallback_env_username_var}/{fallback_env_password_var}",
            )

    log.debug("No Gerrit credentials found from any source")
    return None
