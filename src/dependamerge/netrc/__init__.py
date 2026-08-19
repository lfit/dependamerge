# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Netrc file parsing for Gerrit authentication credentials.

This module provides functionality to parse .netrc files and retrieve
credentials for authenticating with Gerrit servers. It follows the
standard netrc format as documented at:
https://everything.curl.dev/usingcurl/netrc.html

The module supports:
- Standard netrc tokens: machine, login, password, default
- Quoted strings (curl 7.84.0+) with escape sequences
- Multiple search locations (local directory, home directory)
- Windows compatibility (_netrc fallback)
"""

from __future__ import annotations

from .models import (
    CredentialSource,
    GerritCredentials,
    NetrcCredentials,
    NetrcParseError,
)
from .parser import (
    _NETRC_ESCAPES,
    _TOKEN_DEFAULT,
    _TOKEN_LOGIN,
    _TOKEN_MACDEF,
    _TOKEN_MACHINE,
    _TOKEN_PASSWORD,
    NetrcParser,
    _normalize_host_for_netrc_lookup,
)
from .resolve import (
    check_netrc_permissions,
    find_netrc_file,
    get_credentials_for_host,
    load_netrc,
    log,
    resolve_gerrit_credentials,
)

__all__ = [
    "CredentialSource",
    "GerritCredentials",
    "NetrcCredentials",
    "NetrcParseError",
    "NetrcParser",
    "check_netrc_permissions",
    "find_netrc_file",
    "get_credentials_for_host",
    "load_netrc",
    "resolve_gerrit_credentials",
]
