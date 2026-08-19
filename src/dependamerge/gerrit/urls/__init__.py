# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Gerrit URL construction utilities.

This package provides a centralized way to construct Gerrit URLs with
consistent handling of base paths (e.g., "/infra/") that some Gerrit
servers require.

Usage:
    from dependamerge.gerrit.urls import GerritUrlBuilder

    builder = GerritUrlBuilder("gerrit.linuxfoundation.org", base_path="infra")
    api_url = builder.api_url("/changes/")
    change_url = builder.change_url("releng/project", 12345)
"""

from __future__ import annotations

from .builder import GerritUrlBuilder, create_url_builder
from .discovery import (
    _BASE_PATH_CACHE,
    _CIRCUIT_BREAKER,
    _CIRCUIT_BREAKER_RESET_SECONDS,
    _CIRCUIT_BREAKER_THRESHOLD,
    _check_circuit_breaker,
    _extract_base_path,
    _NoRedirect,
    _record_circuit_breaker_failure,
    _reset_circuit_breaker,
    discover_base_path,
    log,
)

__all__ = [
    "GerritUrlBuilder",
    "create_url_builder",
    "discover_base_path",
]
