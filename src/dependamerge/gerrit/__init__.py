# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Gerrit integration package for dependamerge.

This package provides Gerrit Code Review support, enabling bulk review
and submission of similar changes across a Gerrit server.

Modules:
    client: REST client with retry and timeout handling
    urls: URL construction utilities with base path support
    models: Pydantic models for Gerrit data structures
    service: High-level service layer for Gerrit operations
    comparator: Change comparison logic for similarity matching
    submit_manager: Parallel submit operations

Usage:
    from dependamerge.gerrit import GerritRestClient, build_client

    client = build_client("gerrit.example.org")
    changes = client.get("/changes/?q=status:open")
"""

from dependamerge.gerrit.client import (
    GerritAuthError,
    GerritNotFoundError,
    GerritRestClient,
    GerritRestError,
    build_client,
)
from dependamerge.gerrit.comparator import (
    AUTOMATION_INDICATORS,
    GerritChangeComparator,
    create_gerrit_comparator,
)
from dependamerge.gerrit.models import (
    GerritChangeInfo,
    GerritChangeStatus,
    GerritComparisonResult,
    GerritFileChange,
    GerritFileStatus,
    GerritLabelInfo,
    GerritSubmitResult,
)
from dependamerge.gerrit.service import (
    DEFAULT_CHANGE_OPTIONS,
    DEFAULT_LIST_OPTIONS,
    GerritService,
    GerritServiceError,
    create_gerrit_service,
)
from dependamerge.gerrit.submit_manager import (
    GerritSubmitManager,
    SubmitStatus,
    create_submit_manager,
)
from dependamerge.gerrit.urls import (
    GerritUrlBuilder,
    create_url_builder,
    discover_base_path,
)

__all__ = [
    # Client
    "GerritAuthError",
    "GerritNotFoundError",
    "GerritRestClient",
    "GerritRestError",
    "build_client",
    # Models
    "GerritChangeInfo",
    "GerritChangeStatus",
    "GerritComparisonResult",
    "GerritFileChange",
    "GerritFileStatus",
    "GerritLabelInfo",
    "GerritSubmitResult",
    # Comparator
    "AUTOMATION_INDICATORS",
    "GerritChangeComparator",
    "create_gerrit_comparator",
    # Submit Manager
    "GerritSubmitManager",
    "SubmitStatus",
    "create_submit_manager",
    # Service
    "DEFAULT_CHANGE_OPTIONS",
    "DEFAULT_LIST_OPTIONS",
    "GerritService",
    "GerritServiceError",
    "create_gerrit_service",
    # URLs
    "GerritUrlBuilder",
    "create_url_builder",
    "discover_base_path",
]
