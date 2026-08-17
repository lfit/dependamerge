# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Gerrit data models for dependamerge.

This module defines Pydantic models for Gerrit changes and related data,
paralleling the GitHub PR models used elsewhere in the codebase.

These models provide:
- Type-safe representations of Gerrit API responses
- Factory methods for parsing raw API data
- Comparison result structures for similarity matching
"""

from __future__ import annotations

from .change import GerritChangeInfo
from .values import (
    GerritChangeStatus,
    GerritComparisonResult,
    GerritFileChange,
    GerritFileStatus,
    GerritLabelInfo,
    GerritSubmitResult,
)

__all__ = [
    "GerritChangeInfo",
    "GerritChangeStatus",
    "GerritComparisonResult",
    "GerritFileChange",
    "GerritFileStatus",
    "GerritLabelInfo",
    "GerritSubmitResult",
]
