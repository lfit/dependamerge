# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Gerrit change comparator for similarity matching.

This module provides comparison logic for Gerrit changes, enabling the
identification of similar changes for bulk review and submit operations.

The comparator follows the same patterns as the GitHub PR comparator,
adapting the comparison logic for Gerrit-specific fields and conventions.
"""

from __future__ import annotations

from .comparison import (
    AUTOMATION_INDICATORS,
    GerritChangeComparator,
    create_gerrit_comparator,
)

__all__ = [
    "AUTOMATION_INDICATORS",
    "GerritChangeComparator",
    "create_gerrit_comparator",
]
