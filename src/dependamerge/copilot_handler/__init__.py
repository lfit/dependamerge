# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Copilot comment handler for detecting and managing GitHub Copilot review comments.

This module provides functionality to:
- Identify Copilot-generated review comments
- Filter and categorize Copilot feedback
- Dismiss unresolved Copilot comments to unblock PR merging
"""

from __future__ import annotations

from .handler import (
    COMMON_COPILOT_PATTERNS,
    CopilotCommentHandler,
    logger,
)

__all__ = [
    "COMMON_COPILOT_PATTERNS",
    "CopilotCommentHandler",
]
