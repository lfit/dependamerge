# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Error type raised by the Gerrit service layer.

This mirrors :mod:`dependamerge.gerrit._client_errors`: the exception
lives on its own so both :mod:`dependamerge.gerrit.service` and the
sibling mixins that make up ``GerritService`` can raise it without
importing each other.  It stays re-exported from
``dependamerge.gerrit.service``, where it has always been reachable.
"""

from __future__ import annotations


class GerritServiceError(Exception):
    """Raised for service-level errors."""
