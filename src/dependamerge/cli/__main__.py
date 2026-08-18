# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Entry point for ``python -m dependamerge.cli``.

Before this package existed, ``cli`` was a module ending in the usual
``if __name__ == "__main__": app()`` guard, so it could be executed
directly. A package cannot be executed that way, so the guard moves
here to keep that invocation working.
"""

from __future__ import annotations

from . import app

if __name__ == "__main__":
    app()
