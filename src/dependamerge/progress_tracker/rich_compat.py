# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Optional-dependency shim for the Rich terminal rendering library.

Rich drives the live progress display, but it is not a hard
requirement: when it is missing the trackers fall back to plain
carriage-return output.  This module owns that decision once, so every
tracker imports the same ``RICH_AVAILABLE`` flag and the same
``Live`` / ``Text`` / ``Console`` names, whether they are the real Rich
classes or the inert stand-ins defined below.
"""

from __future__ import annotations

from typing import Any

try:
    from rich.console import Console  # pyright: ignore[reportAssignmentType]
    from rich.live import Live  # pyright: ignore[reportAssignmentType]
    from rich.text import Text  # pyright: ignore[reportAssignmentType]

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

    class Live:  # type: ignore[no-redef]  # pyright: ignore[reportRedefinition]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def update(self, *args: Any) -> None:
            pass

        def refresh(self) -> None:
            pass

    class Text:  # type: ignore[no-redef]  # pyright: ignore[reportRedefinition]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def append(self, *args: Any, **kwargs: Any) -> None:
            pass

    class Console:  # type: ignore[no-redef]  # pyright: ignore[reportRedefinition]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
