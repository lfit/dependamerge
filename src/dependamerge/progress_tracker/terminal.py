# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Terminal detection and logging-quieting support for progress trackers.

The live progress display owns a region of the terminal while it is on
screen, so it has to know whether it is attached to a real TTY and it
has to stop terminal-bound logging handlers from writing straight past
Rich and desyncing that region.  Neither concern depends on what is
being tracked, so both live here as a mixin shared by every tracker.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import logging
import sys
from typing import Any


class _TerminalLoggingMixin:
    """Terminal-detection and logging-quieting helpers for trackers."""

    # Terminal-bound logging handlers silenced while the live display
    # is on screen; each entry is ``(handler, saved_level)`` so the
    # original level can be restored on stop.  The concrete tracker
    # initialises this in its constructor.
    _quieted_handlers: list[tuple[logging.Handler, int]]

    @staticmethod
    def _stream_is_tty(stream: Any) -> bool:
        """Return True only when ``stream`` is a real terminal.

        A stream may be a proxy or capture object that lacks ``isatty``
        (or whose ``isatty`` raises); treat a missing or failing
        ``isatty`` as non-TTY rather than letting an ``AttributeError``
        escape and break display setup or teardown.
        """
        isatty = getattr(stream, "isatty", None)
        if not callable(isatty):
            return False
        try:
            return bool(isatty())
        except Exception:
            return False

    @staticmethod
    def _stdout_is_tty() -> bool:
        """Return True only when stdout is a real terminal."""
        return _TerminalLoggingMixin._stream_is_tty(sys.stdout)

    def _quiet_terminal_logging(self) -> None:
        """Silence terminal-bound logging while the live display runs.

        ``logging.basicConfig`` binds a ``StreamHandler`` to the real
        ``sys.stderr`` at startup, before the Rich ``Live`` display
        swaps in its own stdout/stderr proxies.  Because the handler
        cached the original stream, a ``WARNING``/``ERROR`` logged
        while the live region is on screen writes straight past Rich
        to the terminal and desyncs the region: the top line is
        orphaned and the whole block shifts down a row (the reported
        duplicated-header artifact seen when a merge failed).

        Real-merge progress is conveyed by the live counters and
        explained in the end-of-run summary, so while the display is
        active we raise such handlers above ``CRITICAL`` and restore
        them when it stops.  Only stream handlers writing to a real
        terminal are touched: the stream must be one of the process's
        std streams *and* report ``isatty()`` true.  This leaves file
        handlers, pytest capture, and handlers whose ``stderr`` has been
        redirected to a file untouched, so their warnings/errors are
        never lost (there is no Rich desync risk when the target is not
        a terminal).
        """
        self._quieted_handlers = []
        if not self._stdout_is_tty():
            return
        terminal_streams = {
            stream
            for stream in (sys.__stdout__, sys.__stderr__, sys.stdout, sys.stderr)
            if stream is not None
        }
        try:
            handlers = list(logging.getLogger().handlers)
        except Exception:
            return
        for handler in handlers:
            stream = getattr(handler, "stream", None)
            if (
                isinstance(handler, logging.StreamHandler)
                and stream in terminal_streams
                and self._stream_is_tty(stream)
            ):
                self._quieted_handlers.append((handler, handler.level))
                handler.setLevel(logging.CRITICAL + 1)

    def _restore_terminal_logging(self) -> None:
        """Restore logging handlers quieted by :meth:`_quiet_terminal_logging`."""
        for handler, level in self._quieted_handlers:
            try:
                handler.setLevel(level)
            except Exception:
                # Best-effort restore: never let logging teardown
                # raise out of display teardown.
                pass
        self._quieted_handlers = []
