# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Periodic progress output while merges are in flight.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio

from ._base import _MergeManagerBase


class _StatusTickerMixin(_MergeManagerBase):
    """Periodic progress output while merges are in flight."""

    async def _wait_status_ticker(self) -> None:
        """Update the progress display while PRs wait for required checks.

        Runs as a background task for the lifetime of
        ``merge_prs_parallel``. Once per second it samples
        ``self._waiting_prs`` and pushes a single-line status
        message into the progress tracker (which uses Rich Live
        for in-place updates) so the user can see how much
        longer the tool will block before returning the shell
        prompt.

        The countdown uses the latest (worst-case) deadline across
        all waiting PRs so the displayed value reflects the longest
        remaining wait, not an arbitrary one. When no PRs are
        waiting, the message is cleared.

        When the progress tracker is in non-Rich (fallback) mode,
        the per-update line would print to stdout, which would spam
        logs every second. In that case we delegate to the plain
        ticker (15s cadence) instead.
        """
        if not self.progress_tracker:
            # Fallback: emit a periodic plain console line so the
            # user still gets feedback even without Rich progress.
            await self._wait_status_ticker_plain()
            return

        # If the tracker exists but Rich is unavailable (non-TTY,
        # no Rich library, etc.), it falls back to per-update
        # ``print()`` calls. Updating every second would spam the
        # user's terminal/logs, so use the slower plain ticker
        # cadence instead.
        rich_available = bool(getattr(self.progress_tracker, "rich_available", False))
        if not rich_available:
            await self._wait_status_ticker_plain()
            return

        last_message: str | None = None
        try:
            while True:
                async with self._waiting_lock:
                    snapshot = dict(self._waiting_prs)

                if snapshot:
                    now = asyncio.get_running_loop().time()
                    remaining = max(
                        0.0,
                        max(snapshot.values()) - now,
                    )
                    count = len(snapshot)
                    noun = "PR" if count == 1 else "PRs"
                    message = (
                        f"⏳ Waiting for {count} {noun} "
                        f"to complete checks [{int(remaining)}s]"
                    )
                else:
                    message = ""

                if message != last_message:
                    try:
                        self.progress_tracker.update_operation(message)
                    except Exception:
                        # Defensive: a failing tracker must never
                        # take down the whole merge run.
                        pass
                    last_message = message

                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            # Best-effort clear on shutdown so the final tracker
            # state isn't stuck on a stale countdown.
            if last_message:
                try:
                    self.progress_tracker.update_operation("")
                except Exception:
                    # Defensive: a failing tracker must never take down
                    # the shutdown path.
                    pass
            raise

    async def _wait_status_ticker_plain(self) -> None:
        """Plain-text countdown when no Rich progress tracker is present.

        Emits one console line every 15 seconds while PRs are
        waiting on required checks. Less granular than the Rich
        in-place update, but still gives the user visibility into
        why the tool is blocking.
        """
        last_emit: float = 0.0
        try:
            while True:
                async with self._waiting_lock:
                    snapshot = dict(self._waiting_prs)

                if snapshot:
                    now = asyncio.get_running_loop().time()
                    if now - last_emit >= 15.0:
                        remaining = max(0.0, max(snapshot.values()) - now)
                        count = len(snapshot)
                        noun = "PR" if count == 1 else "PRs"
                        try:
                            self._console.print(
                                f"⏳ Waiting for {count} {noun} "
                                f"to complete checks "
                                f"[{int(remaining)}s remaining]"
                            )
                        except Exception:
                            # Console output is best-effort; ignore
                            # write errors on unusual terminals.
                            pass
                        last_emit = now
                else:
                    last_emit = 0.0

                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise
