# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Guard the two ways this tool can be launched.

``dependamerge`` is installed as a console script, and ``cli`` could be
executed directly with ``python -m dependamerge.cli`` back when it was a
single module ending in an ``if __name__ == "__main__":`` guard. Turning
it into a package silently removed the second, because a package is not
executable without a ``__main__`` submodule.

Both are exercised as subprocesses rather than through Typer's test
runner, since the failure being guarded against is in module resolution
and only appears when Python launches the process itself.
"""

from __future__ import annotations

import subprocess
import sys


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


class TestEntryPoints:
    """Both documented invocations must start the CLI."""

    def test_module_execution_works(self) -> None:
        """``python -m dependamerge.cli`` must launch the app."""
        result = _run([sys.executable, "-m", "dependamerge.cli", "--version"])
        assert result.returncode == 0, (
            f"python -m dependamerge.cli failed ({result.returncode}):\n{result.stderr}"
        )
        assert "dependamerge" in result.stdout.lower()

    def test_package_execution_reports_clearly(self) -> None:
        """``python -m dependamerge`` has no entry point and must say so.

        Asserting the current behaviour so that adding one later is a
        deliberate change rather than an accident.
        """
        result = _run([sys.executable, "-m", "dependamerge", "--version"])
        assert result.returncode != 0
