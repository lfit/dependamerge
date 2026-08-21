# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Typer application object and the console every command prints to.

``app`` is the object ``[project.scripts]`` points at, and ``console`` is
the single Rich console the whole package writes through.  Both live here
so that no command module has to import another command module to reach
them.
"""

import sys

import typer
from rich.console import Console

from .._version import __version__

# Constants
MAX_RETRIES = 2

# Default owner-wide global wait ceiling (seconds) for `--max-wait`.
# 15 minutes balances giving dependabot's rebase + CI time to land
# against not blocking the shell indefinitely.  0 disables waiting
# (fire-and-forget).
DEFAULT_MAX_WAIT = 900.0


def version_callback(value: bool):
    """Callback to show version and exit."""
    if value:
        console.print(f"🏷️ dependamerge version {__version__}")
        raise typer.Exit()


class CustomTyper(typer.Typer):
    """Custom Typer class that shows version in help."""

    def __call__(self, *args, **kwargs):
        # Check if help is being requested
        if "--help" in sys.argv or "-h" in sys.argv:
            console.print(f"🏷️ dependamerge version {__version__}")
        return super().__call__(*args, **kwargs)


app = CustomTyper(
    help="Find blocked PRs in GitHub organizations and automatically merge pull requests"
)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """
    Dependamerge command line interface.
    """
    # The actual handling is done via the version_callback.
    # This callback exists only to expose --version at the top level.


console = Console(markup=False)
