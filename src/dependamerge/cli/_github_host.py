# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Applying the ``--github-host`` option at command entry.

Every target-taking command accepts the flag, and every one of them
needs the same three things: record it, validate it, and report a bad
value as a message rather than a traceback.

Validation has to happen here rather than deep in the parsers.  A host
naming a port is refused --- ports cannot reach the API base URLs --- and
that refusal is raised by the configuration readers themselves, which
run before any command's error handling.  Left alone it surfaces as an
uncaught exception on an ordinary mistake.
"""

from __future__ import annotations

import typer

from ..url_parser import (
    UrlParseError,
    default_github_host,
    enterprise_hosts,
    set_github_host,
)
from ._app import console


def apply_github_host(value: str | None) -> None:
    """Record ``--github-host`` and validate the resolved configuration.

    Also probes the environment variables, so a bad ``GH_HOST`` is
    reported here with the same wording rather than escaping from
    whichever parser happens to read it first.

    Args:
        value: The raw flag value, or None when it was omitted.

    Raises:
        typer.Exit: When the flag or the environment names an
            unusable host.
    """
    try:
        set_github_host(value)
        # Force the readers to run now, while there is somewhere
        # sensible to report a failure.
        default_github_host()
        enterprise_hosts()
    except UrlParseError as exc:
        console.print(f"❌ {exc}")
        raise typer.Exit(1) from None
