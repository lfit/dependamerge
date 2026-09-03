# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Applying the ``--github-host`` option at command entry.

Every target-taking command accepts the flag, and every one of them
needs the same three things: record it, validate it, and report a bad
value as a message rather than a traceback.

Validation of the *flag* happens here because the configuration
readers run before any command's error handling, so a host naming a
port --- ports cannot reach the API base URLs --- would otherwise
surface as an uncaught exception on an ordinary mistake.  The
environment variables are deliberately not probed here: see
:func:`apply_github_host`.
"""

from __future__ import annotations

import typer

from ..url_parser import UrlParseError, set_github_host
from ._app import console


def apply_github_host(value: str | None) -> None:
    """Record ``--github-host`` and validate what the operator typed.

    Only the flag is validated here.  The environment variables are
    left to the point of use, because a target that never consults a
    GitHub default --- an explicit Gerrit URL, say --- should not be
    blocked by an unrelated ``GH_HOST`` it does not read.  When a
    shorthand *does* need the default, the failure surfaces through the
    parsers and reaches the operator by way of the usual reporting.

    Args:
        value: The raw flag value, or None when it was omitted.

    Raises:
        typer.Exit: When the flag names an unusable host.
    """
    try:
        set_github_host(value)
    except UrlParseError as exc:
        console.print(f"❌ {exc}")
        raise typer.Exit(1) from None
