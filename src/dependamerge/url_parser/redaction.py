# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Making a target safe to show before it reaches a message.

Every parser reports the target it refused, which is what makes the
message actionable --- and a target is operator input that may carry a
credential.  This branch has now leaked one four separate times: a git
remote logged with its password, a configuration value echoed verbatim,
a network-path reference that skipped stripping, and an API authority
read from ``netloc``.  Each was fixed where it was found, and a fifth
set appeared in the errors added to fix the fourth.

Fixing them one at a time is not working, so the sanitising lives here
and every site that interpolates a target calls it.  A new message is
then safe by default rather than safe if its author remembered.
"""

from __future__ import annotations

import re

__all__ = ["redact_target"]

#: The userinfo a URL may carry, in both the ``scheme://`` and the
#: scheme-less ``//host/path`` forms.  Both name an authority, so both
#: can hide a credential in front of it; requiring an authority marker
#: is what leaves a bare ``a@b`` *path* segment alone.
_USERINFO_RE = re.compile(r"\A((?:[A-Za-z][A-Za-z0-9+.-]*:)?//)[^/@\s]+@")


def redact_target(value: str) -> str:
    """Strip anything credential-bearing from a target for display.

    Removes URL userinfo, the query string and the fragment --- the
    three places a token is conventionally written --- while keeping
    the scheme, host and path, which are what make a message useful.

    The query and fragment go even though normalisation deliberately
    *preserves* them: they are kept so a Gerrit search URL parses, and
    that is a parsing concern.  Nothing downstream of an error needs
    them, and a token in ``?token=`` is the likeliest way one reaches
    a terminal.

    Args:
        value: The target as the operator supplied it, or as
            normalisation rewrote it.

    Returns:
        The target with credentials removed.
    """
    if not value:
        return value
    redacted = _USERINFO_RE.sub(r"\1***@", value)
    return redacted.split("?", 1)[0].split("#", 1)[0]
