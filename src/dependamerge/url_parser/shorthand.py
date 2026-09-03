# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Normalisation of abbreviated and git-remote target forms.

Every parser in this package used to begin by pasting ``https://`` onto
anything that lacked a scheme.  That works for ``github.com/owner/repo``
and fails silently for everything else: ``lfreleng-actions/dependamerge``
became the host ``lfreleng-actions`` with the path ``/dependamerge``, and
``git@github.com:owner/repo.git`` became a hostname of ``git@github.com``.

:func:`normalize_target` replaces that step for all of them, so the
shorthand forms are understood once rather than four times, and the
parsers downstream keep receiving ordinary absolute URLs.

The one genuinely ambiguous case is a two-segment input: is
``a/b`` an owner and a repository, or a host and an owner?  GitHub logins
are restricted to alphanumerics and hyphens --- they cannot contain a dot
--- so a dotted first segment is a host and an undotted one is a login.
Ports and ``localhost`` are covered for completeness.  Repository names
*may* contain dots, but a repository name is never the first segment of
a shorthand, so the rule is not lossy.
"""

from __future__ import annotations

import re

from .git_suffix import strip_git_suffix, strips_git_suffix
from .host_config import (
    DEFAULT_GITHUB_HOST,
    default_github_host,
    enterprise_hosts,
    github_host_override,
    iter_enterprise_hosts,
    set_github_host,
)
from .models import UrlParseError

# Re-exported so the long-standing ``url_parser.shorthand`` import path
# keeps working now that host declaration lives in its own module.
__all__ = [
    "DEFAULT_GITHUB_HOST",
    "default_github_host",
    "enterprise_hosts",
    "github_host_override",
    "iter_enterprise_hosts",
    "looks_like_host",
    "looks_like_owner",
    "normalize_target",
    "set_github_host",
    "strip_git_suffix",
]

# aislop-ignore-file ai-slop/hardcoded-url -- This module parses and builds
# GitHub/Gerrit URLs, so URL literals here are the subject matter, not
# stray configuration.  Enterprise hosts are always derived from the
# caller's input or from an explicit environment override.

# scp-style remote: [user@]host:path, with no scheme and no leading
# slash on the path.  Whether a bare ``host:something`` is scp or
# ``host:port`` is decided in code --- see :func:`_is_scp_remote`.
_SCP_REMOTE_RE = re.compile(
    r"\A(?:(?P<user>[^@/]+)@)?(?P<host>[^@/:]+):(?P<path>[^\s]+)\Z"
)

# A path that is purely a number, i.e. indistinguishable from a port.
_NUMERIC_PATH_RE = re.compile(r"\A\d+(?:/|\Z)")

#: Schemes that already name a web URL, kept as given.
_WEB_SCHEMES = frozenset({"http", "https"})

#: Git transports whose remotes map onto a web URL by dropping the
#: scheme, credentials and port.  Anything outside both sets is left
#: alone for the parsers to reject, rather than being rewritten into a
#: request that would then be made for real.
_GIT_TRANSPORT_SCHEMES = frozenset({"ssh", "git", "git+ssh", "ssh+git", "rsync"})

# A path segment that names a host rather than an owner.
_PORT_SUFFIX_RE = re.compile(r":\d+\Z")

# GitHub login grammar: alphanumerics and hyphens, no leading or
# trailing hyphen, at most 39 characters.  Used to decide whether a
# bare token is plausibly an owner before treating it as one, so that
# genuine rubbish still fails fast with "Invalid URL" instead of being
# expanded into a request for a repository that cannot exist.
#
# Hyphen *placement* is not policed, only structure.  GitHub's user
# signup form forbids consecutive and trailing hyphens, but real
# accounts sit on the wrong side of both rules: ``a--b--t`` is an
# organisation, and ``johan--`` is a user with about two thousand
# repositories.  This gate stops obvious rubbish rather than
# reimplementing GitHub's account policy --- being too permissive
# costs a clear 404 from the API, whereas being too strict reports
# "Invalid URL" for an owner that exists, and previously *regressed*
# ``status``/``blocked``, which accepted any bare token before this
# gate was added.  Enterprise adds a second reason, since accounts
# provisioned over LDAP or SAML need not follow the dotcom grammar at
# all.
#
# A *leading* hyphen is still refused: no such account is known, and
# the likelier cause is a mistyped command-line flag, which is worth
# reporting as a bad target rather than expanding into a request.
_OWNER_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9-]{0,38}\Z")

#: First segments that GitHub reserves as URL routes, so they can never
#: be an owner.  Expanding a shorthand beginning with one would produce
#: a URL meaning something other than ``owner/repo`` --- and in the case
#: of ``orgs``, something broader.
_URL_ONLY_ROUTES = frozenset({"orgs"})


def looks_like_host(segment: str) -> bool:
    """Report whether a leading path segment names a host.

    GitHub owner logins are alphanumerics and hyphens only, so a dot
    settles it, and so does a port.

    Bare ``localhost`` deliberately does *not* qualify.  It satisfies
    the login grammar and names a real GitHub account, so treating it
    as a host would contradict the documented rule and make
    ``localhost/widget`` unreachable.  A local server still works when
    named with a port or an explicit scheme.

    Args:
        segment: The first segment of a schemeless target.

    Returns:
        True when the segment should be read as a hostname.
    """
    segment = segment.strip().lower()
    if not segment:
        return False
    if _PORT_SUFFIX_RE.search(segment):
        return True
    return "." in segment


def looks_like_owner(segment: str) -> bool:
    """Report whether a segment is a plausible GitHub owner login.

    Logins are alphanumerics and hyphens, no leading or trailing
    hyphen, at most 39 characters.  Checking the shape keeps the
    shorthand from swallowing arbitrary text: ``lfreleng-actions`` is an
    owner, ``not a url`` is not.

    Args:
        segment: The first segment of a schemeless target.

    Returns:
        True when the segment could name an owner.
    """
    return bool(_OWNER_RE.match(segment.strip()))


def _is_scp_remote(match: re.Match[str]) -> bool:
    """Decide whether an ambiguous ``host:tail`` is an scp remote.

    ``git@github.com:29418/widget.git`` is a clone URL for an owner
    named ``29418``; ``ghe.example.com:8443/acme`` is a host and a
    port.  Userinfo settles it --- a port never follows one --- and
    without userinfo a purely numeric leading segment is read as a
    port, which is the commoner intent.
    """
    if match.group("user"):
        return True
    return not _NUMERIC_PATH_RE.match(match.group("path"))


def normalize_target(value: str, *, default_host: str | None = None) -> str:
    """Expand an abbreviated or git-remote target into an absolute URL.

    An ordinary ``http(s)://`` URL is rewritten rather than passed
    through: any credentials are removed, and a trailing ``.git`` comes
    off *conditionally*.  The suffix survives on a change path and
    inside a Gerrit query, where removing it would either repair a
    malformed URL into a live reference or alter the value searched
    for --- see :mod:`dependamerge.url_parser.git_suffix`.

    Abbreviated and remote forms expand as follows:

    ==============================  =====================================
    Input                           Result
    ==============================  =====================================
    ``owner``                       ``https://github.com/owner``
    ``owner/repo``                  ``https://github.com/owner/repo``
    ``github.com/owner``            ``https://github.com/owner``
    ``ghe.example.com/owner/repo``  ``https://ghe.example.com/owner/repo``
    ``git@github.com:owner/repo``   ``https://github.com/owner/repo``
    ``ssh://git@host:29418/proj``   ``https://host/proj``
    ==============================  =====================================

    Args:
        value: The raw target as typed, pasted, or read from a remote.
        default_host: Host to assume for a shorthand that names none.
            Defaults to :func:`default_github_host`.

    Returns:
        An absolute ``http(s)://`` URL.  Empty input is returned
        unchanged so the existing "URL cannot be empty" errors keep
        their wording and their callers.
    """
    value = (value or "").strip()
    if not value:
        return value

    # A network-path reference ("//host/path") names a host despite
    # starting with a slash, so it has to be handled before the rooted
    # exemption below.  Left to that branch it was returned untouched,
    # which skipped credential stripping entirely: ``urlparse`` reads
    # ``user:SECRET@github.com`` as the authority, so the secret
    # survived into the parsed URL and into any error quoting the
    # netloc.  Treated as the web URL it is, which strips userinfo and
    # keeps the port for the port check to reject.
    if value.startswith("//"):
        return _rebuild("https://" + _strip_userinfo(value[2:], strip_port=False))

    # A rooted path names no host and is not a shorthand --- nobody
    # types "/owner/repo" to mean an abbreviation.  Left alone so the
    # parsers still reject it with "URL must include a hostname" rather
    # than silently resolving it against the default host.
    if value.startswith("/"):
        return value

    # ssh://, git://, and friends: keep the host and path, drop the
    # scheme, any credentials, and any port.  A Gerrit SSH remote such
    # as ssh://user@gerrit.example.org:29418/releng/tool differs from
    # its web URL only in those three things.
    scheme_match = re.match(r"\A([A-Za-z][A-Za-z0-9+.-]*)://(.*)\Z", value, re.DOTALL)
    if scheme_match:
        scheme = scheme_match.group(1).lower()
        if scheme in _WEB_SCHEMES:
            # Credentials are stripped even here, where the scheme and
            # port are kept.  A clone remote may embed a token, and the
            # normalised URL is printed back to the operator when a
            # target is inferred from a checkout --- which would put
            # that token in the terminal and in any captured log.
            return _rebuild(
                f"{scheme}://"
                + _strip_userinfo(scheme_match.group(2), strip_port=False)
            )
        if scheme not in _GIT_TRANSPORT_SCHEMES:
            # Anything else is refused outright.  Returning it unchanged
            # is not enough: the parsers read only the netloc and path,
            # so ``javascript://github.com/acme/widget`` would parse as
            # an ordinary repository and trigger a real operation.
            raise UrlParseError(
                f"Unsupported URL scheme {scheme!r}. Targets must be "
                "http(s) URLs, git remotes, or shorthand."
            )
        remainder = scheme_match.group(2)
        # A git transport carries a port --- Gerrit's 29418, for
        # instance --- which has no bearing on the web URL, so it goes
        # along with any credentials.
        return _rebuild("https://" + _strip_userinfo(remainder, strip_port=True))

    # scp-style remote, which has no scheme at all.
    scp = _SCP_REMOTE_RE.match(value)
    if scp and _is_scp_remote(scp):
        return _rebuild(f"https://{scp.group('host')}/{scp.group('path').lstrip('/')}")

    # Schemeless.  Whether the first segment is a host or an owner is
    # what decides between a URL and a shorthand.
    first = value.split("/", 1)[0]
    if looks_like_host(first):
        # Schemeless but host-shaped, so this is a web URL missing only
        # its scheme.  Any port is part of that URL and is kept.
        return _rebuild("https://" + _strip_userinfo(value, strip_port=False))

    if not looks_like_owner(first):
        # Neither a host nor a possible login.  Returned unchanged so
        # the parsers reject it on their own terms, rather than being
        # expanded into a plausible-looking URL that cannot resolve.
        return value

    remainder = value.split("/", 1)[1].strip("/") if "/" in value else ""
    if first.lower() in _URL_ONLY_ROUTES:
        # ``orgs`` is a route segment in a GitHub URL, not an owner, and
        # GitHub reserves it, so no such account can exist.  Left to
        # expand, ``orgs/acme`` produces ``https://github.com/orgs/acme``
        # --- the canonical owner-wide URL --- so a two-segment
        # shorthand naming one repository silently widens into a merge
        # of everything ``acme`` owns.  Bare ``orgs`` expands to a path
        # the parsers then reject as a malformed repository URL, which
        # explains nothing.  Both are refused here, where the reason is
        # known.
        if remainder:
            raise UrlParseError(
                f"{value!r} is ambiguous: {first!r} is a path segment in a "
                "GitHub URL, not an owner. For every repository owned by "
                f"{remainder!r}, give the owner on its own; "
                "for a single repository, give the full URL."
            )
        raise UrlParseError(
            f"{first!r} is a path segment in a GitHub URL, not an owner, "
            "so no account has that name. Give an owner, an owner/repo "
            "pair, or a full URL."
        )

    # Only now is a default host needed.  Resolving it earlier makes an
    # explicit URL --- a Gerrit one, even --- fail on an unrelated
    # GitHub host misconfiguration it never consults.
    host = (default_host or default_github_host()).strip().lower()
    return _rebuild(f"https://{host}/{value.lstrip('/')}")


def _strip_userinfo(authority_and_path: str, *, strip_port: bool) -> str:
    """Drop ``user@`` --- and optionally ``:port`` --- from ``host/path``."""
    parts = authority_and_path.split("/", 1)
    authority = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    if "@" in authority:
        authority = authority.rsplit("@", 1)[1]
    if strip_port:
        authority = _PORT_SUFFIX_RE.sub("", authority)
    return f"{authority}/{rest}" if rest else authority


def _rebuild(url: str, *, strip_git: bool = True) -> str:
    """Reassemble ``url``, optionally dropping a ``.git`` path suffix."""
    if not strip_git:
        return url
    match = re.match(r"\A(https?://[^/]+)(/.*)?\Z", url, re.DOTALL)
    if not match:
        return url
    authority = match.group(1)
    path = match.group(2) or ""
    if not path:
        return authority
    # Preserve any query or fragment; only the path carries ``.git``.
    split = re.match(r"\A([^?#]*)(.*)\Z", path, re.DOTALL)
    assert split is not None
    path_part = split.group(1)
    # The same path means different things on the two platforms, so the
    # decision needs the host as well.
    netloc = authority.split("://", 1)[-1].rsplit("@", 1)[-1]
    host = _PORT_SUFFIX_RE.sub("", netloc).lower()
    if not strips_git_suffix(path_part, host):
        return url
    return f"{authority}{strip_git_suffix(path_part)}{split.group(2)}"
