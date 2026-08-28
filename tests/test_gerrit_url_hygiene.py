# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""Architectural tests enforcing Gerrit URL construction via GerritUrlBuilder.

All Gerrit URL construction in production code MUST go through the centralised
``GerritUrlBuilder`` class (``dependamerge.gerrit.urls``).  Building URLs with
ad-hoc f-strings bypasses base-path handling and leads to inconsistent URLs
when a Gerrit instance is deployed behind a reverse-proxy prefix (e.g.
``/infra``).

These tests scan production source files for patterns that indicate direct
Gerrit URL construction and fail if any are found outside the builder itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src" / "dependamerge"

# Files that are *allowed* to construct Gerrit URLs directly.
# gerrit/urls/builder.py  — the canonical builder implementation
_ALLOWED_FILES: set[str] = {
    str(_SRC_ROOT / "gerrit" / "urls" / "builder.py"),
}

# Files that only *detect* / *parse* Gerrit URL shapes (pattern matching,
# not construction) — these legitimately reference /c/ and /+/ in regexes,
# docstrings, and error messages but never emit a URL.
_DETECTION_ONLY_FILES: set[str] = {
    str(_SRC_ROOT / "url_parser" / "change.py"),
    str(_SRC_ROOT / "url_parser" / "topic.py"),
}

# Files that *rewrite* a URL the caller supplied — adding a missing
# scheme, expanding a shorthand, converting a git remote — rather than
# synthesising a Gerrit endpoint from parts.
#
# The distinction that matters to this guard is base-path handling.  A
# violation is dangerous because it invents a path and so can omit a
# deployment's ``/infra`` prefix.  Normalisation carries the caller's
# path through untouched, so there is no path for it to get wrong; the
# only component it supplies is the scheme.  Anything here that starts
# composing Gerrit endpoints belongs in GerritUrlBuilder instead.
_NORMALISATION_FILES: set[str] = {
    str(_SRC_ROOT / "url_parser" / "shorthand.py"),
}

# ---------------------------------------------------------------------------
# Patterns that indicate direct Gerrit URL construction
# ---------------------------------------------------------------------------

# Each entry is (compiled_regex, human-readable description).
# The regexes are intentionally broad to catch creative variants.

_DIRECT_URL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # f"https://{host}/c/{project}/+/{number}"  — change URL
    (
        re.compile(
            r"""f["']https?://\{[^}]+\}/c/\{[^}]+\}/\+/\{[^}]+\}""",
        ),
        "Direct Gerrit change URL construction (use GerritUrlBuilder.change_url())",
    ),
    # f"https://{host}/{base_path}/c/{project}/+/{number}"  — change URL with base_path
    (
        re.compile(
            r"""f["']https?://\{[^}]+\}/\{[^}]+\}/c/\{[^}]+\}/\+/\{[^}]+\}""",
        ),
        "Direct Gerrit change URL with base_path (use GerritUrlBuilder.change_url())",
    ),
    # f"https://{host}/q/{change_id}"  — search/query URL
    (
        re.compile(
            r"""f["']https?://\{[^}]+\}/q/\{[^}]+\}""",
        ),
        "Direct Gerrit search URL construction (use GerritUrlBuilder.web_url())",
    ),
    # f"https://{host}/changes/"  — REST API changes endpoint
    (
        re.compile(
            r"""f["']https?://\{[^}]+\}/changes/""",
        ),
        "Direct Gerrit changes API URL (use GerritUrlBuilder.changes_api_url())",
    ),
    # f"https://{host}/a/changes/"  — authenticated REST API changes endpoint
    (
        re.compile(
            r"""f["']https?://\{[^}]+\}/a/changes/""",
        ),
        "Direct Gerrit authenticated API URL (use GerritUrlBuilder.api_url())",
    ),
    # f"{base}/q/{change_id}"  — search URL built from a base variable
    (
        re.compile(
            r"""f["']\{[^}]+\}/q/\{[^}]+\}""",
        ),
        "Direct Gerrit search URL from base variable (use GerritUrlBuilder.web_url())",
    ),
]

# Pattern for constructing a Gerrit base URL from host + base_path.
# This catches: f"https://{host}/{base_path}/"  and similar shapes.
# Only flagged outside urls/builder.py AND the low-level client bootstrap
# (gerrit/client.py:build_client is the HTTP transport layer and needs
# its own base URL before the builder exists).
_BASE_URL_PATTERN = re.compile(
    r"""f["']https?://\{[^}]+\}/\{[^}]+(?:\.strip\([^)]*\))?\}/?\s*["']""",
)

_BASE_URL_ALLOWED_FILES: set[str] = {
    str(_SRC_ROOT / "gerrit" / "urls" / "builder.py"),
    # The REST client bootstraps its own base URL before the builder is
    # available — this is acceptable at the transport layer.
    str(_SRC_ROOT / "gerrit" / "client.py"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _production_python_files() -> list[Path]:
    """Return all production .py files under src/dependamerge/."""
    return sorted(_SRC_ROOT.rglob("*.py"))


def _is_comment_line(line: str) -> bool:
    """Check whether *line* is a ``#`` comment (ignoring leading whitespace)."""
    return line.lstrip().startswith("#")


def _scan_file(
    path: Path,
    patterns: list[tuple[re.Pattern[str], str]],
) -> list[tuple[int, str, str]]:
    """Scan a single file for pattern violations.

    Skips:
    - ``#`` comment lines
    - Triple-quoted docstring / string-literal blocks whose opening
      delimiter (``\"\"\"``, ``'''``) appears at the start of the line
      (after whitespace).  The *entire* block is skipped — opening
      line, interior lines, and closing line — so that example URLs
      in docstrings do not trigger false positives.
    - Lines starting with ``f\"\"\"`` / ``f'''`` are **not** skipped
      because f-strings are executed code whose interpolations may
      construct URLs.

    **Limitation:** triple-quoted strings that start mid-line (e.g.
    ``text = \"\"\"...``) are not detected and will be scanned normally.
    This is acceptable for this project's codebase where docstrings
    and multi-line string literals consistently start on their own
    line.  A ``tokenize``/AST approach would be needed for full
    coverage.

    Returns list of (line_number, line_content, description) tuples.
    """
    violations: list[tuple[int, str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return violations

    # Stateful tracking of triple-quoted blocks.
    # When inside a block, *in_triple_quote* holds the delimiter
    # (either '"""' or "'''"); otherwise it is None.
    in_triple_quote: str | None = None

    for line_no, line in enumerate(lines, start=1):
        stripped = line.lstrip()

        # --- Inside a triple-quoted block: skip until closing delimiter ---
        if in_triple_quote is not None:
            if in_triple_quote in stripped:
                # Closing delimiter found — end the block, skip this line
                in_triple_quote = None
            continue

        # --- Comment line ---
        if _is_comment_line(line):
            continue

        # --- Opening a new triple-quoted block (non-f-string) ---
        for delim in ('"""', "'''"):
            if stripped.startswith(delim):
                # Count occurrences of the delimiter on this line.
                # A single-line docstring (e.g. """text""") has ≥ 2;
                # only enter the "inside block" state for multi-line
                # strings that open but don't close on the same line.
                if stripped.count(delim) == 1:
                    in_triple_quote = delim
                # Either way, skip the opening / single-line docstring
                break
        else:
            # Not a comment, not a docstring — scan for violations
            for pattern, description in patterns:
                if pattern.search(line):
                    violations.append((line_no, line.strip(), description))

    return violations


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoDirectGerritUrlConstruction:
    """Ensure production code uses GerritUrlBuilder for all Gerrit URLs."""

    def test_no_direct_gerrit_url_construction_in_production_code(self) -> None:
        """Production files must not construct Gerrit URLs with f-strings.

        All Gerrit URL building should go through GerritUrlBuilder to
        ensure the detected base_path (``/infra``, etc.) is consistently
        applied.  Direct f-string construction bypasses this and causes
        broken URLs when a base path is present.
        """
        all_violations: list[str] = []

        for py_file in _production_python_files():
            file_str = str(py_file)

            if file_str in _ALLOWED_FILES:
                continue
            if file_str in _DETECTION_ONLY_FILES:
                continue

            violations = _scan_file(py_file, _DIRECT_URL_PATTERNS)
            for line_no, line_content, description in violations:
                rel = py_file.relative_to(_PROJECT_ROOT)
                all_violations.append(
                    f"  {rel}:{line_no}: {description}\n    {line_content}"
                )

        if all_violations:
            msg = (
                "Direct Gerrit URL construction detected in production code.\n"
                "Use GerritUrlBuilder (from dependamerge.gerrit.urls) instead.\n\n"
                + "\n".join(all_violations)
            )
            pytest.fail(msg)

    def test_no_direct_gerrit_base_url_construction(self) -> None:
        """Production files must not build Gerrit base URLs manually.

        The pattern ``f"https://{host}/{base_path}/"`` should only appear
        in ``gerrit/urls/builder.py`` (the builder) and ``gerrit/client.py``
        (transport-layer bootstrap).  Everywhere else, use
        ``GerritUrlBuilder._build_base_url()`` or its public methods.
        """
        all_violations: list[str] = []

        for py_file in _production_python_files():
            file_str = str(py_file)

            if file_str in _BASE_URL_ALLOWED_FILES:
                continue
            if file_str in _DETECTION_ONLY_FILES:
                continue
            if file_str in _NORMALISATION_FILES:
                continue

            violations = _scan_file(
                py_file, [(_BASE_URL_PATTERN, "Direct Gerrit base URL construction")]
            )
            for line_no, line_content, description in violations:
                rel = py_file.relative_to(_PROJECT_ROOT)
                all_violations.append(
                    f"  {rel}:{line_no}: {description}\n    {line_content}"
                )

        if all_violations:
            msg = (
                "Direct Gerrit base URL construction detected.\n"
                "Use GerritUrlBuilder (from dependamerge.gerrit.urls) instead.\n\n"
                + "\n".join(all_violations)
            )
            pytest.fail(msg)

    def test_builder_exposes_required_methods(self) -> None:
        """GerritUrlBuilder must expose all the URL-building methods that
        production code might need, so there is no excuse to bypass it."""
        from dependamerge.gerrit.urls import GerritUrlBuilder

        required_methods = [
            "api_url",
            "web_url",
            "change_url",
            "changes_api_url",
            "change_api_url",
            "review_url",
            "submit_url",
        ]
        for method_name in required_methods:
            assert hasattr(GerritUrlBuilder, method_name), (
                f"GerritUrlBuilder is missing '{method_name}' — "
                f"add it before production code resorts to direct construction"
            )
            assert callable(getattr(GerritUrlBuilder, method_name)), (
                f"GerritUrlBuilder.{method_name} must be callable"
            )

    def test_builder_respects_base_path(self) -> None:
        """Verify that the builder correctly includes the base path in all
        URL types — this is the whole reason we centralise construction."""
        from dependamerge.gerrit.urls import GerritUrlBuilder

        builder = GerritUrlBuilder(
            host="gerrit.example.org",
            base_path="infra",
            auto_discover=False,
        )

        # change_url must include base path
        url = builder.change_url("my-project", 12345)
        assert "/infra/" in url, f"change_url missing base path: {url}"
        assert "/c/my-project/+/12345" in url

        # web_url must include base path
        url = builder.web_url("dashboard")
        assert "/infra/" in url, f"web_url missing base path: {url}"

        # api_url must include base path
        url = builder.api_url("/changes/")
        assert "/infra/" in url, f"api_url missing base path: {url}"

    def test_builder_works_without_base_path(self) -> None:
        """Verify the builder works correctly when no base path is set."""
        from dependamerge.gerrit.urls import GerritUrlBuilder

        builder = GerritUrlBuilder(
            host="gerrit.example.org",
            base_path=None,
            auto_discover=False,
        )

        url = builder.change_url("project", 99999)
        assert url == "https://gerrit.example.org/c/project/+/99999"

        # No double slashes
        assert "org//" not in url

    def test_allowed_files_exist(self) -> None:
        """Ensure the allow-lists reference files that actually exist.

        Prevents stale entries hiding violations after file renames.
        """
        for allowed in (
            _ALLOWED_FILES
            | _BASE_URL_ALLOWED_FILES
            | _DETECTION_ONLY_FILES
            | _NORMALISATION_FILES
        ):
            assert Path(allowed).exists(), (
                f"Allow-listed file does not exist: {allowed}\n"
                "Update the allow-list in test_gerrit_url_hygiene.py"
            )
