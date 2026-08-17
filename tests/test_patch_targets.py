# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Guard against substitutions that stop reaching the code they cover.

A call site resolves a global from its *own* module's namespace. So when
a test substitutes ``dependamerge.cli._deps.GitHubClient`` but a sibling
module has bound that name directly with ``from ..github_client import
GitHubClient``, the substitution still succeeds --- the attribute exists
--- while the code under test keeps using the real class. Nothing fails;
the test simply stops testing, and in the worst case starts talking to
GitHub.

Splitting a module into a package is what creates the gap, because names
that used to share one namespace no longer do. This check cross-refers
every patch target in the suite against the names each sibling module
binds at run time, so the gap cannot reopen silently.

Imports guarded by ``if TYPE_CHECKING:`` are exempt: they never execute,
so they cannot shadow a substitution.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src" / "dependamerge"
_TESTS = _ROOT / "tests"

_TARGET = re.compile(
    r'"dependamerge\.([a-z_0-9]+(?:\.[a-z_0-9]+)*)\.([A-Za-z_][A-Za-z_0-9]*)"'
)


def _patch_targets() -> dict[str, set[str]]:
    """Map ``dependamerge.<module>`` to the names tests substitute on it."""
    targets: dict[str, set[str]] = defaultdict(set)
    for path in _TESTS.rglob("*.py"):
        for match in _TARGET.finditer(path.read_text(encoding="utf-8")):
            targets[match.group(1)].add(match.group(2))
    return targets


def _runtime_relative_imports(tree: ast.Module) -> list[ast.ImportFrom]:
    """Relative imports that actually execute, ignoring TYPE_CHECKING blocks."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_type_checking = (
            isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
        ) or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
        if is_type_checking:
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    guarded.add(id(sub))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level and id(node) not in guarded
    ]


class TestSubstitutionsReachTheirCallSites:
    """Patched names must not be shadowed by a sibling's direct binding."""

    def test_no_sibling_shadows_a_patch_target(self) -> None:
        violations: list[str] = []
        for module, names in sorted(_patch_targets().items()):
            target = _SRC / (module.replace(".", "/") + ".py")
            if not target.exists():
                continue
            package = target.parent
            if package == _SRC:
                # A top-level module owns the namespace being patched, so a
                # binding there is exactly what the substitution replaces.
                continue
            for sibling in sorted(package.glob("*.py")):
                if sibling.name in ("__init__.py", target.name):
                    continue
                tree = ast.parse(sibling.read_text(encoding="utf-8"))
                referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                for node in _runtime_relative_imports(tree):
                    for alias in node.names:
                        bound = alias.asname or alias.name
                        if (
                            alias.name in names
                            and bound == alias.name
                            and alias.name in referenced
                        ):
                            violations.append(
                                f"  {sibling.relative_to(_ROOT)}:{node.lineno} binds "
                                f"{alias.name}, which tests substitute at "
                                f"dependamerge.{module}.{alias.name}"
                            )
        assert not violations, (
            "Substituted names shadowed by a direct import:\n"
            + "\n".join(sorted(set(violations)))
            + "\n\nReach these through the module object instead "
            "(`from . import _deps` … `_deps.Name`), so one substitution "
            "is seen by every caller."
        )

    def test_check_finds_patch_targets(self) -> None:
        """Guard the guard: an empty scan would pass vacuously."""
        targets = _patch_targets()
        total = sum(len(v) for v in targets.values())
        assert total > 20, (
            f"Only {total} patch targets found across {len(targets)} modules; "
            "the scan is probably not reaching the test suite."
        )
