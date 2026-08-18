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


def _module_aliases(tree: ast.Module) -> dict[str, str]:
    """Map local names bound to a ``dependamerge`` module onto its path.

    Covers ``import dependamerge.x as m``, ``from dependamerge import x``
    and ``from dependamerge import x as m``. These appear inside test
    function bodies as often as at module level, so the whole tree is
    walked.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dependamerge" or alias.name.startswith(
                    "dependamerge."
                ):
                    bound = alias.asname or alias.name
                    aliases[bound] = alias.name.removeprefix("dependamerge.")
        elif isinstance(node, ast.ImportFrom) and node.module == "dependamerge":
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
    return aliases


def _aliased_targets(tree: ast.Module) -> set[tuple[str, str]]:
    """Find ``setattr(m, "name", …)`` and ``patch.object(m, "name")`` calls.

    The substitutions that motivated this guard are written this way
    rather than as dotted strings, so a scan of string literals alone
    would miss precisely the cases it exists to protect.
    """
    aliases = _module_aliases(tree)
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in ("setattr", "object"):
            continue
        target, attr = node.args[0], node.args[1]
        if not (isinstance(attr, ast.Constant) and isinstance(attr.value, str)):
            continue
        # ``m`` or ``m.submodule``
        parts: list[str] = []
        while isinstance(target, ast.Attribute):
            parts.insert(0, target.attr)
            target = target.value
        if not isinstance(target, ast.Name) or target.id not in aliases:
            continue
        module = ".".join([aliases[target.id], *parts])
        found.add((module, attr.value))
    return found


def _patch_targets() -> dict[str, set[str]]:
    """Map ``dependamerge.<module>`` to the names tests substitute on it."""
    targets: dict[str, set[str]] = defaultdict(set)
    for path in _TESTS.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for match in _TARGET.finditer(source):
            targets[match.group(1)].add(match.group(2))
        for module, name in _aliased_targets(ast.parse(source)):
            targets[module].add(name)
    return targets


def _module_scope_relative_imports(tree: ast.Module) -> list[ast.ImportFrom]:
    """Relative imports bound at module scope, which shadow for the whole module.

    Imports inside a function body are re-resolved on every call, so a
    substitution applied before the call is still observed. Only a
    module-scope binding freezes the name at import time, and only that
    can shadow silently. ``TYPE_CHECKING`` blocks never execute.
    """
    imports: list[ast.ImportFrom] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level:
            imports.append(node)
        elif isinstance(node, ast.If):
            test = node.test
            is_type_checking = (
                isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
            ) or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
            if is_type_checking:
                continue
            imports.extend(
                child
                for child in node.body
                if isinstance(child, ast.ImportFrom) and child.level
            )
    return imports


def _resolve_target(module: str) -> Path | None:
    """Return the file owning ``dependamerge.<module>``'s namespace.

    A target may name a module (``cli._deps``) or a package
    (``github_async``). For a package the namespace lives in its
    ``__init__.py``; resolving only the ``.py`` form silently skipped
    every package target, including the three that motivated this guard.
    """
    as_module = _SRC / (module.replace(".", "/") + ".py")
    if as_module.exists():
        return as_module
    as_package = _SRC / module.replace(".", "/") / "__init__.py"
    if as_package.exists():
        return as_package
    return None


class TestSubstitutionsReachTheirCallSites:
    """Patched names must not be shadowed by a sibling's direct binding."""

    def test_no_sibling_shadows_a_patch_target(self) -> None:
        violations: list[str] = []
        for module, names in sorted(_patch_targets().items()):
            target = _resolve_target(module)
            if target is None:
                continue
            package = target.parent
            if package == _SRC and target.name != "__init__.py":
                # A top-level module owns the namespace being patched, so a
                # binding there is exactly what the substitution replaces.
                continue
            for sibling in sorted(package.glob("*.py")):
                if sibling.name in ("__init__.py", target.name):
                    continue
                tree = ast.parse(sibling.read_text(encoding="utf-8"))
                referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                for node in _module_scope_relative_imports(tree):
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
