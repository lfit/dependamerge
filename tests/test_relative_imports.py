# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Guard against relative imports that name a non-existent sibling.

Splitting a module into a package changes what a single-dot import
means: inside ``dependamerge/copilot_handler/threads.py``,
``from .github_graphql import X`` resolves to
``dependamerge.copilot_handler.github_graphql`` rather than the
top-level module it did before the split, and raises ``ImportError``.

Function-level imports make this invisible to both the type checkers and
the test suite until the branch that contains them runs in production,
so this check is static: it walks every relative import in the package
and asserts the target exists.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "dependamerge"


def _module_exists(base: Path, dotted: str) -> bool:
    """Report whether ``dotted`` resolves to a module or package under ``base``."""
    target = base.joinpath(*dotted.split("."))
    return target.with_suffix(".py").exists() or target.is_dir()


def _package_exports(base: Path) -> set[str]:
    """Names bound at module scope by ``base/__init__.py``.

    ``from . import x`` is satisfied by a submodule *or* by a name the
    package's ``__init__`` defines or re-exports, so both are collected.

    Only module scope counts. Descending into function and class bodies
    would treat a local variable as an export, letting a broken
    ``from . import x`` pass whenever some unrelated local happened to
    share the name.
    """
    init = base / "__init__.py"
    if not init.exists():
        return set()
    return _bound_at_module_scope(ast.parse(init.read_text(encoding="utf-8")).body)


def _bound_at_module_scope(body: list[ast.stmt]) -> set[str]:
    """Collect names bound by ``body``, descending only into control flow.

    ``if`` / ``try`` branches still bind at module scope, so they are
    followed; function and class bodies introduce their own scope and
    are not.
    """
    names: set[str] = set()
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.If):
            names |= _bound_at_module_scope(node.body)
            names |= _bound_at_module_scope(node.orelse)
        elif isinstance(node, ast.Try):
            names |= _bound_at_module_scope(node.body)
            names |= _bound_at_module_scope(node.orelse)
            names |= _bound_at_module_scope(node.finalbody)
            for handler in node.handlers:
                names |= _bound_at_module_scope(handler.body)
    return names


def _unresolved(base: Path, node: ast.ImportFrom) -> list[str]:
    """Return the parts of ``node`` that do not resolve under ``base``.

    Two shapes need checking. ``from .a.b import X`` names a module, and
    the whole dotted path must resolve --- not merely its first
    component. ``from . import x`` names either a submodule or an
    attribute of the package, and raises ``ImportError`` when it is
    neither, so each alias is resolved individually.
    """
    if node.module:
        return [] if _module_exists(base, node.module) else [node.module]
    exports = _package_exports(base)
    return [
        alias.name
        for alias in node.names
        if alias.name != "*"
        and not _module_exists(base, alias.name)
        and alias.name not in exports
    ]


def _relative_imports() -> list[tuple[Path, ast.ImportFrom]]:
    """Return every relative import with the file that contains it."""
    found: list[tuple[Path, ast.ImportFrom]] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                found.append((path, node))
    return found


class TestRelativeImportsResolve:
    """Every relative import must name something that exists."""

    def test_relative_import_targets_exist(self) -> None:
        violations: list[str] = []
        for path, node in _relative_imports():
            # ``level`` dots walk up from the containing directory.
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            for missing in _unresolved(base, node):
                rel = path.relative_to(_SRC_ROOT.parent.parent)
                dots = "." * node.level
                what = (
                    f"{dots}{node.module}"
                    if node.module
                    else f"{dots} import {missing}"
                )
                violations.append(
                    f"  {rel}:{node.lineno}: from {what}\n"
                    f"    {base.joinpath(*missing.split('.'))} does not exist, "
                    f"and no such name is exported by {base.name}"
                )
        assert not violations, (
            "Relative imports naming something that does not exist:\n"
            + "\n".join(violations)
            + "\n\nA single dot means 'this package'. After a module becomes a "
            "package, imports of *other* top-level modules need two dots."
        )

    def test_check_covers_the_package(self) -> None:
        """Guard the guard: a silent zero-import walk would pass vacuously."""
        imports = _relative_imports()
        assert len(imports) > 50, (
            f"Only {len(imports)} relative imports found; the walk is probably "
            "not reaching the package."
        )
