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


def _target_exists(base: Path, module: str | None) -> bool:
    """Report whether ``module`` resolves to a file or package under ``base``."""
    if module is None:
        # ``from . import x`` — the package itself always exists.
        return True
    head = module.split(".")[0]
    return (base / f"{head}.py").exists() or (base / head).is_dir()


def _relative_imports() -> list[tuple[Path, int, str, int]]:
    """Return every relative import as (path, line, module, level)."""
    found: list[tuple[Path, int, str, int]] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                found.append((path, node.lineno, node.module or "", node.level))
    return found


class TestRelativeImportsResolve:
    """Every relative import must name something that exists."""

    def test_relative_import_targets_exist(self) -> None:
        violations: list[str] = []
        for path, line, module, level in _relative_imports():
            # ``level`` dots walk up from the containing directory.
            base = path.parent
            for _ in range(level - 1):
                base = base.parent
            if not _target_exists(base, module or None):
                rel = path.relative_to(_SRC_ROOT.parent.parent)
                dots = "." * level
                violations.append(
                    f"  {rel}:{line}: from {dots}{module} import …\n"
                    f"    resolves to {base / module.split('.')[0]}, which does not exist"
                )
        assert not violations, (
            "Relative imports naming a non-existent module:\n"
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
