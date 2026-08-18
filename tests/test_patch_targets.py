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


def _module_aliases(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    """Map local names bound to a ``dependamerge`` module onto its path.

    Covers ``import dependamerge.x``, ``import dependamerge.x as m``,
    ``from dependamerge import x`` and ``from dependamerge import x as m``.
    These appear inside test function bodies as often as at module level,
    so the whole tree is walked.

    ``import a.b`` without ``as`` binds only the root name ``a``, so it
    maps to an empty prefix and the attribute chain at the call site
    supplies the rest. Treating the dotted name as the bound one would
    miss ``setattr(dependamerge.github_async, …)`` entirely.
    """
    aliases: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "dependamerge" and not alias.name.startswith(
                    "dependamerge."
                ):
                    continue
                suffix = alias.name.removeprefix("dependamerge").lstrip(".")
                parts = tuple(p for p in suffix.split(".") if p)
                if alias.asname:
                    aliases[alias.asname] = parts
                else:
                    aliases["dependamerge"] = ()
        elif isinstance(node, ast.ImportFrom) and node.module == "dependamerge":
            for alias in node.names:
                aliases[alias.asname or alias.name] = (alias.name,)
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
        components = [*aliases[target.id], *parts]
        if not components:
            # ``setattr(dependamerge, "x", …)`` patches the root package,
            # which owns no module namespace of its own here.
            continue
        found.add((".".join(components), attr.value))
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


def _is_type_checking(test: ast.expr) -> bool:
    """Recognise ``if TYPE_CHECKING:`` and ``if typing.TYPE_CHECKING:``.

    The attribute form is matched only against ``typing``, so an
    unrelated runtime flag such as ``settings.TYPE_CHECKING`` is not
    mistaken for the sentinel and skipped.
    """
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
    )


def _executing_imports(tree: ast.Module) -> list[tuple[ast.ImportFrom, bool]]:
    """Every executing import of a ``dependamerge`` module, and its scope.

    Relative and absolute forms both bind the same object:
    ``from dependamerge.github_async import _now`` shadows a package
    substitution exactly as ``from . import _now`` does, so restricting
    this to relative imports left an ordinary spelling unguarded.

    ``TYPE_CHECKING`` bodies are excluded because they never execute.
    Both scopes are collected: being inside a function is not on its own
    enough to make an import safe --- see :func:`_reads_target_namespace`.
    """
    found: list[tuple[ast.ImportFrom, bool]] = []

    def collect(node: ast.ImportFrom) -> bool:
        return bool(node.level) or bool(
            node.module
            and (
                node.module == "dependamerge" or node.module.startswith("dependamerge.")
            )
        )

    def walk(body: list[ast.stmt], *, local: bool) -> None:
        for node in body:
            if isinstance(node, ast.ImportFrom) and collect(node):
                found.append((node, local))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(node.body, local=True)
            elif isinstance(node, ast.ClassDef):
                walk(node.body, local=local)
            elif isinstance(node, ast.If):
                if not _is_type_checking(node.test):
                    walk(node.body, local=local)
                walk(node.orelse, local=local)
            elif isinstance(node, ast.Try):
                walk(node.body, local=local)
                walk(node.orelse, local=local)
                walk(node.finalbody, local=local)
                for handler in node.handlers:
                    walk(handler.body, local=local)
            elif isinstance(node, (ast.With, ast.AsyncWith, ast.For, ast.While)):
                walk(node.body, local=local)

    walk(tree.body, local=False)
    return found


def _import_source(sibling: Path, node: ast.ImportFrom) -> Path:
    """Resolve the namespace an import reads from, as a path."""
    if not node.level:
        # Absolute: dependamerge.a.b -> <src>/a/b
        suffix = (node.module or "").removeprefix("dependamerge").lstrip(".")
        return _SRC.joinpath(*[p for p in suffix.split(".") if p])
    base = sibling.parent
    for _ in range(node.level - 1):
        base = base.parent
    return base.joinpath(*node.module.split(".")) if node.module else base


def _reads_target_namespace(source: Path, target: Path) -> bool:
    """Report whether ``source`` names the same namespace as ``target``.

    A substitution replaces an attribute of one namespace. Only an import
    that reads *that* namespace can observe it. A function-local
    ``from ._errors import _now`` is re-resolved per call but still reads
    ``_errors``, so it ignores a substitution on the package just as
    silently as a module-scope binding would.
    """
    namespace = target.parent if target.name == "__init__.py" else target
    return source == namespace or source.with_suffix(".py") == namespace


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


def _dotted_name(path: Path) -> str:
    """Return the ``dependamerge``-relative dotted name of a module file."""
    rel = path.relative_to(_SRC)
    parts = list(rel.parts[:-1])
    if rel.stem != "__init__":
        parts.append(rel.stem)
    return ".".join(parts)


class TestSubstitutionsReachTheirCallSites:
    """Patched names must not be shadowed by a sibling's direct binding."""

    def test_no_sibling_shadows_a_patch_target(self) -> None:
        all_targets = _patch_targets()
        violations: list[str] = []
        for module, names in sorted(all_targets.items()):
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
                for node, is_local in _executing_imports(tree):
                    if is_local and _reads_target_namespace(
                        _import_source(sibling, node), target
                    ):
                        # Re-resolved per call, from the namespace being
                        # substituted, so it observes the substitution.
                        continue
                    for alias in node.names:
                        # An alias freezes the original object just as an
                        # unaliased import does, so what matters is whether
                        # the *bound* name is used, not what it is called.
                        bound = alias.asname or alias.name
                        if alias.name not in names or bound not in referenced:
                            continue
                        if alias.name in all_targets.get(_dotted_name(sibling), set()):
                            # The sibling owns its own binding and tests
                            # substitute it there, so it is deliberate
                            # rather than a shadow of this target.
                            continue
                        renamed = f" as {bound}" if bound != alias.name else ""
                        violations.append(
                            f"  {sibling.relative_to(_ROOT)}:{node.lineno} binds "
                            f"{alias.name}{renamed}, which tests substitute at "
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

    def test_check_finds_module_object_substitutions(self) -> None:
        """The aggregate count alone cannot protect the AST scan.

        Scanning string literals by itself already finds enough targets
        to satisfy the count above, so a regression that dropped the
        ``setattr(mod, "name", …)`` handling would pass unnoticed --- and
        those are the substitutions this guard exists for. They are named
        explicitly here because they are reachable *only* through the AST
        path.
        """
        found = _patch_targets().get("github_async", set())
        expected = {
            "_now",
            "_is_transient_server_error",
            "_APPROVE_RETRY_BASE_DELAY",
        }
        assert expected <= found, (
            f"Module-object substitutions missing from the scan: "
            f"{sorted(expected - found)}. These are written as "
            f'monkeypatch.setattr(mod, "name", …) and are invisible to a '
            "scan of dotted string literals."
        )
