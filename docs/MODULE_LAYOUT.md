<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2026 The Linux Foundation
-->

# Module layout

Most of `src/dependamerge/` uses packages rather than single modules. This
note records why, the conventions each package follows, and two invariants
that are easy to break and expensive to debug.

## Why packages

The `aislop` scanner reports `complexity/file-too-large` above about 330
code lines and `complexity/function-too-long` above 80. Measured against
the whole tree, the metric counts non-blank, non-comment lines rather than
the physical line count it prints, so a file can display as 430 lines and
still pass.

Practical budgets, with margin:

| Unit | Budget |
| --- | --- |
| Module | 320 code lines |
| Function | 80 code lines |
| Parameters | 6 |
| Nesting | 5 levels |

`aislop ci` runs in under a second and is authoritative; treat the budgets
above as planning figures and the scanner as the gate.

## Package conventions

Each package follows the same shape:

- An SPDX header and a module docstring on every file, describing that
  module's role rather than repeating the package's.
- `from __future__ import annotations` throughout.
- Relative imports between siblings.
- An `__init__.py` that re-exports the package's whole surface — private
  names included — with `__all__`, so `dependamerge.<pkg>.<name>` keeps
  resolving for every existing import and test.
- Multi-line parenthesised imports in `__init__.py`. The scanner's
  `ai-slop/unused-import` rule does not parse them, and re-exports are
  legitimately "unused" in the file that performs them.

Incidental stdlib imports stay out of the re-export list. Keeping them
would mean unused imports in `__init__.py`; the package still exposes every
name the original module defined.

### Splitting a large class

Where a class exceeds a module's budget, a mixin splits it: a `_XxxMixin`
holds the method bodies verbatim and the concrete class inherits them,
leaving the method surface untouched.

Mixins declare the attributes their concrete class constructs, which
basedpyright reports as uninitialised instance variables. Those modules
carry `# pyright: reportUninitializedInstanceVariable=false` on the line
above `from __future__ import annotations`. The suppression is per-file and
deliberate, so the rule keeps covering every real class.

## Invariant 1: relative imports must name a real sibling

Inside `copilot_handler/threads.py`, `from .github_graphql import X`
resolves to `dependamerge.copilot_handler.github_graphql`, not the
top-level module it named before the split. Use two dots.

Function-level imports hide this from mypy, basedpyright, ruff and the
test suite until the branch that contains them runs. It reached `main`
once, in code that would have raised `ImportError` in production.

`tests/test_relative_imports.py` walks every relative import in the
package and asserts its target exists.

## Invariant 2: reach substitutable names through a module object

A call site resolves a global from **its own module's** namespace. So if a
test substitutes `dependamerge.cli._deps.GitHubClient` while a sibling has
bound that name directly with `from ..github_client import GitHubClient`,
the substitution still succeeds — the attribute exists — and the code under
test keeps using the real class.

Nothing fails. The test stops testing, and in the worst case starts making
live API calls.

Where tests substitute a name, reach it through the module object:

```python
from . import _deps

client = _deps.GitHubClient(token)     # substitutable
```

not:

```python
from ._deps import GitHubClient

client = GitHubClient(token)           # shadows the substitution
```

`cli/_deps.py` is the canonical home for the CLI's substitutable
collaborators. `github_async` uses the same technique for `_now`,
`_is_transient_server_error` and `_APPROVE_RETRY_BASE_DELAY`, reached as
`_pkg.<name>`.

`tests/test_patch_targets.py` cross-refers every patch target in the suite
against the names each sibling module binds at run time. Imports guarded by
`if TYPE_CHECKING:` are exempt, since they never execute.

### When to split a module, and when not to

A module whose names tests patch *in its own namespace* should stay a
module and shed siblings instead, unless the patch targets move with it.
`gerrit/service.py`, `gerrit/submit_manager.py` and `gerrit/client.py` stay
modules for this reason: between them they carry 138 patch sites against
names their own code calls.

Where the retarget went the other way — `git_ops`, `progress_tracker`,
`rebase`, `cli` — the work first checked that the tests fail against the
old target. A retarget that stops biting is worse than no refactor.
