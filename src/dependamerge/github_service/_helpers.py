# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""Field coercion helpers and the removable rate-limit callback chain."""

from __future__ import annotations

import inspect
import logging
from typing import Any


def _str_or_none(value: Any) -> str | None:
    """Return ``value`` as a string when truthy, else None.

    Used when populating optional ``PullRequestInfo`` fields from
    GraphQL responses where the field may be missing entirely or
    explicitly null.
    """
    if isinstance(value, str) and value:
        return value
    return None


def _bool_or_none(value: Any) -> bool | None:
    """Coerce ``value`` to bool when present, else None.

    Mirrors :func:`_str_or_none` for boolean GraphQL fields
    (``isFork``).
    """
    if isinstance(value, bool):
        return value
    return None


def _clone_url_with_git_suffix(url: Any) -> str | None:
    """Synthesise a canonical ``.git`` clone URL from a GraphQL ``url``.

    GraphQL's ``Repository.url`` returns the HTTPS URL without the
    ``.git`` suffix that REST's ``clone_url`` includes.  This
    helper appends ``.git`` so both code paths produce the same
    string and downstream consumers (notably
    :func:`rebase.local_rebase_pr`) can treat them uniformly.

    Returns None when the input is missing or empty so the
    PullRequestInfo field stays unset rather than holding a
    bogus ``".git"`` string.
    """
    if isinstance(url, str) and url:
        return f"{url}.git"
    return None


class _CallbackChain:
    """A callback made of individually removable links.

    Composing closures produces a chain that can only be undone by
    restoring a snapshot --- and a snapshot is wrong the moment anything
    else registers.  It clobbers callbacks added after the snapshot was
    taken, and with two services borrowing one client, closing the first
    would drop the second's callbacks while closing the second would
    resurrect the first's.  Holding the links in a list lets each owner
    remove exactly its own and leave the rest alone.

    A failure in one link must not suppress the others: these are
    observability hooks, and losing the rate-limit flag because a
    progress tracker raised would reintroduce the very bug this exists
    to prevent.
    """

    __slots__ = ("links",)

    def __init__(self, links: list[Any]) -> None:
        self.links = links

    async def __call__(self, *args: Any) -> None:
        # Iterate a copy: a link may detach itself while being invoked.
        for callback in list(self.links):
            try:
                result = callback(*args)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # pragma: no cover - observability only
                logging.getLogger("dependamerge.github_service").debug(
                    "Rate-limit callback failed", exc_info=True
                )


def _chain_callbacks(existing: Any, added: Any) -> Any:
    """Return a callback invoking *existing* then *added*.

    Callbacks may be plain functions or coroutines, and either side may
    be absent.  Appends to an existing chain rather than nesting one, so
    every link stays individually removable.
    """
    if added is None:
        return existing
    if existing is None:
        return added
    if isinstance(existing, _CallbackChain):
        existing.links.append(added)
        return existing
    return _CallbackChain([existing, added])


def _unchain_callback(current: Any, remove: Any) -> Any:
    """Return *current* with *remove* taken out of it.

    Removes only what it can see.  If something replaced the chain
    wholesale, the replacement is left untouched rather than guessed
    at --- silently restoring an older callback would be worse than
    leaving the current one in place.
    """
    if current is None:
        return None
    if isinstance(current, _CallbackChain):
        if remove in current.links:
            current.links.remove(remove)
        if not current.links:
            return None
        if len(current.links) == 1:
            return current.links[0]
        return current
    # Bound methods are rebuilt on each attribute access, so identity
    # would not hold here; equality compares ``__self__``/``__func__``.
    return None if current == remove else current
