# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Repairing Dependabot title / commit-subject mismatches.

Dependabot writes a pull request **title** that frequently differs from
the single commit's **subject**.  It shortens the subject by removing the
`` from <old> to <new>`` fragment, while the title keeps it::

    title:   Chore: Bump cryptography from 49.0.0 to 50.0.0 in the uv group
    subject: Chore: Bump cryptography in the uv group

When the org-mandated ``Semantic Pull Request`` check runs with
``validateSingleCommitMatchesPrTitle`` the mismatch fails the check, the
required status is never satisfied, and the merge waits out its full
timeout before reporting failure.

The upstream reusable workflow relaxes the exact match when the subject
is a *leading substring* of the title, which covers Dependabot dropping a
**trailing** fragment.  It does not cover the more common shape, where
trailing context (`` in /path``, `` in the <name> group``) keeps the
removed fragment in the **middle** --- so the subject is not a prefix and
the check fails correctly.  Measured over 112 real mismatches, the prefix
rule covered 109 and missed 3.

This module holds the pure decision logic for repairing such a PR by
aligning its title to the commit subject; the orchestration lives in
``merge_manager``.  Keeping it separate makes the rules directly
testable, which matters because the cost of getting them wrong is
rewriting somebody's pull request title.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "SEMANTIC_CHECK_PATTERN",
    "is_semantic_check_name",
    "single_commit_subject",
    "describe_title_change",
    "version_fragment_removed",
]

# Matches the org check however it is titled or namespaced --- GitHub
# reports it variously as "Semantic Pull Request",
# "Semantic Pull Request 🛠️" and "Semantic Pull Request / Semantic Pull
# Request" depending on whether a reusable workflow wraps it.
SEMANTIC_CHECK_PATTERN = re.compile(r"semantic\s+pull\s+request", re.IGNORECASE)

# The fragment Dependabot elides: " from <old> to <new>".
_VERSION_FRAGMENT = re.compile(r"^from\s+\S+\s+to\s+\S+$")


def is_semantic_check_name(name: str) -> bool:
    """Whether *name* denotes the semantic pull request title check."""
    return bool(SEMANTIC_CHECK_PATTERN.search(name or ""))


def single_commit_subject(commits: Iterable[Mapping[str, Any]]) -> str | None:
    """Subject of the sole non-merge commit, or ``None``.

    Merge commits (two or more parents) are excluded, mirroring how the
    upstream check picks the squash subject.  Returns ``None`` unless
    exactly one non-merge commit remains, because a title can only be
    said to "match the commit" when there is one commit to match.
    """
    subjects: list[str] = []
    for entry in commits:
        if not isinstance(entry, Mapping):
            continue
        parents = entry.get("parents")
        if isinstance(parents, list) and len(parents) >= 2:
            continue
        commit = entry.get("commit")
        if not isinstance(commit, Mapping):
            continue
        message = commit.get("message")
        if not isinstance(message, str) or not message:
            continue
        subjects.append(message.split("\n", 1)[0])
    if len(subjects) != 1:
        return None
    return subjects[0]


def version_fragment_removed(title: str, subject: str) -> str | None:
    """The version fragment deleted from *title* to yield *subject*.

    Returns the removed text when *subject* is *title* with a single
    contiguous run of **whole words** cut out, and that run is a
    `` from <old> to <new>`` version fragment.  ``None`` otherwise.

    The comparison works on whitespace-separated tokens rather than raw
    characters.  Character-level prefix/suffix matching cannot express
    "a whole fragment was removed" unambiguously: the same deletion can
    be framed with the space on either side, so a boundary test on the
    span reads differently depending on which way the greedy match fell.
    Tokens make the intent direct, and reject a cut that begins inside a
    word --- ``Bump xabcfrom 1 to 2 y`` → ``Bump xabc y`` truncates a
    token rather than excising a fragment.

    Distinguishing this from other mismatches matters: a title that
    differs because the *versions* differ (Dependabot rebased to a newer
    release while the commit kept the old one) is genuine drift that the
    check exists to catch, and must not be papered over by rewriting the
    title.
    """
    if not title or not subject or len(title) <= len(subject):
        return None

    title_words = title.split()
    subject_words = subject.split()
    if len(title_words) <= len(subject_words):
        return None

    # Longest common run of whole words from each end.
    prefix = 0
    while prefix < len(subject_words) and title_words[prefix] == subject_words[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < len(subject_words) - prefix
        and title_words[len(title_words) - 1 - suffix]
        == subject_words[len(subject_words) - 1 - suffix]
    ):
        suffix += 1

    # The two runs must account for every word of the subject, otherwise
    # more than one edit separates the strings.
    if prefix + suffix != len(subject_words):
        return None

    removed = title_words[prefix : len(title_words) - suffix]
    span = " ".join(removed)
    return span if _VERSION_FRAGMENT.match(span) else None


def describe_title_change(title: str, subject: str) -> str:
    """Human-readable note for the audit trail."""
    removed = version_fragment_removed(title, subject)
    if removed:
        return (
            f"aligned PR title with the commit subject "
            f"(removed {removed!r} to satisfy the semantic check)"
        )
    return "aligned PR title with the commit subject"
