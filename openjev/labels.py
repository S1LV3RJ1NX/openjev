"""Working out what a label set actually means.

Shared by the mixture builder and the retyping tool, so a label set is
classified the same way whether it arrives from the Hub or from disk.
"""

from __future__ import annotations

import re

# Prefixes that negate the stem they attach to. Matching requires the rest of
# the label to be identical, so `no_hate_speech`/`hate_speech` qualifies and
# `national`/`constituency` does not.
NEG_PREFIXES = ("not", "no", "non", "un", "in", "im", "ir", "dis", "anti")


def _norm(label) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


def negation_pair(names: list) -> int | None:
    """Index of the positive label, if these two are X and not-X.

    A yes/no question dressed as a two-option menu does not teach the `noul`
    primitive. The builder previously recognised only a hardcoded list of
    eight label names, which left `not_paraphrase`/`paraphrase`,
    `no_hate_speech`/`hate_speech`, `notsarc`/`sarc`, `valid`/`invalid` and
    `Not-Related`/`Related` all emitted as `choice` — real `noul` data thrown
    away while the held-out binary task sat at chance.
    """
    if len(names) != 2:
        return None
    norm = [_norm(n) for n in names]
    if not all(norm) or norm[0] == norm[1]:
        return None
    for pos, neg in ((0, 1), (1, 0)):
        for p in NEG_PREFIXES:
            if norm[neg] == p + norm[pos]:
                return pos
    return None


# Label sets that are a yes/no question by meaning rather than by spelling,
# where neither label is the other with a prefix bolted on.
EXPLICIT_BINARY: tuple[frozenset[str], str] = (
    (frozenset({"yes", "no"}), "yes"),
    (frozenset({"true", "false"}), "true"),
    (frozenset({"entailment", "notentailment"}), "entailment"),
    (frozenset({"acceptable", "unacceptable"}), "acceptable"),
    (frozenset({"entailment", "contradiction"}), "entailment"),
)


def noul_positive(names: list) -> int | None:
    """Index of the label that means "yes", if this set is a yes/no question."""
    idx = negation_pair(names)
    if idx is not None:
        return idx
    if len(names) != 2:
        return None
    norm = [_norm(n) for n in names]
    for members, positive in EXPLICIT_BINARY:
        if frozenset(norm) == members:
            return norm.index(positive)
    return None
