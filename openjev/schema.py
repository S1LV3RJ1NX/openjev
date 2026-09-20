"""The typed-decision format.

Deliberately mirrors the TypeSafe/Jev API surface, so anything written against
Jev works here and a comparison is apples-to-apples. Three question types, a
`state` that is text or JSON, and answers keyed by question id.

On disk a task is a directory:

    tasks/healthcare_router/
        task.json          the schema: questions, costs, holdout policy
        train.jsonl        one example per line
        dev.jsonl
        test.jsonl

A contributor brings either a `tasksource` name, a CSV, or these files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

JSONContent = str | dict | list


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------


@dataclass
class Choice:
    """Pick exactly one of N labelled options."""

    instructions: str
    criteria: dict[str, JSONContent | None]
    type: Literal["choice"] = "choice"

    @property
    def labels(self) -> list[str]:
        return list(self.criteria)


@dataclass
class Score:
    """Place the state on an ordered rubric, index 0 upward."""

    instructions: str
    criteria: list[JSONContent]
    type: Literal["score"] = "score"

    @property
    def labels(self) -> list[str]:
        return [str(i) for i in range(len(self.criteria))]


@dataclass
class Noul:
    """A yes/no question returning P(true)."""

    instructions: str
    criteria: dict[str, JSONContent | None] | None = None
    type: Literal["noul"] = "noul"

    @property
    def labels(self) -> list[str]:
        return ["false", "true"]


Question = Choice | Score | Noul

_QTYPES = {"choice": Choice, "score": Score, "noul": Noul}


def question_from_dict(d: dict) -> Question:
    d = dict(d)
    cls = _QTYPES[d.pop("type")]
    return cls(**d)


# --------------------------------------------------------------------------
# examples and tasks
# --------------------------------------------------------------------------


@dataclass
class Example:
    """One state, plus the gold answer to every question that applies to it.

    `answers` maps question id -> gold value:
        choice -> the label string
        score  -> the level index (int)
        noul   -> bool
    A question id may be absent, meaning "not annotated for this example",
    which is how partially-labelled contributions are supported.
    """

    state: str | dict
    answers: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    # Per-example option sets, for questions whose menu varies by row:
    # {question_id: {label: description_or_None}}. Multiple-choice data works
    # this way — "which of these four endings?" has different endings every
    # row — and it is arguably better training signal for zero-shot, because
    # a menu that changes every example cannot be memorised.
    # Overrides the task-level criteria for that question on this example.
    criteria: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def tier(self) -> str:
        return self.meta.get("tier", "default")


@dataclass
class Task:
    """A schema plus its examples.

    `costs` is optional but strongly recommended: expected cost per decision at
    each model's own optimal threshold is the metric that should decide what
    ships, and it needs per-class prices. `cost_escalate` is what a human
    review costs; `costs[qid][label]` is what getting that label wrong costs.
    """

    name: str
    questions: dict[str, Question]
    examples: list[Example] = field(default_factory=list)
    costs: dict[str, dict[str, float]] = field(default_factory=dict)
    cost_escalate: float | None = None
    recall_floors: dict[str, dict[str, float]] = field(default_factory=dict)
    description: str = ""

    # Tasks named here must never appear in a training mixture used to
    # evaluate this one. `assert_disjoint` enforces it.
    holdout_of: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    def tiers(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.examples:
            out[e.tier] = out.get(e.tier, 0) + 1
        return out

    # -- io ----------------------------------------------------------------

    def save(self, root: str | Path, split: str = "test") -> Path:
        root = Path(root) / self.name
        root.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "description": self.description,
            "questions": {k: asdict(v) for k, v in self.questions.items()},
            "costs": self.costs,
            "cost_escalate": self.cost_escalate,
            "recall_floors": self.recall_floors,
            "holdout_of": self.holdout_of,
        }
        (root / "task.json").write_text(json.dumps(meta, indent=1))
        with open(root / f"{split}.jsonl", "w") as f:
            for e in self.examples:
                f.write(json.dumps(asdict(e)) + "\n")
        return root

    @classmethod
    def load(cls, root: str | Path, split: str = "test") -> Task:
        root = Path(root)
        meta = json.loads((root / "task.json").read_text())
        examples = []
        path = root / f"{split}.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    examples.append(Example(**json.loads(line)))
        return cls(
            name=meta["name"],
            description=meta.get("description", ""),
            questions={k: question_from_dict(v) for k, v in meta["questions"].items()},
            examples=examples,
            costs=meta.get("costs", {}),
            cost_escalate=meta.get("cost_escalate"),
            recall_floors=meta.get("recall_floors", {}),
            holdout_of=meta.get("holdout_of", []),
        )

    # -- hygiene -----------------------------------------------------------

    def assert_disjoint(self, training_task_names: list[str]) -> None:
        """Fail loudly if a held-out task leaked into the training mixture.

        Generalization claims are the easiest thing in this project to fake by
        accident, since `tasksource` contains Banking77 and friends. Call this
        before every training run.
        """
        leaked = sorted(set(self.holdout_of) & set(training_task_names))
        if leaked:
            raise ValueError(
                f"Task {self.name!r} is held out from {self.holdout_of}, but the "
                f"training mixture contains {leaked}. Generalization numbers "
                f"measured this way are meaningless."
            )

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the task is well formed."""
        problems = []
        for i, e in enumerate(self.examples):
            for qid, gold in e.answers.items():
                if qid not in self.questions:
                    problems.append(f"example {i}: unknown question id {qid!r}")
                    continue
                q = self.questions[qid]
                menu = e.criteria.get(qid) or (q.criteria if isinstance(q, Choice) else None)
                if isinstance(q, Choice) and gold is not None and gold not in (menu or {}):
                    problems.append(f"example {i}: {qid}={gold!r} not in criteria")
                elif isinstance(q, Score) and gold is not None:
                    if not isinstance(gold, int) or not 0 <= gold < len(q.criteria):
                        problems.append(f"example {i}: {qid}={gold!r} out of rubric range")
                elif isinstance(q, Noul) and gold is not None and not isinstance(gold, bool):
                    problems.append(f"example {i}: {qid}={gold!r} is not a bool")
        per_example = {qid for e in self.examples for qid in (e.criteria or {})}
        for qid, q in self.questions.items():
            if isinstance(q, Choice) and len(q.criteria) < 2 and qid not in per_example:
                problems.append(f"question {qid!r}: choice needs >= 2 options")
            if qid in per_example:
                thin = [
                    i for i, e in enumerate(self.examples)
                    if qid in e.answers and len(e.criteria.get(qid, {})) < 2
                ]
                if thin:
                    problems.append(
                        f"question {qid!r}: {len(thin)} examples have fewer than 2 "
                        f"options in their own menu (first: {thin[0]})"
                    )
            if isinstance(q, Choice) and len(q.criteria) > 255:
                problems.append(f"question {qid!r}: {len(q.criteria)} options exceeds 255")
        return problems


# --------------------------------------------------------------------------
# prediction container
# --------------------------------------------------------------------------


@dataclass
class Answer:
    """What a model returns for one question on one example."""

    probabilities: dict[str, float]

    @property
    def label(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def p_max(self) -> float:
        return max(self.probabilities.values())

    @property
    def confidence(self) -> float:
        """Jev's formula, reproduced so comparisons use the same number.

        For choice this is chance-corrected max probability; both are exact
        identities we verified against the live API.
        """
        k = len(self.probabilities)
        return (self.p_max - 1 / k) / (1 - 1 / k) if k > 1 else 1.0

    def ordinal_confidence(self) -> float:
        """The `score` variant: chance-corrected dispersion about the mode."""
        p = [self.probabilities[k] for k in sorted(self.probabilities, key=int)]
        k = len(p)
        m = max(range(k), key=p.__getitem__)
        d_k = (k * k // 4) / k
        return max(0.0, 1 - sum(pi * abs(i - m) for i, pi in enumerate(p)) / d_k)
