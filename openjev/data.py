"""Turning a `Task` into batches the model can train on.

The only real work here is mapping a gold answer onto the index of the marker
that should win its question's softmax:

    choice -> position of the label among the criteria keys
    score  -> the level index directly
    noul   -> 1 for true, 0 for false (markers are packed [no, yes])

Questions with no gold on a given example are skipped rather than guessed at,
which is how partially-annotated contributions stay usable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .encode import Packed, Packer
from .schema import Choice, Example, Noul, Question, Score, Task


def target_index(q: Question, gold) -> int | None:
    if gold is None:
        return None
    if isinstance(q, Choice):
        keys = list(q.criteria)
        return keys.index(gold) if gold in keys else None
    if isinstance(q, Score):
        # Order matters: bool is a subclass of int in Python, and `True == 1`,
        # so a noul gold landing here would silently become level 1.
        return int(gold) if isinstance(gold, int) and not isinstance(gold, bool) else None
    if isinstance(q, Noul):
        return int(bool(gold))
    return None


def option_labels(q: Question) -> list[str]:
    if isinstance(q, Choice):
        return list(q.criteria)
    if isinstance(q, Score):
        return [str(i) for i in range(len(q.criteria))]
    return ["false", "true"]


@dataclass
class Sample:
    packed: Packed
    targets: dict[str, int]      # question id -> winning option index
    example: Example


class TaskDataset(Dataset):
    """Packs on the fly so option-order shuffling is fresh every epoch.

    Shuffling matters: on a production system we measured the argmax flipping
    across option orderings on 7.5% of confusable items, at 2.8x the
    resampling floor. That is per-item instability rather than a positional
    prior, so it cannot be corrected post hoc — it has to be trained out.
    """

    def __init__(
        self,
        task: Task,
        packer: Packer,
        shuffle_options: bool = True,
        max_questions: int | None = None,
        seed: int = 0,
    ):
        self.task = task
        self.packer = packer
        self.shuffle_options = shuffle_options
        self.max_questions = max_questions
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.task.examples)

    def _questions_for(self, ex: Example) -> dict[str, Question]:
        qs = {k: v for k, v in self.task.questions.items() if k in ex.answers}
        if self.max_questions and len(qs) > self.max_questions:
            keep = self.rng.sample(list(qs), self.max_questions)
            qs = {k: qs[k] for k in keep}
        if not self.shuffle_options:
            return qs
        out = {}
        for qid, q in qs.items():
            if isinstance(q, Choice):
                keys = list(q.criteria)
                self.rng.shuffle(keys)
                out[qid] = Choice(
                    instructions=q.instructions,
                    criteria={k: q.criteria[k] for k in keys},
                )
            else:
                out[qid] = q  # score is ordinal and noul is fixed; order is meaning
        return out

    def __getitem__(self, i: int) -> Sample:
        ex = self.task.examples[i]
        qs = self._questions_for(ex)
        packed = self.packer.pack(ex.state, qs)
        packed.labels = {qid: option_labels(q) for qid, q in qs.items()}  # type: ignore[attr-defined]
        targets = {}
        for qid, q in qs.items():
            t = target_index(q, ex.answers.get(qid))
            if t is not None:
                targets[qid] = t
        return Sample(packed=packed, targets=targets, example=ex)


def collate(samples: list[Sample], packer: Packer) -> dict:
    """Batch, and resolve each target to its position in the flat marker array."""
    batch = packer.collate([s.packed for s in samples])

    target_flat, target_group = [], []
    offset, group = 0, 0
    for s in samples:
        for qid, k in zip(s.packed.question_ids, s.packed.n_options):
            if qid in s.targets:
                target_flat.append(offset + s.targets[qid])
                target_group.append(group)
            offset += k
            group += 1

    batch["target_flat"] = torch.tensor(target_flat, dtype=torch.long)
    batch["target_group"] = torch.tensor(target_group, dtype=torch.long)
    return batch
