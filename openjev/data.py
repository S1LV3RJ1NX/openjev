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

import math
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .encode import Packed, Packer
from .schema import Choice, Example, Noul, Question, Score, Task


# Ordinal scale augmentation.
#
# Real ordinal data is scarce here. A 143-task mixture yielded 9 `score`
# questions and 6 of those are the identical negative/neutral/positive scale,
# against a held-out task asking for a 5-level helpfulness judgement. That is
# the cardinality problem from label-space augmentation one level down, and it
# has the same answer: synthesize the diversity instead of hunting for
# datasets that happen to have it.
#
# Two transformations, both label-preserving:
#   coarsening  merge adjacent levels and remap the gold to its group
#   rewording   restate the scale in another vocabulary of the same length
#
# Rewording stays inside a semantic family. Restating a star rating as
# "never ... always" would teach the model to read position and ignore
# meaning, which is the opposite of what an ordinal head is for.
ORDINAL_FAMILIES: dict[str, dict[int, list[list[str]]]] = {
    "valence": {
        2: [["negative", "positive"], ["unfavourable", "favourable"]],
        3: [
            ["negative", "neutral", "positive"],
            ["unfavourable", "neutral", "favourable"],
            ["critical", "mixed", "appreciative"],
            ["disapproving", "ambivalent", "approving"],
        ],
        5: [
            ["very negative", "negative", "neutral", "positive", "very positive"],
            ["strongly unfavourable", "unfavourable", "neutral", "favourable", "strongly favourable"],
        ],
    },
    "quality": {
        2: [["poor", "good"], ["unsatisfactory", "satisfactory"]],
        3: [["poor", "average", "good"], ["low quality", "acceptable", "high quality"]],
        4: [["poor", "fair", "good", "excellent"], ["bad", "mediocre", "good", "outstanding"]],
        5: [
            ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
            ["terrible", "poor", "average", "good", "excellent"],
            ["very poor", "poor", "acceptable", "good", "very good"],
            ["not at all helpful", "slightly helpful", "moderately helpful", "very helpful", "extremely helpful"],
        ],
        7: [["worst", "very poor", "poor", "average", "good", "very good", "best"]],
    },
    "magnitude": {
        2: [["low", "high"], ["small", "large"]],
        3: [["low", "medium", "high"], ["small", "moderate", "large"], ["weak", "moderate", "strong"]],
        4: [["none", "slight", "moderate", "severe"], ["very low", "low", "high", "very high"]],
        5: [
            ["very low", "low", "medium", "high", "very high"],
            ["not at all", "slightly", "moderately", "very", "extremely"],
        ],
        6: [["lowest", "very low", "low", "high", "very high", "highest"]],
        7: [["lowest", "very low", "low", "medium", "high", "very high", "highest"]],
    },
    "agreement": {
        2: [["disagree", "agree"], ["no", "yes"]],
        3: [["disagree", "neutral", "agree"], ["unlikely", "uncertain", "likely"]],
        5: [["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]],
    },
    "frequency": {
        2: [["never", "always"], ["rarely", "often"]],
        3: [["rarely", "sometimes", "often"], ["seldom", "occasionally", "frequently"]],
        5: [["never", "rarely", "sometimes", "often", "always"]],
    },
}

_FAMILY_CUES: dict[str, tuple[str, ...]] = {
    "valence": ("negative", "positive", "neutral", "favourable", "favorable", "sentiment",
                "critical", "appreciative", "approving", "disapproving", "ambivalent", "mixed"),
    "quality": ("star", "poor", "excellent", "good", "bad", "helpful", "quality", "fair",
                "terrible", "mediocre", "outstanding", "average", "acceptable",
                "satisfactory", "unsatisfactory", "worst", "best"),
    "agreement": ("agree", "disagree", "likely", "unlikely", "yes", "no", "uncertain"),
    "frequency": ("never", "always", "often", "rarely", "sometimes",
                  "seldom", "occasionally", "frequently"),
    "magnitude": ("low", "high", "depth", "small", "large", "strong", "weak", "severe",
                  "slight", "moderate", "extremely", "lowest", "highest", "none"),
}


def scale_family(levels: list[str]) -> str | None:
    """Which semantic family a scale belongs to, or None if unrecognised."""
    text = " ".join(str(x).lower() for x in levels)
    best, best_hits = None, 0
    for fam, cues in _FAMILY_CUES.items():
        hits = sum(1 for c in cues if c in text)
        if hits > best_hits:
            best, best_hits = fam, hits
    return best


def coarsen(levels: list[str], m: int, gold: int | None) -> tuple[list[str], int | None]:
    """Merge K ordered levels into m, remapping the gold to its group.

    Labels are derived by naming the span, so the result stays true to the
    source scale even when no family vocabulary of that length exists.
    """
    k = len(levels)
    bounds = [round(j * k / m) for j in range(m + 1)]
    merged = []
    for j in range(m):
        lo, hi = bounds[j], bounds[j + 1] - 1
        merged.append(str(levels[lo]) if lo == hi else f"{levels[lo]} to {levels[hi]}")
    new_gold = None
    if gold is not None:
        new_gold = next(j for j in range(m) if bounds[j] <= gold < bounds[j + 1])
    return merged, new_gold


def _clip_options(q: Question, limit: int) -> Question:
    """Shorten option text, keeping option identity and menu order intact."""
    if isinstance(q, Choice):
        return Choice(
            instructions=q.instructions,
            criteria={k: (str(v)[:limit] if v else v) for k, v in q.criteria.items()},
        )
    if isinstance(q, Score):
        return Score(instructions=q.instructions,
                     criteria=[str(c)[:limit] for c in q.criteria])
    return q


def target_index(q: Question, gold) -> int | None:
    """Index of the winning option, against the menu actually packed.

    `q` must be the resolved question — after any per-example override and
    after shuffling — or the index will point at the wrong option.
    """
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
        distractors: list[tuple[str, str]] | None = None,
        distractor_prob: float = 0.0,
        max_options: int = 128,
        scale_prob: float = 0.0,
        noul_prob: float = 0.0,
    ):
        self.task = task
        self.packer = packer
        self.shuffle_options = shuffle_options
        self.max_questions = max_questions
        self.rng = random.Random(seed)
        # Label-space augmentation: pad a menu with labels borrowed from other
        # tasks. The gold answer is unchanged and still correct, so the example
        # stays valid, but the model has to discriminate against a large menu.
        #
        # This exists because a mixture assembled from ordinary classification
        # datasets tops out around 20 options while real deployments ask for
        # 77 or 151, and a model that has never seen a big menu collapses onto
        # one label when handed one.
        self.distractors = distractors or []
        self.distractor_prob = distractor_prob
        self.max_options = max_options
        self.scale_prob = scale_prob
        self.noul_prob = noul_prob

    def _pad_menu(self, q: Choice, shrink: int = 0) -> Choice:
        if not self.distractors or self.rng.random() >= self.distractor_prob:
            return q
        have = {k.strip().lower() for k in q.criteria}
        # Sample a target size log-uniformly so small menus stay common and
        # large ones appear often enough to matter.
        hi = max(len(q.criteria) + 1, self.max_options // (2 ** shrink))
        target = int(math.exp(self.rng.uniform(math.log(len(q.criteria) + 1), math.log(hi))))
        extra: dict[str, str] = {}
        for _ in range(8 * (target - len(q.criteria))):
            if len(q.criteria) + len(extra) >= target:
                break
            lab, desc = self.rng.choice(self.distractors)
            key = lab.strip()
            # Never add a distractor that could actually be correct.
            if not key or key.lower() in have or key.lower() in {k.lower() for k in extra}:
                continue
            extra[key] = desc
        if not extra:
            return q
        merged = {**q.criteria, **extra}
        keys = list(merged)
        self.rng.shuffle(keys)
        return Choice(instructions=q.instructions, criteria={k: merged[k] for k in keys})

    def __len__(self) -> int:
        return len(self.task.examples)

    def _vary_scale(self, q: Score, gold) -> tuple[Score, int | None]:
        """Reword or coarsen an ordinal scale, keeping the gold correct."""
        levels = [str(x) for x in q.criteria]
        k = len(levels)
        if k < 2 or self.rng.random() >= self.scale_prob:
            return q, gold
        if not (isinstance(gold, int) and not isinstance(gold, bool) and 0 <= gold < k):
            return q, gold  # nothing to remap safely
        fam = scale_family(levels)
        # Coarsen sometimes, and only when there is room to lose a level.
        m = k
        if k >= 4 and self.rng.random() < 0.5:
            m = self.rng.randint(2, k - 1)
        if m != k:
            levels, gold = coarsen(levels, m, gold)
        # Reword, if this family has another vocabulary of the right length.
        alts = [v for v in ORDINAL_FAMILIES.get(fam or "", {}).get(m, []) if v != levels]
        if alts:
            levels = list(self.rng.choice(alts))
        return Score(instructions=q.instructions, criteria=levels), gold

    def _as_noul(self, q: Choice, gold) -> tuple[Noul, bool] | None:
        """Turn "which of these?" into "is it this one?".

        `noul` is the primitive the mixture is poorest in — 13 tasks against
        128 `choice` — and a held-out binary task sits at exactly chance,
        predicting one class for every input. Choice data answers the question
        already: if the gold is `refill`, then "is this a refill?" is true and
        "is this a store-hours question?" is false. No labels are invented,
        and asking about the gold half the time keeps the classes balanced.
        """
        labels = list(q.criteria)
        if gold not in labels or len(labels) < 2:
            return None
        if self.rng.random() < 0.5:
            pick, ans = gold, True
        else:
            pick, ans = self.rng.choice([l for l in labels if l != gold]), False
        desc = str(q.criteria[pick] or pick).strip()
        stem = q.instructions.strip().rstrip("?.")
        return Noul(instructions=f'{stem}. Specifically, does "{desc}" apply?'), ans

    def _questions_for(
        self, ex: Example, shrink: int = 0
    ) -> tuple[dict[str, Question], dict[str, int | bool]]:
        """Resolved questions, plus golds that augmentation moved."""
        qs = {k: v for k, v in self.task.questions.items() if k in ex.answers}
        # A per-example menu replaces the task-level one for that question.
        for qid, menu in (ex.criteria or {}).items():
            if qid in qs and menu:
                qs[qid] = Choice(instructions=qs[qid].instructions, criteria=dict(menu))
        if self.max_questions and len(qs) > self.max_questions:
            keep = self.rng.sample(list(qs), self.max_questions)
            qs = {k: qs[k] for k in keep}
        if not self.shuffle_options:
            return qs, {}
        out: dict[str, Question] = {}
        golds: dict[str, int | bool] = {}
        for qid, q in qs.items():
            if isinstance(q, Choice):
                if self.noul_prob and self.rng.random() < self.noul_prob:
                    conv = self._as_noul(q, ex.answers.get(qid))
                    if conv is not None:
                        out[qid], golds[qid] = conv
                        continue
                q = self._pad_menu(q, shrink)
                keys = list(q.criteria)
                self.rng.shuffle(keys)
                out[qid] = Choice(
                    instructions=q.instructions,
                    criteria={k: q.criteria[k] for k in keys},
                )
            elif isinstance(q, Score):
                # Levels are never shuffled — their order is the meaning — but
                # the scale itself can be restated or merged.
                q, g = self._vary_scale(q, ex.answers.get(qid))
                out[qid] = q
                if g is not None:
                    golds[qid] = g
            else:
                out[qid] = q  # noul is fixed
        return out, golds

    def _subsample(self, q: Question, gold, cap: int) -> Question:
        """Cut a choice menu down to `cap` options, never dropping the gold."""
        if not isinstance(q, Choice) or len(q.criteria) <= cap:
            return q
        keys = [k for k in q.criteria if k != gold]
        self.rng.shuffle(keys)
        keep = keys[: max(1, cap - 1)]
        if gold in q.criteria:
            keep.append(gold)
        self.rng.shuffle(keep)
        return Choice(instructions=q.instructions, criteria={k: q.criteria[k] for k in keep})

    def __getitem__(self, i: int) -> Sample:
        ex = self.task.examples[i]
        # Padding a menu can push the sequence over the context budget. An
        # augmentation that kills a multi-hour run is worse than no
        # augmentation, so back off to progressively smaller menus and, in
        # the limit, to the original one.
        golds: dict[str, int] = {}
        for attempt in range(6):
            qs, golds = self._questions_for(ex, shrink=attempt)
            try:
                packed = self.packer.pack(ex.state, qs)
                break
            except ValueError:
                continue
        else:
            # Even the unaugmented menu does not fit. This is real: a
            # 174-option task rendered with a per-option template can exceed
            # the budget on its own. Subsample the menu, always keeping the
            # gold option, and halve until it fits. Training on a subsampled
            # menu is still a valid example — the correct answer is present
            # and the model still has to pick it out.
            qs = {k: v for k, v in self.task.questions.items() if k in ex.answers}
            for qid, menu in (ex.criteria or {}).items():
                if qid in qs and menu:
                    qs[qid] = Choice(instructions=qs[qid].instructions, criteria=dict(menu))
            golds = {}  # this path uses the original questions, so original golds
            cap = 64
            while True:
                trimmed = {
                    qid: self._subsample(q, ex.answers.get(qid), cap)
                    for qid, q in qs.items()
                }
                try:
                    packed = self.packer.pack(ex.state, trimmed)
                    qs = trimmed
                    break
                except ValueError:
                    pass
                # Two options can still overflow on their own when the options
                # are passage-length, which some MultipleChoice sources are:
                # state truncated to half the budget plus two 840-token
                # options is 2,700 tokens against a 2,048 limit. Shorten the
                # option text rather than raising, since a run that dies
                # thousands of steps in costs far more than a clipped
                # distractor.
                try:
                    clipped = {qid: _clip_options(q, 400) for qid, q in trimmed.items()}
                    packed = self.packer.pack(ex.state, clipped)
                    qs = clipped
                    break
                except ValueError:
                    cap //= 2
                    if cap < 2:
                        raise
        packed.labels = {qid: option_labels(q) for qid, q in qs.items()}  # type: ignore[attr-defined]
        targets = {}
        for qid, q in qs.items():
            t = target_index(q, golds.get(qid, ex.answers.get(qid)))
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
