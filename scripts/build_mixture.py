"""Build a multi-task training mixture from tasksource, minus anything held out.

Our zero-shot measurement showed the warm start alone gives nothing usable
(see docs/results.md). Task diversity is where the capability comes from, so
this is the load-bearing data step, not a nicety.

Contamination is filtered **here**, at build time, rather than trusted to a
check at train time. `tasksource` contains Banking77, SST-2 and friends, and
not always under a recognisable name — `rotten_tomatoes` carries 77% of
SST-5's test sentences — so the exclusion list is the 140-name holdout union
and matching is done on dataset name, config, and the full task id.

Run in the pinned data environment, because tasksource's loader needs an
older `datasets`:

    uv venv .venv-data -p 3.12
    VIRTUAL_ENV=.venv-data uv pip install "datasets==2.21.0" tasksource \\
        "fsspec<=2024.6.1" "huggingface_hub<0.26"
    VIRTUAL_ENV=.venv-data uv run --no-project python scripts/build_mixture.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import only the pure-python schema; openjev has no heavy deps.
from openjev.schema import Choice, Example, Noul, Score, Task  # noqa: E402

OUT = ROOT / "tasks" / "mixture"

# Loading every task pulls tens of GB and most of it is redundant. These are
# the families that map cleanly onto typed decisions.
SKIP_PATTERNS = [
    r"^babi_nli", r"lm_task", r"^blimp", r"probe",  # synthetic / diagnostic
]


def holdout_names() -> set[str]:
    with open(ROOT / "tasks" / "heldout" / "manifest.json") as f:
        m = json.load(f)
    names = set()
    for n in m["holdout_of_union"]:
        n = n.strip().lower()
        names.add(n)
        names.add(n.rsplit("/", 1)[-1])
    return names


def is_excluded(row, banned: set[str]) -> str | None:
    """Return the offending name if this task must not be trained on."""
    candidates = {
        str(row.id).lower(),
        str(row.dataset_name).lower(),
        f"{row.dataset_name}/{row.config_name}".lower(),
        str(row.dataset_name).lower().rsplit("/", 1)[-1],
    }
    for c in candidates:
        if c in banned:
            return c
        for b in banned:
            if b and (c == b or c.startswith(b + "/") or c.split("/")[0] == b):
                return b
    return None


def humanize(label: str) -> str:
    s = re.sub(r"[_\-]+", " ", str(label)).strip()
    return s if s else str(label)


def safe_name(task_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", task_id.lower()).strip("_")


def _mc_task(task_id, tr, choice_cols, cols, names, max_rows, rng) -> Task | None:
    """Multiple choice: the menu is different on every row.

    These are valuable precisely because the option set changes per example,
    so the model cannot memorise a label space and has to read the option
    text — which is the capability zero-shot transfer depends on.
    """
    ctx = next((c for c in ("sentence1", "inputs", "premise", "question") if c in cols), None)
    if ctx is None:
        return None

    qid = "answer"
    examples = []
    for r in tr.select(range(min(len(tr), max_rows))):
        lab = r["labels"]
        if not isinstance(lab, int) or isinstance(lab, bool):
            continue
        opts, seen = {}, set()
        for c in choice_cols:
            v = r.get(c)
            if v is None or not str(v).strip():
                continue
            s = str(v).strip()
            if s.lower() in seen:  # duplicate options make the gold ambiguous
                continue
            seen.add(s.lower())
            opts[s] = None
        if len(opts) < 2 or not (0 <= lab < len(choice_cols)):
            continue
        gold_raw = r.get(choice_cols[lab])
        if gold_raw is None:
            continue
        gold = str(gold_raw).strip()
        if gold not in opts or not str(r[ctx]).strip():
            continue
        examples.append(
            Example(
                state=str(r[ctx]),
                answers={qid: gold},
                criteria={qid: opts},
                meta={"tier": "mixture_mc", "source": task_id},
            )
        )

    if len(examples) < 32:
        return None
    rng.shuffle(examples)
    return Task(
        name=safe_name(task_id),
        description=f"tasksource {task_id}",
        # Placeholder menu; every example carries its own.
        questions={qid: Choice(instructions="Which of these fits best?", criteria={})},
        examples=examples,
    )


ORDINAL_PATTERNS = [
    re.compile(r"^\s*(\d+)\s*stars?\s*$", re.I),          # yelp: '1 star' .. '5 stars'
    re.compile(r"^\s*depth[_\s-]?(\d+)\s*$", re.I),        # tree depth
    re.compile(r"^\s*(?:level|rating|score|grade)[_\s-]?(\d+)\s*$", re.I),
]


def ordinal_rank(label: str) -> int | None:
    """The rank a label denotes, if it unambiguously denotes one.

    Deliberately conservative. Bare integers are *not* treated as ordinal:
    the mixture contains `recast_kg_relations` with labels '1'..'6' that are
    arbitrary relation classes, and imposing an order on those would teach
    the model a ranking that does not exist. Only patterns that carry an
    explicit ordinal word ('3 stars', 'depth_7', 'level_2') qualify.
    """
    for pat in ORDINAL_PATTERNS:
        m = pat.match(str(label))
        if m:
            return int(m.group(1))
    return None


# Ordered vocabularies. A label set that is a subset of one of these, in any
# order, is ordinal — sentiment and Likert scales are rankings, and emitting
# them as unordered menus throws the ranking away. Matching against a known
# scale is far safer than guessing from bare integers.
ORDERED_VOCABS = [
    ["very negative", "negative", "neutral", "positive", "very positive"],
    ["strongly negative", "negative", "neutral", "positive", "strongly positive"],
    ["negative", "neutral", "positive"],
    ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    # NLI (contradiction / neutral / entailment) is deliberately NOT here.
    # It is arguably an ordered scale of support, but the mixture contains
    # dozens of NLI tasks, so including it would convert a large fraction of
    # what currently works into a different primitive on the strength of a
    # contestable judgement call. Conventional treatment is categorical.
    ["never", "rarely", "sometimes", "often", "always"],
    ["poor", "fair", "good", "very good", "excellent"],
    ["terrible", "bad", "okay", "good", "great"],
    ["none", "low", "medium", "high"],
    ["low", "medium", "high"],
    ["not helpful", "slightly helpful", "helpful", "very helpful"],
    ["unacceptable", "acceptable"],
    ["worse", "same", "better"],
]


def _vocab_ranks(names: list) -> list[int] | None:
    norm = [re.sub(r"[_\-]+", " ", str(n)).strip().lower() for n in names]
    if len(set(norm)) != len(norm):
        return None
    for vocab in ORDERED_VOCABS:
        if len(norm) < 3:  # a two-way split is not usefully ordinal
            continue
        if all(n in vocab for n in norm):
            return [vocab.index(n) for n in norm]
    return None


def as_ordinal(names: list) -> list[int] | None:
    """Ranks for a label set, if it unambiguously denotes an ordering.

    Two routes, both conservative: an explicit ordinal pattern in the label
    itself ('3 stars', 'depth_7'), or membership in a known ordered
    vocabulary. Bare integers still do not qualify — the mixture contains
    relation classes labelled '1'..'6' with no ordering at all.
    """
    ranks = [ordinal_rank(n) for n in names]
    if not any(r is None for r in ranks) and len(set(ranks)) == len(ranks):
        return ranks  # type: ignore[return-value]
    return _vocab_ranks(names)


def to_task(task_id: str, dd, max_rows: int, rng: random.Random) -> Task | None:
    """tasksource's standardised columns -> one openjev Task."""
    if "train" not in dd:
        return None
    tr = dd["train"]
    cols = tr.column_names
    if "labels" not in cols:
        return None

    feat = tr.features["labels"]
    names = getattr(feat, "names", None)
    if names is None:  # Sequence(ClassLabel) = multi-label; out of scope for v1
        return None
    if not 2 <= len(names) <= 255:
        return None

    choice_cols = sorted(c for c in cols if c.startswith("choice"))
    if choice_cols:
        return _mc_task(task_id, tr, choice_cols, cols, names, max_rows, rng)

    text_cols = [c for c in ("sentence1", "sentence2") if c in cols]
    if not text_cols:
        return None

    qid = "answer"

    # Ordinal label sets become `score` questions. Emitting them as `choice`
    # discards the ordering, and the mixture previously contained zero score
    # questions at all -- which is exactly why the held-out ordinal task sat
    # at chance while every choice task cleared it.
    ranks = as_ordinal(names)
    if ranks is not None and len(names) >= 3:
        order = sorted(range(len(names)), key=lambda i: ranks[i])
        levels = [humanize(names[i]) for i in order]
        pos = {orig: new for new, orig in enumerate(order)}
        examples = []
        for r in tr.select(range(min(len(tr), max_rows))):
            lab = r["labels"]
            if not isinstance(lab, int) or isinstance(lab, bool):
                continue
            if not 0 <= lab < len(names):
                continue
            text_cols_o = [c for c in ("sentence1", "sentence2") if c in cols]
            if not text_cols_o:
                continue
            state = ({c: r[c] for c in text_cols_o} if len(text_cols_o) > 1
                     else r[text_cols_o[0]])
            if not state:
                continue
            examples.append(Example(
                state=state, answers={qid: pos[lab]},
                meta={"tier": "mixture_ordinal", "source": task_id},
            ))
        if len(examples) < 32:
            return None
        rng.shuffle(examples)
        return Task(
            name=safe_name(task_id),
            description=f"tasksource {task_id}",
            questions={qid: Score(
                instructions="Where on this scale does the input fall?",
                criteria=levels,
            )},
            examples=examples,
        )

    binary = len(names) == 2
    criteria = {str(n): humanize(n) for n in names}
    question = (
        Noul(instructions=f"Is the answer to this {humanize(names[1])}?")
        if binary and set(map(str.lower, map(str, names))) <= {"yes", "no", "true", "false",
                                                               "entailment", "not_entailment",
                                                               "acceptable", "unacceptable"}
        else Choice(instructions="Which of these applies?", criteria=criteria)
    )

    examples = []
    n = min(len(tr), max_rows)
    for r in tr.select(range(n)):
        lab = r["labels"]
        if lab is None or not isinstance(lab, int) or not 0 <= lab < len(names):
            continue
        state = (
            {c: r[c] for c in text_cols} if len(text_cols) > 1 else r[text_cols[0]]
        )
        if not state:
            continue
        if isinstance(question, Noul):
            answers = {qid: bool(lab == 1)}
        else:
            answers = {qid: str(names[lab])}
        examples.append(
            Example(state=state, answers=answers, meta={"tier": "mixture", "source": task_id})
        )

    if len(examples) < 32:
        return None
    rng.shuffle(examples)
    return Task(
        name=safe_name(task_id),
        description=f"tasksource {task_id}",
        questions={qid: question},
        examples=examples,
    )


def load_with_backoff(tasksource, tid: str, max_rows: int, retries: int):
    """The Hub rate-limits hard when you pull hundreds of datasets in a row.

    The first build lost 238 of 285 failures to HfHubHTTPError, which is not a
    property of the datasets but of asking for them too fast. Retrying with
    backoff recovers most of them.
    """
    delay = 4.0
    for attempt in range(retries + 1):
        try:
            return tasksource.load_task(tid, max_rows=max_rows)
        except Exception as e:  # noqa: BLE001
            transient = any(
                s in f"{type(e).__name__}{e}"
                for s in ("HfHubHTTPError", "429", "Too Many Requests",
                          "ConnectionError", "ReadTimeout", "504", "502")
            )
            if not transient or attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tasks", type=int, default=250)
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--throttle", type=float, default=0.4,
                    help="seconds between successful loads, to stay under Hub limits")
    ap.add_argument("--out", default=str(OUT),
                    help="output directory for the task dirs")
    ap.add_argument("--min-k", type=int, default=0,
                    help="only keep tasks with at least this many options")
    args = ap.parse_args()

    import tasksource

    rng = random.Random(args.seed)
    banned = holdout_names()
    listing = tasksource.list_tasks()
    listing = listing[listing.task_type == "Classification"]
    print(f"{len(listing)} Classification tasks in tasksource")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    built, skipped_holdout, failed, rejected = 0, [], [], 0
    manifest = []

    for row in listing.itertuples():
        if built >= args.max_tasks:
            break
        tid = str(row.id)
        if any(re.search(p, tid) for p in SKIP_PATTERNS):
            continue
        hit = is_excluded(row, banned)
        if hit:
            skipped_holdout.append((tid, hit))
            continue
        try:
            dd = load_with_backoff(tasksource, tid, args.max_rows, args.retries)
            task = to_task(tid, dd, args.max_rows, rng)
        except Exception as e:  # noqa: BLE001
            failed.append((tid, f"{type(e).__name__}: {str(e)[:60]}"))
            continue
        time.sleep(args.throttle)
        if task is None:
            rejected += 1
            continue
        if args.min_k and len(next(iter(task.questions.values())).labels) < args.min_k:
            rejected += 1
            continue
        task.save(out_dir, split="train")
        built += 1
        k = len(next(iter(task.questions.values())).labels)
        manifest.append({"name": task.name, "source": tid, "n": len(task), "K": k})
        print(f"[{built:3d}] {task.name[:44]:<46} n={len(task):<5} K={k}")

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "tasks": manifest,
                "n_tasks": built,
                "n_rows": sum(m["n"] for m in manifest),
                "excluded_as_heldout": [{"task": t, "matched": h} for t, h in skipped_holdout],
                "failed": [{"task": t, "error": e} for t, e in failed],
            },
            indent=1,
        )
    )
    print(f"\nbuilt {built} tasks, {sum(m['n'] for m in manifest)} rows")
    print(f"excluded as held-out: {len(skipped_holdout)}")
    for t, h in skipped_holdout[:12]:
        print(f"   {t:<44} matched {h!r}")
    print(f"rejected (shape/size): {rejected}   load failures: {len(failed)}")

    from collections import Counter

    ks = Counter(m["K"] for m in manifest)
    print(f"\noption-count distribution: {dict(sorted(ks.items()))}")
    high = sum(v for k, v in ks.items() if k >= 20)
    if high < max(1, built // 10):
        print(
            f"!! only {high} of {built} tasks have 20+ options. The held-out suite\n"
            f"   runs to K=151, so high-cardinality transfer is unlikely to appear\n"
            f"   from this mixture. Source more many-way tasks before concluding\n"
            f"   anything about the architecture from a poor result there."
        )


if __name__ == "__main__":
    main()
