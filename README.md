# OpenJev

**Open System One models you can train on your own data.**

A System One model takes some state — a support ticket, an email, a JSON
document — plus a set of typed questions, and returns a typed answer with a
probability for every option. It does not generate text. There is nothing to
parse, no schema to validate, and no free-text field for an answer to escape
through.

```python
result = model.predict(
    state="I need a refill on my thyroid tablets, and are you open on Sunday?",
    questions={
        "intent":      Choice(instructions="What is this about?",
                              criteria={"refill": "...", "store_hours": "...", ...}),
        "urgency":     Score(instructions="How time-critical?",
                             criteria=["can wait", "today", "immediate"]),
        "needs_human": Noul(instructions="Does this need a human?"),
    },
)
result["intent"].probabilities    # {"refill": 0.71, "store_hours": 0.24, ...}
result["intent"].label            # "refill"
result["urgency"].score           # 1.30 — the expected level, not the argmax
result["urgency"].is_unimodal     # False means that 1.30 describes nothing
result["needs_human"].probabilities["true"]
```

`score` returns a **number**, which is the point of it being a separate
primitive: a distribution of `[0.01, 0.69, 0.29, 0.01]` over "can wait / a few
days / today / blocking" gives 1.30, so mostly *a few days* with real pull
toward *today*. That is something you can threshold or price. The argmax, `1`,
throws it away.

Check `is_unimodal` before trusting it. On an input that is either trivial or
an emergency, `[0.45, 0.02, 0.03, 0.50]` averages to 1.58 — a level with 2%
probability, looking like a calm middling answer while the model believes two
contradictory things.

OpenJev is an attempt to build, in the open, the class of model TypeSafe AI
introduced with Jev. It started by using Jev — reading its docs, calling its
API, and measuring its behaviour on tasks we cared about — and then asking what
it would take to build something with the same shape that anyone can train on
their own data and run on their own hardware. The credit for the idea, and for
the API design we deliberately follow, belongs to them.

Three primitives, borrowed from that API shape so that code written against
either runs against the other and comparisons are like-for-like:

| | |
|---|---|
| `choice` | pick one of N labelled options |
| `score` | place the state on an ordered rubric |
| `noul` | a yes/no question returning P(true) |

---

## Status: first result in, no released checkpoint

The architecture trains. On Banking77, fine-tuned from ModernBERT-base in
**13.5 minutes on one H100**, it reaches **92.3% accuracy** (macro-F1 0.923,
ECE 0.029 after calibration) on a 77-way task, against 82.0% for a zero-shot
commercial reference on the same test split.

**Read that carefully**: we trained on 9,839 Banking77 examples and the
reference did not. It measures what having labels buys you, not which model is
better, and it says nothing about zero-shot ability. Full numbers and caveats
in [`docs/results.md`](docs/results.md).

What exists today:

- `openjev/` — the typed-decision format, a metric suite, the packer and model
- `tasks/` — two evaluation suites, with reference baselines measured on an
  existing commercial system so we know what we are aiming at
- `scripts/` — a readable training loop and dataset builders
- `docs/` — the architecture, the evaluation, prior art, results

Multi-task training on 61 tasksource tasks learns those tasks (0.84 on ethos,
0.675 on three-way MNLI) but transfers **nothing** to held-out schemas —
0.9x chance, no better than no training at all. That is a mixture problem,
not an architecture verdict: 61 tasks against the ~282 where Flan says the
gain accrues, with option counts capped at 20 while the held-out suite runs
to 151.

Not done yet: a mixture large and diverse enough to actually test
generalization, the `score` and `noul` primitives, and a released
checkpoint.

---

## Why build this

Most agent pipelines spend a large fraction of their calls on classification,
routing and tool selection — deciding *which* of a known set of things applies.
That work does not need a model that writes. Removing generation removes the
token-by-token decode loop, which is where most of the latency and cost sits.

The properties that matter for that job are unglamorous: the output is always a
valid member of the set you supplied, you get a probability on every option so
your own code can decide when to escalate, and the whole thing is small enough
to self-host, which matters when the data cannot leave your network.

## What is planned

- A ModernBERT-based encoder with one shared scoring head over option markers,
  so a new label set changes the *input* rather than the weights
- Many questions about one state in a single forward pass, via a shared state
  prefix with block-diagonal attention
- Ordinal handling for `score` that a plain softmax cannot express
- Calibration fitted per question type and option count
- A trainer that runs on a consumer GPU, and a converter so you can bring a CSV
  or a `tasksource` task

See [`docs/architecture.md`](docs/architecture.md) for the design, and
[`docs/results.md`](docs/results.md) for measurements. The
reasoning behind each choice.

## The evaluation

Built before the model, on purpose. An evaluation written after you see your
results is an evaluation you tuned.

**`tasks/healthcare_router`** — 988 synthetic pharmacy-routing items across 18
tiers, testing compound utterances, negation, oblique safety signals,
transcription noise and adversarial input. Ours, handwritten, no real user data.

**`tasks/heldout`** — seven public tasks, 4,200 rows, covering all three
primitives with option counts from 2 to 151. This is the generalization check:
entire *schemas* are held out, never random rows.

Contamination here is the default rather than the exception. `tasksource`, the
obvious training source, contains most common benchmarks — and not always under
a recognisable name. We found that `rotten_tomatoes` contains 77% of SST-5's
test sentences verbatim, and that `toxic_conversations` is 100% Civil Comments
rows. So:

```python
from openjev.heldout import assert_training_mixture_clean
assert_training_mixture_clean(mixture_names)   # call BEFORE training
```

See [`docs/evaluation.md`](docs/evaluation.md).

## On the numbers in these docs

Where a figure is attributed to TypeSafe's Jev, we measured it ourselves
against the public API (model `jev-1.13.0`) on 20 September 2026, using the
sample sizes stated alongside each number. Those are black-box measurements of
a hosted endpoint on one day from one network location, not statements about
how that system is built — its architecture, size and training data are not
public. Treat them as a benchmark reference point, reproduce them before
relying on them, and read the caveats, which are recorded next to the results
rather than buried.

We have published no comparison in our own favour, because we have nothing to
compare yet.

## Install

```bash
uv add openjev        # not yet on PyPI; clone for now
```

```bash
git clone https://github.com/<owner>/openjev && cd openjev && uv sync
uv run python -c "from openjev import Task; print(len(Task.load('tasks/heldout/banking77')))"
```

## Contributing

The most useful contributions right now are adversarial: find a place where the
evaluation is saturated, mislabelled, or measuring the wrong thing. Several
tiers in the router suite are already at ceiling and are documented as
regression floors rather than scoring targets — more of those are worth
knowing about.

## Licence

Apache 2.0. Dataset rows retain their upstream licences; see
[`tasks/heldout/README.md`](tasks/heldout/README.md). Two datasets with no
usable licence are not redistributed here and are regenerated locally instead.
