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
result["intent"].probabilities   # {"refill": 0.71, "store_hours": 0.24, ...}
```

Three primitives, borrowed deliberately from the API shape TypeSafe introduced
with Jev, so that code written against either runs against the other and
comparisons are like-for-like:

| | |
|---|---|
| `choice` | pick one of N labelled options |
| `score` | place the state on an ordered rubric |
| `noul` | a yes/no question returning P(true) |

---

## Status: no model yet

**There is no trained checkpoint in this repository.** Nothing here has been
validated as a model, because the model has not been built.

What exists today:

- `openjev/` — the typed-decision format and a metric suite
- `tasks/` — two evaluation suites, with baselines measured on an existing
  commercial system so we know what we are aiming at
- `docs/` — the architecture we intend to build, and why

We are publishing at this stage so the evaluation and the claims can be
scrutinised *before* there are results to defend. If you are here for a model
to download, come back later.

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

See [`docs/architecture.md`](docs/architecture.md) for the design and the
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
