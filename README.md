<h1 align="center">OpenJev</h1>

<p align="center">
  <b>Typed decisions from a small model you can train yourself.</b><br>
  State in, typed answers with calibrated probabilities out. No text generation.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20preview-orange">
  <img alt="Backbone" src="https://img.shields.io/badge/backbone-ModernBERT%20%7C%20Qwen3-green">
  <a href="docs/results.md"><img alt="Results" src="https://img.shields.io/badge/results-measured%2C%20with%20CIs-brightgreen"></a>
  <a href="https://buymeacoffee.com/prathams"><img alt="Buy Me A Coffee" src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-prathams-FFDD00?logo=buy-me-a-coffee&logoColor=black"></a>
</p>

---

## What this is, and what it is not

**It is a fine-tune-first tool.** Give it roughly 400 labelled examples and it
is peer-level with the commercial API it was built to understand: a
statistical tie on multi-label compound routing, ahead on obliquely-worded
clinical risk (1.000 against 0.793) and on scope gating, behind on
single-label intent (0.899 against 0.941). Training takes 38 seconds on one
GPU and the data never leaves your machine.

**It is not a zero-shot replacement.** On a schema it has never seen,
Banking77, it scores 0.355 against roughly 0.820. That gap is real and this
README will say so until a measurement says otherwise.

### Why, and where you come in

Zero-shot ability on typed decisions does not come from the architecture. We
tested that: a strong encoder with a clever output format scores **0.7x
chance** untrained, and a 61-task mixture transferred nothing at all. It
comes from **task diversity in training data**. Our mixture is 143 tasks
scraped from `tasksource`; the Flan work suggests the gain keeps accruing
past ~282, and ours is lopsided besides — 100 `choice` tasks against 34
`noul` and 9 `score`, which is exactly why the two non-`choice` held-out
tasks sit at chance while every `choice` task clears it.

Assembling a genuinely diverse typed-decision dataset is more than one person
can do. **If the community builds that dataset, a general encoder with real
zero-shot ability is reachable**, and everything needed to try is already
here: the [format](docs/dataset-format.md), a
[CSV scaffold](scripts/make_task.py), a contamination guard that refuses a
mixture overlapping the evaluation suite, and an
[evaluation harness](docs/evaluation.md) with a held-in control so "no
transfer" cannot be confused with a bug.

What would help most, in order: ordinal (`score`) tasks, which we have almost
none of; yes/no (`noul`) tasks; and anything with a menu past 20 options.

**One untested hypothesis.** Everything here is base-sized: ModernBERT-base
and Qwen3-1.7B. Scaling the backbone may move zero-shot transfer and we have
not measured it, so we are not claiming it either way.

## Quick start

```bash
git clone https://github.com/S1LV3RJ1NX/openjev && cd openjev
uv sync
```

Train a decision model on your own task in about a minute:

```bash
uv run python scripts/train.py --task tasks/healthcare_router --epochs 6 --bs 8
uv run python scripts/eval_router.py --ckpt checkpoints/healthcare_router/model.pt
```

Score any held-out schema with no training at all:

```bash
uv run python scripts/eval_heldout.py --decoder --backbone Qwen/Qwen3-1.7B --limit 200
```

## How to use it

Give it a state and some typed questions. Every question is answered in **one
forward pass over a shared state**, so asking twenty costs about the same as
asking one.

```python
from openjev import Choice, Score, Noul

result = model.predict(
    state="I need a refill on my thyroid tablets, and are you open on Sunday?",
    questions={
        "intent":      Choice(instructions="What is this about?",
                              criteria={"refill": "another fill of a prescription",
                                        "store_hours": "when a branch is open"}),
        "urgency":     Score(instructions="How time-critical?",
                             criteria=["can wait", "today", "immediate"]),
        "needs_human": Noul(instructions="Does this need a human?"),
    },
)

result["intent"].probabilities      # {"refill": 0.71, "store_hours": 0.24, ...}
result["intent"].confidence         # chance-corrected, comparable across menus
result["urgency"].score             # 1.30 — the expected level, not the argmax
result["urgency"].is_unimodal       # False means that 1.30 describes nothing
result["needs_human"].probabilities["true"]
```

| primitive | returns | use for |
|---|---|---|
| `choice` | one of N options, with probabilities | routing, intent, classification |
| `score` | a **number** on an ordered rubric | urgency, severity, quality |
| `noul` | P(true) | flags, gates, multi-label |

Multi-label is `noul` per label, not `choice`: a message can be about a refill
*and* opening hours, and a softmax cannot say so.

## Train on your own data

Clone, bring a CSV, train, use. Four steps.

**1. Build a task from your data.**

```bash
uv run python scripts/make_task.py --csv mydata.csv \
    --text-column message --label-column intent --name my_task
```

That writes `tasks/my_task/` with a stratified split and a `choice` question.
Or write the files by hand — the format is two file types and is specified in
**[docs/dataset-format.md](docs/dataset-format.md)**, with a copy-paste
template.

**2. Write the option descriptions.** The scaffold leaves `TODO` placeholders.
This is the step worth your time: descriptions stating what distinguishes a
label from its neighbours were worth **+5 accuracy points** on Banking77,
while descriptions that merely restated the label name were worth nothing
(p = 0.75).

**3. Train.** Two recipes.

```bash
# A — straight fine-tune. 395 examples, 38 seconds, one GPU.
uv run python scripts/train.py --task tasks/my_task --epochs 6 --bs 8

# B — start from the general checkpoint. Same data, same time, +36 points.
hf download s1lv3rj1nx/openjev-encoder-general model.pt --local-dir checkpoints/general
uv run python scripts/train.py --task tasks/my_task \
    --init-from checkpoints/general/model.pt --epochs 6 --bs 8
```

Recipe B is the project's central claim and it is measured: intent
0.544 → 0.899, multi-label 0.453 → 0.789, **p = 6e-18**. A safety gate that
never fired at all under A reaches 0.857 recall under B from seven positive
examples, and oblique-clinical recall goes to 1.000.

**4. Evaluate, then use.** Training fits calibration temperatures on `dev`
automatically and saves them with the checkpoint.

```bash
uv run python scripts/eval_router.py --ckpt checkpoints/my_task/model.pt
```

Before trusting the numbers, check per-class recall rather than accuracy. A
class with a handful of examples gets learned as "never predict this" — ours
scored 0.000 recall while looking 0.984 accurate. `docs/dataset-format.md`
lists the other traps we hit.

## Where Jev wins

We measured TypeSafe's Jev against our own suite and it is ahead on the things
that matter most, so use it if those matter more than self-hosting:

- **Zero-shot accuracy.** 0.820 on Banking77 having never seen it; our best
  never-trained-on-it number is 0.355.
- **Routing quality.** Router intent 0.941 against our 0.899 (p = 0.04), and
  the injection gate 0.996 against our 0.978 (p = 0.008).
- **Gate precision.** `G_clinical` false-positive rate 0.006 against our 0.026.
- **Scale.** 255 options and a 32k context, out of the box.

Where we are level or ahead, once fine-tuned on the task: obliquely-worded
clinical risk, where Jev misses one in five and we catch all 29 (1.000 against
0.793, p = 0.03); the `G_pharmacy` scope gate (0.947 against 0.880, p = 3e-04);
the multi-label compound set, now a statistical tie (0.789 against 0.822,
p = 0.14); `G_clinical` correctness (p = 1.00); full-precision probabilities,
where Jev quantizes to 0.01 and puts 71.9% of values at a hard zero;
determinism, which Jev has none of and offers no seed for; and cost, since
this runs on your own hardware with no data leaving it.

All paired on the same 450 items with exact McNemar, reproducible via
`scripts/compare_to_jev.py`.

## Where this architecture fits

**Good fit:** high-volume routing, triage, guardrails and tool selection where
you have labels; long states asked many questions at once; regulated data that
cannot leave your network; anything needing a real probability to threshold on.

**Poor fit:** you have no labels and need it to work on arbitrary schemas
today — use Jev. Short states with one or two questions — the shared prefix
buys nothing and a plain classifier is simpler.

## Docs

| | |
|---|---|
| [Model on the Hub](https://huggingface.co/s1lv3rj1nx/openjev-encoder-general) | The general checkpoint, 17.6x chance on held-out schemas |
| [Dataset format](docs/dataset-format.md) | The spec, a template, and the mistakes we made |
| [Results](docs/results.md) | Every measurement, what worked and what did not |
| [Architecture](docs/architecture.md) | The design, with diagrams, and why each choice |
| [Evaluation](docs/evaluation.md) | The two suites and the contamination guard |
| [Prior art](docs/prior-art.md) | UniMC, PCW, CORN, ADB — most of this exists |

## Honesty policy

Every Jev figure was measured by us against the public API (`jev-1.13.0`,
20 Sep 2026) at stated sample sizes, with the caveats kept next to the
numbers. Negative results are documented as prominently as positive ones, and
[docs/results.md](docs/results.md) contains a list of claims we retracted after
better measurement. No claim of parity with Jev is made anywhere, because the
numbers do not show it.

## Support

Everything here stays free. If it saved you time,
[buy me a coffee](https://buymeacoffee.com/prathams).

## Licence

Apache 2.0. Dataset rows keep their upstream licences; see
[`tasks/heldout/README.md`](tasks/heldout/README.md).
