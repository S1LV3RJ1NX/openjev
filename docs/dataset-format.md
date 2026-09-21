# Dataset format

A task is a directory. Two kinds of file, no code required.

```
tasks/my_task/
    task.json      the schema: questions, optional costs, holdout policy
    train.jsonl    one example per line
    dev.jsonl      optional, used to fit calibration temperatures
    test.jsonl
```

If you would rather not write JSON, `scripts/make_task.py` builds this from a
CSV — see [From a CSV](#from-a-csv) at the bottom.

## Template

Copy this and edit. It is a complete, valid task.

**`task.json`**

```json
{
  "name": "my_task",
  "description": "What this task is and where the data came from.",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Which topic does this message concern?",
      "criteria": {
        "billing": "payments, invoices and refunds",
        "technical": "bugs, outages and integration problems",
        "other": "anything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How time-critical is this?",
      "criteria": ["can wait a week", "this week", "today", "blocking right now"]
    },
    "needs_human": {
      "type": "noul",
      "instructions": "Does this need a human rather than automation?"
    }
  },
  "costs": {},
  "cost_escalate": null,
  "recall_floors": {},
  "holdout_of": []
}
```

**`train.jsonl`** — one JSON object per line, no commas between lines:

```json
{"state": "I was charged twice for March.", "answers": {"intent": "billing", "urgency": 1, "needs_human": false}, "meta": {"tier": "single"}}
{"state": "The API returns 500 on every call.", "answers": {"intent": "technical", "urgency": 3, "needs_human": true}, "meta": {"tier": "single"}}
```

Then:

```bash
uv run python -c "from openjev import Task; t=Task.load('tasks/my_task','train'); print(t.validate() or 'clean', len(t))"
```

## The three question types

| type | `criteria` | gold value in `answers` |
|---|---|---|
| `choice` | `{label: description}` | the label string, e.g. `"billing"` |
| `score` | `[level0, level1, ...]` ordered | the level **index**, e.g. `2` |
| `noul` | omit, or `{"true": ..., "false": ...}` | `true` / `false` |

Descriptions are optional but they matter. On Banking77, descriptions that
state what distinguishes a label from its neighbours were worth about **+5
accuracy points**; descriptions that merely restated the label name were worth
nothing (p = 0.75). Write the distinction, not a synonym.

A question may be left out of `answers` on any example. That row simply does
not supervise it, which is how partially-annotated data stays usable.

## Patterns worth knowing

### Multi-label: use `noul` per label, not `choice`

A message can be about a refill *and* opening hours. A `choice` softmax cannot
say so — it must pick one.

```json
"questions": {
  "is_refill":      {"type": "noul", "instructions": "Does this ask for a refill?"},
  "is_store_hours": {"type": "noul", "instructions": "Does this ask about opening hours?"}
}
```

Measured: on three-intent messages a single `choice` scores 0.581 while the
per-label nouls reach F1 0.955. A cheap trick if you want both — ask the
`choice` *and* the nouls in the same call (it costs one forward pass) and use
the choice's **second-highest probability** as a signal that more than one
intent is present. That detector scores AUROC 0.924.

### Ambiguous items: record the acceptable set

When more than one label is defensible, omit the gold and put the acceptable
set in `meta`. Forcing a single answer marks a correct model wrong.

```json
{"state": "Where's my refill?", "answers": {}, "meta": {"acceptable": ["refill", "order_status"]}}
```

### Safety gates: keep them out of the intent menu

A softmax makes a safety option compete with the intents, so "I need a refill
and I've been bruising a lot" can lose its clinical flag to `refill`. Ask gates
as independent `noul` questions instead; they are evaluated regardless of what
the intent question decided, and they cost nothing extra.

### Costs, if you have them

`expected_cost()` uses these to pick a per-class escalation threshold, which is
the metric that should actually decide deployment.

```json
"costs":         {"intent": {"lost_or_stolen_card": 400.0, "store_hours": 0.5}},
"cost_escalate": 1.20,
"recall_floors": {"intent": {"lost_or_stolen_card": 0.98}}
```

### `holdout_of`, if this is an evaluation task

Names listed here must never appear in a training mixture used to score this
task. `assert_training_mixture_clean()` enforces it. Include every alias:
`tasksource` calls Banking77 `banking77`, the Hub calls it `PolyAI/banking77`
and `mteb/banking77`.

## Mistakes we made, so you don't have to

**Splits must be disjoint by content, not by row.** If you paraphrase or add
noisy variants, keep every variant of an item in the same split. We group by
content and merge at Jaccard ≥ 0.62.

**A slice that scores 1.000 measures nothing.** Our first router set had four
tiers at ceiling; they could not distinguish any two models. Build items hard
enough to have headroom, and treat saturated slices as regression floors.

**Minority classes need enough examples to learn from.** Our `G_abusive` gate
had 7 positives against 443 negatives and learned to never fire — 0.000 recall
while scoring 0.984 "accuracy". Check per-class counts before trusting any
aggregate.

**Small tiers cannot prove anything.** Our `clinical_oblique` tier has 29 items.
With zero losses, the best possible paired result is p = 0.0625, so that tier
can never reach significance no matter how good the model is.

**`score` gold is an `int`, `noul` gold is a `bool`.** In Python `True == 1`,
so a careless conversion rewrites score levels 0 and 1 as `false`/`true`. That
corrupted 240 of 600 rows for us and looked exactly like "the ordinal head
doesn't work". `validate()` now catches it.

## From a CSV

```bash
uv run python scripts/make_task.py --csv mydata.csv \
    --text-column message --label-column intent --name my_task
```

That writes `tasks/my_task/` with an 80/10/10 split, a `choice` question built
from the distinct labels, and placeholder descriptions for you to fill in —
which is the part actually worth your time.
