---
license: apache-2.0
task_categories:
  - text-classification
  - zero-shot-classification
  - multiple-choice
tags:
  - typed-decisions
  - multi-task
  - instruction-tuning
  - openjev
size_categories:
  - 100K<n<1M
---

# OpenJev training mixture

279 classification and multiple-choice tasks, 323,466 rows, normalised into
one typed-decision format so a single model can be trained across all of
them. Assembled from [`tasksource`](https://huggingface.co/tasksource) plus
two curated ordinal datasets.

Built for [OpenJev](https://github.com/S1LV3RJ1NX/openjev). The format is
plain JSONL, so nothing here requires that code.

## Composition

| primitive | tasks | what it is |
|---|---|---|
| `choice` | 234 | pick one of K options |
| `noul` | 34 | yes/no |
| `score` | 11 | one level on an ordered scale |

Option counts run from 2 to 174. 80 of the tasks carry **per-example
criteria**, meaning each row brings its own answer menu rather than sharing
a fixed one, which is what the MultipleChoice family needs.

## What this is disjoint from

Every task was checked against the
[held-out suite](https://huggingface.co/datasets/s1lv3rj1nx/openjev-heldout)
before being written, matching on full names, owner-stripped basenames and
glob patterns rather than exact strings. 23 tasks were excluded for
overlapping it, including several `civil_comments` configs and `banking77`
under three different names.

A mixture that quietly contains your evaluation data produces excellent
numbers that mean nothing, so this is enforced in code rather than
promised.

## Cleaning applied, and why

**Contradictory labels removed, 1,994 rows.** The same state carrying
different answers under the same menu cannot all be correct, so the
gradient is noise. `ethics_deontology` alone contributed 195.

**Oversized menus removed, 1,787 rows.** Some MultipleChoice options are
multi-paragraph passages, and a menu alone can exceed a 2,048-token budget.

**Option order shuffled.** `tasksource` ships MultipleChoice
correct-answer-first: measured, the gold sat at index 0 in **100% of rows
across all 83 ingested tasks**. Stored that way it teaches "pick the first
option" to anything reading the menu as written. Shuffled at ingestion so
the data on disk is honest.

**Degenerate tasks dropped.** Any task whose gold is one constant string,
or still lands at a fixed index after shuffling.

Verify with `scripts/audit_data.py`, which reports zero errors on this
mixture, and `scripts/check_packing.py`, which confirms all 323,466 rows
pack inside a 2,048-token budget.

## Known limitations

**Lopsided toward `choice`.** 234 against 34 and 11. Models trained here
transfer well on `choice` and poorly on the other two, and that is a
property of this mixture rather than of any architecture. Ordinal data in
particular is scarce: most `score` tasks are three-level sentiment scales.

**19 tasks have a class with under 10 examples**, which is learned as
"never predict this" rather than learned at all.

**Not large enough to be finished.** The Flan work suggests gain keeps
accruing past ~282 tasks. This is 279, and adding more diverse tasks,
especially ordinal and yes/no ones, is the most useful contribution
anyone could make.

## Format

Each task is a directory with `task.json` and `train.jsonl`. Specified in
[docs/dataset-format.md](https://github.com/S1LV3RJ1NX/openjev/blob/main/docs/dataset-format.md).

```json
{"state": "the text", "answers": {"q": "gold label"}, "meta": {"source": "tasksource id"}}
```

Per-example menus add a `criteria` field carrying that row's options.

## Licence

Apache 2.0 for the assembly. Underlying datasets keep their own licences;
each task names its source in `description`.
