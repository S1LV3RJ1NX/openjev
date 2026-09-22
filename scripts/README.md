# Scripts

There are a lot of these. Most exist because a result in the report needed
producing and we keep the thing that produced it. **If you are just using
OpenJev, you need four of them.**

## The four you need

| | |
|--|--|
| [`make_task.py`](make_task.py) | Turn a CSV into a task directory. Start here. See [dataset format](../docs/dataset-format.md). |
| [`train.py`](train.py) | Train an adapter on your task. The recommended recipe is in the [README](../README.md#train-a-task-adapter-on-your-own-data). |
| [`eval_router.py`](eval_router.py) | Score a checkpoint on a multi-question task, per tier, with confidence intervals. |
| [`eval_heldout.py`](eval_heldout.py) | Score zero-shot transfer on schemas the model never trained on, with a held-in control. |

Everything below is for reproducing the paper or for diagnosing a training
run that is misbehaving.

## Reproducing the results

| | |
|--|--|
| [`compare_to_jev.py`](compare_to_jev.py) | Paired McNemar against the commercial reference, using the predictions in [`baselines/`](../baselines/). No API key needed. |
| [`plot_comparison.py`](plot_comparison.py) | The four-panel figure in the README and the report. |
| [`plot_results.py`](plot_results.py) | Training curves and held-out transfer across mixture generations. |
| [`plot_mask.py`](plot_mask.py) | The block-diagonal attention mask diagram. |

## Benchmarks

Run these on an idle GPU. A contended benchmark is not a cheap
approximation of an uncontended one, it is a different measurement, and
mixing the two put two disagreeing sets of latency numbers in our report
for a while.

| | |
|--|--|
| [`bench_latency.py`](bench_latency.py) | Per-state p50 and p95 at batch 1, the interactive case. |
| [`bench_throughput.py`](bench_throughput.py) | Sustained throughput across batch sizes, the serving case. |
| [`bench_packing.py`](bench_packing.py) | Packing against one sequence per question. |
| [`baseline_separate.py`](baseline_separate.py) | Independent per-question classifiers, the obvious alternative to packing. |

## Building the training mixture

You only need these if you are rebuilding the 279-task mixture rather than
[downloading it](https://huggingface.co/datasets/s1lv3rj1nx/openjev-mixture).
Run them in this order.

| | |
|--|--|
| [`build_mixture.py`](build_mixture.py) | Pull tasks from `tasksource`, with backoff. Slow and rate-limited. |
| [`add_multiplechoice.py`](add_multiplechoice.py) | Add the MultipleChoice family, which needs per-example option sets. |
| [`add_ordinal_tasks.py`](add_ordinal_tasks.py) | Add curated star-rating datasets, because ordinal data is scarce. |
| [`retype_mixture.py`](retype_mixture.py) | Retype negation-pair `choice` tasks as `noul`. |
| [`clean_mixture.py`](clean_mixture.py) | Drop contradictory and degenerate rows. |

## Building the evaluation suites

| | |
|--|--|
| [`build_heldout.py`](build_heldout.py) | Rebuild the held-out suite. Two tasks are not redistributable and are fetched rather than committed. |
| [`heldout_criteria.py`](heldout_criteria.py) | Attach per-example option sets to held-out tasks. |
| [`heldout_criteria_banking77.py`](heldout_criteria_banking77.py) | The same for Banking77, which needs its own label handling. |
| [`verify_heldout_lineage.py`](verify_heldout_lineage.py) | Check no held-out dataset leaked into the mixture. |
| [`build_banking77_specialist.py`](build_banking77_specialist.py) | An in-task specialist split, for the contamination experiment. |

## Diagnosing a bad run

Reach for these when a number looks wrong. Each exists because something
did go wrong once.

| | |
|--|--|
| [`audit_data.py`](audit_data.py) | Gold correctness, class balance, contradictions, degenerate tasks. Found 3,781 bad rows. |
| [`check_packing.py`](check_packing.py) | Sweep every example through the packer on CPU before spending GPU hours. |
| [`diagnose_binary.py`](diagnose_binary.py) | Tells apart "no signal" from "miscalibrated" on binary tasks, which argmax accuracy cannot. |
| [`eval_zeroshot.py`](eval_zeroshot.py) | Score an untrained backbone, to check how much the output format alone buys. It buys very little. |
| [`run_general_decoder.sh`](run_general_decoder.sh) | The exact command that trains the general adapter, with its settings explained. |
