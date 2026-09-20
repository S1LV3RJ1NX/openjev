# Results

Everything measured by us, with the command to reproduce it. Numbers move when
you rerun them; intervals are bootstrap over examples.

## Banking77 specialist

**In-task, not a generalization result.** This model was fine-tuned on 9,839
Banking77 training examples. The reference system was not: it answered the same
question zero-shot from the option text alone. These measure different things,
and the comparison below is only meaningful as "what does having labels buy
you", not as "which model is better".

`tasks/heldout/banking77` is therefore **disqualified as a held-out task for
this checkpoint**. The trainer refuses to run on it without `--allow-heldout`,
and records the override in the checkpoint file.

```bash
uv run python scripts/build_banking77_specialist.py
uv run python scripts/train.py --task tasks/banking77_specialist \
    --epochs 3 --bs 16 --max-len 4096 --allow-heldout
```

ModernBERT-base (149.6M params, of which the shared scorer is 593K), 3 epochs,
batch 16, 1,845 steps, **13.5 minutes on one H100**. Test is the same 600 rows
as `tasks/heldout/banking77`.

| | accuracy | 95% CI | macro-F1 | ECE | Brier |
|---|---|---|---|---|---|
| OpenJev, raw | 0.9233 | [0.900, 0.945] | 0.9230 | 0.056 | 0.1300 |
| OpenJev, temperature-scaled | **0.9233** | [0.900, 0.945] | **0.9230** | **0.029** | **0.1194** |
| *reference: Jev `1.13.0`, zero-shot* | *0.820* | *[0.777, 0.863]* | *0.806* | *0.068* | *0.277* |

The reference row was measured against the public TypeSafe API on 20 September
2026 on 300 stratified rows of the same test split, so it is a different sample
from our 600 and should be read as approximate.

### What this does and does not support

**Supports:** the architecture learns. A shared scoring head over option
markers, with no per-class parameters, reaches 92.3% on a 77-way task — in line
with what ordinary fine-tuned encoders have long achieved on Banking77. The
head is not the bottleneck, and a 150M model trained for 13 minutes on one GPU
is enough.

**Supports:** if you have labels for your task, fine-tuning a small encoder
beats a general decision model on that task, by about ten points here. That is
the project's actual pitch and it is the least surprising result in the table.

**Does not support:** any claim about zero-shot ability, generalization, or
being "better than Jev". We trained on the task. The interesting comparison —
a held-out schema neither model has seen — has not been run yet.

### Temperature scaling did exactly what it should

Accuracy is **identical** before and after (0.9233 both), while ECE halves from
0.056 to 0.029 and Brier improves from 0.1300 to 0.1194. That is the expected
behaviour and a useful check on the implementation: temperature scaling is
monotonic, so it cannot reorder predictions and cannot change accuracy. If it
had, something would be wrong.

The fitted temperature is **1.917**, comfortably above 1, meaning the model was
overconfident — the standard consequence of training to convergence on
cross-entropy, and the reason the calibration step exists at all.

## Packing benchmark

See [`architecture.md`](architecture.md#measured-is-packing-worth-its-complexity).
Flat in question count; 24x over the naive one-sequence-per-question
alternative at a 2.8k-token state; no benefit at all on short states.

## Not yet measured

- Held-out schema transfer, which is the number that matters for the zero-shot
  claim and the one we expect to lose on.
- The healthcare router suite, including the oblique-clinical gate.
- Anything on `score` or `noul` primitives — Banking77 is `choice` only.
- Latency served the same way as the reference system. Our 9.6 ms is local
  compute with no network; the reference's ~400 ms included a 331 ms round
  trip. Comparing them directly would be dishonest.
