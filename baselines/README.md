# Baseline predictions

## `jev_healthcare_router.json`

Per-item predictions from TypeSafe's Jev API on all 450 test items of
`tasks/healthcare_router`, collected in a single pass. This is what makes
the paired comparisons in our results reproducible: a McNemar test needs
both systems' answers on the same items, and quoting only aggregate
accuracies would let anyone check the deltas but not the significance.

Structure, keyed by the input string:

```json
{
  "I need to refill my blood pressure prescription.": {
    "A_pred": "refill",
    "A_probs": {"refill": 1.0, "order_status": 0.0, ...},
    "nouls": {"C_refill": 0.95, "G_clinical": 0.04, ...}
  }
}
```

Reproduce our comparison:

```bash
python scripts/eval_router.py --ckpt <your.pt> --dump /tmp/ours.json
python scripts/compare_to_jev.py \
  --ours /tmp/ours.json --jev baselines/jev_healthcare_router.json
```

### Provenance and caveats

We authored every input in this dataset; these are one vendor's outputs
on them, recorded so others can check our arithmetic. We are not
redistributing their model, weights, or any part of their system.

Three things to know before you rely on these numbers:

- **One pass, one point in time.** The API is not deterministic across
  calls and the service changes. A rerun today will not match this file
  exactly, so treat it as the baseline *we measured*, not as a fixed
  property of the product.
- **Probabilities are quantized** to two decimals, which is what the API
  returns. Thresholding at exactly 0.5 is therefore slightly coarse.
- **The comparison this supports is asymmetric.** Our models are
  fine-tuned on this task and Jev is zero-shot on it. That favours us,
  and it is the reason we state the fine-tuned and zero-shot results
  separately everywhere rather than reporting a single headline.
