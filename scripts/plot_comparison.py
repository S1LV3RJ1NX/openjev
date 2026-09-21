"""Encoder, decoder and Jev on the same router task: accuracy and latency.

    uv run python scripts/plot_comparison.py --out docs/plots

The numbers are measured, not estimated, and each carries its provenance
below. They come from different runs on the same 450 test items, which is
why they are written here rather than parsed from one log.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Accuracy: scripts/eval_router.py on tasks/healthcare_router test, 450 items.
# Paired significance from scripts/compare_to_jev.py (exact McNemar).
# Jev is jev-1.13.0, measured 20 September 2026.
METRICS = [
    #  label,                     encoder, decoder LoRA,  jev
    ("intent",                      0.899,        0.979, 0.941),
    ("multi-label\nexact set",      0.789,        0.909, 0.822),
    ("G_clinical",                  0.980,        0.987, 0.978),
    ("G_abusive",                   0.998,        0.993, 0.996),
    ("G_injection",                 0.978,        0.993, 0.996),
    ("G_pharmacy",                  0.947,        0.978, 0.880),
    ("oblique clinical\nrecall",    1.000,        0.966, 0.793),
]

# Latency per state with the full router asked in one call.
#
# Ours: scripts/bench_latency.py, batch size 1 on an H100, 10 questions and
# 24 options, median of 60 timed calls after warmup.
#
# Jev: jev-reverse-engineering, full 8-question router shape, p50 over
# repeated calls to the hosted API.
#
# These are not the same measurement. Ours is local GPU compute; Jev's is an
# end-to-end call to a hosted service and therefore includes network
# round-trip. It is the latency a user experiences, not a claim about the
# speed of their model.
LATENCY = [
    ("encoder\n150M, local",  19.9, 20.3),
    ("decoder LoRA\n1.7B+16r, local",  22.4, 55.3),
    ("Jev\nhosted API",      407.0, None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/plots")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    enc, dec, jev = "#4C72B0", "#DD8452", "#55A868"

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.4, 1]}
    )

    # ---- accuracy --------------------------------------------------------
    labels = [m[0] for m in METRICS]
    xs = range(len(METRICS))
    w = 0.27
    for off, idx, colour, name in ((-w, 1, enc, "OpenJev encoder"),
                                   (0.0, 2, dec, "OpenJev decoder, LoRA r=16"),
                                   (w, 3, jev, "Jev 1.13.0")):
        ax1.bar([x + off for x in xs], [m[idx] for m in METRICS],
                width=w, color=colour, label=name)
    ax1.set_ylim(0.75, 1.02)
    ax1.set_ylabel("accuracy on 450 held-out test items")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_title("Router accuracy, fine-tuned on 395 examples")
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    ax1.grid(alpha=0.25, axis="y", linewidth=0.5)
    ax1.spines[["top", "right"]].set_visible(False)

    # ---- latency ---------------------------------------------------------
    names = [row[0] for row in LATENCY]
    p50 = [row[1] for row in LATENCY]
    ax2.bar(range(3), p50, width=0.55, color=[enc, dec, jev])
    for i, (row, v) in enumerate(zip(LATENCY, p50)):
        ax2.text(i, v * 1.12, f"{v:.0f} ms", ha="center", fontsize=9)
        if row[2]:
            ax2.plot([i, i], [v, row[2]], color="black", linewidth=1)
            ax2.text(i + 0.3, row[2], f"p95 {row[2]:.0f}", fontsize=7, va="center")
    ax2.set_yscale("log")
    ax2.set_ylim(10, 1200)
    ax2.set_ylabel("median latency per state, ms (log)")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(names, fontsize=8)
    ax2.set_title("Latency, whole router in one call")
    ax2.grid(alpha=0.25, axis="y", linewidth=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("OpenJev against Jev on the same healthcare router", fontsize=13)
    fig.tight_layout()
    p = out / "comparison_router.png"
    fig.savefig(p, dpi=150)
    print(f"wrote {p}")
    print("note: Jev latency is an end-to-end hosted API call and includes "
          "network;\n      ours is local GPU compute. Not a like-for-like "
          "model-speed claim.")


if __name__ == "__main__":
    main()
