"""Bar charts comparing the encoder, the decoder and the commercial reference.

Four panels, because the honest answer differs by question and a single
headline number hides that: zero-shot transfer, task-specific accuracy after
fine-tuning, single-request latency, and sustained throughput. The reference
system wins the first, we win the other three, and a reader deciding what to
deploy needs all four in one place.

Every number is measured and sourced in docs/results.md. Nothing here is
interpolated or estimated.

    uv run python scripts/plot_comparison.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ENC, DEC, JEV = "#4C72B0", "#DD8452", "#55A868"
OUT = "docs/plots/comparison.png"


def bars(ax, labels, values, colours, title, ylabel, fmt="{:.3f}", note=None):
    x = range(len(labels))
    b = ax.bar(x, values, color=colours, width=0.62, edgecolor="none")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(title, fontsize=10.5, pad=9, loc="left")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    top = max(values)
    for r, v in zip(b, values):
        ax.text(r.get_x() + r.get_width() / 2, v + top * 0.02, fmt.format(v),
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, top * 1.22)
    if note:
        ax.text(0.0, -0.30, note, transform=ax.transAxes, fontsize=7.6,
                va="top", color="#555555")


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    fig.subplots_adjust(hspace=0.62, wspace=0.26)

    # 1. Zero-shot. The one the reference system wins, so it goes first.
    bars(
        axes[0][0],
        ["encoder", "decoder", "Jev"],
        [17.2, 29.6, 38.3],
        [ENC, DEC, JEV],
        "Zero-shot transfer, 7 held-out tasks",
        "mean multiple of chance",
        "{:.1f}x",
        note="Schemas none of the three was trained on, 600 rows each, scored at full menu size.\n"
             "Jev wins five of seven and ties two. This is the gap we did not close.",
    )

    # 2. Fine-tuned. The one we win, and the project's actual claim.
    bars(
        axes[0][1],
        ["enc\nfrom base", "enc\nfrom ckpt", "dec\nfrom ckpt", "dec\nfrom base", "Jev\nzero-shot"],
        [0.544, 0.899, 0.9615, 0.979, 0.941],
        [ENC, ENC, DEC, DEC, JEV],
        "Healthcare router intent, after fine-tuning",
        "accuracy",
        note="395 labelled examples. Note the two paths differ: the encoder gains +36 points from a\n"
             "general checkpoint, the decoder loses 1.8 by starting there rather than from base.",
    )

    # 3. Latency. Idle GPU, batch 1, so the numbers are comparable.
    bars(
        axes[1][0],
        ["encoder", "decoder", "Jev\n(compute)", "Jev\n(observed)"],
        [20.4, 42.9, 70.0, 401.0],
        [ENC, DEC, JEV, JEV],
        "Latency per state, single request",
        "p50 milliseconds",
        "{:.0f} ms",
        note="Ours on an idle H100. Jev's observed p50 from India is 401 ms against a 331 ms network\n"
             "floor, so at most about 70 ms of it is their compute. The rest is transport you cannot remove.",
    )

    # 4. Throughput, the measurement that most changes a deployment decision.
    bars(
        axes[1][1],
        ["encoder", "decoder", "Jev"],
        [654.6, 84.4, 47.0],
        [ENC, DEC, JEV],
        "Sustained throughput, concurrent load",
        "states per second",
        "{:.0f}/s",
        note="Each state answers ten questions, so the encoder peak is 6,546 decisions per second.\n"
             "Ours is one process with no HTTP layer or network, so it bounds the hardware, not a deployment.",
    )

    fig.suptitle(
        "OpenJev encoder and decoder against the commercial reference",
        fontsize=13, y=0.985, x=0.09, ha="left",
    )
    fig.text(
        0.09, 0.945,
        "The answer differs by question: they are better with no labels, we are better with a few hundred, "
        "and on serving cost it is not close.",
        fontsize=9, color="#555555", ha="left",
    )
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
