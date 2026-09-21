"""Draw the block-diagonal attention mask, for the paper.

The mask is the least obvious part of the architecture and the easiest to
describe wrongly in prose. A picture of which positions may attend to which
settles it in one glance.

    uv run python scripts/plot_mask.py --out report/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    # A small packed sequence: shared state, then three questions.
    blocks = [("shared\nstate", 0, 6), ("q1\nintent", 1, 4),
              ("q2\nclinical", 2, 3), ("q3\nurgency", 3, 3)]
    ids, ticks, labels = [], [], []
    pos = 0
    for name, bid, n in blocks:
        ids += [bid] * n
        ticks.append(pos + n / 2 - 0.5)
        labels.append(name)
        pos += n
    b = np.array(ids)
    n = len(b)

    # Eq. 1: attend if the key is in the shared prefix, or in my own block.
    allowed = (b[None, :] == 0) | (b[None, :] == b[:, None])
    causal = np.tril(np.ones((n, n), dtype=bool))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, m, title in (
        (axes[0], allowed, "Encoder backbone\n(bidirectional within the rule)"),
        (axes[1], allowed & causal, "Decoder backbone\n(same rule, composed with causality)"),
    ):
        # 0 = blocked, 1 = allowed to the shared prefix, 2 = allowed within block
        shade = np.zeros_like(m, dtype=float)
        shade[m & (b[None, :] == 0)] = 1.0
        shade[m & (b[None, :] != 0)] = 2.0
        ax.imshow(shade, cmap=matplotlib.colors.ListedColormap(
            ["#EDEDED", "#9EC5E8", "#F3B98B"]), vmin=0, vmax=2,
            interpolation="nearest")
        # Block boundaries.
        edges = np.where(np.diff(b) != 0)[0] + 0.5
        for e in edges:
            ax.axhline(e, color="white", linewidth=2)
            ax.axvline(e, color="white", linewidth=2)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel("key position (attended to)", fontsize=9)
        ax.set_ylabel("query position (attending)", fontsize=9)
        ax.set_title(title, fontsize=10)

    fig.legend(handles=[
        Patch(facecolor="#9EC5E8", label="may attend: shared state prefix"),
        Patch(facecolor="#F3B98B", label="may attend: own question block"),
        Patch(facecolor="#EDEDED", label="blocked"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "attention_mask.png"
    fig.savefig(p, dpi=170)
    print(f"wrote {p}")
    print("Every question sees the state and itself, never another question,")
    print("so an answer cannot depend on what else was asked in the call.")


if __name__ == "__main__":
    main()
