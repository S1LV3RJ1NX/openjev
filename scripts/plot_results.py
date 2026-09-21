"""Plot training curves and held-out results from the logs a run leaves behind.

    uv run python scripts/plot_results.py --log /tmp/train_final.log \\
        --out docs/plots --title "mixture_final"

Reads the trainer's own stdout rather than a separate metrics file, so any
run that has already happened can be plotted after the fact. Writes PNGs
suitable for the docs and the model card.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

STEP = re.compile(r"ep(\d+)\s+step\s+(\d+)/(\d+)\s+loss\s+([\d.]+)\s+(\d+)s")
ROW = re.compile(
    r"^(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(?:\[[\d.,]+\]\s+)?([\d.]+)?\s*([\d.]+)x"
)


def parse_curve(text: str) -> tuple[list[int], list[float], list[int]]:
    steps, losses, secs = [], [], []
    for m in STEP.finditer(text):
        steps.append(int(m.group(2)))
        losses.append(float(m.group(4)))
        secs.append(int(m.group(5)))
    return steps, losses, secs


def parse_heldout(text: str) -> list[tuple[str, int, float, float]]:
    """(task, K, accuracy, multiple of chance) from the held-out table."""
    out = []
    block = text.split("HELD-OUT SUITE")[-1]
    for line in block.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0][0].isalpha():
            continue
        try:
            k, n, acc = int(parts[1]), int(parts[2]), float(parts[3])
        except ValueError:
            continue
        mult = next((float(p.rstrip("x")) for p in parts if p.endswith("x")), acc * k)
        out.append((parts[0], k, acc, mult))
    return out


def smooth(ys: list[float], w: int) -> list[float]:
    if w < 2 or len(ys) < w:
        return ys
    out = []
    for i in range(len(ys)):
        lo = max(0, i - w // 2)
        out.append(sum(ys[lo : i + 1]) / (i + 1 - lo))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, nargs="+")
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--out", default="docs/plots")
    ap.add_argument("--title", default="")
    ap.add_argument("--smooth", type=int, default=50)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = args.label or [Path(p).stem for p in args.log]
    written = []

    # ---- training curves -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_curve = False
    for path, label in zip(args.log, labels):
        text = Path(path).read_text(errors="ignore")
        steps, losses, _ = parse_curve(text)
        if not steps:
            continue
        any_curve = True
        # Draw the smoothed line first so the raw trace can borrow its colour;
        # letting matplotlib cycle independently gives each series two colours
        # and makes the legend wrong.
        line, = ax.plot(steps, smooth(losses, args.smooth), linewidth=1.8, label=label)
        ax.plot(steps, losses, alpha=0.18, linewidth=0.8, color=line.get_color())
    if any_curve:
        ax.set_xlabel("step")
        ax.set_ylabel("training loss")
        ax.set_title(f"Training loss{' — ' + args.title if args.title else ''}")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        p = out / "training_loss.png"
        fig.savefig(p, dpi=150)
        written.append(p)
    plt.close(fig)

    # ---- held-out transfer ----------------------------------------------
    series = {}
    for path, label in zip(args.log, labels):
        rows = parse_heldout(Path(path).read_text(errors="ignore"))
        if rows:
            series[label] = rows
    if series:
        tasks = [r[0] for r in next(iter(series.values()))]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        width = 0.8 / max(1, len(series))
        for i, (label, rows) in enumerate(series.items()):
            vals = {r[0]: r[3] for r in rows}
            xs = [j + i * width for j in range(len(tasks))]
            ax.bar(xs, [vals.get(t, 0) for t in tasks], width=width, label=label)
        # Chance is 1.0 on this axis, so the line is the thing to clear.
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        # Label it below the line and hard left: the rightmost tasks are the
        # ones that sit closest to chance, so anything above the line there
        # lands on top of the bars it is meant to explain.
        ax.text(-0.45, 0.94, "chance", fontsize=8, ha="left", va="top")
        ax.set_yscale("log")
        ax.set_ylabel("multiple of chance (log scale)")
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(tasks))])
        ax.set_xticklabels([t.replace("_", "\n", 1) for t in tasks], fontsize=8)
        ax.set_title(f"Held-out transfer{' — ' + args.title if args.title else ''}")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, axis="y", linewidth=0.5)
        fig.tight_layout()
        p = out / "heldout_transfer.png"
        fig.savefig(p, dpi=150)
        written.append(p)
        plt.close(fig)

    if not written:
        raise SystemExit("nothing parsed from those logs; is the format right?")
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
