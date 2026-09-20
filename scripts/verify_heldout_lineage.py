"""Measure the row overlaps that `holdout_of` asserts, instead of assuming them.

    uv run python scripts/verify_heldout_lineage.py

A `holdout_of` list is a claim: "these datasets contain the same rows as my
test set." Claims in this project get measured. Two of ours were worth
checking and one of them changed what we believe:

* **Civil Comments -> toxic_conversations.** Asserted from the dataset card
  ("an exact replica of the data released for the Jigsaw Unintended Bias
  challenge"). Measured against all 1,999,514 Civil Comments rows: **100.0%**
  of both `mteb/toxic_conversations_50k` and `SetFit/toxic_conversations`
  appear verbatim. They are these comments under another name.

  Note the sharding trap: compare against only the first train shard and the
  figure reads 50.9%, which looks like a partial overlap worth debating
  rather than an identity. This script reads every shard present and says so.

* **SST-5 -> Rotten Tomatoes.** *Not* originally the suspected path. The
  suspected path was GLUE SST-2, and GLUE turns out to respect the split:
  only 0.3% of SST-5 test sentences occur in `glue/sst2` train. The real leak
  is `rotten_tomatoes`, whose **train** split contains **77.0%** of SST-5's
  **test** sentences verbatim, with binary labels over the same text. It is
  task #378 in `tasksource`, so this is a live path into the intended mixture
  and nothing about the name would warn you.

  `glue/sst2` validation separately contains 75.4% of SST-5's *dev* sentences
  (and 0% of test), which is SST's dev split resurfacing under another name.

Network access required. Nothing here is needed to build or run the suite; it
exists so the README's numbers can be re-derived.
"""

from __future__ import annotations

import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/ojcache")
UA = {"User-Agent": "openjev-heldout/0.1"}
CIVIL_COMMENTS_TOTAL = 1_999_514  # 1,804,874 train + 97,320 validation + 97,320 test


def _norm(s: object) -> str:
    """Compare content, not tokenisation: mirrors differ on spacing and case."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _grab(ds: str, config: str, split: str, shards: int = 1) -> pd.DataFrame:
    idx = json.load(
        urllib.request.urlopen(f"https://huggingface.co/api/datasets/{ds}/parquet", timeout=90)
    )
    frames = []
    for url in idx[config][split][:shards]:
        raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900).read()
        frames.append(pd.read_parquet(io.BytesIO(raw)))
    return pd.concat(frames, ignore_index=True)


def _text_col(df: pd.DataFrame) -> str:
    for c in ("text", "sentence", "comment_text"):
        if c in df.columns:
            return c
    return df.columns[0]


def _local(ds: str, config: str, split: str) -> pd.DataFrame | None:
    p = CACHE / f"{ds.replace('/', '__')}__{config}__{split}__0.parquet"
    return pd.read_parquet(p) if p.exists() else None


def check_sst5() -> None:
    print("\n== SST-5 test/dev sentences, found in which other datasets ==")
    local = {s: _local("SetFit/sst5", "default", s) for s in ("validation", "test")}
    if any(v is None for v in local.values()):
        print("   skipped: run build_heldout.py --fetch first")
        return
    probes = [
        ("glue/sst2 (train)", "stanfordnlp/sst2", "default", "train"),
        ("glue/sst2 (validation)", "stanfordnlp/sst2", "default", "validation"),
        ("rotten_tomatoes (train)", "cornell-movie-review-data/rotten_tomatoes", "default", "train"),
        ("rotten_tomatoes (test)", "cornell-movie-review-data/rotten_tomatoes", "default", "test"),
    ]
    print(f"   {'pool':30s} {'rows':>7s}  {'% of SST-5 test':>16s}  {'% of SST-5 dev':>15s}")
    for label, ds, cfg, split in probes:
        try:
            df = _grab(ds, cfg, split)
        except Exception as exc:  # noqa: BLE001
            print(f"   {label:30s}  ERROR {type(exc).__name__}")
            continue
        pool = set(df[_text_col(df)].map(_norm))
        a = local["test"].text.map(_norm).isin(pool).mean() * 100
        b = local["validation"].text.map(_norm).isin(pool).mean() * 100
        print(f"   {label:30s} {len(df):7d}  {a:15.1f}%  {b:14.1f}%")


def _ensure_civil_comments_shards() -> None:
    """Download any Civil Comments shards not already cached."""
    idx = json.load(
        urllib.request.urlopen(
            "https://huggingface.co/api/datasets/google/civil_comments/parquet", timeout=90
        )
    )
    for split, urls in idx["default"].items():
        for n, url in enumerate(urls):
            out = CACHE / f"google__civil_comments__default__{split}__{n}.parquet"
            if out.exists():
                continue
            print(f"   fetching {out.name} ...")
            out.write_bytes(
                urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=1800).read()
            )


def check_civil_comments() -> None:
    print("\n== toxic_conversations rows, found in Civil Comments ==")
    # Every shard, not just shard 0: the train split is 2 shards and reading
    # one turns a 100% identity into a 50.9% "partial overlap". The build does
    # not need the train split, so fetch it here rather than making every
    # rebuild pay ~380 MB for it.
    _ensure_civil_comments_shards()
    shards = sorted(CACHE.glob("google__civil_comments__default__*.parquet"))
    if not shards:
        print("   skipped: run build_heldout.py --fetch first")
        return
    pool_df = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
    pool = set(pool_df.text.map(_norm))
    complete = len(pool_df) >= CIVIL_COMMENTS_TOTAL
    print(f"   compared against {len(pool_df)} of {CIVIL_COMMENTS_TOTAL} Civil Comments rows "
          f"({len(pool)} distinct) from {len(shards)} shard(s)"
          + ("" if complete else "  -- INCOMPLETE, figures below are lower bounds"))
    for ds in ("mteb/toxic_conversations_50k", "SetFit/toxic_conversations"):
        try:
            df = _grab(ds, "default", "test")
        except Exception as exc:  # noqa: BLE001
            print(f"   {ds:34s}  ERROR {type(exc).__name__}")
            continue
        share = df[_text_col(df)].map(_norm).isin(pool).mean() * 100
        print(f"   {ds:34s} {len(df):6d} rows  ->  {share:5.1f}% also in Civil Comments")


def main() -> None:
    if "--help" in sys.argv:
        print(__doc__)
        return
    check_sst5()
    check_civil_comments()
    print(
        "\nAnything above 0% means the named dataset must stay out of any training\n"
        "mixture used to report numbers on the corresponding held-out task."
    )


if __name__ == "__main__":
    main()
