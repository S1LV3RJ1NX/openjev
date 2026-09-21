"""Check a task directory or mixture for the data faults that cost us runs.

Training is the expensive step, so the faults worth catching are the ones
that survive a run and show up as a bad number hours later. Every check here
corresponds to one we actually hit.

    uv run python scripts/audit_data.py tasks/mixture_ord3
    uv run python scripts/audit_data.py tasks/heldout --deep

Exits non-zero if anything at ERROR level is found.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openjev.schema import Choice, Noul, Score, Task  # noqa: E402

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def audit_task(path: Path, deep: bool) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    splits = {}
    for s in ("train", "dev", "test"):
        if (path / f"{s}.jsonl").exists():
            try:
                splits[s] = Task.load(path, s)
            except Exception as e:  # noqa: BLE001
                out.append((ERROR, f"{s} will not load: {str(e)[:70]}"))
    if not splits:
        return [(ERROR, "no loadable split")]

    for s, t in splits.items():
        for p in t.validate():
            out.append((ERROR, f"{s}: {p[:90]}"))

        for qid, q in t.questions.items():
            answered = [e for e in t.examples if qid in e.answers]
            if not answered:
                continue

            # --- type-specific gold correctness ---------------------------
            if isinstance(q, Score):
                k = len(q.criteria)
                for e in answered[:2000]:
                    g = e.answers[qid]
                    if isinstance(g, bool):
                        out.append((ERROR, f"{s}/{qid}: score gold is a bool. "
                                           f"True == 1 in Python, so levels 0 and 1 "
                                           f"get silently rewritten"))
                        break
                    if not isinstance(g, int) or not 0 <= g < k:
                        out.append((ERROR, f"{s}/{qid}: score gold {g!r} outside 0..{k-1}"))
                        break
            elif isinstance(q, Noul):
                bad = [e for e in answered[:2000] if not isinstance(e.answers[qid], bool)]
                if bad:
                    out.append((ERROR, f"{s}/{qid}: noul gold is "
                                       f"{type(bad[0].answers[qid]).__name__}, not bool"))
            elif isinstance(q, Choice):
                for e in answered[:2000]:
                    menu = (e.criteria or {}).get(qid) or q.criteria
                    if e.answers[qid] not in menu:
                        out.append((ERROR, f"{s}/{qid}: gold {e.answers[qid]!r} "
                                           f"is not in its own menu"))
                        break

            # --- class balance -------------------------------------------
            # Only meaningful for a fixed menu. With per-example criteria the
            # answer text is often unique per row by construction, and
            # counting it reported 76 "minority classes" on tasks that have
            # no classes at all. Position balance is what matters there, and
            # the ingester checks that.
            per_example = any(e.criteria for e in answered[:50])
            dist = collections.Counter(str(e.answers[qid]) for e in answered)
            if per_example:
                dist = collections.Counter()
            if not dist:
                pass  # per-example menu, checked at ingestion instead
            elif len(dist) == 1:
                out.append((ERROR, f"{s}/{qid}: one class only, teaches nothing"))
            else:
                worst, n_worst = min(dist.items(), key=lambda kv: kv[1])
                if n_worst < 10 and len(answered) > 100:
                    out.append((WARN, f"{s}/{qid}: class {worst!r} has {n_worst} "
                                      f"of {len(answered)} examples; it will be learned "
                                      f"as 'never predict this'"))
                top = max(dist.values()) / len(answered)
                if top > 0.95:
                    out.append((WARN, f"{s}/{qid}: {top:.0%} one class, so a constant "
                                      f"answer scores {top:.3f}"))

            # --- option text quality --------------------------------------
            if isinstance(q, Noul) and q.criteria:
                a, b = norm(q.criteria.get("true", "")), norm(q.criteria.get("false", ""))
                if a and b:
                    wa, wb = set(a.split()), set(b.split())
                    j = len(wa & wb) / max(1, len(wa | wb))
                    if j > 0.5:
                        out.append((WARN, f"{qid}: the two noul descriptions overlap "
                                          f"{j:.0%}. They restate one judgement from "
                                          f"opposite sides, and scoring both measures "
                                          f"the overlap"))
            if isinstance(q, Choice) and q.criteria:
                descs = [norm(v) for v in q.criteria.values() if v]
                if descs and len(set(descs)) < len(descs):
                    out.append((WARN, f"{qid}: duplicate option descriptions"))
                placeholder = [k for k, v in q.criteria.items()
                               if str(v).strip().upper().startswith("TODO")]
                if placeholder:
                    out.append((WARN, f"{qid}: {len(placeholder)} option(s) still "
                                      f"hold a TODO description"))
                bare = sum(1 for k, v in q.criteria.items() if norm(k) == norm(v))
                if bare and bare == len(q.criteria):
                    out.append((INFO, f"{qid}: descriptions just restate the labels; "
                                      f"ours measured that as worth nothing (p = 0.75)"))

    # --- split hygiene ----------------------------------------------------
    # Compare whole states. Truncating first reported 169 false duplicates on
    # a task whose states are long and share a prompt prefix, and an auditor
    # that invents faults is worse than none.
    if len(splits) > 1 and deep:
        seen: dict[str, str] = {}
        for s, t in splits.items():
            for e in t.examples:
                key = norm(str(e.state))
                if key in seen and seen[key] != s:
                    out.append((ERROR, f"state appears in both {seen[key]} and {s}"))
                    break
                seen[key] = s

    for s, t in splits.items():
        qid0 = next(iter(t.questions), None)
        groups: dict[tuple, list] = collections.defaultdict(list)
        for e in t.examples:
            # Key on the menu as well as the state. With per-example criteria
            # the same state legitimately carries a different gold when it is
            # offered different options, and ignoring that flagged 1,500
            # perfectly good rows on a preference dataset as contradictory.
            menu = tuple(sorted((e.criteria or {}).get(qid0, {}))) if e.criteria else ()
            groups[(norm(str(e.state)), menu)].append(e.answers.get(qid0))
        dupes = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
        # A repeated state is only a fault if its labels disagree, which makes
        # those items unanswerable however good the model is.
        conflicting = sum(len(v) for v in groups.values()
                          if len(v) > 1 and len({str(x) for x in v}) > 1)
        if conflicting:
            out.append((ERROR, f"{s}: {conflicting} examples share a state with a "
                               f"different gold; they cannot all be answered correctly"))
        elif dupes > len(t.examples) * 0.05:
            out.append((WARN, f"{s}: {dupes} repeated states of {len(t.examples)}, "
                              f"labels consistent"))

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--deep", action="store_true", help="also check split overlap")
    ap.add_argument("--quiet", action="store_true", help="only show ERROR and WARN")
    args = ap.parse_args()

    root = Path(args.root)
    dirs = ([root] if (root / "task.json").exists()
            else sorted(p for p in root.iterdir() if (p / "task.json").exists()))
    if not dirs:
        raise SystemExit(f"no tasks under {root}")

    totals = collections.Counter()
    by_level: dict[str, list[str]] = collections.defaultdict(list)
    types = collections.Counter()

    for d in dirs:
        findings = audit_task(d, args.deep)
        try:
            t = Task.load(d, "train" if (d / "train.jsonl").exists() else "test")
            for q in t.questions.values():
                types[type(q).__name__] += 1
        except Exception:  # noqa: BLE001
            pass
        for level, msg in findings:
            if args.quiet and level == INFO:
                continue
            totals[level] += 1
            by_level[level].append(f"{d.name}: {msg}")

    print(f"audited {len(dirs)} task(s) in {root}")
    print(f"question types: {dict(types)}")
    for level in (ERROR, WARN, INFO):
        msgs = by_level.get(level, [])
        if not msgs:
            continue
        print(f"\n== {level} ({len(msgs)}) ==")
        for m in msgs[:40]:
            print(f"  {m}")
        if len(msgs) > 40:
            print(f"  ... and {len(msgs) - 40} more")

    if not totals:
        print("\nclean")
    sys.exit(1 if totals[ERROR] else 0)


if __name__ == "__main__":
    main()
