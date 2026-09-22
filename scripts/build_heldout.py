"""Build the held-out generalization suite under `tasks/heldout/`.

    uv run python scripts/build_heldout.py              # build everything
    uv run python scripts/build_heldout.py --dump-train # per-label train dumps
    uv run python scripts/build_heldout.py --check      # validate, do not write

What this script is for
-----------------------
`docs/evaluation.md` commits us to holding out *entire question schemas*,
never random rows. The training mixture comes from `tasksource`, which contains
Banking77, AG News, all seven Civil Comments configs and all five HelpSteer
configs. Left alone, our held-out numbers would be contaminated by default
rather than by mistake.

So every task carries a `holdout_of` list naming every alias of its source we
could find, `tasks/heldout/manifest.json` carries the union of those lists, and
`openjev.heldout.assert_training_mixture_clean` refuses a mixture that
intersects it.

Three rules the code enforces rather than documents
---------------------------------------------------
1. Option descriptions are checked to cover exactly the source label set, so a
   hand-written key that matches no real label cannot slip through.
2. Labels keep the source dataset's own surface form. The only rename is
   CLINC's `oos` -> `none_of_the_above`, which the brief asks for, and it is
   asserted to be the only one.
3. Rows for datasets whose licence does not clearly permit redistribution are
   generated locally and git-ignored, never vendored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import heldout_criteria as C  # noqa: E402
from openjev.schema import Choice, Example, Noul, Score, Task  # noqa: E402

OUT = ROOT / "tasks" / "heldout"
CACHE = Path("/tmp/ojcache")
SEED = 0
CAP = 600

# Our reference measurement of Jev (0.820 accuracy / 0.806 macro-F1 / ECE 0.068,
# `jev-1.13.0`, 20 Sep 2026) was taken on 300 stratified rows of this exact
# parquet. If you hold that row list, point OPENJEV_B77_REFERENCE at it and the
# reference set becomes a strict subset of the 600 sampled here, rather than a
# different draw that would need caveating. Absent it, sampling is plain
# stratified and the reference numbers are simply a different sample.
B77_REFERENCE_METRICS = Path(
    os.environ.get("OPENJEV_B77_REFERENCE", "/nonexistent/metrics_banking77.json")
)
B77_PARQUET_SHA256 = "9575f636fdeae0c94a0f3f2d926ca8f9a2833b998cf94108936dfd5d72322bf9"


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


@dataclass
class Source:
    """Where a task's rows come from, and what we may do with them."""

    hf_id: str
    config: str
    homepage: str
    license: str
    redistributable: bool
    license_note: str
    # Names that must never appear in a training mixture used to score this
    # task. Inclusion rule (see tasks/heldout/README.md): the dataset itself
    # under every alias, anything sharing source documents with it, and any
    # subsample or relabelling of it. NOT merely same-topic datasets.
    holdout_of: list[str]
    # Same-topic datasets we deliberately did *not* exclude, recorded so the
    # decision is auditable rather than invisible.
    related_not_excluded: list[str] = field(default_factory=list)
    # Task ids that are actually present in `tasksource` AND leak into this
    # test set. This is the field that matters: a task can be safe under its
    # own name and still be contaminated through a differently-named source,
    # which is exactly what happens to sst5 via `rotten_tomatoes`.
    tasksource_leak_paths: list[str] = field(default_factory=list)
    description_source: str = ""


SOURCES: dict[str, Source] = {
    "banking77": Source(
        hf_id="mteb/banking77",
        config="default",
        homepage="https://huggingface.co/datasets/PolyAI/banking77",
        license="CC-BY-4.0",
        redistributable=True,
        license_note=(
            "Upstream PolyAI/banking77 is CC-BY-4.0; the mteb mirror we read is "
            "labelled MIT. Both permit redistribution with attribution."
        ),
        tasksource_leak_paths=["banking77"],
        description_source="train split (verbatim from the measured +5.1pt description set)",
        holdout_of=[
            "banking77",
            "PolyAI/banking77",
            "mteb/banking77",
            "legacy-datasets/banking77",
            "DeepPavlov/banking77",
            "FastFit/banking_77",
            "mteb/BankingClassification",
            "mteb/Banking77Classification",
            "gtfintechlab/banking77",
            "haizelabs/banking77",
            "willcb/banking77",
            "AutoIntent/banking77_aug",
            "DeepPavlov/banking77-translated",
            "Donoe/Multilingual_Banking77",
        ],
    ),
    "clinc_oos": Source(
        hf_id="clinc/clinc_oos",
        config="plus",
        homepage="https://huggingface.co/datasets/clinc/clinc_oos",
        license="CC-BY-3.0",
        redistributable=True,
        license_note="CC-BY-3.0 permits redistribution with attribution.",
        tasksource_leak_paths=[],
        description_source="train split",
        holdout_of=[
            "clinc_oos",
            "clinc/clinc_oos",
            "clinc150",
            "CLINC150",
            "DeepPavlov/clinc_oos",
            "DeepPavlov/clinc150",
            "DeepPavlov/clinc150_subset",
            "contemmcm/clinc150",
            "FastFit/clinc_150",
            "Syedtahaali94/clinc_oos",
            "cmaldona/Generalization-MultiClass-CLINC150-ROSTD",
            "cmaldona/All-Generalization-OOD-CLINC150",
            "AutoIntent/clinc150_aug_qwen2.5-7b-awq",
            "OpenVoiceOS/ovos-intent-bench-clinc150",
            "OpenVoiceOS/ovos-intent-bench-clinc150-train",
            "mteb/CLINC150Classification",
        ],
    ),
    "massive_intent": Source(
        hf_id="mteb/amazon_massive_intent",
        config="en",
        homepage="https://huggingface.co/datasets/mteb/amazon_massive_intent",
        license="CC-BY-4.0",
        redistributable=True,
        license_note=(
            "Upstream Amazon MASSIVE is CC-BY-4.0; the mteb mirror is labelled "
            "Apache-2.0. Both permit redistribution."
        ),
        tasksource_leak_paths=[],
        description_source="train split",
        holdout_of=[
            "massive",
            "AmazonScience/massive",
            "amazon_massive_intent",
            "mteb/amazon_massive_intent",
            "mteb/MassiveIntentClassification",
            "MassiveIntentClassification",
            "SetFit/amazon_massive_intent_en-US",
            # The scenario labels are the same utterances with a coarser label.
            "amazon_massive_scenario",
            "mteb/amazon_massive_scenario",
            "mteb/MassiveScenarioClassification",
            "SetFit/amazon_massive_scenario_en-US",
            # Non-English MASSIVE is the same sentences translated. Training on
            # them would be cross-lingual leakage of this exact test set.
            "SetFit/amazon_massive_intent_*",
            "SetFit/amazon_massive_scenario_*",
        ],
        related_not_excluded=[
            "snips_built_in_intents",
            "bigbench/intent_recognition",
        ],
    ),
    "sst5": Source(
        hf_id="SetFit/sst5",
        config="default",
        homepage="https://huggingface.co/datasets/SetFit/sst5",
        license="unknown",
        redistributable=False,
        license_note=(
            "SetFit/sst5 carries no licence metadata at all and upstream "
            "stanfordnlp/sst is tagged 'unknown'. Absence of a licence is not "
            "permission, so rows are generated locally and git-ignored."
        ),
        # sst5 is not a tasksource task, but two tasksource tasks contain its
        # rows: rotten_tomatoes (#378) holds 77.0% of our TEST sentences and
        # glue/sst2 (#319) holds 75.4% of our DEV sentences. Measured, see
        # scripts/verify_heldout_lineage.py.
        tasksource_leak_paths=["rotten_tomatoes", "glue/sst2"],
        description_source="dataset documentation (the SST five-way scale)",
        holdout_of=[
            "sst5",
            "SetFit/sst5",
            "sst-5",
            "Realgon/sst5",
            "Samsoup/SST5",
            "VirtualRoyalty/SST5",
            "gimmaru/SetFit-sst5",
            # SST-2 is the SAME Rotten Tomatoes sentences with the neutral band
            # dropped and the scale collapsed to binary. glue/sst2 IS in
            # tasksource, so this is the live contamination risk for this task.
            "sst2",
            "glue/sst2",
            "stanfordnlp/sst2",
            "SetFit/sst2",
            "gpt3mix/sst2",
            # The treebank itself, and the phrase-level and Rotten Tomatoes
            # sources it was cut from.
            "sst",
            "stanfordnlp/sst",
            "sst/default",
            "rotten_tomatoes",
            "cornell-movie-review-data/rotten_tomatoes",
            "mattymchen/mr",
            "SetFit/rotten_tomatoes",
        ],
        related_not_excluded=[
            "tweet_eval/sentiment",
            "poem_sentiment",
            "amazon_polarity",
            "imdb",
        ],
    ),
    "ag_news": Source(
        hf_id="fancyzhx/ag_news",
        config="default",
        homepage="https://huggingface.co/datasets/fancyzhx/ag_news",
        license="unknown",
        redistributable=False,
        license_note=(
            "The dataset card declares 'license: unknown'. The original AG "
            "corpus terms permit non-commercial academic use of the corpus but "
            "do not grant redistribution rights we can rely on, so rows are "
            "generated locally and git-ignored."
        ),
        tasksource_leak_paths=["ag_news"],
        description_source="dataset documentation (ClassLabel names) and train split",
        holdout_of=[
            "ag_news",
            "fancyzhx/ag_news",
            "SetFit/ag_news",
            "sh0416/ag_news",
            "pietrolesci/agnews",
            "sentence-transformers/agnews",
            "contemmcm/ag_news",
            "r-three/ag_news_subset",
            "ag_news_subset",
            "Recognai/ag_news_corrected_labels",
            "Recognai/corrected_labels_ag_news",
            "Lots-of-LoRAs/task379_agnews_topic_classification",
            "Lots-of-LoRAs/task1541_agnews_classification",
        ],
    ),
    "civil_comments_toxicity": Source(
        hf_id="google/civil_comments",
        config="default",
        homepage="https://huggingface.co/datasets/google/civil_comments",
        license="CC0-1.0",
        redistributable=True,
        license_note="CC0-1.0 public domain dedication; redistribution is unrestricted.",
        tasksource_leak_paths=[
            "civil_comments/toxicity", "civil_comments/severe_toxicity",
            "civil_comments/obscene", "civil_comments/threat", "civil_comments/insult",
            "civil_comments/identity_attack", "civil_comments/sexual_explicit",
            "jigsaw_toxicity", "toxic_conversations",
        ],
        description_source="dataset documentation (the Jigsaw crowd-rating question)",
        holdout_of=[
            "civil_comments",
            "google/civil_comments",
            # tasksource exposes seven label columns over the SAME comments.
            "civil_comments/toxicity",
            "civil_comments/severe_toxicity",
            "civil_comments/obscene",
            "civil_comments/threat",
            "civil_comments/insult",
            "civil_comments/identity_attack",
            "civil_comments/sexual_explicit",
            "civil_comments__toxicity",
            "civil_comments__severe_toxicity",
            "civil_comments__obscene",
            "civil_comments__threat",
            "civil_comments__insult",
            "civil_comments__identity_attack",
            "civil_comments__sexual_explicit",
            # The card states this data is "an exact replica of the data
            # released for the Jigsaw Unintended Bias Kaggle challenge", so
            # every derivative of that challenge is the same comments.
            "jigsaw_unintended_bias",
            "google/jigsaw_unintended_bias",
            "jigsaw_toxicity",
            "tasksource/jigsaw_toxicity",
            "toxic_conversations",
            "SetFit/toxic_conversations",
            "SetFit/toxic_conversations_50k",
            "mteb/toxic_conversations_50k",
            "mteb/ToxicConversationsClassification",
            "civilcomments_wilds",
            "pietrolesci/civilcomments-wilds",
            "shlomihod/civil-comments-wilds",
            "lighteval/civil_comments_helm",
            "Lots-of-LoRAs/task1720_civil_comments_toxicity_classification",
            "Lots-of-LoRAs/task1721_civil_comments_obscenity_classification",
            "Lots-of-LoRAs/task1722_civil_comments_threat_classification",
            "Lots-of-LoRAs/task1723_civil_comments_sexuallyexplicit_classification",
            "Lots-of-LoRAs/task1724_civil_comments_insult_classification",
            "Lots-of-LoRAs/task1725_civil_comments_severtoxicity_classification",
        ],
        related_not_excluded=[
            "tweet_eval/hate",
            "hate_speech18",
            "hate_speech_offensive",
            "implicit-hate-stg1",
            "dynahate",
            "google/jigsaw_toxicity_pred",
        ],
    ),
    "helpsteer_helpfulness": Source(
        hf_id="nvidia/HelpSteer",
        config="default",
        homepage="https://huggingface.co/datasets/nvidia/HelpSteer",
        license="CC-BY-4.0",
        redistributable=True,
        license_note="CC-BY-4.0 permits redistribution with attribution.",
        tasksource_leak_paths=[
            "HelpSteer/helpfulness", "HelpSteer/correctness", "HelpSteer/coherence",
            "HelpSteer/complexity", "HelpSteer/verbosity",
            "HelpSteer2/helpfulness", "HelpSteer2/correctness", "HelpSteer2/coherence",
            "HelpSteer2/complexity", "HelpSteer2/verbosity",
        ],
        description_source="dataset documentation (card definition) and train split",
        holdout_of=[
            "HelpSteer",
            "nvidia/HelpSteer",
            "helpsteer",
            # tasksource exposes five attributes over the SAME prompt/response
            # pairs; training on any of them shows the model these exact rows.
            "HelpSteer/helpfulness",
            "HelpSteer/correctness",
            "HelpSteer/coherence",
            "HelpSteer/complexity",
            "HelpSteer/verbosity",
            "helpsteer__helpfulness",
            "helpsteer__correctness",
            "helpsteer__coherence",
            "helpsteer__complexity",
            "helpsteer__verbosity",
            # HelpSteer2/3 are separate collections but same project, same
            # annotation guidelines and overlapping prompt pools. Excluded
            # conservatively: a false positive only costs us a training task.
            "HelpSteer2",
            "nvidia/HelpSteer2",
            "HelpSteer2/helpfulness",
            "HelpSteer2/correctness",
            "HelpSteer2/coherence",
            "HelpSteer2/complexity",
            "HelpSteer2/verbosity",
            "helpsteer_2__helpfulness",
            "helpsteer_2__correctness",
            "helpsteer_2__coherence",
            "helpsteer_2__complexity",
            "helpsteer_2__verbosity",
            "HelpSteer3",
            "nvidia/HelpSteer3",
            "Weyaxi/HelpSteer-filtered",
            "RLHFlow/Helpsteer-preference-standard",
            "Columbia-NLP/DPO-HelpSteer",
        ],
        related_not_excluded=[
            "oasst2_dense_flat/helpfulness",
            "oasst2_dense_flat/quality",
            "openai/summarize_from_feedback",
        ],
    ),
}


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def stratified(labels: list, cap: int, seed: int = SEED, pinned: list[int] | None = None) -> list[int]:
    """Row positions spread as evenly as possible over `labels`, capped at `cap`.

    Classes with fewer rows than their quota contribute everything they have
    and their shortfall is redistributed, so a rare class never silently
    shrinks the sample. `pinned` positions are always included and count
    against their class quota, which is how the Banking77 reference rows stay
    inside our draw.
    """
    rng = random.Random(seed)
    by_label: dict[object, list[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)
    for idxs in by_label.values():
        rng.shuffle(idxs)

    pinned_set = set(pinned or [])
    chosen = sorted(pinned_set)
    # pinned rows first within their class, so a quota of q keeps up to q pinned
    for lab, idxs in by_label.items():
        idxs.sort(key=lambda i: (i not in pinned_set,))

    remaining = cap - len(chosen)
    if remaining <= 0:
        return sorted(chosen)[:cap]

    pools = {lab: [i for i in idxs if i not in pinned_set] for lab, idxs in by_label.items()}
    quota = {lab: 0 for lab in pools}
    # water-filling: hand out one row at a time to the class furthest below its
    # share, which degrades gracefully when classes are exhausted
    order = sorted(pools, key=lambda l: (len(pools[l]), str(l)))
    while remaining > 0:
        progressed = False
        for lab in order:
            if remaining == 0:
                break
            if quota[lab] < len(pools[lab]):
                quota[lab] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    for lab, q in quota.items():
        chosen.extend(pools[lab][:q])
    return sorted(chosen)


def _read(ds: str, config: str, split: str, shard: int = 0) -> pd.DataFrame:
    path = CACHE / f"{ds.replace('/', '__')}__{config}__{split}__{shard}.parquet"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nRun: uv run python scripts/build_heldout.py --fetch"
        )
    return pd.read_parquet(path).reset_index(drop=True)


def fetch_all() -> None:
    """Pull the source parquet into /tmp/ojcache. Network, run once."""
    want = {
        ("mteb/banking77", "default"): ["train", "test"],
        ("clinc/clinc_oos", "plus"): ["train", "validation", "test"],
        ("mteb/amazon_massive_intent", "en"): ["train", "validation", "test"],
        ("SetFit/sst5", "default"): ["train", "validation", "test"],
        ("fancyzhx/ag_news", "default"): ["train", "test"],
        ("google/civil_comments", "default"): ["validation", "test"],
        ("nvidia/HelpSteer", "default"): ["train", "validation"],
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    for (ds, cfg), splits in want.items():
        idx = json.load(
            urllib.request.urlopen(f"https://huggingface.co/api/datasets/{ds}/parquet", timeout=60)
        )
        for split in splits:
            for n, url in enumerate(idx[cfg][split]):
                out = CACHE / f"{ds.replace('/', '__')}__{cfg}__{split}__{n}.parquet"
                if out.exists():
                    continue
                req = urllib.request.Request(url, headers={"User-Agent": "openjev-heldout/0.1"})
                with urllib.request.urlopen(req, timeout=900) as r:
                    out.write_bytes(r.read())
                print(f"  fetched {out.name}")
    names_path = CACHE / "clinc_intent_names.json"
    if not names_path.exists():
        info = json.load(
            urllib.request.urlopen(
                "https://datasets-server.huggingface.co/info?dataset=clinc%2Fclinc_oos", timeout=60
            )
        )
        names_path.write_text(
            json.dumps(info["dataset_info"]["plus"]["features"]["intent"]["names"])
        )


def _clinc_names() -> list[str]:
    return json.loads((CACHE / "clinc_intent_names.json").read_text())


# --------------------------------------------------------------------------
# task builders
# --------------------------------------------------------------------------


def _check_labels(task: str, criteria: dict[str, str], source_labels: list[str]) -> None:
    missing = [l for l in source_labels if l not in criteria]
    extra = [k for k in criteria if k not in source_labels]
    if missing or extra:
        raise SystemExit(f"{task}: criteria mismatch. missing={missing} extra={extra}")


def build_banking77() -> tuple[Task, list[Example]]:
    train = _read("mteb/banking77", "default", "train")
    test = _read("mteb/banking77", "default", "test")
    labels = sorted(train.label_text.unique().tolist())
    _check_labels("banking77", C.BANKING77, labels)

    pinned: list[int] = []
    if B77_REFERENCE_METRICS.exists():
        digest = hashlib.sha256(
            (CACHE / "mteb__banking77__default__test__0.parquet").read_bytes()
        ).hexdigest()
        if digest == B77_PARQUET_SHA256:
            pinned = json.loads(B77_REFERENCE_METRICS.read_text())["sample_ids"]["main"]

    idx = stratified(test.label_text.tolist(), CAP, pinned=pinned)
    pin = set(pinned)
    examples = [
        Example(
            state=test.at[i, "text"],
            answers={"intent": test.at[i, "label_text"]},
            meta={"row": int(i), "jev_reference_300": i in pin},
        )
        for i in idx
    ]
    q = Choice(
        instructions=C.BANKING77_INSTRUCTIONS,
        criteria={l: C.BANKING77[l] for l in labels},
    )
    task = Task(
        name="banking77",
        questions={"intent": q},
        description=(
            "Banking77 customer-support intent routing, 77 options. The anchor task: "
            "Jev scored 0.820 accuracy / 0.806 macro-F1 / ECE 0.068 on 300 rows that are "
            "a strict subset of this sample."
        ),
        holdout_of=SOURCES["banking77"].holdout_of,
    )
    task.examples = examples
    return task, []


def build_clinc_oos() -> tuple[Task, list[Example]]:
    names = _clinc_names()
    in_scope = [n for n in names if n != "oos"]
    _check_labels("clinc_oos", C.CLINC, in_scope)

    def rows(split: str) -> pd.DataFrame:
        df = _read("clinc/clinc_oos", "plus", split)
        df["label"] = df["intent"].map(lambda i: names[i])
        return df

    def sample(df: pd.DataFrame, cap: int) -> list[Example]:
        # Keep the benchmark's own out-of-scope prevalence rather than giving
        # `oos` 1/151 of the sample: the escape option is the whole point of
        # this task and even stratification would leave it with ~4 rows.
        oos_share = float((df.label == "oos").mean())
        n_oos = round(cap * oos_share)
        ins = df[df.label != "oos"]
        oos = df[df.label == "oos"]
        i_in = stratified(ins.label.tolist(), cap - n_oos)
        i_oos = stratified(["oos"] * len(oos), n_oos, seed=SEED + 1)
        out = []
        for i in i_in:
            r = ins.iloc[i]
            out.append(Example(state=r.text, answers={"intent": r.label},
                               meta={"source_label": r.label, "out_of_scope": False}))
        for i in i_oos:
            r = oos.iloc[i]
            out.append(Example(state=r.text, answers={"intent": "none_of_the_above"},
                               meta={"source_label": "oos", "out_of_scope": True}))
        return out

    criteria = {l: C.CLINC[l] for l in in_scope}
    criteria["none_of_the_above"] = C.CLINC_NONE_OF_THE_ABOVE
    task = Task(
        name="clinc_oos",
        questions={"intent": Choice(instructions=C.CLINC_INSTRUCTIONS, criteria=criteria)},
        description=(
            "CLINC150 assistant intents with a genuine out-of-scope class, 150 in-scope "
            "options plus none_of_the_above. The only public set that tests the escape "
            "option directly."
        ),
        holdout_of=SOURCES["clinc_oos"].holdout_of,
    )
    task.examples = sample(rows("test"), CAP)
    return task, sample(rows("validation"), CAP)


def build_massive_intent() -> tuple[Task, list[Example]]:
    train = _read("mteb/amazon_massive_intent", "en", "train")
    labels = sorted(train.label_text.unique().tolist())
    _check_labels("massive_intent", C.MASSIVE, labels)

    def sample(split: str) -> list[Example]:
        df = _read("mteb/amazon_massive_intent", "en", split)
        df = df[df.lang == "en"].reset_index(drop=True)
        return [
            Example(state=df.at[i, "text"], answers={"intent": df.at[i, "label_text"]},
                    meta={"row": int(i), "lang": "en"})
            for i in stratified(df.label_text.tolist(), CAP)
        ]

    task = Task(
        name="massive_intent",
        questions={
            "intent": Choice(
                instructions=C.MASSIVE_INSTRUCTIONS,
                criteria={l: C.MASSIVE[l] for l in labels},
            )
        },
        description=(
            "Amazon MASSIVE smart-speaker intents, English, 60 options. High cardinality "
            "in a different domain from Banking77."
        ),
        holdout_of=SOURCES["massive_intent"].holdout_of,
    )
    task.examples = sample("test")
    return task, sample("validation")


def build_sst5() -> tuple[Task, list[Example]]:
    def sample(split: str) -> list[Example]:
        df = _read("SetFit/sst5", "default", split)
        return [
            Example(state=df.at[i, "text"], answers={"sentiment": int(df.at[i, "label"])},
                    meta={"row": int(i), "source_label_text": df.at[i, "label_text"]})
            for i in stratified(df.label.tolist(), CAP)
        ]

    task = Task(
        name="sst5",
        questions={"sentiment": Score(instructions=C.SST5_INSTRUCTIONS, criteria=list(C.SST5))},
        description=(
            "SST-5 fine-grained sentiment as a 5-level ordinal score. Tests whether "
            "ordinal questions transfer at all; Jev's score head was measured not to be "
            "genuinely ordinal."
        ),
        holdout_of=SOURCES["sst5"].holdout_of,
    )
    task.examples = sample("test")
    return task, sample("validation")


def build_ag_news() -> tuple[Task, list[Example]]:
    names = ["World", "Sports", "Business", "Sci/Tech"]
    _check_labels("ag_news", C.AG_NEWS, names)
    df = _read("fancyzhx/ag_news", "default", "test")
    task = Task(
        name="ag_news",
        questions={
            "topic": Choice(instructions=C.AG_NEWS_INSTRUCTIONS,
                            criteria={l: C.AG_NEWS[l] for l in names})
        },
        description=(
            "AG News topic classification, 4 options. A deliberately easy "
            "low-cardinality control that separates 'the model is broken' from 'the task "
            "is hard'."
        ),
        holdout_of=SOURCES["ag_news"].holdout_of,
    )
    task.examples = [
        Example(state=df.at[i, "text"], answers={"topic": names[int(df.at[i, "label"])]},
                meta={"row": int(i)})
        for i in stratified(df.label.tolist(), CAP)
    ]
    return task, []


def build_civil_comments() -> tuple[Task, list[Example]]:
    def sample(split: str, seed: int) -> list[Example]:
        df = _read("google/civil_comments", "default", split)
        df = df[df.text.str.strip().str.len() > 0].reset_index(drop=True)
        gold = (df.toxicity >= 0.5)
        idx = stratified(gold.tolist(), CAP, seed=seed)
        return [
            Example(
                state=df.at[i, "text"],
                answers={"toxic": bool(gold[i])},
                # the raw rater fraction is kept so the 0.5 cut can be revisited
                # and the balanced sample reweighted to natural prevalence
                meta={"row": int(i), "toxicity": round(float(df.at[i, "toxicity"]), 4)},
            )
            for i in idx
        ]

    task = Task(
        name="civil_comments_toxicity",
        questions={"toxic": Noul(instructions=C.CIVIL_INSTRUCTIONS, criteria=dict(C.CIVIL))},
        description=(
            "Civil Comments toxicity as a binary noul question. Real crowd labels, CC0, "
            "and the canonical shape of a production gate. Sampled 50/50, which is NOT "
            "the deployment prevalence of about 8%."
        ),
        holdout_of=SOURCES["civil_comments_toxicity"].holdout_of,
    )
    task.examples = sample("test", SEED)
    return task, sample("validation", SEED)


def build_helpsteer() -> tuple[Task, list[Example]]:
    df = _read("nvidia/HelpSteer", "default", "validation")
    idx = stratified(df.helpfulness.tolist(), CAP)
    task = Task(
        name="helpsteer_helpfulness",
        questions={
            "helpfulness": Score(instructions=C.HELPSTEER_INSTRUCTIONS, criteria=list(C.HELPSTEER))
        },
        description=(
            "HelpSteer response helpfulness on the annotators' 0-4 Likert scale. A "
            "non-sentiment ordinal rubric with real human labels, so `score` is not "
            "evaluated purely on sentiment. The state is a JSON object, which also "
            "exercises the structured-state path."
        ),
        holdout_of=SOURCES["helpsteer_helpfulness"].holdout_of,
    )
    task.examples = [
        Example(
            state={"prompt": df.at[i, "prompt"], "response": df.at[i, "response"]},
            answers={"helpfulness": int(df.at[i, "helpfulness"])},
            meta={
                "row": int(i),
                # the other four attributes are *not* gold for this question,
                # they are kept for error analysis only
                "other_attributes": {
                    k: int(df.at[i, k]) for k in ("correctness", "coherence", "complexity", "verbosity")
                },
            },
        )
        for i in idx
    ]
    return task, []


BUILDERS = {
    "banking77": build_banking77,
    "clinc_oos": build_clinc_oos,
    "massive_intent": build_massive_intent,
    "sst5": build_sst5,
    "ag_news": build_ag_news,
    "civil_comments_toxicity": build_civil_comments,
    "helpsteer_helpfulness": build_helpsteer,
}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _primitive(task: Task) -> tuple[str, int]:
    q = next(iter(task.questions.values()))
    return q.type, len(q.labels)


def _coverage(task: Task, examples: list[Example]) -> dict:
    """How many of the menu's options actually occur as gold, and how thinly.

    Worth recording per task rather than assuming: MASSIVE's `cooking_query`
    has zero rows in the English test split, so its menu offers 60 options but
    only 59 can ever be right, and macro-F1 over the full menu is not
    comparable to macro-F1 over the labels present.
    """
    qid, q = next(iter(task.questions.items()))
    counts = Counter(e.answers[qid] for e in examples)
    menu = q.labels if not isinstance(q, Noul) else ["false", "true"]
    present = [str(l) for l in menu if counts.get(l if not isinstance(q, Score) else int(l), 0)
               or counts.get(l, 0)]
    per_class = [counts.get(l, 0) if not isinstance(q, Score) else counts.get(int(l), 0)
                 for l in menu]
    if isinstance(q, Noul):
        per_class = [counts.get(False, 0), counts.get(True, 0)]
        present = [l for l, c in zip(["false", "true"], per_class) if c]
    return {
        "options_in_menu": len(menu),
        "options_present_as_gold": len(present),
        "options_never_gold": [str(l) for l, c in zip(menu, per_class) if c == 0],
        "min_per_option": min(per_class),
        "max_per_option": max(per_class),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_train() -> None:
    """Per-label train dumps, the only thing consulted when writing criteria."""
    out = ROOT / "tasks" / "heldout" / "_train_dumps"
    out.mkdir(parents=True, exist_ok=True)
    names = _clinc_names()
    specs = [
        ("banking77", _read("mteb/banking77", "default", "train"), "text", "label_text", None),
        ("clinc_oos", _read("clinc/clinc_oos", "plus", "train"), "text", "intent", names),
        ("massive_intent", _read("mteb/amazon_massive_intent", "en", "train"), "text", "label_text", None),
        ("sst5", _read("SetFit/sst5", "default", "train"), "text", "label_text", None),
        ("ag_news", _read("fancyzhx/ag_news", "default", "train"), "text", "label", ["World", "Sports", "Business", "Sci/Tech"]),
    ]
    for name, df, textcol, labcol, mapping in specs:
        lab = df[labcol].map(lambda i: mapping[i]) if mapping else df[labcol].astype(str)
        rng = random.Random(SEED)
        lines = []
        for value in sorted(lab.unique()):
            sub = df[lab == value][textcol].tolist()
            lines.append(f"{value}  (n={len(sub)})")
            for s in rng.sample(sub, min(4, len(sub))):
                lines.append("    | " + " ".join(str(s).split())[:110])
        (out / f"{name}.txt").write_text("\n".join(lines) + "\n")
        print(f"  wrote _train_dumps/{name}.txt")


README_BEGIN = "<!-- BEGIN GENERATED holdout_of -->"
README_END = "<!-- END GENERATED holdout_of -->"


def render_readme_appendix(manifest_tasks: dict, union: list[str]) -> None:
    """Rewrite the README's holdout_of appendix from the manifest.

    Generated rather than hand-written so the prose and the thing the guard
    actually enforces cannot drift apart.
    """
    readme = OUT / "README.md"
    if not readme.exists():
        return
    lines = [
        README_BEGIN,
        "",
        "## Appendix: the full `holdout_of` lists",
        "",
        f"Generated from `manifest.json` by `scripts/build_heldout.py`. "
        f"{len(union)} distinct names across {len(manifest_tasks)} tasks.",
    ]
    for name, meta in manifest_tasks.items():
        leaks = meta.get("tasksource_leak_paths") or []
        lines += [
            "",
            f"### `{name}` — {len(meta['holdout_of'])} names",
            "",
            (
                "In `tasksource` and reaching this test set: "
                + ", ".join(f"`{t}`" for t in leaks)
                if leaks
                else "No `tasksource` task is known to reach this test set."
            ),
            "",
            "```",
        ]
        lines += [f"{n}" for n in meta["holdout_of"]]
        lines.append("```")
    lines += [
        "",
        "### Union",
        "",
        "This is what `assert_training_mixture_clean()` checks against.",
        "",
        "```",
        *union,
        "```",
        "",
        README_END,
    ]
    text = readme.read_text()
    start, end = text.find(README_BEGIN), text.find(README_END)
    if start == -1 or end == -1:
        raise SystemExit("README.md is missing the generated-section markers")
    readme.write_text(text[:start] + "\n".join(lines) + text[end + len(README_END):])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download source parquet first")
    ap.add_argument("--dump-train", action="store_true", help="write per-label train dumps")
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = ap.parse_args()

    if args.fetch:
        fetch_all()
    if args.dump_train:
        dump_train()
        return

    OUT.mkdir(parents=True, exist_ok=True)
    rows, manifest_tasks, problems = [], {}, []

    for name, build in BUILDERS.items():
        src = SOURCES[name]
        task, dev = build()
        assert task.name == name, f"{name}: builder produced {task.name!r}"

        probs = task.validate()
        # a rename that is not the one sanctioned rename is a bug
        for qid, q in task.questions.items():
            if isinstance(q, Choice):
                renamed = [l for l in q.criteria if l == "none_of_the_above"]
                if renamed and name != "clinc_oos":
                    probs.append(f"{qid}: unsanctioned rename to none_of_the_above")
        if dev:
            shadow = Task(name=task.name, questions=task.questions, examples=dev)
            probs += [f"dev: {p}" for p in shadow.validate()]
        if probs:
            problems.append((name, probs))

        prim, k = _primitive(task)
        n_test = len(task.examples)
        if not args.check:
            task.save(OUT, "test")
            if dev:
                keep = task.examples
                task.examples = dev
                task.save(OUT, "dev")
                task.examples = keep
            d = OUT / name
            src_json = {
                "task": name,
                "hf_dataset": src.hf_id,
                "config": src.config,
                "homepage": src.homepage,
                "license": src.license,
                "redistributable": src.redistributable,
                "license_note": src.license_note,
                "tasksource_leak_paths": src.tasksource_leak_paths,
                "description_source": src.description_source,
                "primitive": prim,
                "n_options": k,
                "n_test": n_test,
                "n_dev": len(dev),
                "seed": SEED,
                "cap": CAP,
                "test_label_coverage": _coverage(task, task.examples),
                "related_not_excluded": src.related_not_excluded,
                "sha256": {
                    f: _sha256(d / f) for f in ("test.jsonl", "dev.jsonl") if (d / f).exists()
                },
            }
            (d / "source.json").write_text(json.dumps(src_json, indent=1) + "\n")

        manifest_tasks[name] = {
            "primitive": prim,
            "n_options": k,
            "n_test": n_test,
            "n_dev": len(dev),
            "test_label_coverage": _coverage(task, task.examples),
            "hf_dataset": src.hf_id,
            "config": src.config,
            "license": src.license,
            "redistributable": src.redistributable,
            "tasksource_leak_paths": src.tasksource_leak_paths,
            "rows_vendored": src.redistributable,
            "holdout_of": src.holdout_of,
        }
        rows.append((name, prim, k, n_test, len(dev), src.hf_id,
                     "yes" if src.redistributable else "LOADER"))

    union = sorted({n for s in SOURCES.values() for n in s.holdout_of})
    if not args.check:
        (OUT / "manifest.json").write_text(
            json.dumps(
                {
                    "suite": "openjev-heldout-v1",
                    "seed": SEED,
                    "cap_per_task": CAP,
                    "n_tasks": len(manifest_tasks),
                    "tasks": manifest_tasks,
                    "holdout_of_union": union,
                    "note": (
                        "Any training mixture used to report numbers on this suite must "
                        "contain none of holdout_of_union. Enforced by "
                        "openjev.heldout.assert_training_mixture_clean()."
                    ),
                },
                indent=1,
            )
            + "\n"
        )
        render_readme_appendix(manifest_tasks, union)

    w = [max(len(str(r[i])) for r in rows + [("task", "prim", "K", "n", "dev", "source", "rows")])
         for i in range(7)]
    head = ("task", "primitive", "K", "n_test", "n_dev", "source", "rows")
    w = [max(w[i], len(head[i])) for i in range(7)]
    line = "  ".join("-" * x for x in w)
    print("\n" + "  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
    print(line)
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    print(line)
    print(f"{len(rows)} tasks, {sum(r[3] for r in rows)} test rows, "
          f"{sum(r[4] for r in rows)} dev rows, {len(union)} names in holdout_of union")

    if problems:
        print("\nVALIDATION PROBLEMS")
        for name, probs in problems:
            for p in probs[:10]:
                print(f"  {name}: {p}")
        raise SystemExit(1)
    print("Task.validate(): clean on all tasks")


if __name__ == "__main__":
    main()
