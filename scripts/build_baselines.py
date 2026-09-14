"""Phase 5: fit the baselines and record what they are made of.

Zero LLM calls - deliberately, so baselines cost no token budget and can be
refit freely.

Training data for the TF-IDF baselines needs care, because the obvious source
(the golden set) is the test set. Two regimes are fitted and both reported:

* **silver** - the 100 dev_silver rows. The only genuinely labelled non-test
  data available, and very data-poor at ~9 rows per class.
* **cluster** - ~7,700 subsample rows pseudo-labelled by their Phase 2 cluster,
  with golden and silver threads excluded. Free and far larger, but the labels
  are cluster assignments rather than reviewed intents, so they are noisier.

Reporting both is the point. If the LLM pipeline only beats the data-poor
version, its advantage is largely "does not need labelled data" rather than
"reasons better", and the report should say so.

Usage:  python scripts/build_baselines.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import baselines, config, retrieval, taxonomy  # noqa: E402

OUT_DIR = config.INTERIM_DIR
OUT_INFO = config.RESULTS_DIR / "baselines_info.json"


def cluster_intent_map() -> dict[int, str]:
    """cluster id -> intent name, from the taxonomy's recorded provenance."""
    return {i.cluster: i.name for i in taxonomy.INTENTS if i.cluster is not None}


def main() -> int:
    if not config.GOLDEN_JSONL.exists():
        sys.exit("no golden set; run scripts/finalize_golden.py")
    t0 = time.perf_counter()

    golden = [json.loads(l) for l in config.GOLDEN_JSONL.open(encoding="utf-8")]
    silver_path = config.GOLDEN_DIR / "dev_silver_labeled_raw.jsonl"
    if not silver_path.exists():
        sys.exit("no labelled dev_silver; run label_golden.py --which silver")
    silver = [json.loads(l) for l in silver_path.open(encoding="utf-8")]

    index = retrieval.RetrievalIndex.load()
    sub = pd.read_parquet(config.SUBSAMPLE_PARQUET)

    # Cluster assignment must match Phase 2 exactly, so recompute with the same
    # settings rather than trusting a stale column.
    from sklearn.cluster import KMeans
    from threadpilot import embeddings
    X = embeddings.embed(sub["customer_msg"].tolist(), normalize=True,
                         show_progress=False)
    sub["cluster"] = KMeans(n_clusters=10, random_state=config.SEED,
                            n_init=10).fit_predict(X)

    held_out = retrieval.held_out_thread_ids()
    c2i = cluster_intent_map()

    # ---- training pools --------------------------------------------------
    silver_texts = [r["customer_msg"] for r in silver
                    if isinstance(r.get("label_A"), dict)
                    and "error" not in r["label_A"]]
    silver_intents = [r["label_A"]["intent"] for r in silver
                      if isinstance(r.get("label_A"), dict)
                      and "error" not in r["label_A"]]

    train_c = sub[~sub["thread_id"].isin(held_out)].copy()
    train_c["intent"] = train_c["cluster"].map(c2i)
    train_c = train_c[train_c["intent"].notna()]
    cluster_texts = train_c["customer_msg"].tolist()
    cluster_intents = train_c["intent"].tolist()

    print(f"training pools:")
    print(f"  silver   {len(silver_texts):>6,} rows, "
          f"{len(set(silver_intents))} classes")
    print(f"  cluster  {len(cluster_texts):>6,} rows, "
          f"{len(set(cluster_intents))} classes (golden+silver threads excluded)")

    # ---- templates -------------------------------------------------------
    templates = baselines.derive_templates(index, {v: k for k, v in c2i.items()},
                                           sub)
    print(f"\ntemplates derived for {len(templates)} intents "
          f"(from real brand replies nearest each cluster centroid)")

    # ---- fit -------------------------------------------------------------
    trivial = baselines.TrivialBaseline.fit(cluster_intents)
    simple_silver = baselines.SimpleBaseline.fit(
        silver_texts, silver_intents, templates, name="simple_tfidf_silver")
    simple_cluster = baselines.SimpleBaseline.fit(
        cluster_texts, cluster_intents, templates, name="simple_tfidf_cluster")
    nn = baselines.Retrieval1NNBaseline.build(index,
                                              classifier=simple_cluster.pipeline)

    for obj, fname in ((trivial, "baseline_trivial.pkl"),
                       (simple_silver, "baseline_simple_silver.pkl"),
                       (simple_cluster, "baseline_simple_cluster.pkl")):
        with (OUT_DIR / fname).open("wb") as fh:
            pickle.dump(obj, fh)

    # ---- sanity: how do they behave on a couple of golden rows? ----------
    sample = [g["text"] for g in golden[:3]]
    print("\nsanity check on 3 golden messages:")
    for b in (trivial, simple_silver, simple_cluster, nn):
        res = b.triage(sample)
        print(f"  {b.name:<24} " + ", ".join(
            f"{r.classification.intent}/{r.decision.decision[:4]}" for r in res))

    info = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": config.SEED, "brand": config.BRAND,
        "llm_calls": 0,
        "baselines": {
            "trivial": {"majority_intent": trivial.majority_intent,
                        "reply": baselines.CANNED_REPLY,
                        "policy": "always escalate"},
            "simple_tfidf_silver": {
                "train_rows": len(silver_texts),
                "train_classes": len(set(silver_intents)),
                "source": "dev_silver, single-labeller labels"},
            "simple_tfidf_cluster": {
                "train_rows": len(cluster_texts),
                "train_classes": len(set(cluster_intents)),
                "source": "subsample rows pseudo-labelled by Phase 2 cluster, "
                          "golden+silver threads excluded"},
            "retrieval_1nn": {
                "index_size": len(index),
                "reply_source": "nearest precedent's brand reply, verbatim",
                "intent_source": "simple_tfidf_cluster classifier"},
        },
        "templates": templates,
        "cluster_intent_map": {str(k): v for k, v in c2i.items()},
        "note": "No baseline makes an LLM call. retrieval_1nn is the sharpest "
                "comparison: it isolates what generation adds over simply "
                "copying the nearest historical reply.",
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwrote 3 pickles + {OUT_INFO.name}  ({time.perf_counter()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
