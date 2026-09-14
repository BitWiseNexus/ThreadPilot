"""Phase 3 step 1: sample golden-set and dev-silver candidates.

Sampling happens BEFORE any labelling and before the pipeline exists, so the
pipeline cannot be built to the test (decision_log.md D14).

## The chicken-and-egg problem, and how it is handled

The plan says "stratified across intents". But intent labels are what the golden
set is being built to produce - stratifying on them would require already having
them. Stratifying on the Phase 2 **cluster assignment** instead is the honest
substitute: clusters are what the intents were derived from, so they are a
data-derived proxy that is available now. The consequence, which is reported
rather than smoothed over: the realised per-INTENT counts will not exactly match
the per-CLUSTER targets, because the human naming step merged clusters 0 and 1
and because `account_security` has no cluster of its own.

## Three pools, deliberately not proportional

1. **Cluster strata** - equal allocation across the 10 clusters, not
   proportional. Proportional sampling would give the rarest classes too few
   examples to measure, and per-class F1 on 5 examples is noise.
2. **account_security** - targeted keyword pool. At ~2.5% of traffic a
   proportional sample yields ~5 examples, which cannot support a per-class
   metric, yet it is the highest-stakes intent in the taxonomy.
3. **Hard cases** - deliberately sought, not avoided: very short messages,
   messages carrying an order ID (require private data), multi-intent signals,
   possible sarcasm, and messages with no near neighbour in the corpus. A golden
   set drawn only from cluster centres would measure the easy middle and report
   it as overall performance.

All three make the golden set **deliberately NOT distribution-matched**. That
biases any headline accuracy figure and is why `eval/results/golden_strata.json`
records the realised composition next to the true traffic shares, for the
report's "what is misleading" section.

dev_silver is sampled separately and kept THREAD-DISJOINT from golden, because
it is used to tune thresholds that the golden set then reports on (D5). It is
sampled closer to the true distribution, since a tuning set should resemble
production traffic.

Usage:
    python scripts/sample_golden.py
    python scripts/sample_golden.py --n-golden 200 --n-silver 100
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import config, embeddings  # noqa: E402

OUT_GOLDEN = config.GOLDEN_DIR / "golden_candidates.jsonl"
OUT_SILVER = config.GOLDEN_DIR / "dev_silver_candidates.jsonl"
OUT_STRATA = config.RESULTS_DIR / "golden_strata.json"

ORDER_ID_RE = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")
ACCOUNT_RE = re.compile(
    r"\b(password|log ?in|login|sign ?in|locked out|hack(?:ed)?|"
    r"account (?:locked|suspended|closed|access|hacked|compromised)|"
    r"unauthoris|unauthoriz|fraud|someone (?:else|has) (?:used|accessed|hacked))\b",
    re.IGNORECASE)
# Praise words in a support complaint are usually sarcastic. Imperfect on
# purpose: this only needs to SURFACE candidates for review, not label them.
SARCASM_RE = re.compile(
    r"\b(thanks a (?:lot|bunch)|great job|well done|brilliant|fantastic|"
    r"love how|cheers for|nice one|top marks|10/10|impressive)\b", re.IGNORECASE)

# Coarse keyword signals per intent, used ONLY to detect multi-intent messages.
# Never used to assign a label - that is the labeller's job.
INTENT_SIGNALS = {
    "delivery": r"\b(deliver|delivery|arriv|late|shipping|dispatch|track)\b",
    "refund": r"\b(refund|charge|money|cashback|billed|payment|price)\b",
    "damage": r"\b(damag|broken|crush|packag|box)\b",
    "digital": r"\b(app|alexa|echo|kindle|fire tv|prime video|music|stream)\b",
    "prime": r"\bprime\b",
    "account": r"\b(account|login|password|hack)\b",
    "complaint": r"\b(customer service|awful|terrible|worst|pathetic|useless)\b",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_hard_case_flags(df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """Flag the message types a lazy golden set would under-represent."""
    df = df.copy()
    df["hc_very_short"] = df["customer_msg"].str.len() < 60
    df["hc_order_id"] = df["customer_msg"].str.contains(ORDER_ID_RE, na=False)
    df["hc_account"] = df["customer_msg"].str.contains(ACCOUNT_RE, na=False)
    df["hc_sarcasm"] = df["customer_msg"].str.contains(SARCASM_RE, na=False)

    sig = pd.DataFrame({
        k: df["customer_msg"].str.contains(v, case=False, regex=True, na=False)
        for k, v in INTENT_SIGNALS.items()})
    df["n_intent_signals"] = sig.sum(axis=1)
    df["hc_multi_intent"] = df["n_intent_signals"] >= 3

    # Novelty: nearest neighbour OTHER than itself. A message with no close
    # neighbour is one retrieval will struggle to ground, which is exactly the
    # case the escalation gate G2 exists for - so the golden set must contain
    # some. Computed in blocks to keep the 8000x8000 similarity matrix off-heap.
    log("computing nearest-neighbour similarity for novelty flag")
    best = np.empty(len(X), dtype=np.float32)
    block = 512
    for i in range(0, len(X), block):
        sims = X[i:i + block] @ X.T
        for r in range(sims.shape[0]):
            sims[r, i + r] = -1.0          # exclude self-match
        best[i:i + block] = sims.max(axis=1)
    df["nn_similarity"] = best
    df["hc_novel"] = df["nn_similarity"] < np.quantile(best, 0.05)
    return df


def take(pool: pd.DataFrame, n: int, used: set[str], rng: np.random.Generator
         ) -> pd.DataFrame:
    """Seeded sample of n unused rows from pool."""
    avail = pool[~pool["pair_id"].isin(used)]
    if avail.empty or n <= 0:
        return avail.head(0)
    n = min(n, len(avail))
    idx = rng.permutation(len(avail))[:n]
    picked = avail.iloc[idx]
    used.update(picked["pair_id"])
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-golden", type=int, default=200)
    ap.add_argument("--n-silver", type=int, default=100)
    args = ap.parse_args()

    if not config.SUBSAMPLE_PARQUET.exists():
        sys.exit("missing subsample. Run: python tasks.py subsample")

    rng = np.random.default_rng(config.SEED)
    df = pd.read_parquet(config.SUBSAMPLE_PARQUET)
    log(f"subsample: {len(df):,} pairs, {df['thread_id'].nunique():,} threads")

    X = embeddings.embed(df["customer_msg"].tolist(), normalize=True,
                         show_progress=False)

    # Cluster assignment, recomputed with the Phase 2 settings so the strata
    # match the taxonomy derivation exactly.
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=10, random_state=config.SEED, n_init=10)
    df["cluster"] = km.fit_predict(X)
    log(f"cluster sizes: {sorted(df['cluster'].value_counts().to_dict().items())}")

    df = add_hard_case_flags(df, X)
    hc_cols = [c for c in df.columns if c.startswith("hc_")]
    log("hard-case pool sizes: " +
        ", ".join(f"{c[3:]}={int(df[c].sum())}" for c in hc_cols))

    used: set[str] = set()
    picked: list[pd.DataFrame] = []
    strata_log: dict[str, int] = {}

    # --- pool 1: equal allocation across clusters --------------------------
    n_clusters = df["cluster"].nunique()
    per_cluster = int(round(args.n_golden * 0.65 / n_clusters))
    for c in sorted(df["cluster"].unique()):
        got = take(df[df["cluster"] == c], per_cluster, used, rng)
        got = got.assign(stratum=f"cluster_{c}")
        picked.append(got)
        strata_log[f"cluster_{c}"] = len(got)

    # --- pool 2: account_security, deliberately oversampled ----------------
    got = take(df[df["hc_account"]], 20, used, rng).assign(stratum="account_security")
    picked.append(got)
    strata_log["account_security"] = len(got)

    # --- pool 3: hard cases ------------------------------------------------
    remaining = args.n_golden - sum(len(p) for p in picked)
    hard_kinds = ["hc_order_id", "hc_very_short", "hc_multi_intent",
                  "hc_novel", "hc_sarcasm"]
    per_hard = max(1, remaining // len(hard_kinds))
    for col in hard_kinds:
        got = take(df[df[col]], per_hard, used, rng).assign(stratum=col[3:])
        picked.append(got)
        strata_log[col[3:]] = len(got)

    # --- top up to exactly n_golden from the general pool ------------------
    golden = pd.concat(picked, ignore_index=True)
    short = args.n_golden - len(golden)
    if short > 0:
        top = take(df, short, used, rng).assign(stratum="general")
        golden = pd.concat([golden, top], ignore_index=True)
        strata_log["general"] = len(top)
    golden = golden.head(args.n_golden)

    # --- dev_silver: thread-disjoint, closer to the true distribution ------
    golden_threads = set(golden["thread_id"])
    silver_pool = df[~df["thread_id"].isin(golden_threads) &
                     ~df["pair_id"].isin(used)]
    sidx = rng.permutation(len(silver_pool))[:args.n_silver]
    silver = silver_pool.iloc[sidx].assign(stratum="silver_random")

    # --- disjointness is the whole point; assert it rather than assume it --
    assert not (set(golden["thread_id"]) & set(silver["thread_id"])), \
        "golden and dev_silver share threads - threshold tuning would leak"
    assert golden["pair_id"].is_unique
    assert len(golden) == args.n_golden, f"got {len(golden)} golden rows"

    keep = ["pair_id", "thread_id", "parent_id", "reply_id", "customer_msg",
            "brand_reply", "reply_kind", "usable_precedent", "implies_escalation",
            "has_url", "has_info_request", "n_chars", "dup_count", "cluster",
            "stratum", "nn_similarity", "n_intent_signals"] + hc_cols

    for frame, path in ((golden, OUT_GOLDEN), (silver, OUT_SILVER)):
        out = frame[keep].copy()
        out["nn_similarity"] = out["nn_similarity"].round(4)
        # Empty label fields, to be filled by labelling then human review.
        for col in ("intent", "auto_or_escalate", "escalation_reason_notes",
                    "ideal_reply_notes", "label_confidence", "ambiguity_flag"):
            out[col] = None
        out.sort_values("pair_id").to_json(path, orient="records", lines=True,
                                           force_ascii=False)
        log(f"wrote {path.name}: {len(out)} rows")

    true_shares = (df["cluster"].value_counts(normalize=True).sort_index()
                   .round(4).to_dict())
    got_shares = (golden["cluster"].value_counts(normalize=True).sort_index()
                  .round(4).to_dict())
    OUT_STRATA.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": config.SEED, "brand": config.BRAND,
        "n_golden": len(golden), "n_silver": len(silver),
        "strata": strata_log,
        "true_cluster_shares": {str(k): v for k, v in true_shares.items()},
        "golden_cluster_shares": {str(k): v for k, v in got_shares.items()},
        "hard_case_counts_in_golden": {
            c[3:]: int(golden[c].sum()) for c in hc_cols},
        "thread_disjoint_from_silver": True,
        "warning": "The golden set is DELIBERATELY not distribution-matched: "
                   "clusters are equally allocated, account_security is "
                   "oversampled ~8x, and hard cases are deliberately sought. "
                   "Headline accuracy on this set is therefore NOT an estimate "
                   "of production accuracy. See report 'what is misleading'.",
    }, indent=2), encoding="utf-8")

    print(f"\n{'='*68}")
    print(f"golden {len(golden)}  |  silver {len(silver)}  |  thread-disjoint OK")
    print(f"\nstrata: {json.dumps(strata_log)}")
    print(f"\nhard cases present in golden:")
    for c in hc_cols:
        print(f"  {c[3:]:<14} {int(golden[c].sum()):>3}")
    print(f"\ncluster share  true -> golden (equal allocation is intentional):")
    for c in sorted(true_shares):
        print(f"  c{c}: {true_shares[c]:.1%} -> {got_shares.get(c, 0):.1%}")
    print(f"\nwrote {OUT_STRATA.name}")
    print("\nNEXT: python scripts/label_golden.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
