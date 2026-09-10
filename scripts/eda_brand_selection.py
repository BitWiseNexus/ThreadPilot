"""Phase 1: choose a brand by measuring the criteria pre-registered in
docs/requirements.md section 2.1 -- C1 volume, C2 intent diversity,
C3 in-thread resolution rate, C4 thread depth, C5 escalation signal.

The criteria and the hypothesis (docs/requirements.md 2.2: AppleSupport fails C3
on channel deflection; an airline wins) were committed BEFORE this script ran.
That ordering is the point: brand choice should be a measurement, not a
rationalisation. Whatever this prints, the report records whether the
pre-registered hypothesis held or was falsified.

C3 is computed from the five-way reply classifier in threadpilot.data.clean --
see that module for why a single 'deflection' flag was the wrong abstraction.
The classifier is validated against an LLM rater in
scripts/validate_reply_proxy.py; C3 should not be trusted until that agreement
is reported.

Outputs:
    data/interim/structure.parquet         cached structural frame (no text)
    data/interim/pairs_<brand>.parquet     sampled pairs w/ text, for reuse
    eval/results/brand_selection.json      every number, machine-readable
    eval/results/brand_selection.md        the table that goes in the report

Usage:
    python scripts/eda_brand_selection.py --top 12 --sample 4000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import config  # noqa: E402
from threadpilot.data.clean import (  # noqa: E402
    ReplyKind, classify_reply, has_mojibake, normalize,
)

STRUCTURE_PARQUET = config.INTERIM_DIR / "structure.parquet"
OUT_JSON = config.RESULTS_DIR / "brand_selection.json"
OUT_MD = config.RESULTS_DIR / "brand_selection.md"

# C5: does the brand receive BOTH plainly-automatable and plainly-human traffic?
# If everything sits at one pole, the auto/escalate task is degenerate.
ESCALATE_POLE = re.compile(
    r"\brefund|\bcharged?\b|billing|invoice|fraud|hack(?:ed)?|stolen|scam|"
    r"password|locked out|account (?:locked|suspended|closed)|lawsuit|legal|"
    r"compensation|reimburse|cancel my|close my account|data breach|"
    r"unauthori[sz]ed", re.IGNORECASE)
AUTO_POLE = re.compile(
    r"\b(?:how (?:do|can|would) i|how to|what time|what are your|"
    r"when (?:will|does|is|do)|where (?:is|can)|is there|do you (?:have|offer|ship)|"
    r"what is your|any update|track(?:ing)? (?:my|number)|status of)\b",
    re.IGNORECASE)


def to_md(df: pd.DataFrame) -> str:
    """Minimal markdown table. pandas .to_markdown() needs `tabulate`, and
    adding a dependency to format one table works against the lean-core-install
    decision (D17)."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_structure(force: bool = False) -> pd.DataFrame:
    """Pass A: structure only (no text) -- cheap enough to hold all 2.81M rows."""
    if STRUCTURE_PARQUET.exists() and not force:
        log(f"reusing cached {STRUCTURE_PARQUET.name}")
        return pd.read_parquet(STRUCTURE_PARQUET)

    log("pass A: reading structure (no text column)")
    chunks = []
    reader = pd.read_csv(
        config.RAW_CSV,
        usecols=["tweet_id", "author_id", "inbound", "in_response_to_tweet_id"],
        dtype={"tweet_id": "int64", "author_id": "string", "inbound": "boolean"},
        chunksize=500_000, low_memory=False,
    )
    for i, ch in enumerate(reader, 1):
        # Nullable Int64 keeps a missing parent meaningful, rather than letting
        # float coercion turn it into 0.0 and invent a parent tweet.
        ch["in_response_to_tweet_id"] = pd.to_numeric(
            ch["in_response_to_tweet_id"], errors="coerce").astype("Int64")
        chunks.append(ch)
        log(f"  chunk {i}: {len(ch):,} rows")
    df = pd.concat(chunks, ignore_index=True)
    df["author_id"] = df["author_id"].astype("category")
    df.to_parquet(STRUCTURE_PARQUET, index=False)
    log(f"pass A done: {len(df):,} rows -> {STRUCTURE_PARQUET.name}")
    return df


def load_texts(needed: set[int]) -> dict[int, str]:
    """Pass B: fetch text only for the tweet_ids we actually need."""
    log(f"pass B: fetching text for {len(needed):,} tweet_ids")
    out: dict[int, str] = {}
    reader = pd.read_csv(
        config.RAW_CSV, usecols=["tweet_id", "text"],
        dtype={"tweet_id": "int64", "text": "string"},
        chunksize=500_000, low_memory=False,
    )
    for i, ch in enumerate(reader, 1):
        hit = ch[ch["tweet_id"].isin(needed)]
        out.update(dict(zip(hit["tweet_id"], hit["text"])))
    log(f"  got {len(out):,} texts")
    return out


def diversity_entropy(texts: list[str], k: int = 12, seed: int = config.SEED) -> float:
    """C2 proxy: normalised entropy of KMeans cluster sizes over TF-IDF+SVD.

    Deliberately TF-IDF rather than MiniLM here: it needs no torch, so brand
    selection stays runnable on a minimal install. Phase 2 derives the actual
    taxonomy with MiniLM embeddings; this is only a spread proxy for ranking
    brands, and a rank-order proxy suffices for that.

    1.0 = uniform spread across k clusters; 0.0 = everything in one cluster.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(texts) < k * 10:
        return float("nan")
    X = TfidfVectorizer(max_features=20_000, stop_words="english",
                        min_df=3, ngram_range=(1, 2)).fit_transform(texts)
    if X.shape[1] < 50:
        return float("nan")
    Z = TruncatedSVD(n_components=50, random_state=seed).fit_transform(X)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Z)
    counts = np.bincount(labels, minlength=k).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(k))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not config.RAW_CSV.exists():
        sys.exit(f"missing {config.RAW_CSV}. Run: python tasks.py data")

    rng = np.random.default_rng(config.SEED)
    t0 = time.perf_counter()
    df = load_structure(force=args.force)
    n_in = int((df["inbound"] == True).sum())   # noqa: E712
    n_out = int((df["inbound"] == False).sum())  # noqa: E712
    log(f"rows {len(df):,}  inbound {n_in:,}  outbound {n_out:,}")

    # ---- C1: volume ------------------------------------------------------
    out = df[df["inbound"] == False]  # noqa: E712
    vol = out["author_id"].value_counts()
    vol = vol[vol > 0]
    log(f"distinct brand accounts: {len(vol):,}")
    candidates = list(vol.head(args.top).index)

    # ---- pair construction: brand reply -> its parent customer tweet -----
    # The dataset ships flat rows with parent pointers; this reconstructs the
    # (customer message, brand reply) unit the whole project depends on.
    parent_of = dict(zip(df["tweet_id"], df["in_response_to_tweet_id"]))
    inbound_ids = set(df.loc[df["inbound"] == True, "tweet_id"])  # noqa: E712

    per_brand_pairs: dict[str, list[tuple[int, int]]] = {}
    for brand in candidates:
        sub = out[out["author_id"] == brand]
        pid = sub["in_response_to_tweet_id"]
        ok = pid.notna()
        per_brand_pairs[brand] = [
            (int(p), int(r)) for p, r in
            zip(pid[ok].astype("int64"), sub["tweet_id"][ok]) if int(p) in inbound_ids
        ]
    log("C1: " + ", ".join(f"{b}={len(per_brand_pairs[b]):,}" for b in candidates[:6]) + " ...")

    sampled, needed = {}, set()
    for brand, pairs in per_brand_pairs.items():
        idx = rng.permutation(len(pairs))[: args.sample] if pairs else []
        sampled[brand] = [pairs[i] for i in idx]
        for p, r in sampled[brand]:
            needed.add(p); needed.add(r)

    texts = load_texts(needed)

    # ---- per-brand metrics ----------------------------------------------
    rows, examples = [], {}
    for brand in candidates:
        s = sampled[brand]
        recs = []
        for p, r in s:
            ct, rt = texts.get(p), texts.get(r)
            if not isinstance(ct, str) or not isinstance(rt, str):
                continue
            f = classify_reply(rt)
            recs.append({
                "parent_id": p, "reply_id": r,
                "customer_msg": normalize(ct), "brand_reply": normalize(rt),
                "raw_reply": rt, "kind": f.kind.value, "usable": f.usable,
                "has_url": f.has_url, "n_chars": f.n_chars,
                "multiturn": pd.notna(parent_of.get(p, pd.NA)),
                "mojibake": has_mojibake(ct) or has_mojibake(rt),
            })
        if not recs:
            continue
        pdf = pd.DataFrame(recs)
        pdf.to_parquet(config.INTERIM_DIR / f"pairs_{brand}.parquet", index=False)
        n = len(pdf)
        comp = Counter(pdf["kind"])

        cust = pdf["customer_msg"].tolist()
        row = {
            "brand": brand,
            "c1_usable_pairs": len(per_brand_pairs[brand]),
            "n_scored": n,
            "c3_usable_rate": round(float(pdf["usable"].mean()), 4),
        }
        # Composition over EVERY ReplyKind, derived from the enum rather than
        # hardcoded: adding a class must not silently drop it from the report.
        for k in ReplyKind:
            row[k.value] = round(comp[k.value] / n, 4)
        rows.append({
            **row,
            "median_reply_chars": int(pdf["n_chars"].median()),
            "c4_multiturn_rate": round(float(pdf["multiturn"].mean()), 4),
            "c2_diversity_entropy": round(diversity_entropy(cust), 4),
            "c5_escalate_pole_rate": round(
                float(np.mean([bool(ESCALATE_POLE.search(t)) for t in cust])), 4),
            "c5_auto_pole_rate": round(
                float(np.mean([bool(AUTO_POLE.search(t)) for t in cust])), 4),
            "mojibake_rate": round(float(pdf["mojibake"].mean()), 5),
        })
        # Real examples per class, so the heuristics can be audited not trusted.
        examples[brand] = {
            k.value: pdf.loc[pdf["kind"] == k.value, "brand_reply"].head(3).tolist()
            for k in ReplyKind
        }
        log(f"  {brand:<18} n={n:<5} usable={rows[-1]['c3_usable_rate']:.2f} "
            f"(self={rows[-1][ReplyKind.SELF_CONTAINED.value]:.2f} "
            f"link={rows[-1][ReplyKind.LINK_REFERRAL.value]:.2f} "
            f"ask={rows[-1][ReplyKind.DIAGNOSTIC_ASK.value]:.2f} "
            f"handoff={rows[-1][ReplyKind.HANDOFF_WITH_ASK.value]:.2f}) "
            f"bare={rows[-1][ReplyKind.CHANNEL_SWITCH_BARE.value]:.2f} "
            f"div={rows[-1]['c2_diversity_entropy']:.3f}")

    res = pd.DataFrame(rows)

    # ---- composite score -------------------------------------------------
    # Weighting reflects the pre-registered claim that C3 is decisive: a brand
    # with no usable precedent cannot satisfy the grounding requirement at all,
    # whereas the other criteria only make the task easier or harder. Volume
    # enters as log, since past ~20k pairs more data adds little.
    def norm(col: pd.Series) -> pd.Series:
        lo, hi = col.min(), col.max()
        return pd.Series(0.5, index=col.index) if hi == lo else (col - lo) / (hi - lo)

    res["score"] = (
        0.45 * norm(res["c3_usable_rate"])
        + 0.20 * norm(res["c2_diversity_entropy"].fillna(0))
        + 0.15 * norm(np.log10(res["c1_usable_pairs"].clip(lower=1)))
        + 0.10 * norm(res["c4_multiturn_rate"])
        + 0.10 * norm(res[["c5_escalate_pole_rate", "c5_auto_pole_rate"]].min(axis=1))
    ).round(4)
    res = res.sort_values("score", ascending=False).reset_index(drop=True)

    OUT_JSON.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": config.SEED, "sample_per_brand": args.sample,
        "total_rows": int(len(df)), "n_inbound": n_in, "n_outbound": n_out,
        "criteria": "docs/requirements.md 2.1 (pre-registered)",
        "c3_method": "5-way classifier in threadpilot.data.clean; "
                     "usable = self_contained + link_referral",
        "score_weights": {"c3_usable": 0.45, "c2_diversity": 0.20,
                          "c1_volume_log": 0.15, "c4_multiturn": 0.10,
                          "c5_both_poles": 0.10},
        "table": res.to_dict(orient="records"), "examples": examples,
    }, indent=2), encoding="utf-8")

    cols = (["brand", "c1_usable_pairs", "c3_usable_rate"]
            + [k.value for k in ReplyKind]
            + ["c2_diversity_entropy", "c4_multiturn_rate",
               "c5_escalate_pole_rate", "c5_auto_pole_rate", "score"])
    OUT_MD.write_text(
        "# Phase 1 - brand selection\n\n"
        f"Criteria pre-registered in `docs/requirements.md` 2.1 *before* this ran. "
        f"Seed {config.SEED}; {args.sample} sampled pairs per brand; "
        f"{len(df):,} total rows ({n_in:,} inbound / {n_out:,} outbound).\n\n"
        f"`c3_usable_rate` = sum of the classes admitted to the retrieval "
        f"index: {', '.join('`'+k.value+'`' for k in ReplyKind if k.usable_as_precedent)}. "
        f"Excluded: {', '.join('`'+k.value+'`' for k in ReplyKind if not k.usable_as_precedent)}. "
        f"See `threadpilot.data.clean` for why a single deflection flag was the "
        f"wrong abstraction, and `c3_proxy_validation.md` for the measured "
        f"agreement of this classifier with an independent rater.\n\n"
        + to_md(res[cols]) + "\n", encoding="utf-8")

    print("\n" + res[cols].to_string(index=False))
    print(f"\nwrote {OUT_JSON.name}, {OUT_MD.name}  ({time.perf_counter()-t0:.1f}s)")
    print(f"\nTOP BY COMPOSITE SCORE: {res.iloc[0]['brand']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
