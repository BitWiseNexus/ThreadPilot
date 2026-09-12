"""Build the committed, deterministic subsample for the chosen brand.

This is the artifact everything downstream reads, and the one committed to git so
the pipeline runs without a Kaggle account. It must be byte-reproducible from the
same seed, so every ordering step here is explicitly seeded and no step depends
on dict or filesystem iteration order.

Three things it produces that later phases depend on:

* **thread_id** - the dataset ships flat rows with parent pointers and no
  conversation id, so root ids are computed by walking `in_response_to_tweet_id`
  to the root. This is what the Phase 4 leakage guard keys on: golden-set threads
  must be excluded from the retrieval index, and without a real thread id the
  guard would silently pass while leaking.

* **duplicate grouping, not duplicate deletion.** "where is my order" recurs
  thousands of times. Dropping those outright would distort the true message
  distribution, which the golden set is supposed to represent; keeping them all
  would waste retrieval slots and make top-k redundant. So duplicates are
  *grouped* and one canonical member flagged: the index uses canonical rows only,
  while sampling can still see real frequency via `dup_count`.

* **reply classification** from threadpilot.data.clean, so index admission is
  decided once, here, rather than re-derived inconsistently later.

Usage:
    python scripts/build_subsample.py
    python scripts/build_subsample.py --n-pairs 8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import config  # noqa: E402
from threadpilot.data.clean import (  # noqa: E402
    ReplyKind, classify_reply, has_mojibake, normalize,
)

STRUCTURE_PARQUET = config.INTERIM_DIR / "structure.parquet"
OUT_PARQUET = config.SUBSAMPLE_PARQUET
OUT_META = config.PROCESSED_DIR / "subsample_meta.json"

DEDUPE_KEY_RE = re.compile(r"[^a-z0-9 ]+")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dedupe_key(text: str) -> str:
    """Aggressive normalisation for grouping near-identical customer messages."""
    t = DEDUPE_KEY_RE.sub(" ", str(text).lower())
    t = re.sub(r"\s+", " ", t).strip()
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]


def build_thread_ids(parent_of: dict[int, object], ids: list[int]) -> dict[int, int]:
    """Root tweet id per tweet, by walking parent pointers with memoisation.

    Iterative rather than recursive: chains are short but the dataset is large
    enough that recursion depth is not worth risking. Cycles should not exist in
    a reply graph, but a visited-set guard is cheap and a cycle here would
    otherwise hang the whole build.
    """
    root: dict[int, int] = {}
    for tid in ids:
        path, cur = [], tid
        seen = set()
        while True:
            if cur in root:
                r = root[cur]
                break
            p = parent_of.get(cur)
            if p is None or pd.isna(p) or int(p) not in parent_of:
                r = cur
                break
            if cur in seen:      # defensive: malformed cycle
                r = cur
                break
            seen.add(cur)
            path.append(cur)
            cur = int(p)
        for node in path:
            root[node] = r
        root[tid] = r
    return root


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", default=config.BRAND)
    ap.add_argument("--n-pairs", type=int, default=config.SUBSAMPLE_PAIRS)
    args = ap.parse_args()

    if not STRUCTURE_PARQUET.exists():
        sys.exit("missing data/interim/structure.parquet. "
                 "Run scripts/eda_brand_selection.py first (it caches it).")

    rng = np.random.default_rng(config.SEED)
    t0 = time.perf_counter()
    df = pd.read_parquet(STRUCTURE_PARQUET)
    log(f"structure: {len(df):,} rows")

    parent_of = dict(zip(df["tweet_id"], df["in_response_to_tweet_id"]))
    inbound_ids = set(df.loc[df["inbound"] == True, "tweet_id"])  # noqa: E712

    brand_rows = df[(df["inbound"] == False) &  # noqa: E712
                    (df["author_id"].astype("string") == args.brand)]
    pid = brand_rows["in_response_to_tweet_id"]
    ok = pid.notna()
    pairs = [(int(p), int(r)) for p, r in
             zip(pid[ok].astype("int64"), brand_rows["tweet_id"][ok])
             if int(p) in inbound_ids]
    log(f"{args.brand}: {len(brand_rows):,} replies -> {len(pairs):,} usable pairs")
    if not pairs:
        sys.exit(f"no pairs for brand {args.brand!r}")

    # Oversample before filtering: classification and dedupe both drop rows, and
    # we want to land at n_pairs *after* those, not before.
    order = rng.permutation(len(pairs))
    take = min(len(pairs), args.n_pairs * 4)
    cand = [pairs[i] for i in order[:take]]
    log(f"candidate pool: {len(cand):,} (oversampled 4x to survive filtering)")

    needed = {i for pr in cand for i in pr}
    log(f"reading text for {len(needed):,} tweet_ids")
    texts: dict[int, str] = {}
    reader = pd.read_csv(config.RAW_CSV, usecols=["tweet_id", "text", "created_at"],
                         dtype={"tweet_id": "int64", "text": "string",
                                "created_at": "string"},
                         chunksize=500_000, low_memory=False)
    created: dict[int, str] = {}
    for ch in reader:
        hit = ch[ch["tweet_id"].isin(needed)]
        texts.update(dict(zip(hit["tweet_id"], hit["text"])))
        created.update(dict(zip(hit["tweet_id"], hit["created_at"])))
    log(f"got {len(texts):,} texts")

    thread_of = build_thread_ids(parent_of, [p for p, _ in cand])

    recs = []
    for parent_id, reply_id in cand:
        ct, rt = texts.get(parent_id), texts.get(reply_id)
        if not isinstance(ct, str) or not isinstance(rt, str):
            continue
        cm, br = normalize(ct), normalize(rt)
        if not cm or not br:
            continue
        f = classify_reply(rt)
        recs.append({
            "pair_id": f"{parent_id}-{reply_id}",
            "thread_id": int(thread_of.get(parent_id, parent_id)),
            "parent_id": parent_id, "reply_id": reply_id,
            "customer_msg": cm, "brand_reply": br,
            "raw_customer": ct, "raw_reply": rt,
            "created_at": created.get(parent_id, ""),
            "reply_kind": f.kind.value,
            "usable_precedent": bool(f.usable),
            "implies_escalation": bool(f.kind.implies_escalation),
            "has_url": f.has_url,
            "has_info_request": f.has_info_request,
            "n_chars": f.n_chars,
            "mojibake": has_mojibake(ct) or has_mojibake(rt),
            "dedupe_key": dedupe_key(cm),
        })
    sub = pd.DataFrame(recs)
    log(f"built {len(sub):,} records")

    # Duplicate GROUPING, not deletion - see module docstring.
    sub["dup_count"] = sub.groupby("dedupe_key")["pair_id"].transform("size")
    # Canonical = first within group under the seeded shuffle, so the choice is
    # deterministic but not biased toward the earliest tweet_id.
    sub = sub.iloc[rng.permutation(len(sub))].reset_index(drop=True)
    sub["is_canonical"] = ~sub.duplicated("dedupe_key", keep="first")
    log(f"duplicate groups: {sub['dedupe_key'].nunique():,} "
        f"({int((~sub['is_canonical']).sum()):,} non-canonical rows)")

    # Trim to target, preferring canonical rows so the retrieval index stays as
    # large as possible, but keeping some duplicates so dup_count stays meaningful.
    if len(sub) > args.n_pairs:
        canon = sub[sub["is_canonical"]]
        rest = sub[~sub["is_canonical"]]
        n_rest = min(len(rest), max(0, args.n_pairs - len(canon)))
        keep = pd.concat([canon.head(args.n_pairs), rest.head(n_rest)])
        sub = keep.iloc[rng.permutation(len(keep))].reset_index(drop=True)

    # Stable, seed-independent ordering on disk so the parquet is byte-comparable
    # across runs regardless of how the shuffles fell.
    sub = sub.sort_values("pair_id").reset_index(drop=True)
    sub.to_parquet(OUT_PARQUET, index=False)

    kinds = sub["reply_kind"].value_counts().to_dict()
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "brand": args.brand, "seed": config.SEED,
        "n_pairs": int(len(sub)),
        "n_threads": int(sub["thread_id"].nunique()),
        "n_usable_precedent": int(sub["usable_precedent"].sum()),
        "n_canonical": int(sub["is_canonical"].sum()),
        "n_implies_escalation": int(sub["implies_escalation"].sum()),
        "mojibake_rows": int(sub["mojibake"].sum()),
        "reply_kind_counts": {k: int(v) for k, v in kinds.items()},
        "total_brand_pairs_available": len(pairs),
        "note": "Committed subsample. usable_precedent rows with is_canonical "
                "form the retrieval index, minus golden-set thread_ids (Phase 4 "
                "leakage guard).",
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n{'='*66}")
    print(f"brand           {args.brand}")
    print(f"pairs           {len(sub):,}")
    print(f"threads         {sub['thread_id'].nunique():,}")
    print(f"canonical       {int(sub['is_canonical'].sum()):,}")
    print(f"usable          {int(sub['usable_precedent'].sum()):,} "
          f"({sub['usable_precedent'].mean():.1%})")
    print(f"index candidates{int((sub['usable_precedent'] & sub['is_canonical']).sum()):,}"
          "  (usable AND canonical)")
    print(f"escalation-signal rows {int(sub['implies_escalation'].sum()):,}")
    print(f"\nreply kinds:")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {k:<22} {v:>6,}  {v/len(sub):>6.1%}")
    size_mb = OUT_PARQUET.stat().st_size / 1024 / 1024
    print(f"\nwrote {OUT_PARQUET.name} ({size_mb:.1f}MB) and {OUT_META.name}"
          f"  ({time.perf_counter()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
