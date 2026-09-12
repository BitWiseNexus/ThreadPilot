"""Validate the C3 reply classifier against an independent LLM rater.

Why this exists: brand selection turns on `c3_usable_rate`, which is produced by
a regex classifier I wrote myself (threadpilot.data.clean). Choosing a brand on
an unvalidated heuristic of my own design would be exactly the sort of
unexamined headline number this project is supposed to avoid. So the heuristic
gets measured against a rater that has never seen it.

Method:
  * stratified sample across all five ReplyKind classes per brand, so rare
    classes are actually tested rather than swamped by the majority class;
  * the rater is asked the DOWNSTREAM question ("is this usable precedent?"),
    never "does this match my regex" - it is given no hint of the heuristic;
  * the judge model is the JUDGE family, not the generator family, for the same
    self-preference reason as D7;
  * report Cohen's kappa + raw agreement + the disagreement cases themselves.

A high kappa means C3 can be trusted. A low kappa is also a result: it would
mean the brand table is built on sand and must be rebuilt or re-weighted.

Usage:
    python scripts/validate_reply_proxy.py --brands AmazonHelp SpotifyCares British_Airways
    python scripts/validate_reply_proxy.py --per-class 12 --offline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import config, llm  # noqa: E402
from threadpilot.data.clean import ReplyKind  # noqa: E402

OUT_JSON = config.RESULTS_DIR / "c3_proxy_validation.json"
OUT_MD = config.RESULTS_DIR / "c3_proxy_validation.md"

# The rater is given the DEFINITION the project cares about, not the heuristic.
# Deliberately no mention of DMs, URLs, lengths or part markers: if the rater
# were told the rule it would be grading the rule against itself.
SYSTEM = """You are auditing a customer-support dataset.

For each brand reply you are shown the customer message it answered. Decide
whether the brand reply would be USEFUL AS PRECEDENT for training a support
assistant - that is, whether it shows how the brand actually resolves this kind
of issue.

USABLE means the reply does at least one of:
  - answers the question or explains the situation
  - gives a concrete next step, instruction, or specific resource
  - asks a specific diagnostic question that moves resolution forward

NOT USABLE means the reply does none of those, e.g.:
  - it only moves the conversation to another channel with no answer
  - it is only sympathy, thanks, or pleasantry
  - it is an incomplete sentence fragment that cannot stand alone
  - it is marketing or chit-chat rather than support

Reply with ONLY compact JSON: {"usable": true|false, "why": "<12 words max>"}"""


def build_sample(brands: list[str], per_class: int, seed: int) -> pd.DataFrame:
    """Stratified across ReplyKind so rare classes are genuinely tested."""
    rng = np.random.default_rng(seed)
    frames = []
    for brand in brands:
        f = config.INTERIM_DIR / f"pairs_{brand}.parquet"
        if not f.exists():
            sys.exit(f"missing {f}. Run scripts/eda_brand_selection.py first.")
        df = pd.read_parquet(f)
        for kind in ReplyKind:
            pool = df[df["kind"] == kind.value]
            if pool.empty:
                continue
            take = min(per_class, len(pool))
            idx = rng.permutation(len(pool))[:take]
            s = pool.iloc[idx].copy()
            s["brand"] = brand
            frames.append(s)
    out = pd.concat(frames, ignore_index=True)
    # Shuffle so the rater does not see runs of one class and drift.
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def rate(row: pd.Series, *, offline: bool, model: str) -> tuple[bool | None, str]:
    user = (f"CUSTOMER: {row['customer_msg']}\n\n"
            f"BRAND REPLY: {row['brand_reply']}")
    try:
        r = llm.complete(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": user}],
            model=model, temperature=0.0,
            max_tokens=config.MAX_TOKENS_JUDGE, json_mode=True,
            offline=offline, tag="c3_proxy_validation",
        )
        d = r.json()
        return bool(d["usable"]), str(d.get("why", ""))[:80]
    except llm.CacheMissInOfflineMode:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] rating failed: {type(exc).__name__}: {exc}")
        return None, f"ERROR {type(exc).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brands", nargs="+",
                    default=["AmazonHelp", "SpotifyCares", "British_Airways"])
    ap.add_argument("--per-class", type=int, default=12,
                    help="samples per ReplyKind per brand (5 classes)")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--model", default=None, help="defaults to JUDGE_MODEL")
    args = ap.parse_args()
    model = args.model or config.JUDGE_MODEL

    sample = build_sample(args.brands, args.per_class, config.SEED)
    print(f"rating {len(sample)} replies across {len(args.brands)} brands "
          f"with {model} (offline={args.offline})")

    t0 = time.perf_counter()
    llm_usable, whys = [], []
    for i, row in sample.iterrows():
        u, why = rate(row, offline=args.offline, model=model)
        llm_usable.append(u)
        whys.append(why)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(sample)}  {llm.STATS.report()}", flush=True)
    sample["llm_usable"] = llm_usable
    sample["llm_why"] = whys

    ok = sample[sample["llm_usable"].notna()].copy()
    if ok.empty:
        sys.exit("no successful ratings")

    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    a = ok["usable"].astype(bool).to_numpy()      # my regex classifier
    b = ok["llm_usable"].astype(bool).to_numpy()  # independent LLM rater
    agree = float((a == b).mean())
    kappa = float(cohen_kappa_score(a, b))
    cm = confusion_matrix(a, b, labels=[False, True]).tolist()

    per_class = (ok.assign(match=(a == b))
                   .groupby("kind")
                   .agg(n=("match", "size"), agreement=("match", "mean"),
                        regex_usable=("usable", "mean"),
                        llm_usable=("llm_usable", "mean"))
                   .round(3).reset_index())
    per_brand = (ok.assign(match=(a == b))
                   .groupby("brand")
                   .agg(n=("match", "size"), agreement=("match", "mean"),
                        regex_usable=("usable", "mean"),
                        llm_usable=("llm_usable", "mean"))
                   .round(3).reset_index())

    disagreements = ok[a != b][
        ["brand", "kind", "usable", "llm_usable", "llm_why",
         "customer_msg", "brand_reply"]
    ].head(20)

    verdict = ("STRONG - C3 can be trusted as reported" if kappa >= 0.6 else
               "MODERATE - C3 is usable but its error is reported alongside it"
               if kappa >= 0.4 else
               "WEAK - C3 is unreliable; brand table must be re-weighted")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rater_model": model, "generator_model": config.GEN_MODEL,
        "n_rated": int(len(ok)), "n_failed": int(sample["llm_usable"].isna().sum()),
        "raw_agreement": round(agree, 4), "cohen_kappa": round(kappa, 4),
        "verdict": verdict,
        "confusion_matrix": {"labels": ["not_usable", "usable"],
                             "rows_regex_cols_llm": cm},
        "per_class": per_class.to_dict(orient="records"),
        "per_brand": per_brand.to_dict(orient="records"),
        "disagreements": disagreements.to_dict(orient="records"),
        "cache": llm.cache_stats(),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        return "\n".join(["| " + " | ".join(cols) + " |",
                          "|" + "|".join("---" for _ in cols) + "|",
                          *["| " + " | ".join(str(v) for v in r) + " |"
                            for r in df.itertuples(index=False)]])

    OUT_MD.write_text(
        "# C3 proxy validation\n\n"
        f"The `c3_usable_rate` that decides brand selection comes from a regex "
        f"classifier I wrote. This checks it against an independent rater "
        f"(`{model}`) which was given the downstream definition of usable "
        f"precedent and **no hint of the heuristic**.\n\n"
        f"- n rated: **{len(ok)}** (stratified across all 5 reply classes)\n"
        f"- raw agreement: **{agree:.1%}**\n"
        f"- Cohen kappa: **{kappa:.3f}** -> {verdict}\n\n"
        f"## Agreement by reply class\n\n{md(per_class)}\n\n"
        f"## Agreement by brand\n\n{md(per_brand)}\n",
        encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"n={len(ok)}  raw agreement={agree:.1%}  Cohen kappa={kappa:.3f}")
    print(f"verdict: {verdict}")
    print(f"{'='*70}\n")
    print("by reply class:\n" + per_class.to_string(index=False))
    print("\nby brand:\n" + per_brand.to_string(index=False))
    print(f"\n{llm.STATS.report()}  ({time.perf_counter()-t0:.1f}s)")
    print(f"wrote {OUT_JSON.name}, {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
