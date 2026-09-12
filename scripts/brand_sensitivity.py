"""Is the brand choice robust to how "usable precedent" is defined?

Context. The C3 classifier scored Cohen kappa 0.374 against an independent rater
(eval/results/c3_proxy_validation.md). Per-class agreement is 0.83-0.92 wherever
the definition is unambiguous, and collapses on exactly two classes:
`truncated` (0.12) and `fragment` (0.47). Reading those cases, the divergence is
definitional rather than noise - my classifier asks "is this a complete,
well-formed reply?" while the rater asks "does the visible text contain help?"
Both questions are legitimate.

The wrong response to that is to keep editing the classifier until it agrees
with the rater; that is just fitting one heuristic to another and would be a
third round of the same mistake. The right response is to ask whether the
decision the metric feeds actually depends on the disagreement.

So: recompute the brand ranking under four defensible definitions of usable and
see whether the winner survives. A decision that holds under all four is safe to
make on a kappa of 0.374. One that flips is not, and would have to be reported as
undecidable on this evidence.

Usage:
    python scripts/brand_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import config  # noqa: E402
from threadpilot.data.clean import ReplyKind  # noqa: E402

K = ReplyKind
OUT_JSON = config.RESULTS_DIR / "brand_sensitivity.json"
OUT_MD = config.RESULTS_DIR / "brand_sensitivity.md"

# Four definitions, from most to least demanding.
DEFINITIONS: dict[str, tuple[str, set[str]]] = {
    "A_strict_answers_only": (
        "Only replies that answer in the tweet itself or name a specific "
        "resource. Rejects anything that defers to another channel.",
        {K.SELF_CONTAINED.value, K.LINK_REFERRAL.value},
    ),
    "B_project_default": (
        "A, plus diagnostic questions and handoffs that name what is needed. "
        "This is what the pipeline actually uses (D21/D22).",
        {K.SELF_CONTAINED.value, K.LINK_REFERRAL.value,
         K.DIAGNOSTIC_ASK.value, K.HANDOFF_WITH_ASK.value},
    ),
    "C_rater_aligned": (
        "B, plus source-truncated replies - the rater judged 89% of those "
        "usable because the visible text still contains help.",
        {K.SELF_CONTAINED.value, K.LINK_REFERRAL.value, K.DIAGNOSTIC_ASK.value,
         K.HANDOFF_WITH_ASK.value, K.TRUNCATED.value},
    ),
    "D_most_inclusive": (
        "C, plus split-reply fragments. Everything except bare acknowledgements "
        "and bare channel switches.",
        {K.SELF_CONTAINED.value, K.LINK_REFERRAL.value, K.DIAGNOSTIC_ASK.value,
         K.HANDOFF_WITH_ASK.value, K.TRUNCATED.value, K.FRAGMENT.value},
    ),
}


def norm(col: pd.Series) -> pd.Series:
    lo, hi = col.min(), col.max()
    return pd.Series(0.5, index=col.index) if hi == lo else (col - lo) / (hi - lo)


def main() -> int:
    base = json.loads((config.RESULTS_DIR / "brand_selection.json").read_text("utf-8"))
    rows = {r["brand"]: r for r in base["table"]}

    frames = {}
    for brand in rows:
        f = config.INTERIM_DIR / f"pairs_{brand}.parquet"
        if f.exists():
            frames[brand] = pd.read_parquet(f)
    if not frames:
        sys.exit("no pairs_*.parquet found. Run scripts/eda_brand_selection.py first.")

    results, rankings = {}, {}
    for name, (desc, usable_kinds) in DEFINITIONS.items():
        recs = []
        for brand, df in frames.items():
            r = rows[brand]
            recs.append({
                "brand": brand,
                "usable_rate": round(float(df["kind"].isin(usable_kinds).mean()), 4),
                "c1_usable_pairs": r["c1_usable_pairs"],
                "c2_diversity_entropy": r["c2_diversity_entropy"],
                "c4_multiturn_rate": r["c4_multiturn_rate"],
                "c5_min_pole": min(r["c5_escalate_pole_rate"], r["c5_auto_pole_rate"]),
            })
        d = pd.DataFrame(recs)
        # Identical weights to eda_brand_selection.py; only the definition moves.
        d["score"] = (
            0.45 * norm(d["usable_rate"])
            + 0.20 * norm(d["c2_diversity_entropy"].fillna(0))
            + 0.15 * norm(np.log10(d["c1_usable_pairs"].clip(lower=1)))
            + 0.10 * norm(d["c4_multiturn_rate"])
            + 0.10 * norm(d["c5_min_pole"])
        ).round(4)
        d = d.sort_values("score", ascending=False).reset_index(drop=True)
        d["rank"] = d.index + 1
        results[name] = {"description": desc,
                         "usable_kinds": sorted(usable_kinds),
                         "table": d.to_dict(orient="records")}
        rankings[name] = list(d["brand"])
        print(f"\n--- {name} ---\n{desc}")
        print(d[["rank", "brand", "usable_rate", "score"]].head(5).to_string(index=False))

    winners = {n: r[0] for n, r in rankings.items()}
    unique_winners = set(winners.values())
    top3 = {n: set(r[:3]) for n, r in rankings.items()}
    always_top3 = set.intersection(*top3.values())

    robust = len(unique_winners) == 1
    verdict = (
        f"ROBUST - {next(iter(unique_winners))} wins under all "
        f"{len(DEFINITIONS)} definitions, so the kappa 0.374 disagreement does "
        f"not affect the decision."
        if robust else
        f"NOT ROBUST - the winner changes by definition ({winners}). The brand "
        f"choice cannot be justified on this metric alone and must be reported "
        f"as such."
    )

    payload = {
        "context": "C3 classifier vs independent rater: kappa 0.374; per-class "
                   "agreement 0.83-0.92 except truncated (0.12) and fragment (0.47)",
        "definitions": results, "winners": winners,
        "always_in_top3": sorted(always_top3), "robust": robust, "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Brand-choice sensitivity to the definition of usable precedent", "",
        "The C3 classifier agrees with an independent rater at **kappa 0.374**. "
        "Per-class agreement is 0.83-0.92 where the definition is unambiguous and "
        "collapses on `truncated` (0.12) and `fragment` (0.47) - a definitional "
        "divergence, not classification noise. Rather than tune the classifier "
        "until it matches the rater, this asks whether the *decision* depends on "
        "the disagreement at all.", "",
        "Same scoring weights throughout; only the set of admitted reply classes "
        "changes.", "",
        "## Ranking under each definition", "",
        "| definition | 1st | 2nd | 3rd | admits |", "|---|---|---|---|---|",
    ]
    for name, r in rankings.items():
        lines.append(f"| `{name}` | **{r[0]}** | {r[1]} | {r[2]} | "
                     f"{len(results[name]['usable_kinds'])} classes |")
    lines += ["", f"**{verdict}**", "",
              f"Brands in the top 3 under every definition: "
              f"{', '.join(sorted(always_top3)) or 'none'}", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{'='*72}\n{verdict}\n{'='*72}")
    print(f"always top-3: {sorted(always_top3)}")
    print(f"wrote {OUT_JSON.name}, {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
