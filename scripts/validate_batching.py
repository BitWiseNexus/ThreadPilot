"""Does batching change the answers?

Batching was adopted to fit a 200k-token/day budget (D39). It amortises a
~1,200-token taxonomy prompt across N items, which is the difference between the
remaining phases taking ~1.5 days and ~5. But it introduces a risk that cannot
be argued away: an item's label might be influenced by its neighbours in the
batch in a way an isolated call would not be.

So it is measured rather than assumed. The same messages are classified twice -
once batched, once one-per-call - and the agreement is reported. Any result in
this project produced via batching is quoted next to this number.

A high agreement means batching is a pure cost saving. A low one means the
headline numbers are partly an artifact of batch composition, which would have
to be reported as such (and batch size dropped).

Usage:
    python scripts/validate_batching.py --n 30
    python scripts/validate_batching.py --n 30 --offline
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
from threadpilot import batching, classify, config, llm  # noqa: E402

OUT = config.RESULTS_DIR / "batching_validation.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=batching.DEFAULT_BATCH_SIZE)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    if not config.GOLDEN_JSONL.exists():
        sys.exit("no golden set; run scripts/finalize_golden.py first")
    rows = [json.loads(l) for l in config.GOLDEN_JSONL.open(encoding="utf-8")]

    # Seeded sample so this is reproducible and not cherry-picked.
    rng = np.random.default_rng(config.SEED)
    idx = rng.permutation(len(rows))[:args.n]
    sample = [rows[i] for i in sorted(idx)]
    texts = [r["text"] for r in sample]
    print(f"comparing batched (size {args.batch_size}) vs unbatched on "
          f"{len(texts)} golden messages")

    t0 = time.perf_counter()
    st_b = batching.BatchStats()
    batched = classify.classify_many(texts, batch_size=args.batch_size,
                                     offline=args.offline, stats=st_b)
    t_batched = time.perf_counter() - t0
    calls_batched = llm.STATS.n

    t0 = time.perf_counter()
    st_s = batching.BatchStats()
    single = classify.classify_many(texts, batch_size=1,
                                    offline=args.offline, stats=st_s)
    t_single = time.perf_counter() - t0
    calls_single = llm.STATS.n - calls_batched

    agree_intent = [b.intent == s.intent for b, s in zip(batched, single)]
    conf_delta = [abs(b.confidence - s.confidence) for b, s in zip(batched, single)]

    from sklearn.metrics import cohen_kappa_score
    kappa = float(cohen_kappa_score([b.intent for b in batched],
                                    [s.intent for s in single]))

    disagreements = [
        {"pair_id": r["pair_id"], "text": r["text"][:200],
         "batched": b.intent, "batched_conf": round(b.confidence, 2),
         "unbatched": s.intent, "unbatched_conf": round(s.confidence, 2),
         "gold": r["intent"]}
        for r, b, s in zip(sample, batched, single) if b.intent != s.intent
    ]

    # Which one is closer to the golden label? If batching only relocates
    # errors rather than adding them, that matters more than raw agreement.
    gold = [r["intent"] for r in sample]
    acc_b = float(np.mean([b.intent == g for b, g in zip(batched, gold)]))
    acc_s = float(np.mean([s.intent == g for s, g in zip(single, gold)]))

    rate = float(np.mean(agree_intent))
    verdict = ("SAFE - batching does not materially change classifications"
               if rate >= 0.90 else
               "MARGINAL - report batched results next to this figure"
               if rate >= 0.80 else
               "UNSAFE - reduce batch size; results are batch-composition dependent")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n": len(texts), "batch_size": args.batch_size,
        "model": config.GEN_MODEL,
        "intent_agreement": round(rate, 4),
        "intent_cohen_kappa": round(kappa, 4),
        "mean_abs_confidence_delta": round(float(np.mean(conf_delta)), 4),
        "accuracy_vs_gold_batched": round(acc_b, 4),
        "accuracy_vs_gold_unbatched": round(acc_s, 4),
        "api_calls_batched": calls_batched,
        "api_calls_unbatched": calls_single,
        "call_reduction": f"{calls_single}/{calls_batched}",
        "seconds_batched": round(t_batched, 1),
        "seconds_unbatched": round(t_single, 1),
        "batch_stats": st_b.to_dict(),
        "verdict": verdict,
        "disagreements": disagreements,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print(f"\n{'='*66}")
    print(f"  intent agreement        {rate:.1%}   (kappa {kappa:.3f})")
    print(f"  mean |conf delta|       {np.mean(conf_delta):.3f}")
    print(f"  accuracy vs gold        batched {acc_b:.1%}  unbatched {acc_s:.1%}")
    print(f"  api calls               {calls_batched} batched vs {calls_single} unbatched")
    print(f"  {verdict}")
    print(f"{'='*66}")
    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s):")
        for d in disagreements[:6]:
            print(f"  {d['batched']:<20} vs {d['unbatched']:<20} gold={d['gold']}")
            print(f"    {d['text'][:100]}")
    print(f"\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
