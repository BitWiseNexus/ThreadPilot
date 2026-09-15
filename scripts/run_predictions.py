"""Run every system over the golden set and save raw predictions.

Separated from scoring on purpose. Predictions are the expensive part (LLM
calls); metrics are free. Keeping them apart means the metric code can be
rewritten and re-run any number of times without spending a single token, and
it is why `--offline` can regenerate every reported number later.

Systems evaluated, all on the identical golden rows through the identical code
path:

  pipeline            the real thing: classify -> retrieve -> draft -> gates
  pipeline_no_gates   ablation: does the deterministic rule layer earn its place?
  pipeline_no_retr    ablation: what does grounding actually buy?
  trivial             majority intent + canned reply + always escalate
  simple_tfidf_silver TF-IDF trained on 100 labelled rows
  simple_tfidf_cluster TF-IDF trained on 6,948 cluster-pseudo-labelled rows
  retrieval_1nn       nearest historical reply, verbatim, no generation

Usage:
    python scripts/run_predictions.py                 # all systems
    python scripts/run_predictions.py --only pipeline
    python scripts/run_predictions.py --offline       # replay the cache
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import baselines, config, llm, pipeline, retrieval  # noqa: E402

PRED_DIR = config.RESULTS_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

LLM_SYSTEMS = ("pipeline", "pipeline_no_gates", "pipeline_no_retr")


def load_golden() -> list[dict]:
    if not config.GOLDEN_JSONL.exists():
        sys.exit("no golden set; run scripts/finalize_golden.py")
    return [json.loads(l) for l in config.GOLDEN_JSONL.open(encoding="utf-8")]


def save(name: str, results, stats: dict | None = None) -> Path:
    p = PRED_DIR / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    if stats is not None:
        (PRED_DIR / f"{name}.stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--draft-batch", type=int, default=6)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                        stream=sys.stderr)

    golden = load_golden()
    texts = [g["text"] for g in golden]
    pids = [g["pair_id"] for g in golden]
    index = retrieval.RetrievalIndex.load()
    print(f"golden {len(texts)} rows | index {len(index):,} vectors "
          f"| offline={args.offline}")

    # Belt and braces: the guard already ran at build time, but this is the
    # moment it matters, so verify against the rows actually being scored.
    index.assert_disjoint_from({int(g["thread_id"]) for g in golden})
    print("leakage guard verified against the golden rows being scored\n")

    want = set(args.only) if args.only else None
    done: dict[str, str] = {}

    def wanted(n: str) -> bool:
        return want is None or n in want

    # ---- LLM systems -----------------------------------------------------
    for name, kw in (
        ("pipeline", dict(use_gates=True, use_retrieval=True)),
        ("pipeline_no_gates", dict(use_gates=False, use_retrieval=True)),
        ("pipeline_no_retr", dict(use_gates=True, use_retrieval=False)),
    ):
        if not wanted(name):
            continue
        print(f"--- {name} ---", flush=True)
        t0 = time.perf_counter()
        st = pipeline.RunStats()
        try:
            res = pipeline.triage_many(
                texts, index=index, pair_ids=pids, offline=args.offline,
                draft_batch_size=args.draft_batch, stats=st, **kw)
        except llm.DailyQuotaExhausted as e:
            print(f"[!] {name}: {e}")
            print("    Daily token budget gone. Re-run tomorrow; everything "
                  "already completed is cached, so it will resume cheaply.")
            break
        save(name, res, st.to_dict())
        done[name] = f"{len(res)} rows in {time.perf_counter()-t0:.0f}s"
        print(f"    {done[name]}  |  {llm.STATS.report()}\n", flush=True)

    # ---- baselines (no LLM calls) ---------------------------------------
    bl: dict[str, object] = {}
    for n, f in (("trivial", "baseline_trivial.pkl"),
                 ("simple_tfidf_silver", "baseline_simple_silver.pkl"),
                 ("simple_tfidf_cluster", "baseline_simple_cluster.pkl")):
        p = config.INTERIM_DIR / f
        if p.exists():
            with p.open("rb") as fh:
                bl[n] = pickle.load(fh)
    if "simple_tfidf_cluster" in bl:
        bl["retrieval_1nn"] = baselines.Retrieval1NNBaseline.build(
            index, classifier=bl["simple_tfidf_cluster"].pipeline)

    for name, obj in bl.items():
        if not wanted(name):
            continue
        t0 = time.perf_counter()
        res = obj.triage(texts, pair_ids=pids)
        save(name, res, {"llm_calls": 0})
        done[name] = f"{len(res)} rows in {time.perf_counter()-t0:.0f}s"
        print(f"--- {name} --- {done[name]}")

    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_golden": len(texts), "offline": args.offline,
        "index_size": len(index), "systems": done,
        "llm": llm.STATS.report(),
        "note": "Predictions are saved separately from scoring so metric code "
                "can be re-run without spending tokens.",
    }
    (PRED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                            encoding="utf-8")
    print(f"\nwrote {len(done)} prediction files to {PRED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
