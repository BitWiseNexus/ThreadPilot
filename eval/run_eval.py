"""Produce every number in docs/report.md.

Reads saved predictions (scripts/run_predictions.py) and scores them. Splitting
prediction from scoring is what makes `--offline` meaningful: metrics can be
rewritten and re-run indefinitely without spending a token.

## Two sampling decisions, both deliberate

**The judge runs on a PAIRED subsample for cross-system comparison.** Judging
every system on all 200 rows would cost ~200 extra calls against a 200k
token/day budget. Judging them on the *same* seeded subsample instead gives a
paired comparison, which is statistically stronger per row than unpaired full
samples - the same message is scored for every system, so per-message difficulty
cancels out. The main pipeline is additionally judged on all 200 for its own
headline figure.

**Nothing is averaged across the two escalation error types.** A false escalate
costs an agent thirty seconds; a false auto-handle publishes a wrong answer
under the brand's name. Reporting one combined "decision accuracy" would hide
the trade the system exists to make.

Usage:
    python -m eval.run_eval --offline
    python -m eval.run_eval --judge-n 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import batching, config, llm, retrieval  # noqa: E402

from . import judge as judge_mod  # noqa: E402
from . import metrics as M  # noqa: E402

PRED_DIR = config.RESULTS_DIR / "predictions"
OUT_JSON = config.RESULTS_DIR / "eval_results.json"
OUT_MD = config.RESULTS_DIR / "eval_results.md"
JUDGE_DIR = config.RESULTS_DIR / "judge"
JUDGE_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_ORDER = ["trivial", "simple_tfidf_silver", "simple_tfidf_cluster",
                "retrieval_1nn", "pipeline_no_retr", "pipeline_no_gates",
                "pipeline"]


def load_golden() -> list[dict]:
    return [json.loads(l) for l in config.GOLDEN_JSONL.open(encoding="utf-8")]


def load_preds(name: str) -> list[dict] | None:
    p = PRED_DIR / f"{name}.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def inter_labeller_ceiling() -> float | None:
    p = config.RESULTS_DIR / "label_agreement.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["intent_raw_agreement"]


def md_table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, rule, *body])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--judge-n", type=int, default=100,
                    help="paired subsample size for cross-system judging")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--judge-systems", nargs="*", default=None,
                    help="restrict judging to these systems")
    ap.add_argument("--boot", type=int, default=M.DEFAULT_BOOTSTRAP)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                        stream=sys.stderr)
    t0 = time.perf_counter()

    golden = load_golden()
    by_pair = {g["pair_id"]: g for g in golden}
    ceiling = inter_labeller_ceiling()
    print(f"golden {len(golden)} rows | inter-labeller ceiling "
          f"{ceiling:.1%}" if ceiling else "")

    available = [s for s in SYSTEM_ORDER if (PRED_DIR / f"{s}.jsonl").exists()]
    if not available:
        sys.exit("no predictions. Run: python scripts/run_predictions.py")
    print(f"systems with predictions: {', '.join(available)}\n")

    results: dict[str, dict] = {}

    # ---- classification + decision (free) --------------------------------
    for name in available:
        preds = load_preds(name)
        pairs = [(by_pair[p["pair_id"]], p) for p in preds
                 if p["pair_id"] in by_pair]
        y_true = [g["intent"] for g, _ in pairs]
        y_pred = [p["intent"] for _, p in pairs]
        d_true = [g["auto_or_escalate"] for g, _ in pairs]
        d_pred = [p["decision"] for _, p in pairs]

        cls = M.classification_report(y_true, y_pred, ceiling=ceiling,
                                      n_boot=args.boot)
        dec = M.decision_report(d_true, d_pred, n_boot=args.boot)
        results[name] = {
            "n": len(pairs),
            "classification": cls.to_dict(),
            "top_confusions": M.top_confusions(cls),
            "decision": dec.to_dict(),
        }
        print(f"{name:<22} acc {cls.accuracy.fmt():<24} "
              f"macroF1 {cls.macro_f1.fmt(pct=False):<22} "
              f"cov {dec.coverage.point:.0%}  FAH {dec.false_auto_handle.point:.0%}")

    # ---- judge (the token spend) -----------------------------------------
    if not args.skip_judge:
        rng = np.random.default_rng(config.SEED)
        idx = sorted(rng.permutation(len(golden))[:args.judge_n])
        paired_ids = [golden[i]["pair_id"] for i in idx]

        # The index is loaded LAZILY, and only if something actually needs
        # judging. It lives in data/interim/ (gitignored, because its metadata
        # carries tweet text from a CC BY-NC-SA dataset), so a fresh clone does
        # not have it. Eagerly loading it here made `--offline` crash on a clean
        # checkout with FileNotFoundError - i.e. the README's central promise,
        # "reproduce every number with no API key", was false for everyone but
        # me. Found by the Phase 9 fresh-clone pass, which is what it is for.
        _index = {"i": None}

        def evidence_for(pid):
            if _index["i"] is None:
                _index["i"] = retrieval.RetrievalIndex.load()
            return _index["i"].search_text(by_pair[pid]["text"],
                                           k=config.RETRIEVAL_K)

        print(f"\njudging on a paired subsample of {len(paired_ids)} rows "
              f"with {config.JUDGE_MODEL}")
        st = batching.BatchStats()
        # Judge in PRIORITY order, not SYSTEM_ORDER. Under a daily token cap the
        # run can stop part-way, and stopping before the main system is scored
        # wastes the whole budget on baselines - which is what happened once.
        JUDGE_PRIORITY = ["pipeline", "pipeline_no_gates", "pipeline_no_retr",
                          "retrieval_1nn", "simple_tfidf_cluster",
                          "simple_tfidf_silver", "trivial"]
        to_judge = [n for n in JUDGE_PRIORITY if n in available]
        if args.judge_systems:
            to_judge = [n for n in to_judge if n in set(args.judge_systems)]
        for name in to_judge:
            if (JUDGE_DIR / f"{name}.json").exists():
                results[name]["judge"] = {
                    k: v for k, v in
                    json.loads((JUDGE_DIR / f"{name}.json").read_text("utf-8")).items()
                    if k != "scores"}
                print(f"  {name:<22} already judged, reusing")
                continue
            preds = {p["pair_id"]: p for p in load_preds(name)}
            items, ids = [], []
            for pid in paired_ids:
                if pid not in preds:
                    continue
                items.append(judge_mod.JudgeItem(
                    customer_msg=by_pair[pid]["text"],
                    reply=preds[pid]["reply"],
                    precedents=evidence_for(pid)))
                ids.append(pid)
            try:
                scores = judge_mod.judge_many(items, offline=args.offline,
                                              stats=st)
            except llm.DailyQuotaExhausted as e:
                print(f"[!] judging stopped: {e}")
                break
            ok = [(i, s) for i, s in zip(ids, scores) if s is not None]
            if not ok:
                continue
            send = [s.send_unedited for _, s in ok]
            payload = {
                "system": name, "n": len(ok), "n_failed": len(ids) - len(ok),
                "judge_model": config.JUDGE_MODEL,
                "send_unedited_rate": M.bootstrap_ci(send, n_boot=args.boot).to_dict(),
                "criteria_means": {
                    c: round(float(np.mean([getattr(s, c) for _, s in ok])), 3)
                    for c in judge_mod.CRITERIA},
                "scores": {i: s.to_dict() for i, s in ok},
            }
            (JUDGE_DIR / f"{name}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            results[name]["judge"] = {k: v for k, v in payload.items()
                                      if k != "scores"}
            print(f"  {name:<22} send-unedited "
                  f"{payload['send_unedited_rate']['point']:.1%} "
                  f"(n={len(ok)})  {payload['criteria_means']}")
        print(f"  batching: {st.report()}")

    # ---- coverage/risk curve for the main system -------------------------
    if "pipeline" in results:
        preds = load_preds("pipeline")
        pairs = [(by_pair[p["pair_id"]], p) for p in preds
                 if p["pair_id"] in by_pair]
        jp = JUDGE_DIR / "pipeline.json"
        acceptable = None
        if jp.exists():
            js = json.loads(jp.read_text(encoding="utf-8"))["scores"]
            acceptable = [js.get(p["pair_id"], {}).get("send_unedited")
                          for _, p in pairs]
            if any(a is None for a in acceptable):
                acceptable = None   # partial coverage would bias the curve
        results["pipeline"]["coverage_risk_curve"] = M.coverage_risk_curve(
            [g["auto_or_escalate"] for g, _ in pairs],
            [p["confidence"] for _, p in pairs],
            [p["max_similarity"] for _, p in pairs],
            acceptable=acceptable)

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "brand": config.BRAND, "seed": config.SEED,
        "n_golden": len(golden),
        "inter_labeller_ceiling": ceiling,
        "gen_model": config.GEN_MODEL, "judge_model": config.JUDGE_MODEL,
        "judge_paired_n": args.judge_n,
        "bootstrap_resamples": args.boot,
        "systems": results,
        "caveats": [
            "The golden set is NOT distribution-matched (equal cluster "
            "allocation, account_security oversampled ~8x, hard cases sought). "
            "Accuracy here does not estimate production accuracy.",
            f"Accuracy should be read against the inter-labeller ceiling "
            f"({ceiling:.1%} if known), not against 100%.",
            "Golden labels carry ~15% measured residual noise (D37).",
            "The judge shares a provider with the generator and is smaller.",
            "Batched inference shifts ~20% of individual predictions without "
            "changing aggregate accuracy (D44).",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    rows = []
    for name in available:
        r = results[name]
        c, d = r["classification"], r["decision"]
        j = r.get("judge")
        rows.append({
            "system": f"`{name}`",
            "accuracy": M.Interval(**{k: v for k, v in c["accuracy"].items()}).fmt(),
            "macro F1": f"{c['macro_f1']['point']:.3f}",
            "coverage": f"{d['coverage']['point']:.1%}",
            "false auto-handle": f"{d['false_auto_handle_rate']['point']:.1%}",
            "send unedited": (f"{j['send_unedited_rate']['point']:.1%}"
                              if j else "-"),
        })
    OUT_MD.write_text(
        f"# Evaluation results\n\n*Generated {payload['generated']}. "
        f"n={len(golden)} golden rows; judge on a paired subsample of "
        f"{args.judge_n}.*\n\n"
        + md_table(rows, ["system", "accuracy", "macro F1", "coverage",
                          "false auto-handle", "send unedited"])
        + "\n\n## Caveats carried with these numbers\n\n"
        + "\n".join(f"- {c}" for c in payload["caveats"]) + "\n",
        encoding="utf-8")

    print(f"\nwrote {OUT_JSON.name} and {OUT_MD.name} "
          f"({time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
