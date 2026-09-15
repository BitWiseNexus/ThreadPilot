"""Human-vs-judge agreement study. Mandatory, not optional.

An LLM-judged quality number without an agreement statistic is an unvalidated
claim. This measures how far the judge can be trusted, and the result is
reported next to every judge-derived figure.

## The process, and why it is in two commands

    python -m eval.judge_validation worksheet   # blind items, no judge scores
    # ...score them into eval/results/judge/human_scores.jsonl...
    python -m eval.judge_validation compare     # merge and measure

`worksheet` deliberately omits the judge's scores. Scoring while able to see
them is not independent re-scoring, it is agreeing with them - and the resulting
kappa would measure nothing except my willingness to defer. The two-step shape
makes that impossible to do accidentally.

## What is measured

* `send_unedited` - the headline binary. Raw agreement plus Cohen kappa.
* the five ordinals - **quadratic-weighted** kappa, because on a 1-5 scale a
  4-vs-5 disagreement is not the same as a 1-vs-5 and unweighted kappa would
  badly understate agreement.
* the disagreements themselves, quoted, because the direction of a systematic
  disagreement matters more than its size: a judge that is uniformly generous
  can be corrected for, one that is erratic cannot.

## Honest limitation, stated where it cannot be missed

The "human" here is the AI assistant that built this repo, scoring blind against
the written rubric. That is weaker than an independent annotator and is labelled
as such in the output, not quietly called "human". The repo owner re-scoring even
ten of these rows would be worth more than any of it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import config, retrieval  # noqa: E402

from . import judge as judge_mod  # noqa: E402
from . import metrics as M  # noqa: E402

JUDGE_DIR = config.RESULTS_DIR / "judge"
WORKSHEET = JUDGE_DIR / "validation_worksheet.json"
HUMAN = JUDGE_DIR / "human_scores.jsonl"
OUT_JSON = config.RESULTS_DIR / "judge_validation.json"
OUT_MD = config.RESULTS_DIR / "judge_validation.md"

SYSTEM_FOR_VALIDATION = "pipeline"
DEFAULT_N = 40


def cmd_worksheet(args) -> int:
    p = JUDGE_DIR / f"{args.system}.json"
    if not p.exists():
        sys.exit(f"no judge scores for {args.system}; run eval.run_eval first")
    judged = json.loads(p.read_text(encoding="utf-8"))
    scores = judged["scores"]

    golden = {json.loads(l)["pair_id"]: json.loads(l)
              for l in config.GOLDEN_JSONL.open(encoding="utf-8")}
    preds = {json.loads(l)["pair_id"]: json.loads(l)
             for l in (config.RESULTS_DIR / "predictions" /
                       f"{args.system}.jsonl").open(encoding="utf-8")}
    index = retrieval.RetrievalIndex.load()

    rng = np.random.default_rng(config.SEED)
    ids = sorted(scores)
    pick = sorted(ids[i] for i in rng.permutation(len(ids))[:args.n])

    items = []
    for pid in pick:
        prec = index.search_text(golden[pid]["text"], k=3)
        items.append({
            "pair_id": pid,
            "customer_msg": golden[pid]["text"],
            "reply": preds[pid]["reply"],
            "precedent": [{"asked": p.customer_msg[:200],
                           "brand_replied": p.brand_reply[:200],
                           "similarity": round(p.similarity, 3)} for p in prec],
            # NO judge scores here, on purpose.
        })

    WORKSHEET.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": args.system, "n": len(items),
        "instructions": (
            "Score each item WITHOUT looking at eval/results/judge/"
            f"{args.system}.json. Append one JSON object per line to "
            "eval/results/judge/human_scores.jsonl with keys: pair_id, "
            "grounded, on_intent, no_overcommit, voice, actionable (1-5), "
            "send_unedited (bool), note. Rubric is in eval/judge.py."),
        "items": items,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {WORKSHEET} ({len(items)} items, judge scores withheld)")
    print("score them into", HUMAN)
    return 0


def cmd_compare(args) -> int:
    if not HUMAN.exists():
        sys.exit(f"no human scores at {HUMAN}; run `worksheet` first")
    p = JUDGE_DIR / f"{args.system}.json"
    judged = json.loads(p.read_text(encoding="utf-8"))["scores"]
    human = {}
    for line in HUMAN.open(encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("//"):
            r = json.loads(line)
            human[r["pair_id"]] = r

    common = sorted(set(human) & set(judged))
    if not common:
        sys.exit("no overlapping pair_ids between human and judge scores")
    print(f"comparing {len(common)} items")

    hb = [bool(human[i]["send_unedited"]) for i in common]
    jb = [bool(judged[i]["send_unedited"]) for i in common]
    binary_agree = float(np.mean([h == j for h, j in zip(hb, jb)]))
    binary_kappa = M.cohen_kappa(hb, jb) if len(set(hb)) > 1 and len(set(jb)) > 1 \
        else float("nan")

    ordinals = {}
    for c in judge_mod.CRITERIA:
        h = [int(human[i][c]) for i in common]
        j = [int(judged[i][c]) for i in common]
        ordinals[c] = {
            "exact_agreement": round(float(np.mean([a == b for a, b in zip(h, j)])), 4),
            "within_one": round(float(np.mean([abs(a - b) <= 1 for a, b in zip(h, j)])), 4),
            "quadratic_weighted_kappa": (round(M.weighted_kappa(h, j), 4)
                                         if len(set(h)) > 1 and len(set(j)) > 1
                                         else None),
            "human_mean": round(float(np.mean(h)), 3),
            "judge_mean": round(float(np.mean(j)), 3),
            # Direction matters more than magnitude: a uniformly generous judge
            # can be corrected for, an erratic one cannot.
            "judge_bias": round(float(np.mean(j) - np.mean(h)), 3),
        }

    disagreements = [
        {"pair_id": i, "human_send": hb[k], "judge_send": jb[k],
         "human_note": human[i].get("note", ""),
         "judge_why": judged[i].get("why", "")}
        for k, i in enumerate(common) if hb[k] != jb[k]
    ]

    verdict = ("STRONG - judge-derived numbers can be read as stated"
               if binary_kappa >= 0.6 else
               "MODERATE - judge numbers usable, but report this kappa beside them"
               if binary_kappa >= 0.4 else
               "WEAK - judge-derived quality numbers should not carry the report")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": args.system, "n": len(common),
        "judge_model": config.JUDGE_MODEL,
        "rater": "AI assistant that built this repo, scoring blind against the "
                 "written rubric. NOT an independent human annotator - weaker "
                 "evidence than the phrase 'human validation' usually implies.",
        "send_unedited": {"raw_agreement": round(binary_agree, 4),
                          "cohen_kappa": (round(binary_kappa, 4)
                                          if binary_kappa == binary_kappa else None),
                          "human_rate": round(float(np.mean(hb)), 4),
                          "judge_rate": round(float(np.mean(jb)), 4)},
        "ordinals": ordinals,
        "verdict": verdict,
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    L = [f"# Judge validation ({args.system})", "",
         f"*{len(common)} items, blind re-scoring. Judge: `{config.JUDGE_MODEL}`.*", "",
         "> **Who the rater is.** " + payload["rater"], "",
         "## send_unedited (the headline binary)", "",
         "| | value |", "|---|---|",
         f"| raw agreement | {binary_agree:.1%} |",
         f"| Cohen kappa | {payload['send_unedited']['cohen_kappa']} |",
         f"| rater says send-unedited | {np.mean(hb):.1%} |",
         f"| judge says send-unedited | {np.mean(jb):.1%} |", "",
         f"**{verdict}**", "",
         "## Ordinal criteria (quadratic-weighted kappa)", "",
         "| criterion | exact | within 1 | qw-kappa | rater mean | judge mean | judge bias |",
         "|---|---|---|---|---|---|---|"]
    for c, d in ordinals.items():
        L.append(f"| {c} | {d['exact_agreement']:.0%} | {d['within_one']:.0%} | "
                 f"{d['quadratic_weighted_kappa']} | {d['human_mean']} | "
                 f"{d['judge_mean']} | {d['judge_bias']:+.2f} |")
    L += ["", f"## Where they disagreed on send_unedited ({len(disagreements)})", ""]
    for d in disagreements[:12]:
        L.append(f"- `{d['pair_id']}` rater={d['human_send']} judge={d['judge_send']}"
                 f" — judge: \"{d['judge_why']}\" / rater: \"{d['human_note']}\"")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\n  send_unedited agreement {binary_agree:.1%}  kappa "
          f"{payload['send_unedited']['cohen_kappa']}")
    print(f"  {verdict}")
    for c, d in ordinals.items():
        print(f"    {c:<16} qwk={d['quadratic_weighted_kappa']} "
              f"bias={d['judge_bias']:+.2f}")
    print(f"\nwrote {OUT_JSON.name}, {OUT_MD.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worksheet")
    w.add_argument("--system", default=SYSTEM_FOR_VALIDATION)
    w.add_argument("--n", type=int, default=DEFAULT_N)
    w.set_defaults(func=cmd_worksheet)
    c = sub.add_parser("compare")
    c.add_argument("--system", default=SYSTEM_FOR_VALIDATION)
    c.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
