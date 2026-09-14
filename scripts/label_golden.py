"""Phase 3 step 2: draft golden labels with TWO independent LLM labellers.

The brief allows LLM assistance for drafting labels but requires that a human
review and correct every one. This script produces the drafts and, more
importantly, produces the evidence that tells a human WHERE to look.

## Why two labellers instead of one

A single LLM pass gives labels with no way to tell a confident label from a
coin-flip. Two labellers from **different model families** (gpt-oss vs Qwen -
the same separation D7 requires of the judge) give three things a single pass
cannot:

* an inter-labeller agreement statistic, which measures how hard the labelling
  task actually is rather than assuming it is easy;
* automatic surfacing of the genuinely ambiguous items - disagreements are
  exactly where human adjudication is worth spending attention;
* a check on systematic bias: if one model has a pet class, the confusion
  between labellers shows it.

## Two anti-circularity rules, both deliberate

1. **The labeller is NOT told the escalation policy.** `taxonomy.prompt_block()`
   emits definitions and boundary notes but never dispositions. If the labeller
   were told "refund_billing is always escalate", the golden auto/escalate
   labels would be a restatement of gate G1, and later measuring G1 against them
   would be circular - the gate would score perfectly by construction. Instead
   each message is judged on its merits, so G1's agreement with the labels is a
   real measurement.

2. **The labeller never sees the brand's actual reply.** It sees only the
   customer message, which is exactly what the pipeline sees at inference time.
   Showing the historical reply would leak the answer into the label and would
   also import the brand's own mistakes as ground truth.

Outputs:
    eval/golden/golden_labeled_raw.jsonl     both labellers' drafts per row
    eval/results/label_agreement.json        agreement + the disagreement list

Usage:
    python scripts/label_golden.py
    python scripts/label_golden.py --which silver --offline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import config, llm, taxonomy  # noqa: E402

OUT_RAW = config.GOLDEN_DIR / "golden_labeled_raw.jsonl"
OUT_RAW_SILVER = config.GOLDEN_DIR / "dev_silver_labeled_raw.jsonl"
OUT_AGREE = config.RESULTS_DIR / "label_agreement.json"

SYSTEM = """You are labelling customer support tweets sent to Amazon's support account, to build an evaluation set.

You see ONLY the customer's message - the same thing an assistant would see before replying. You do NOT see how Amazon actually replied. Judge the message on its own.

## Step 1: intent

Pick exactly one intent from this taxonomy. The boundary notes exist because adjacent intents are easy to confuse; read them.

{taxonomy}

## Step 2: can this be handled automatically?

Decide INDEPENDENTLY of the intent you chose, on the merits of THIS message.

"auto_handle" means: a support assistant could send a helpful, safe reply drawn from how Amazon has handled similar messages before, and a human would not need to edit it.

"escalate" means a human must handle it. Reasons include, but are not limited to:
  - it needs private account or order data to answer at all
  - it involves money: refunds, disputed charges, compensation
  - the account may be compromised, or fraud is alleged
  - there is a legal, safety, or serious-harm angle
  - the customer is angry enough that an automated reply would make it worse
  - the message is too vague or ambiguous to answer safely

Do not apply a blanket rule per intent. Two messages with the same intent can differ.

## Step 3: notes

- confidence: 0.0-1.0, how sure you are of the intent. Be honest; low confidence is useful signal.
- ambiguous_with: the intent name this could also plausibly be, or null.
- ideal_reply_notes: one short sentence on what a good reply must contain or must avoid.

Reply with ONLY compact JSON:
{{"intent": "...", "confidence": 0.0, "decision": "auto_handle"|"escalate", "reason": "<15 words max>", "ambiguous_with": null, "ideal_reply_notes": "..."}}"""


def label_one(text: str, model: str, *, offline: bool) -> dict | None:
    sys_prompt = SYSTEM.format(taxonomy=taxonomy.prompt_block())
    try:
        r = llm.complete(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": f"CUSTOMER MESSAGE:\n{text}"}],
            model=model, temperature=0.0, max_tokens=1024, json_mode=True,
            offline=offline, tag="golden_labeling",
        )
        d = r.json()
        intent = str(d.get("intent", "")).strip()
        if intent not in taxonomy.BY_NAME:
            # Do not silently coerce an unknown label into `other`: that would
            # hide a prompt/taxonomy mismatch as a legitimate residual class.
            return {"error": f"unknown intent {intent!r}", "raw": d}
        dec = str(d.get("decision", "")).strip()
        if dec not in ("auto_handle", "escalate"):
            return {"error": f"unknown decision {dec!r}", "raw": d}
        amb = d.get("ambiguous_with")
        return {
            "intent": intent,
            "confidence": float(d.get("confidence", 0.0)),
            "decision": dec,
            "reason": str(d.get("reason", ""))[:160],
            "ambiguous_with": amb if amb in taxonomy.BY_NAME else None,
            "ideal_reply_notes": str(d.get("ideal_reply_notes", ""))[:240],
        }
    except llm.CacheMissInOfflineMode:
        raise
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["golden", "silver"], default="golden")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--labellers", choices=["AB", "A", "B"], default="AB",
                    help="AB = two independent labellers (golden). A/B = single "
                         "labeller, for dev_silver where D5 explicitly accepts "
                         "noisier labels because it only tunes thresholds.")
    args = ap.parse_args()

    src = (config.GOLDEN_DIR / "golden_candidates.jsonl" if args.which == "golden"
           else config.GOLDEN_DIR / "dev_silver_candidates.jsonl")
    out = OUT_RAW if args.which == "golden" else OUT_RAW_SILVER
    if not src.exists():
        sys.exit(f"missing {src}. Run scripts/sample_golden.py first.")

    df = pd.read_json(src, lines=True)
    if args.limit:
        df = df.head(args.limit)
    all_labellers = {"A": config.GEN_MODEL, "B": config.JUDGE_MODEL}
    labellers = [(t, all_labellers[t]) for t in args.labellers]
    print(f"labelling {len(df)} {args.which} rows with "
          f"{' and '.join(m for _, m in labellers)} (offline={args.offline})")

    t0 = time.perf_counter()
    rows = []
    for i, rec in enumerate(df.to_dict(orient="records")):
        entry = dict(rec)
        for tag, model in labellers:
            entry[f"label_{tag}"] = label_one(rec["customer_msg"], model,
                                              offline=args.offline)
        rows.append(entry)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(df)}  {llm.STATS.report()}", flush=True)

    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"\nwrote {out.name} ({len(rows)} rows) in {time.perf_counter()-t0:.0f}s")

    # ---- agreement between the two independent labellers -------------------
    if len(labellers) == 1:
        tag = labellers[0][0]
        n_ok = sum(1 for r in rows
                   if isinstance(r.get(f"label_{tag}"), dict)
                   and "error" not in r[f"label_{tag}"])
        print(f"\nsingle-labeller run ({tag}): {n_ok}/{len(rows)} labelled, "
              f"no agreement statistic computable by design")
        print(llm.STATS.report())
        return 0

    ok = [r for r in rows
          if isinstance(r.get("label_A"), dict) and "error" not in r["label_A"]
          and isinstance(r.get("label_B"), dict) and "error" not in r["label_B"]]
    errs = len(rows) - len(ok)
    if not ok:
        sys.exit("no rows labelled successfully by both labellers")

    from sklearn.metrics import cohen_kappa_score

    ia = [r["label_A"]["intent"] for r in ok]
    ib = [r["label_B"]["intent"] for r in ok]
    da = [r["label_A"]["decision"] for r in ok]
    db = [r["label_B"]["decision"] for r in ok]

    intent_agree = sum(a == b for a, b in zip(ia, ib)) / len(ok)
    intent_kappa = float(cohen_kappa_score(ia, ib))
    dec_agree = sum(a == b for a, b in zip(da, db)) / len(ok)
    dec_kappa = float(cohen_kappa_score(da, db))

    disagreements = [
        {"pair_id": r["pair_id"], "stratum": r["stratum"],
         "customer_msg": r["customer_msg"][:300],
         "A_intent": r["label_A"]["intent"], "B_intent": r["label_B"]["intent"],
         "A_decision": r["label_A"]["decision"], "B_decision": r["label_B"]["decision"],
         "A_conf": r["label_A"]["confidence"], "B_conf": r["label_B"]["confidence"],
         "A_reason": r["label_A"]["reason"], "B_reason": r["label_B"]["reason"]}
        for r in ok
        if r["label_A"]["intent"] != r["label_B"]["intent"]
        or r["label_A"]["decision"] != r["label_B"]["decision"]
    ]

    # Which intent pairs get confused: this is the map of where the taxonomy's
    # boundaries are weakest, and it is known BEFORE the classifier is built.
    confusions: dict[str, int] = {}
    for a, b in zip(ia, ib):
        if a != b:
            key = " <-> ".join(sorted([a, b]))
            confusions[key] = confusions.get(key, 0) + 1

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "which": args.which, "n": len(ok), "n_errors": errs,
        "labeller_A": config.GEN_MODEL, "labeller_B": config.JUDGE_MODEL,
        "intent_raw_agreement": round(intent_agree, 4),
        "intent_cohen_kappa": round(intent_kappa, 4),
        "decision_raw_agreement": round(dec_agree, 4),
        "decision_cohen_kappa": round(dec_kappa, 4),
        "n_disagreements": len(disagreements),
        "intent_confusion_pairs": dict(sorted(confusions.items(),
                                              key=lambda kv: -kv[1])),
        "note": "Labellers saw only the customer message, never the brand's "
                "actual reply, and were NOT told the escalation policy - "
                "otherwise the auto/escalate labels would restate gate G1 and "
                "measuring G1 against them would be circular.",
        "disagreements": disagreements,
    }
    if args.which == "golden":
        OUT_AGREE.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"n={len(ok)}  errors={errs}")
    print(f"  intent   agreement {intent_agree:.1%}   kappa {intent_kappa:.3f}")
    print(f"  decision agreement {dec_agree:.1%}   kappa {dec_kappa:.3f}")
    print(f"  rows needing human adjudication: {len(disagreements)}")
    print(f"{'='*70}")
    print("\ntop intent confusions between labellers:")
    for k, v in list(payload["intent_confusion_pairs"].items())[:8]:
        print(f"  {v:>3}  {k}")
    print(f"\n{llm.STATS.report()}")
    print("\nNEXT: review disagreements, then finalise the golden set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
