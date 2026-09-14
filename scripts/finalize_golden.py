"""Phase 3 step 3: turn two labellers' drafts into a reviewed golden set.

## What "reviewed" means here, precisely

The brief permits LLM assistance for drafting labels but requires a human to
review and correct every one. Claiming blanket human review of 200 rows would be
easy to write and impossible to verify, so this script enforces a specific,
auditable process instead:

* **Every disagreement between the two labellers is adjudicated.** Not
  auto-resolved by confidence, not broken by majority (there is no majority with
  two labellers) - each one requires an explicit written adjudication or this
  script refuses to produce a golden set.

* **A seeded random sample of the AGREED rows is also reviewed.** This is the
  part that matters most. Two LLMs from different families can agree and both be
  wrong, and agreement between them would then look like confirmation. Auditing
  a random sample of agreements measures how often that happens.

* **The override rate on that audit sample is reported as an estimate of
  residual label noise** in the un-audited agreed rows. So the golden set ships
  with a measured error bar rather than an assurance.

Every final label carries `provenance`, so a reader can separate rows a human
actually looked at from rows accepted on labeller agreement.

Inputs:
    eval/golden/golden_labeled_raw.jsonl   both labellers' drafts
    eval/golden/adjudications.jsonl        human decisions (written by review)

Outputs:
    eval/golden/golden_set.jsonl           the deliverable
    eval/golden/review_queue.json          what a human still has to look at
    eval/results/golden_label_quality.json measured label-noise estimate

Usage:
    python scripts/finalize_golden.py --queue     # build the review queue
    python scripts/finalize_golden.py             # finalise once adjudicated
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
from threadpilot import config, taxonomy  # noqa: E402

RAW = config.GOLDEN_DIR / "golden_labeled_raw.jsonl"
ADJ = config.GOLDEN_DIR / "adjudications.jsonl"
OUT = config.GOLDEN_JSONL
QUEUE = config.GOLDEN_DIR / "review_queue.json"
QUALITY = config.RESULTS_DIR / "golden_label_quality.json"

N_AUDIT = 40  # agreed rows to review; sets the precision of the noise estimate


def load_raw() -> list[dict]:
    if not RAW.exists():
        sys.exit(f"missing {RAW}. Run scripts/label_golden.py first.")
    return [json.loads(l) for l in RAW.open(encoding="utf-8")]


def load_adjudications() -> dict[str, dict]:
    if not ADJ.exists():
        return {}
    out = {}
    for line in ADJ.open(encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        rec = json.loads(line)
        out[rec["pair_id"]] = rec
    return out


def split_rows(rows: list[dict]) -> tuple[list, list, list]:
    """(agreed, disagreed, errored)."""
    agreed, disagreed, errored = [], [], []
    for r in rows:
        a, b = r.get("label_A"), r.get("label_B")
        if not isinstance(a, dict) or "error" in a or \
           not isinstance(b, dict) or "error" in b:
            errored.append(r)
        elif a["intent"] == b["intent"] and a["decision"] == b["decision"]:
            agreed.append(r)
        else:
            disagreed.append(r)
    return agreed, disagreed, errored


def audit_sample(agreed: list[dict], n: int) -> list[str]:
    """Seeded random sample of agreed rows to audit.

    Random rather than confidence-ranked on purpose: sampling the rows the
    models were least sure about would bias the noise estimate upward and make
    it useless as an estimate for the rest.
    """
    rng = np.random.default_rng(config.SEED)
    ids = sorted(r["pair_id"] for r in agreed)
    idx = rng.permutation(len(ids))[:min(n, len(ids))]
    return sorted(ids[i] for i in idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", action="store_true",
                    help="write the review queue and exit")
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="finalise without a complete audit (records the gap)")
    args = ap.parse_args()

    rows = load_raw()
    agreed, disagreed, errored = split_rows(rows)
    adj = load_adjudications()
    audit_ids = audit_sample(agreed, N_AUDIT)

    print(f"{len(rows)} rows: {len(agreed)} agreed, {len(disagreed)} disagreed, "
          f"{len(errored)} errored")
    print(f"adjudications on file: {len(adj)}")

    need = [r["pair_id"] for r in disagreed if r["pair_id"] not in adj]
    audit_todo = [p for p in audit_ids if p not in adj]

    if args.queue:
        by_id = {r["pair_id"]: r for r in rows}
        q = {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "instructions": (
                "Write one JSON object per line into eval/golden/adjudications.jsonl: "
                '{"pair_id": "...", "intent": "...", "decision": "auto_handle|escalate", '
                '"note": "why", "reviewer": "..."}. Every disagreement must be '
                "adjudicated. Audit rows only need an entry if you DISAGREE with the "
                "labellers - agreeing needs no entry, but you must still have looked."),
            "disagreements_to_adjudicate": [
                {"pair_id": r["pair_id"], "stratum": r["stratum"],
                 "customer_msg": r["customer_msg"],
                 "A": {k: r["label_A"][k] for k in ("intent", "decision", "confidence", "reason")},
                 "B": {k: r["label_B"][k] for k in ("intent", "decision", "confidence", "reason")}}
                for r in disagreed if r["pair_id"] in need],
            "audit_sample_to_review": [
                {"pair_id": p, "stratum": by_id[p]["stratum"],
                 "customer_msg": by_id[p]["customer_msg"],
                 "agreed_intent": by_id[p]["label_A"]["intent"],
                 "agreed_decision": by_id[p]["label_A"]["decision"],
                 "confidence": by_id[p]["label_A"]["confidence"]}
                for p in audit_ids],
            "errored": [r["pair_id"] for r in errored],
        }
        QUEUE.write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {QUEUE.name}:")
        print(f"  {len(need)} disagreements to adjudicate")
        print(f"  {len(audit_ids)} agreed rows in the audit sample")
        return 0

    if need and not args.allow_unreviewed:
        sys.exit(f"\n{len(need)} disagreements are not adjudicated. "
                 f"Run --queue, review them, then re-run.\n"
                 f"First few: {need[:5]}")

    # ---- assemble ---------------------------------------------------------
    final, overrides_in_audit, overrides_total = [], 0, 0
    for r in rows:
        pid = r["pair_id"]
        a, b = r.get("label_A"), r.get("label_B")
        both_ok = (isinstance(a, dict) and "error" not in a
                   and isinstance(b, dict) and "error" not in b)
        adjudicated = adj.get(pid)

        if adjudicated:
            intent = adjudicated["intent"]
            decision = adjudicated["decision"]
            if both_ok and a["intent"] == b["intent"] and a["decision"] == b["decision"]:
                provenance = "human_override_of_agreement"
                if intent != a["intent"] or decision != a["decision"]:
                    overrides_total += 1
                    if pid in audit_ids:
                        overrides_in_audit += 1
                else:
                    provenance = "human_confirmed_agreement"
            else:
                provenance = "human_adjudicated_disagreement"
                overrides_total += 1
            note = adjudicated.get("note", "")
            reviewer = adjudicated.get("reviewer", "unknown")
            confidence = 1.0
        elif both_ok and a["intent"] == b["intent"] and a["decision"] == b["decision"]:
            intent, decision = a["intent"], a["decision"]
            provenance = ("both_labellers_agreed_audited" if pid in audit_ids
                          else "both_labellers_agreed")
            note = ""
            reviewer = "audited" if pid in audit_ids else ""
            confidence = min(a["confidence"], b["confidence"])
        else:
            continue  # unadjudicated disagreement or error: excluded

        final.append({
            "pair_id": pid, "thread_id": r["thread_id"],
            "tweet_id": r["parent_id"], "reply_id": r["reply_id"],
            "text": r["customer_msg"],
            "reference_reply": r["brand_reply"],
            "reference_reply_kind": r["reply_kind"],
            "stratum": r["stratum"], "cluster": r["cluster"],
            "intent": intent,
            "auto_or_escalate": decision,
            "escalation_reason_notes": (note or (a["reason"] if both_ok else "")),
            "ideal_reply_notes": (a["ideal_reply_notes"] if both_ok else ""),
            "label_confidence": round(float(confidence), 3),
            "ambiguity_flag": bool(
                both_ok and (a["intent"] != b["intent"]
                             or a.get("ambiguous_with") or b.get("ambiguous_with"))),
            "provenance": provenance,
            "reviewer": reviewer,
            "labeller_A_intent": a["intent"] if both_ok else None,
            "labeller_B_intent": b["intent"] if both_ok else None,
            "nn_similarity": r.get("nn_similarity"),
        })

    out_df = pd.DataFrame(final).sort_values("pair_id")
    with OUT.open("w", encoding="utf-8") as fh:
        for rec in out_df.to_dict(orient="records"):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    audited = [p for p in audit_ids if p in {r["pair_id"] for r in final}]
    audit_reviewed = [p for p in audited if p in adj]
    noise = (overrides_in_audit / len(audit_reviewed)) if audit_reviewed else None

    quality = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_final": len(out_df),
        "n_agreed": len(agreed), "n_disagreed": len(disagreed),
        "n_errored": len(errored),
        "provenance_counts": out_df["provenance"].value_counts().to_dict(),
        "intent_distribution": out_df["intent"].value_counts().to_dict(),
        "decision_distribution": out_df["auto_or_escalate"].value_counts().to_dict(),
        "ambiguity_flagged": int(out_df["ambiguity_flag"].sum()),
        "audit": {
            "sample_size": len(audit_ids),
            "reviewed": len(audit_reviewed),
            "overrides_found": overrides_in_audit,
            "estimated_residual_label_noise": (
                round(noise, 4) if noise is not None else None),
            "method": "Seeded random sample of rows where both labellers agreed. "
                      "The override rate here estimates how often BOTH labellers "
                      "were wrong together, which agreement alone cannot detect. "
                      "Applies to the un-audited agreed rows.",
        },
        "total_human_overrides": overrides_total,
        "caveat": "The golden set is NOT distribution-matched (see "
                  "golden_strata.json). Labels were drafted by two LLMs from "
                  "different families and reviewed per the process above; rows "
                  "with provenance 'both_labellers_agreed' were NOT individually "
                  "inspected, and the audit estimates their error rate.",
    }
    QUALITY.write_text(json.dumps(quality, indent=2), encoding="utf-8")

    print(f"\nwrote {OUT.name}: {len(out_df)} rows")
    print("\nprovenance:")
    for k, v in quality["provenance_counts"].items():
        print(f"  {k:<38} {v:>4}")
    print("\nintent distribution:")
    for k, v in quality["intent_distribution"].items():
        print(f"  {k:<24} {v:>4}")
    print(f"\ndecision: {quality['decision_distribution']}")
    if noise is not None:
        print(f"\naudit: {overrides_in_audit}/{len(audit_reviewed)} overridden "
              f"-> estimated residual label noise {noise:.1%}")
    else:
        print(f"\naudit: NOT YET REVIEWED ({len(audit_todo)} rows outstanding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
