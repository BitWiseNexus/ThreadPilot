"""Phase 7: mine the golden-set results for failure modes.

Every mode here is derived from the data rather than proposed from intuition,
and each is reported with its frequency, real verbatim examples, a hypothesis
for the cause, and where the fix would have to go (prompt / retrieval / taxonomy
/ nowhere cheap).

Two deliberate inclusions:

* **At least one failure of the EVALUATION, not just the system.** A harness
  that only finds faults in its subject is not looking hard enough.
* **The dangerous failures get the most space.** A wrong escalate costs an agent
  thirty seconds; a wrong auto-handle publishes a wrong answer under the brand's
  name. The modes are ordered by harm, not by frequency.

No LLM calls - this reads saved predictions and the golden set.

Usage:
    python scripts/failure_analysis.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import config, taxonomy  # noqa: E402

PRED = config.RESULTS_DIR / "predictions" / "pipeline.jsonl"
JUDGE = config.RESULTS_DIR / "judge" / "pipeline.json"
OUT_JSON = config.RESULTS_DIR / "failure_analysis.json"
OUT_MD = config.RESULTS_DIR / "failure_analysis.md"


def load():
    golden = {json.loads(l)["pair_id"]: json.loads(l)
              for l in config.GOLDEN_JSONL.open(encoding="utf-8")}
    preds = {json.loads(l)["pair_id"]: json.loads(l)
             for l in PRED.open(encoding="utf-8")}
    judge = {}
    if JUDGE.exists():
        judge = json.loads(JUDGE.read_text(encoding="utf-8")).get("scores", {})
    rows = []
    for pid, p in preds.items():
        g = golden.get(pid)
        if g:
            rows.append({"pair_id": pid, "g": g, "p": p, "j": judge.get(pid)})
    return rows


def ex(rows, n=3, maxlen=150):
    out = []
    for r in rows[:n]:
        out.append({
            "pair_id": r["pair_id"],
            "text": r["g"]["text"][:maxlen],
            "gold_intent": r["g"]["intent"],
            "pred_intent": r["p"]["intent"],
            "gold_decision": r["g"]["auto_or_escalate"],
            "pred_decision": r["p"]["decision"],
            "reply": r["p"]["reply"][:maxlen],
            "gates": r["p"]["gates_fired"],
            "max_similarity": r["p"]["max_similarity"],
        })
    return out


def main() -> int:
    rows = load()
    n = len(rows)
    print(f"analysing {n} golden rows")
    modes = []

    # ---- 1. DANGEROUS: auto-handled when it should have escalated ---------
    fah = [r for r in rows
           if r["p"]["decision"] == "auto_handle"
           and r["g"]["auto_or_escalate"] == "escalate"]
    by_intent = Counter(r["g"]["intent"] for r in fah)
    modes.append({
        "rank": 1,
        "name": "False auto-handle (the dangerous failure)",
        "count": len(fah),
        "denominator": sum(1 for r in rows if r["p"]["decision"] == "auto_handle"),
        "severity": "HIGH - publishes an unreviewed reply under the brand's name",
        "concentrated_in": dict(by_intent.most_common(5)),
        "hypothesis":
            "The model proposed auto_handle on all 200 rows, so every escalate "
            "comes from a deterministic gate. The gates fire on intent (G1), "
            "missing precedent (G2), low confidence (G3) and explicit "
            "identifiers (G4) - none of which detects the two rubric tests that "
            "dominate the gold labels: E5 (anger an automated reply would "
            "worsen) and E2 (a reply that would have to commit a remedy). Those "
            "are properties of TONE and OBLIGATION, and no current gate reads "
            "either.",
        "fix_location":
            "New gate. A sentiment/anger signal and a commitment-language "
            "detector would target the actual gap. Prompt changes will not help "
            "while the model's own proposal is constant.",
        "examples": ex(fah),
    })

    # ---- 2. Over-escalation: cost, not danger -----------------------------
    fe = [r for r in rows
          if r["p"]["decision"] == "escalate"
          and r["g"]["auto_or_escalate"] == "auto_handle"]
    gates = Counter(g for r in fe for g in r["p"]["gates_fired"])
    modes.append({
        "rank": 2,
        "name": "Over-escalation (costly, not dangerous)",
        "count": len(fe),
        "denominator": sum(1 for r in rows if r["p"]["decision"] == "escalate"),
        "severity": "LOW - wastes an agent's time, publishes nothing wrong",
        "gates_responsible": dict(gates.most_common()),
        "hypothesis":
            "G1 escalates by INTENT, unconditionally. But the rubric judges per "
            "message, and a generically-answerable message inside an "
            "always-escalate intent ('how long do refunds normally take?') is "
            "auto-handleable. This is the predicted cost of a blunt "
            "intent-level gate, now measured.",
        "fix_location":
            "Gate design: make G1 conditional on the message rather than the "
            "intent alone. That is the single highest-value change available.",
        "examples": ex(fe),
    })

    # ---- 3. Adjacent-intent confusion -------------------------------------
    wrong = [r for r in rows if r["p"]["intent"] != r["g"]["intent"]]
    pairs = Counter(f"{r['g']['intent']} -> {r['p']['intent']}" for r in wrong)
    modes.append({
        "rank": 3,
        "name": "Adjacent-intent confusion",
        "count": len(wrong),
        "denominator": n,
        "severity": "MEDIUM - wrong intent can mis-route and mis-gate",
        "top_pairs": dict(pairs.most_common(6)),
        "hypothesis":
            "These are the SAME pairs the two independent labellers disagreed "
            "on in Phase 3, recorded before the classifier existed. The "
            "classifier fails where the taxonomy boundary is genuinely "
            "contested, not arbitrarily - so most of this is irreducible "
            "against these labels rather than a model deficiency.",
        "fix_location":
            "Taxonomy, not prompt. Merging delivery_delay/delivery_failure or "
            "sharpening the prime_membership boundary would remove more error "
            "than any prompt edit.",
        "examples": ex(wrong),
    })

    # ---- 4. Ungrounded drafting where retrieval had nothing ---------------
    lowsim = sorted(rows, key=lambda r: r["p"]["max_similarity"])[:20]
    lowsim_auto = [r for r in lowsim if r["p"]["decision"] == "auto_handle"]
    modes.append({
        "rank": 4,
        "name": "Drafting with weak precedent",
        "count": len(lowsim_auto),
        "denominator": len(lowsim),
        "severity": "MEDIUM - a confident reply with nothing behind it",
        "hypothesis":
            "Gate G2 escalates below TAU_SIM=0.45, but similarity is a blunt "
            "proxy for 'is there relevant precedent'. A message can score 0.5 "
            "against a superficially similar but substantively different "
            "precedent and still be auto-handled.",
        "fix_location":
            "Retrieval. Re-ranking, or requiring agreement among the top-k "
            "rather than the single max, would tighten this.",
        "examples": ex(lowsim_auto),
    })

    # ---- 5. A failure of the EVALUATION itself ----------------------------
    # First framing of this mode was wrong and is worth recording: I looked for
    # rows where the classifier FAILED on `other` and found zero. The defect is
    # the opposite - it succeeds perfectly on a class that is not a coherent
    # thing, and that success props up the headline.
    other_rows = [r for r in rows if r["g"]["intent"] == "other"]
    other_right = [r for r in other_rows if r["p"]["intent"] == "other"]
    non_other = [r for r in rows if r["g"]["intent"] != "other"]
    non_other_right = [r for r in non_other if r["p"]["intent"] == r["g"]["intent"]]
    overall = sum(1 for r in rows if r["p"]["intent"] == r["g"]["intent"]) / len(rows)
    without = len(non_other_right) / len(non_other) if non_other else 0.0
    modes.append({
        "rank": 5,
        "name": "EVALUATION failure: a 17% grab-bag inflates the headline",
        "count": len(other_right),
        "denominator": len(other_rows),
        "severity": "N/A - a defect in the measurement, not in the system",
        "overall_accuracy": round(overall, 4),
        "accuracy_excluding_other": round(without, 4),
        "inflation_points": round(overall - without, 4),
        "hypothesis":
            "`other` is the largest gold class (34/200, 17%) and the classifier "
            "scores 100% on it - yet it is not one thing. It holds praise "
            "('Absolutely fantastic customer service'), pre-purchase questions "
            "('When will amazon restock this item?'), checkout problems and "
            "third-party product faults. Scoring perfectly on a bag of unlike "
            "things is not a capability, but it lifts overall accuracy from "
            f"{without:.1%} to {overall:.1%} - {overall - without:.1%} points of "
            "the headline number. Part of that is the same methodological "
            "coupling that makes 80.5% exceed the 75.0% inter-labeller ceiling: "
            "the classifier reads the taxonomy definitions I also used when "
            "adjudicating those rows into `other`.",
        "fix_location":
            "Taxonomy: add `positive_feedback` and `pre_purchase`, which would "
            "break the grab-bag into measurable classes. Frozen before labelling "
            "on purpose (D12/D14), so reported rather than patched - and it is "
            "the highest-value single change in 'what I would do next'.",
        "examples": ex(other_rows),
    })

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": n,
        "judge_scores_available": bool(JUDGE.exists()),
        "ordering": "by harm, not by frequency",
        "modes": modes,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    L = ["# Failure analysis", "",
         f"*{n} golden rows. Modes ordered by HARM, not frequency: a wrong "
         f"auto-handle publishes a wrong answer under the brand's name, a wrong "
         f"escalate costs an agent thirty seconds.*", ""]
    for m in modes:
        L += [f"## {m['rank']}. {m['name']}", "",
              f"**{m['count']} / {m['denominator']}** — {m['severity']}", ""]
        for key, label in (("concentrated_in", "Concentrated in"),
                           ("gates_responsible", "Gates responsible"),
                           ("top_pairs", "Top pairs")):
            if m.get(key):
                L += [f"**{label}:** " + ", ".join(
                    f"`{k}` ({v})" for k, v in m[key].items()), ""]
        L += [f"**Hypothesis.** {m['hypothesis']}", "",
              f"**Where the fix goes.** {m['fix_location']}", "",
              "**Real examples:**", ""]
        for e in m["examples"]:
            L.append(f"- *\"{e['text']}\"*")
            L.append(f"  gold `{e['gold_intent']}`/`{e['gold_decision']}` → "
                     f"predicted `{e['pred_intent']}`/`{e['pred_decision']}`"
                     f" (gates {e['gates'] or 'none'}, sim {e['max_similarity']:.2f})")
            if e["reply"]:
                L.append(f"  reply: \"{e['reply']}\"")
        L.append("")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    print(f"\n{'rank':<5}{'mode':<46}{'count':>10}")
    for m in modes:
        print(f"{m['rank']:<5}{m['name'][:44]:<46}{m['count']:>4}/{m['denominator']}")
    print(f"\nwrote {OUT_MD.name}, {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
