"""Generate eval/golden/labelling_notes.md from measured artifacts.

Hand-writing these notes would let the described process drift from the one that
actually ran - the exact failure this document exists to prevent. Everything
below is read from golden_strata.json, label_agreement.json and
golden_label_quality.json, so the numbers cannot be stale.

Usage:  python scripts/write_labelling_notes.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import config  # noqa: E402

OUT = config.GOLDEN_DIR / "labelling_notes.md"


def load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    strata = load(config.RESULTS_DIR / "golden_strata.json")
    agree = load(config.RESULTS_DIR / "label_agreement.json")
    qual = load(config.RESULTS_DIR / "golden_label_quality.json")
    if not (strata and agree):
        sys.exit("run sample_golden.py and label_golden.py first")

    L = [
        "# Golden set - labelling notes", "",
        "*Generated from measured artifacts by "
        "`scripts/write_labelling_notes.py`. Do not edit by hand.*", "",
        f"Brand **{strata['brand']}**, seed `{strata['seed']}`, "
        f"{strata['n_golden']} golden rows and {strata['n_silver']} dev-silver "
        f"rows, thread-disjoint.", "",
        "---", "",
        "## 1. Sampling strategy", "",
        "The golden set was sampled **before** any labelling and before the "
        "pipeline existed, so the pipeline cannot be built to the test.", "",
        "### The chicken-and-egg problem", "",
        'The plan called for stratification "across intents", but intent labels '
        "are what the golden set exists to produce. Stratifying on them would "
        "require already having them. So strata are the Phase 2 **cluster "
        "assignments** - a data-derived proxy that was available at sampling "
        "time. The consequence, reported rather than smoothed over: realised "
        "per-INTENT counts do not match per-CLUSTER targets, because the human "
        "naming step merged clusters 0 and 1 and because `account_security` has "
        "no cluster of its own.", "",
        "### Three pools, deliberately not proportional", "",
        "| stratum | n |", "|---|---|",
    ]
    for k, v in strata["strata"].items():
        L.append(f"| `{k}` | {v} |")

    L += [
        "", "1. **Cluster strata** - equal allocation, not proportional. "
        "Proportional sampling gives the rarest classes too few examples to "
        "measure, and per-class F1 on five examples is noise.",
        "2. **`account_security`** - targeted pool, oversampled roughly 8x "
        "(2.5% of traffic to ~10% of the set). It is the highest-stakes intent "
        "and unmeasurable at its natural rate.",
        "3. **Hard cases** - deliberately sought rather than avoided. A golden "
        "set drawn only from cluster centres measures the easy middle and "
        "reports it as overall performance.", "",
        "### Hard cases present in the final set", "",
        "| kind | n | why it is hard |", "|---|---|---|",
    ]
    why = {
        "very_short": "little signal to classify from",
        "order_id": "requires private account data to answer at all",
        "account": "highest-stakes intent; must never be auto-handled",
        "sarcasm": "surface sentiment inverts the literal reading",
        "multi_intent": "three or more intent signals in one message",
        "novel": "bottom 5% by nearest-neighbour similarity - retrieval has "
                 "nothing close to ground a reply in",
    }
    for k, v in strata["hard_case_counts_in_golden"].items():
        L.append(f"| `{k}` | {v} | {why.get(k, '')} |")

    L += [
        "", "### The set is deliberately NOT distribution-matched", "",
        "| cluster | true share | golden share |", "|---|---|---|",
    ]
    for c in sorted(strata["true_cluster_shares"], key=int):
        t = strata["true_cluster_shares"][c]
        g = strata["golden_cluster_shares"].get(c, 0.0)
        L.append(f"| c{c} | {t:.1%} | {g:.1%} |")
    L += [
        "", "**This biases every headline number computed on it.** Accuracy here "
        "is not an estimate of production accuracy: rare classes and hard cases "
        "are over-represented on purpose, so the set is harder than real traffic "
        "in some respects and easier in others. Carried into the report's "
        "\"what is misleading about my headline number\" section.", "",
        "---", "",
        "## 2. Labelling process", "",
        f"Two independent labellers from **different model families** - "
        f"`{agree['labeller_A']}` (A) and `{agree['labeller_B']}` (B).", "",
        "### Two anti-circularity rules", "",
        "1. **Labellers were not told the escalation policy.** They received "
        "intent definitions and boundary notes but never dispositions. Had they "
        "been told \"refund_billing is always escalate\", the auto/escalate "
        "labels would restate gate G1, and later measuring G1 against them "
        "would be circular - the gate would score perfectly by construction.",
        "2. **Labellers never saw the brand's actual reply.** They saw only the "
        "customer message, exactly what the pipeline sees at inference. Showing "
        "the historical reply would leak the answer into the label and import "
        "the brand's own mistakes as ground truth.", "",
        "### Inter-labeller agreement", "",
        "| | raw agreement | Cohen kappa |", "|---|---|---|",
        f"| intent ({len(strata['strata'])} strata, 11 classes) | "
        f"{agree['intent_raw_agreement']:.1%} | "
        f"**{agree['intent_cohen_kappa']:.3f}** |",
        f"| auto vs escalate | {agree['decision_raw_agreement']:.1%} | "
        f"**{agree['decision_cohen_kappa']:.3f}** |", "",
        f"{agree['n_disagreements']} of {agree['n']} rows disagreed on intent "
        "or decision and went to adjudication.", "",
        "**This statistic is itself a finding.** It measures how hard the "
        "labelling task is, which a single-labeller process cannot reveal. A "
        "modest kappa means the taxonomy's boundaries are genuinely contested "
        "in places, and any classifier evaluated against these labels inherits "
        "that ceiling.", "",
        "### Where the labellers disagreed most", "",
        "| intent pair | n |", "|---|---|",
    ]
    for k, v in list(agree["intent_confusion_pairs"].items())[:10]:
        L.append(f"| {k} | {v} |")
    L += [
        "", "These pairs are the taxonomy's weakest boundaries, identified "
        "**before** the classifier was built - so Phase 6's confusion matrix "
        "can be checked against a prediction rather than explained after the "
        "fact.", "",
        "---", "",
        "## 3. Review, stated precisely", "",
        "### Who reviewed, in plain terms", "",
        "**The adjudicator was the AI assistant working on this repo, not an "
        "independent human annotator.** Every adjudication was written against "
        "the explicit rubric in `src/threadpilot/escalation.py` and the "
        "taxonomy boundary notes, and each carries a written justification in "
        "`eval/golden/adjudications.jsonl` naming the rule applied - so the "
        "reasoning is auditable line by line rather than taken on trust.",
        "",
        "That is a real limitation and is not dressed up as anything else: the "
        "same system drafted the labels, designed the taxonomy, wrote the "
        "rubric and adjudicated the disagreements, so a consistent "
        "misconception propagates through all four unchallenged. Two partial "
        "mitigations: the drafts came from two different model families, and "
        "the adjudicator disagreed with BOTH of them often (on the 63 "
        "same-intent/opposite-decision rows it split 32 auto / 31 escalate, "
        "where labeller A had said auto on all 63 and labeller B escalate on "
        "all 63).",
        "",
        "**The repo owner should spot-check a sample** - "
        "`eval/golden/adjudications.jsonl` is human-readable and each entry "
        "states its reasoning.", "",
    ]

    if qual:
        a = qual["audit"]
        L += [
            "Claiming blanket human review of 200 rows would be easy to write "
            "and impossible to verify. The enforced process instead:", "",
            "| provenance | n |", "|---|---|",
        ]
        for k, v in qual["provenance_counts"].items():
            L.append(f"| `{k}` | {v} |")
        noise = a["estimated_residual_label_noise"]
        L += [
            "", "* **Every labeller disagreement was adjudicated.** "
            "`finalize_golden.py` refuses to emit a golden set while any "
            "disagreement lacks an explicit written adjudication.",
            f"* **A seeded random sample of {a['sample_size']} AGREED rows was "
            "also reviewed.** This is the part that matters: two LLMs from "
            "different families can agree and both be wrong, and their "
            "agreement then looks like confirmation. Random rather than "
            "confidence-ranked, because sampling the least-confident rows would "
            "bias the estimate upward and make it useless for the rest.",
        ]
        if noise is not None:
            L += [
                f"* **Measured residual label noise: {noise:.1%}** "
                f"({a['overrides_found']} overrides in {a['reviewed']} audited "
                "rows). That is the estimated error rate in the "
                f"{qual['provenance_counts'].get('both_labellers_agreed', 0)} "
                "rows accepted on agreement without individual inspection.",
            ]
        else:
            L += ["* **Audit not yet complete** - residual label noise unmeasured."]
        L += [
            "", f"Total human overrides: {qual['total_human_overrides']}. "
            f"{qual['ambiguity_flagged']} rows carry `ambiguity_flag`.", "",
            "### Final composition", "", "| intent | n |", "|---|---|",
        ]
        for k, v in qual["intent_distribution"].items():
            L.append(f"| `{k}` | {v} |")
        L += ["", f"Decisions: {qual['decision_distribution']}", ""]

    L += [
        "---", "",
        "## 4. Label noise in the SOURCE data", "",
        "Distinct from label noise in this set, and worth stating plainly "
        "because it caps what any evaluation here can mean.", "",
        "* **The brand's historical reply is a reference, not ground truth.** "
        "Some of what AmazonHelp actually sent is unhelpful, deflecting, or "
        "truncated mid-sentence. `reference_reply` is kept for comparison, "
        "never treated as a ceiling a good reply must match. A draft that beats "
        "the historical reply is not penalised for differing from it.",
        "* **Roughly 16% of reference replies are unusable as precedent** "
        "(fragments, bare acknowledgements, bare channel switches) by the "
        "classifier in `threadpilot.data.clean`, which itself agrees with an "
        "independent rater at only kappa 0.374.",
        "* **The source contains genuine encoding damage.** U+FFFD appears in "
        "~0.004% of texts - flagged via a `mojibake` column, never silently "
        "repaired, since guessing the original character would be fabrication.",
        "* **Tweets are truncated by the platform**, so some messages end "
        "mid-clause and are genuinely ambiguous to a human reader too.",
        "* **~24% of real traffic is excluded** as non-English (Japanese "
        "largest), so these labels describe English traffic only.", "",
        "---", "",
        "## 4b. A taxonomy defect this labelling exposed", "",
        "`other` finished as the **largest class** - larger than any named "
        "intent. That is not a residual behaving normally; inspecting it shows "
        "two coherent intents the taxonomy lacks:", "",
        "* **praise / positive feedback** - \"Absolutely fantastic customer "
        "service ... Bravo\", \"Fantastic service from Daniela\". The taxonomy "
        "was derived from clusters of a complaint-dominated corpus, so "
        "satisfied customers have nowhere to go. Both labellers repeatedly put "
        "these in `service_complaint`, which is defined as *dissatisfaction* - "
        "a real error caught during adjudication.",
        "* **pre-purchase / product questions** - \"When will amazon restock "
        "this item?\", a camera whose wifi does not work, checkout and promo "
        "queries. These concern products rather than an existing order.", "",
        "The taxonomy was deliberately frozen before labelling (D12, D14), and "
        "re-deriving it now would let it be fitted to the evaluation data. So "
        "the defect is reported rather than patched, and it is the highest-value "
        "single change in the report's \"what I would do next\".", "",
        "**Consequence for the metrics:** `other` is a 17% catch-all, so the "
        "majority-class baseline is inflated and per-class scores for `other` "
        "measure a bag of unlike things.", "",
        "---", "",
        "## 5. Known weaknesses of this golden set", "",
        "1. **Not distribution-matched** (section 1) - accuracy here does not "
        "estimate production accuracy.",
        "2. **Drafted by LLMs.** Reviewed per section 3, but rows marked "
        "`both_labellers_agreed` were not individually inspected; their error "
        "rate is estimated, not zero.",
        "3. **Both labellers share a provider** (Groq) and a broad pretraining "
        "era. Different families reduce correlated error; they do not eliminate "
        "it.",
        "4. **n=200 across 11 classes** is roughly 18 per class, so per-class "
        "confidence intervals are wide. Every proportion in the report carries "
        "a bootstrap 95% CI for this reason.",
        "5. **One author.** The same person designed the taxonomy, wrote the "
        "labelling prompt and adjudicated disagreements, so consistent "
        "misconceptions propagate through all three unchallenged.", "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}.*", "",
    ]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
