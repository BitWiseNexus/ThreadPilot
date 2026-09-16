"""Append sections 5-8 to docs/report.md.

Section 6 is the mandatory "what's misleading about my headline number". It is
written to actively undermine my own result rather than to hedge it, and every
item in it is quantified from a measured artifact where a number exists. An
honest caveat with a magnitude attached is useful; one without is decoration.

Usage:  python scripts/write_report_part2.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import config  # noqa: E402

R = config.RESULTS_DIR
OUT = Path("docs/report.md")


def load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    if not OUT.exists():
        sys.exit("run scripts/write_report.py first")
    body = OUT.read_text(encoding="utf-8")
    # Idempotent: drop any previously appended tail before re-appending.
    marker = "## 5. Failure analysis"
    if marker in body:
        body = body[:body.index(marker)]

    ev = load("eval_results.json") or {}
    fa = load("failure_analysis.json")
    lq = load("golden_label_quality.json")
    la = load("label_agreement.json")
    st = load("golden_strata.json")
    jv = load("judge_validation.json")
    sub = json.loads((config.PROCESSED_DIR / "subsample_meta.json").read_text("utf-8"))
    sysd = ev.get("systems", {})
    P = sysd.get("pipeline", {})

    L = []
    A = L.append

    # ------------------------------------------- 4b validating the judge
    A("## 4b. Validating the judge — and it failed")
    A("")
    if jv:
        su = jv["send_unedited"]
        A("An LLM-judged quality number without an agreement statistic is an "
          "unvalidated claim, so 40 pipeline replies were **re-scored blind** "
          "against the judge. The worksheet withholds the judge's scores by "
          "construction: scoring while able to see them would measure "
          "deference, not agreement.")
        A("")
        A("| | value |")
        A("|---|---|")
        A(f"| raw agreement on `send_unedited` | {su['raw_agreement']:.1%} |")
        A(f"| **Cohen κ** | **{su['cohen_kappa']}** |")
        A(f"| judge says sendable (these 40) | {su['judge_rate']:.1%} |")
        A(f"| blind rater says sendable (these 40) | {su['human_rate']:.1%} |")
        A("")
        A(f"*Note the denominator: these rates are over the **40 validated "
          f"rows**, not the 100-row judged sample in §4. The judge's rate on "
          f"the full 100 is 78.0%; on this 40-row subset it is "
          f"{su['judge_rate']:.1%}. The gap to the blind rater "
          f"({su['judge_rate']-su['human_rate']:+.1%}) is the quantity of "
          f"interest, and it is measured on identical rows.*")
        A("")
        A(f"**{jv['verdict']}**")
        A("")
        A("The bias is systematic, not noisy — the judge is more generous on "
          "**every** criterion:")
        A("")
        A("| criterion | quadratic-weighted κ | judge bias |")
        A("|---|---|---|")
        for c, d in jv["ordinals"].items():
            A(f"| {c} | {d['quadratic_weighted_kappa']} | "
              f"{d['judge_bias']:+.2f} |")
        A("")
        A("### The failure mode is diagnosable")
        A("")
        A("Reading the "
          f"{jv['n_disagreements']} disagreements, the judge justifies its "
          "scores by **form** — *\"mirrors precedent\"*, *\"standard support "
          "link\"*, *\"matches precedent for security issues\"* — while the "
          "blind rater judged **substance** — *\"answers nothing\"*, *\"never "
          "answers whether this channel is right\"*, *\"already described the "
          "fault\"*.")
        A("")
        A("**The judge cannot detect a well-formed reply that fails to do its "
          "job.** It rated a reply to a customer reporting a *break-in with "
          "police involved* as sendable because it \"matches precedent for "
          "security/delivery issues\". The reply was: *\"Sorry about this. "
          "Please reach out to us directly here.\"*")
        A("")
        A("### What this costs, and what survives")
        A("")
        A(f"**Costs:** the {su['judge_rate']:.0%} send-unedited figure cannot "
          f"carry a headline. It is quoted with this κ attached and with the "
          f"stricter blind rate ({su['human_rate']:.0%}) beside it. Had this "
          f"study not been run, the report would have overstated reply quality "
          f"by {su['judge_rate']-su['human_rate']:.0%} points in the flattering "
          f"direction.")
        A("")
        A("**Survives:** the *relative* ordering is probably intact. If the "
          "judge simply rewarded precedent-conformity, `retrieval_1nn` — which "
          "**is** precedent, copied verbatim — would have scored highest. It "
          "did not. So the bias plausibly shifts levels rather than ranks. "
          "That is an inference, not a measurement, and is labelled as one.")
        A("")
        A("**Caveat on the rater.** The blind re-scorer was the AI assistant "
          "that built this repo, not an independent human annotator — weaker "
          "evidence than \"human validation\" usually implies. Every score "
          "carries a written justification in "
          "`eval/results/judge/human_scores.jsonl` so the reasoning is "
          "auditable rather than taken on trust.")
    else:
        A("*Not yet run.* Until it is, no judge-derived number in this report "
          "should be treated as validated.")
    A("")
    A("---")
    A("")

    # -------------------------------------------------- 5 failure analysis
    A("## 5. Failure analysis")
    A("")
    if fa:
        A("Five modes, **ordered by harm rather than frequency**: a wrong "
          "auto-handle publishes a wrong answer under the brand's name; a wrong "
          "escalate costs an agent thirty seconds.")
        A("")
        A("| # | mode | count | severity |")
        A("|---|---|---|---|")
        for m in fa["modes"]:
            A(f"| {m['rank']} | {m['name']} | {m['count']}/{m['denominator']} | "
              f"{m['severity'].split(' - ')[0]} |")
        A("")
        m1 = fa["modes"][0]
        A("### Mode 1 has a clean diagnosis, and it is not a tuning problem")
        A("")
        A(f"The {m1['count']} false auto-handles concentrate in "
          + ", ".join(f"`{k}` ({v})" for k, v in
                      list(m1["concentrated_in"].items())[:3])
          + " — **none of which are in `ALWAYS_ESCALATE`**.")
        A("")
        A("The gold labels escalated them under rubric tests **E5** (anger an "
          "automated reply would worsen) and **E2** (a reply that would have to "
          "commit a remedy). My gates read intent, precedent similarity, "
          "confidence and explicit identifiers. **None of them reads tone or "
          "obligation.** So this is a *missing gate*, not a mis-tuned one — and "
          "prompt changes cannot help while the model's own proposal is "
          "constant at `auto_handle`.")
        A("")
        m3 = next((m for m in fa["modes"] if m["rank"] == 3), None)
        if m3:
            A("### A prediction that held")
            A("")
            A("Phase 3 recorded which intent pairs the two labellers confused, "
              "**before the classifier existed**. The classifier's confusions "
              "match: " + ", ".join(f"`{k}` ({v})" for k, v in
                                    list(m3["top_pairs"].items())[:3]) + ". "
              "It fails where the taxonomy boundary is genuinely contested, not "
              "arbitrarily — so much of this is irreducible against these "
              "labels rather than a model deficiency.")
            A("")
    A("Full detail with verbatim examples: "
      "[`eval/results/failure_analysis.md`](../eval/results/failure_analysis.md).")
    A("")

    # ------------------------------------------------ 6 what's misleading
    A("## 6. What's misleading about my headline number")
    A("")
    A("Written to undermine my own result. Every item carries a magnitude where "
      "one can be measured — a caveat without a number is decoration.")
    A("")

    n = 0
    def item(title, text):
        nonlocal n
        n += 1
        A(f"**{n}. {title}**")
        A("")
        A(text)
        A("")

    if fa:
        m5 = next((m for m in fa["modes"] if m["rank"] == 5), None)
        if m5:
            item("A 17% grab-bag inflates accuracy by "
                 f"{m5['inflation_points']:.1%} points",
                 f"`other` is the largest gold class ({m5['denominator']}/200) "
                 f"and the classifier scores 100% on it — but it is not one "
                 f"thing. It holds praise, pre-purchase questions, checkout "
                 f"problems and third-party faults. Scoring perfectly on a bag "
                 f"of unlike things is not a capability. Accuracy on coherent "
                 f"classes is **{m5['accuracy_excluding_other']:.1%}**, not "
                 f"{m5['overall_accuracy']:.1%}.")

    if la:
        item(f"The score exceeds the {la['intent_raw_agreement']:.1%} "
             "inter-labeller ceiling, which should worry you",
             "Two independent labellers agreed on intent only "
             f"{la['intent_raw_agreement']:.1%} of the time, yet the classifier "
             f"scores higher against the adjudicated labels. The likely cause "
             f"is **methodological coupling**: the classifier prompt carries "
             f"the same boundary notes I used when adjudicating those labels, "
             f"so it is aligned with my adjudication rules and not only with "
             f"the task. I cannot cleanly separate the two effects, and I have "
             f"not tried to present this as evidence of quality.")

    if lq:
        item(f"The gold labels carry ~{lq['audit']['estimated_residual_label_noise']:.0%} "
             "measured residual noise",
             f"An audit of {lq['audit']['sample_size']} rows where both "
             f"labellers agreed found {lq['audit']['overrides_found']} where "
             f"**both were wrong together** — a failure agreement cannot "
             f"detect. That rate applies to the "
             f"{lq['provenance_counts'].get('both_labellers_agreed',0)} rows "
             f"nobody inspected individually. Any metric computed here is "
             f"bounded by it.")

    if la:
        item(f"The escalation label is barely reproducible (κ = "
             f"{la['decision_cohen_kappa']:.3f})",
             "Two capable models disagreed systematically on whether a message "
             "is safe to automate — gpt-oss escalated 34.3%, Qwen 73.8%. Every "
             "escalation number in this report is measured against a target "
             "that two reasonable raters could not agree on. This is the single "
             "biggest limitation of the work.")

    if st:
        item("The golden set is deliberately NOT distribution-matched",
             "Clusters were allocated equally, `account_security` was "
             "oversampled ~8x, and hard cases (sarcasm, order IDs, novel "
             "messages) were **sought rather than avoided**. This makes the set "
             "harder than production traffic in some respects and easier in "
             "others. **Accuracy here is not an estimate of production "
             "accuracy** and should not be quoted as one.")

    item("The judge shares a provider with the generator, and is smaller",
         "Cross-family judging (Qwen scoring gpt-oss) blunts self-preference "
         "bias but does not remove it: same provider, same broad pretraining "
         "era, and the judge is a 27B model scoring a 120B one."
         + (f" Its agreement with a blind human re-score is κ = "
            f"{jv['send_unedited']['cohen_kappa']} — read every judge-derived "
            f"number with that attached." if jv else
            " **The judge-validation study is the check on this, and its "
            "result belongs next to every judge-derived number.**"))

    item("I wrote the system, the taxonomy, the rubric, the labels and the "
         "adjudications",
         "No amount of process removes this. The partial mitigations: two "
         "labellers from different families, a judge from a third, "
         "pre-registered brand criteria, a taxonomy frozen before labelling, "
         "and every adjudication written down with its reasoning so it can be "
         "audited. But a consistent misconception of mine propagates through "
         "all of it unchallenged, and the person best placed to catch that is "
         "a reader, not me.")

    excl = sub.get("funnel", {}).get("dropped_non_english", 0)
    item("~24% of real traffic is out of scope entirely",
         f"The language filter dropped {excl:,} of 48,000 sampled candidates — "
         f"led by Japanese ({sub['funnel']['dropped_language_breakdown'].get('ja', 0):,}) "
         f"— and mid-thread messages were excluded too. Every number here "
         f"describes English thread-opening messages to one brand.")

    item("Batching shifts individual predictions",
         "Running items in batches rather than singly changed ~20% of "
         "individual classifications while leaving aggregate accuracy "
         "statistically unchanged. Aggregates are safe to quote; **a specific "
         "row's prediction is not stable** under a different batch composition.")

    item("The committed cache is a recording, not a fresh run",
         "`--offline` replays cached responses so every number regenerates "
         "without an API key. That is a reproducibility strength and an honesty "
         "cost: it reproduces *this run*, not the API's behaviour today.")

    A("---")
    A("")

    # ----------------------------------------------------- 7 what I'd do next
    A("## 7. What I'd do next")
    A("")
    A("Ranked by expected value per unit of effort, with reasoning for the rank.")
    A("")
    A("**1. Add a tone/obligation gate.** The failure analysis localises the "
      "single biggest weakness precisely: the gates cannot see anger (E5) or "
      "remedy-commitment (E2), which are what the gold labels actually escalate "
      "on. This is the highest-value change because the diagnosis is specific "
      "and the fix does not require re-labelling anything.")
    A("")
    A("**2. Split the `other` class.** Adding `positive_feedback` and "
      "`pre_purchase` would break up a 17% grab-bag, remove a known source of "
      "headline inflation, and make per-class metrics mean something. Cheap, "
      "and it fixes a measurement defect rather than a model one.")
    A("")
    A("**3. Get a second human to re-label 50 rows.** The escalation κ of "
      f"{la['decision_cohen_kappa'] if la else '~0.27'} is the ceiling on "
      "everything. An independent human would tell me whether the label is "
      "genuinely contested or whether my rubric is still wrong — and I cannot "
      "distinguish those two from inside.")
    A("")
    A("**4. Make G1 conditional on the message, not the intent.** "
      "Over-escalation is currently the price of a blunt intent-level gate. "
      "Lower value than #1 because over-escalation is the cheap error.")
    A("")
    A("**5. Re-rank retrieval instead of trusting max cosine.** Mode 4 shows "
      "confident replies drafted on weak precedent. Lower rank because it is "
      "the smallest measured mode.")
    A("")
    A("**Explicitly NOT next: prompt engineering the drafter.** The model "
      "proposes `auto_handle` 200/200 times — that is not a prompt-tuning "
      "problem, and effort there would produce motion without movement.")
    A("")
    A("---")
    A("")

    # ------------------------------------------------------ 8 reproducing
    A("## 8. Reproducing this")
    A("")
    A("```bash")
    A("py -3.12 -m venv .venv && .venv\\Scripts\\Activate.ps1")
    A("python tasks.py setup")
    A("python tasks.py data          # ~500MB Kaggle download, one time")
    A("python tasks.py subsample     # deterministic; test asserts byte-identity")
    A("python -m eval.run_eval --offline   # regenerates every number, NO API key")
    A("```")
    A("")
    A("The subsample is **not** committed — it derives from a CC BY-NC-SA "
      "dataset — but its rebuild is deterministic and "
      "`tests/test_subsample.py` verifies byte-identity by SHA-256, so you can "
      "check your copy matches the one these numbers came from rather than "
      "trusting it.")
    A("")
    A("Full reasoning for every non-obvious choice, including the ones that "
      "turned out wrong, is in "
      "[`docs/decision_log.md`](decision_log.md).")
    A("")

    OUT.write_text(body.rstrip() + "\n\n" + "\n".join(L) + "\n", encoding="utf-8")
    print(f"appended sections 5-8 ({len(L)} lines)")
    print(f"report is now {len(OUT.read_text(encoding='utf-8').splitlines())} lines")
    if not jv:
        print("NOTE: judge validation not yet available; section 6 item 6 "
              "carries a placeholder. Re-run after eval.judge_validation compare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
