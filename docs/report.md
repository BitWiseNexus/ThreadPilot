# ThreadPilot — evaluation report

*Generated 2026-09-17 from `eval/results/*.json` by `scripts/write_report.py`. Brand **AmazonHelp**, seed `20260909`, n=200 hand-reviewed golden rows.*

---

## The headline, with its caveats attached

**Intent classification: 80.5% [75.0%, 86.5%] accuracy, macro-F1 0.790** against a best-baseline 36.0% — a gap of **+44.5% points** with non-overlapping confidence intervals.

**Escalation: this is where the system is weak.** It auto-handles 62.0% of messages, and **41.1% [33.1%, 49.2%] of those should have gone to a human.**

Three things make the accuracy figure look better than it is, all quantified in [§6](#6-whats-misleading-about-my-headline-number):

1. A 17% `other` grab-bag scored 100% inflates it by **4.0% points** — real accuracy on coherent classes is 76.5%.
2. It **exceeds the 75.0% inter-labeller ceiling**, which is a warning sign rather than an achievement: the classifier reads the same boundary notes I adjudicated the gold labels with.
3. The gold labels carry **~15% measured residual noise**.

---

## 1. Problem framing

A support team gets a stream of public tweets. Before acting, an agent must answer three questions: *what is this person asking for*, *what have we said to people who asked the same thing*, and *can a machine safely send this*. ThreadPilot answers all three for one brand.

**Scope decisions that shaped everything downstream**, each with its cost stated:

| decision | cost |
|---|---|
| One brand (`AmazonHelp`) | Grounding is a per-brand claim; results do not transfer |
| Thread-opening messages only | Half the pairs discarded; no follow-up handling (was already non-goal #5) |
| English only | **~24% of real traffic excluded**, led by Japanese (3,738 of 48k sampled) |
| 8,000-pair subsample | May not represent the full 2.81M rows |

The brand was chosen by criteria **pre-registered before the EDA**. The recorded hypothesis was half wrong — AppleSupport did fail on channel deflection, but "an airline wins" was false — and that is reported rather than rewritten. The composite score could not separate the top two brands at all; a 4-definition sensitivity analysis showed the winner flipping, so `AmazonHelp` was chosen on **robustness**, being the only brand top-3 under every definition.

## 2. What was built

```
inbound tweet
  → classify        intent + confidence          (LLM, batched)
  → retrieve        top-k precedent              (local, exact cosine)
  → draft + propose reply + model's own verdict  (LLM, batched)
  → gates           deterministic, escalate-only (no LLM)
```

The retrieval index holds **6,155 (message → reply) pairs**, down from 6,390 eligible after the leakage guard removed 235 rows belonging to held-out threads.

**The leakage guard is the load-bearing correctness detail.** Golden-set *threads* (not just rows) are excluded from the index. Without it the retriever surfaces the exact reply a golden row was built from, the drafter copies it, and every quality metric is inflated to the point of fraud — with no error message. It is asserted at build time, re-asserted against the rows being scored, and covered by a test.

## 3. How it was evaluated

**Golden set: 200 rows**, sampled *before* the pipeline existed so it could not be built to the test.

| property | value |
|---|---|
| individually reviewed | **153/200** |
| accepted on labeller agreement | 47 |
| **measured residual label noise** | **~15%** |
| inter-labeller κ (intent) | 0.721 |
| inter-labeller κ (auto/escalate) | **0.273** |

**The single most important number in this report is that 0.273.** Two capable models from different families barely agreed on whether a message is safe to automate — and not randomly: gpt-oss escalated 34.3% of messages, Qwen 73.8%. On 63 rows they picked the *same intent* and the *opposite decision*.

That caps the whole exercise. **Any classifier evaluated against escalation labels inherits that ceiling**, and a headline escalation accuracy would largely measure which labeller I anointed. It also exposed that my first rubric was too loose ("needs private data" is true of nearly every support message), which I rewrote around the *reply* rather than the message.

**Two anti-circularity rules.** Labellers were never told the escalation policy — otherwise the gold labels would restate gate G1 and measuring G1 against them would be circular. And they never saw the brand's actual reply, which would leak the answer into the label.

## 4. Results

| system | accuracy | macro-F1 | coverage | false auto-handle | send unedited |
|---|---|---|---|---|---|
| `trivial` | 11.0% [7.0%, 15.0%] | 0.018 | 0.0% | 0.0% [0.0%, 0.0%] | 1.0% |
| `simple_tfidf_silver` | 30.0% [24.0%, 36.0%] | 0.263 | 58.5% | 53.0% [43.6%, 61.5%] | 15.0% |
| `simple_tfidf_cluster` | 36.0% [29.5%, 43.0%] | 0.332 | 82.0% | 54.3% [46.3%, 62.8%] | 17.0% |
| `retrieval_1nn` | 36.0% [29.5%, 43.0%] | 0.332 | 78.5% | 56.0% [48.4%, 63.1%] | 48.0% |
| `pipeline_no_retr` | 80.5% [75.0%, 86.5%] | 0.790 | 0.0% | 0.0% [0.0%, 0.0%] | 52.0% |
| `pipeline_no_gates` | 80.5% [75.0%, 86.5%] | 0.790 | 100.0% | 55.0% [48.0%, 61.0%] | 78.0% |
| **`pipeline`** | 80.5% [75.0%, 86.5%] | 0.790 | 62.0% | 41.1% [33.1%, 49.2%] | 78.0% |

*`coverage` = share auto-handled. `false auto-handle` = of those, the share that should have gone to a human — conditioned on the auto-handled slice, which is the operationally meaningful denominator. `send unedited` is the LLM judge's binary, on a paired 100-row subsample.*

### The trivial baseline proves the metric is gameable

`trivial` scores **0.0% false auto-handle** — a perfect safety score — at **0.0% coverage**, with 11.0% accuracy. It is simultaneously the safest and the most useless system here.

This is why auto-handle is reported as a coverage/risk curve rather than a single number. The argument is now a measurement, not a claim.

### The gates supply 100% of the escalation capability

**The model proposed `auto_handle` for all 200 messages.** It never once proposed escalating. Its self-assessment is worthless here.

| | coverage | false auto-handle | escalate recall |
|---|---|---|---|
| gates OFF | 100.0% | 55.0% | 0.0% |
| gates ON | 62.0% | 41.1% | 53.6% |

The deterministic rule layer earns its place — but note it only moves false auto-handle from 55.0% to 41.1%. Auto-handling everything scores 55%, so the gates buy a real but modest improvement, at a large cost in coverage.

### Both halves are necessary — the ablation is decisive

| configuration | send unedited | grounded |
|---|---|---|
| **pipeline** (LLM + retrieval) | **78%** | **4.51** |
| `pipeline_no_retr` (LLM alone) | 52% | 3.59 |
| `retrieval_1nn` (retrieval alone) | 48% | 3.81 |

Retrieval adds **+26% points** over the LLM alone; generation adds **+30% points** over retrieval alone. **Neither component is close on its own.** Groundedness rises 3.59 → 4.51 when precedent is supplied, which is exactly what retrieval is for.

*Read these with §4b attached: the judge is systematically generous, so the levels are inflated even if the gaps are not.*

### What generation adds over copying

`retrieval_1nn` returns the nearest historical reply **verbatim** — no generation at all — and the judge rates it sendable **48.0%** of the time, with the highest `voice` score of any system (4.25). That is the bar generation has to clear to justify itself.

---

## 4b. Validating the judge — and it failed

An LLM-judged quality number without an agreement statistic is an unvalidated claim, so 40 pipeline replies were **re-scored blind** against the judge. The worksheet withholds the judge's scores by construction: scoring while able to see them would measure deference, not agreement.

| | value |
|---|---|
| raw agreement on `send_unedited` | 70.0% |
| **Cohen κ** | **0.3333** |
| judge says sendable (these 40) | 75.0% |
| blind rater says sendable (these 40) | 60.0% |

*Note the denominator: these rates are over the **40 validated rows**, not the 100-row judged sample in §4. The judge's rate on the full 100 is 78.0%; on this 40-row subset it is 75.0%. The gap to the blind rater (+15.0%) is the quantity of interest, and it is measured on identical rows.*

**WEAK - judge-derived quality numbers should not carry the report**

The bias is systematic, not noisy — the judge is more generous on **every** criterion:

| criterion | quadratic-weighted κ | judge bias |
|---|---|---|
| grounded | 0.3043 | +0.90 |
| on_intent | 0.3968 | +1.07 |
| no_overcommit | 0.4565 | +0.45 |
| voice | 0.1127 | +0.97 |
| actionable | 0.3861 | +0.97 |

### The failure mode is diagnosable

Reading the 12 disagreements, the judge justifies its scores by **form** — *"mirrors precedent"*, *"standard support link"*, *"matches precedent for security issues"* — while the blind rater judged **substance** — *"answers nothing"*, *"never answers whether this channel is right"*, *"already described the fault"*.

**The judge cannot detect a well-formed reply that fails to do its job.** It rated a reply to a customer reporting a *break-in with police involved* as sendable because it "matches precedent for security/delivery issues". The reply was: *"Sorry about this. Please reach out to us directly here."*

### What this costs, and what survives

**Costs:** the 75% send-unedited figure cannot carry a headline. It is quoted with this κ attached and with the stricter blind rate (60%) beside it. Had this study not been run, the report would have overstated reply quality by 15% points in the flattering direction.

**Survives:** the *relative* ordering is probably intact. If the judge simply rewarded precedent-conformity, `retrieval_1nn` — which **is** precedent, copied verbatim — would have scored highest. It did not. So the bias plausibly shifts levels rather than ranks. That is an inference, not a measurement, and is labelled as one.

**Caveat on the rater.** The blind re-scorer was the AI assistant that built this repo, not an independent human annotator — weaker evidence than "human validation" usually implies. Every score carries a written justification in `eval/results/judge/human_scores.jsonl` so the reasoning is auditable rather than taken on trust.

---

## 5. Failure analysis

Five modes, **ordered by harm rather than frequency**: a wrong auto-handle publishes a wrong answer under the brand's name; a wrong escalate costs an agent thirty seconds.

| # | mode | count | severity |
|---|---|---|---|
| 1 | False auto-handle (the dangerous failure) | 51/124 | HIGH |
| 2 | Over-escalation (costly, not dangerous) | 17/76 | LOW |
| 3 | Adjacent-intent confusion | 39/200 | MEDIUM |
| 4 | Drafting with weak precedent | 10/20 | MEDIUM |
| 5 | EVALUATION failure: a 17% grab-bag inflates the headline | 34/34 | N/A |

### Mode 1 has a clean diagnosis, and it is not a tuning problem

The 51 false auto-handles concentrate in `delivery_failure` (11), `unresolved_followup` (9), `delivery_delay` (8) — **none of which are in `ALWAYS_ESCALATE`**.

The gold labels escalated them under rubric tests **E5** (anger an automated reply would worsen) and **E2** (a reply that would have to commit a remedy). My gates read intent, precedent similarity, confidence and explicit identifiers. **None of them reads tone or obligation.** So this is a *missing gate*, not a mis-tuned one — and prompt changes cannot help while the model's own proposal is constant at `auto_handle`.

### A prediction that held

Phase 3 recorded which intent pairs the two labellers confused, **before the classifier existed**. The classifier's confusions match: `delivery_failure -> delivery_delay` (4), `prime_membership -> delivery_delay` (4), `service_complaint -> other` (3). It fails where the taxonomy boundary is genuinely contested, not arbitrarily — so much of this is irreducible against these labels rather than a model deficiency.

Full detail with verbatim examples: [`eval/results/failure_analysis.md`](../eval/results/failure_analysis.md).

## 6. What's misleading about my headline number

Written to undermine my own result. Every item carries a magnitude where one can be measured — a caveat without a number is decoration.

**1. A 17% grab-bag inflates accuracy by 4.0% points**

`other` is the largest gold class (34/200) and the classifier scores 100% on it — but it is not one thing. It holds praise, pre-purchase questions, checkout problems and third-party faults. Scoring perfectly on a bag of unlike things is not a capability. Accuracy on coherent classes is **76.5%**, not 80.5%.

**2. The score exceeds the 75.0% inter-labeller ceiling, which should worry you**

Two independent labellers agreed on intent only 75.0% of the time, yet the classifier scores higher against the adjudicated labels. The likely cause is **methodological coupling**: the classifier prompt carries the same boundary notes I used when adjudicating those labels, so it is aligned with my adjudication rules and not only with the task. I cannot cleanly separate the two effects, and I have not tried to present this as evidence of quality.

**3. The gold labels carry ~15% measured residual noise**

An audit of 40 rows where both labellers agreed found 6 where **both were wrong together** — a failure agreement cannot detect. That rate applies to the 47 rows nobody inspected individually. Any metric computed here is bounded by it.

**4. The escalation label is barely reproducible (κ = 0.273)**

Two capable models disagreed systematically on whether a message is safe to automate — gpt-oss escalated 34.3%, Qwen 73.8%. Every escalation number in this report is measured against a target that two reasonable raters could not agree on. This is the single biggest limitation of the work.

**5. The golden set is deliberately NOT distribution-matched**

Clusters were allocated equally, `account_security` was oversampled ~8x, and hard cases (sarcasm, order IDs, novel messages) were **sought rather than avoided**. This makes the set harder than production traffic in some respects and easier in others. **Accuracy here is not an estimate of production accuracy** and should not be quoted as one.

**6. The judge shares a provider with the generator, and is smaller**

Cross-family judging (Qwen scoring gpt-oss) blunts self-preference bias but does not remove it: same provider, same broad pretraining era, and the judge is a 27B model scoring a 120B one. Its agreement with a blind human re-score is κ = 0.3333 — read every judge-derived number with that attached.

**7. I wrote the system, the taxonomy, the rubric, the labels and the adjudications**

No amount of process removes this. The partial mitigations: two labellers from different families, a judge from a third, pre-registered brand criteria, a taxonomy frozen before labelling, and every adjudication written down with its reasoning so it can be audited. But a consistent misconception of mine propagates through all of it unchallenged, and the person best placed to catch that is a reader, not me.

**8. ~24% of real traffic is out of scope entirely**

The language filter dropped 11,419 of 48,000 sampled candidates — led by Japanese (3,738) — and mid-thread messages were excluded too. Every number here describes English thread-opening messages to one brand.

**9. Batching shifts individual predictions**

Running items in batches rather than singly changed ~20% of individual classifications while leaving aggregate accuracy statistically unchanged. Aggregates are safe to quote; **a specific row's prediction is not stable** under a different batch composition.

**10. The committed cache is a recording, not a fresh run**

`--offline` replays cached responses so every number regenerates without an API key. That is a reproducibility strength and an honesty cost: it reproduces *this run*, not the API's behaviour today.

---

## 7. What I'd do next

Ranked by expected value per unit of effort, with reasoning for the rank.

**1. Add a tone/obligation gate.** The failure analysis localises the single biggest weakness precisely: the gates cannot see anger (E5) or remedy-commitment (E2), which are what the gold labels actually escalate on. This is the highest-value change because the diagnosis is specific and the fix does not require re-labelling anything.

**2. Split the `other` class.** Adding `positive_feedback` and `pre_purchase` would break up a 17% grab-bag, remove a known source of headline inflation, and make per-class metrics mean something. Cheap, and it fixes a measurement defect rather than a model one.

**3. Get a second human to re-label 50 rows.** The escalation κ of 0.2727 is the ceiling on everything. An independent human would tell me whether the label is genuinely contested or whether my rubric is still wrong — and I cannot distinguish those two from inside.

**4. Make G1 conditional on the message, not the intent.** Over-escalation is currently the price of a blunt intent-level gate. Lower value than #1 because over-escalation is the cheap error.

**5. Re-rank retrieval instead of trusting max cosine.** Mode 4 shows confident replies drafted on weak precedent. Lower rank because it is the smallest measured mode.

**Explicitly NOT next: prompt engineering the drafter.** The model proposes `auto_handle` 200/200 times — that is not a prompt-tuning problem, and effort there would produce motion without movement.

---

## 8. Reproducing this

```bash
py -3.12 -m venv .venv && .venv\Scripts\Activate.ps1
python tasks.py setup
python tasks.py data          # ~500MB Kaggle download, one time
python tasks.py subsample     # deterministic; test asserts byte-identity
python -m eval.run_eval --offline   # regenerates every number, NO API key
```

The subsample is **not** committed — it derives from a CC BY-NC-SA dataset — but its rebuild is deterministic and `tests/test_subsample.py` verifies byte-identity by SHA-256, so you can check your copy matches the one these numbers came from rather than trusting it.

Full reasoning for every non-obvious choice, including the ones that turned out wrong, is in [`docs/decision_log.md`](decision_log.md).

