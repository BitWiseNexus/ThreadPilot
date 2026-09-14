# Golden set - labelling notes

*Generated from measured artifacts by `scripts/write_labelling_notes.py`. Do not edit by hand.*

Brand **AmazonHelp**, seed `20260909`, 200 golden rows and 100 dev-silver rows, thread-disjoint.

---

## 1. Sampling strategy

The golden set was sampled **before** any labelling and before the pipeline existed, so the pipeline cannot be built to the test.

### The chicken-and-egg problem

The plan called for stratification "across intents", but intent labels are what the golden set exists to produce. Stratifying on them would require already having them. So strata are the Phase 2 **cluster assignments** - a data-derived proxy that was available at sampling time. The consequence, reported rather than smoothed over: realised per-INTENT counts do not match per-CLUSTER targets, because the human naming step merged clusters 0 and 1 and because `account_security` has no cluster of its own.

### Three pools, deliberately not proportional

| stratum | n |
|---|---|
| `cluster_0` | 13 |
| `cluster_1` | 13 |
| `cluster_2` | 13 |
| `cluster_3` | 13 |
| `cluster_4` | 13 |
| `cluster_5` | 13 |
| `cluster_6` | 13 |
| `cluster_7` | 13 |
| `cluster_8` | 13 |
| `cluster_9` | 13 |
| `account_security` | 20 |
| `order_id` | 10 |
| `very_short` | 10 |
| `multi_intent` | 10 |
| `novel` | 10 |
| `sarcasm` | 10 |

1. **Cluster strata** - equal allocation, not proportional. Proportional sampling gives the rarest classes too few examples to measure, and per-class F1 on five examples is noise.
2. **`account_security`** - targeted pool, oversampled roughly 8x (2.5% of traffic to ~10% of the set). It is the highest-stakes intent and unmeasurable at its natural rate.
3. **Hard cases** - deliberately sought rather than avoided. A golden set drawn only from cluster centres measures the easy middle and reports it as overall performance.

### Hard cases present in the final set

| kind | n | why it is hard |
|---|---|---|
| `very_short` | 17 | little signal to classify from |
| `order_id` | 22 | requires private account data to answer at all |
| `account` | 21 | highest-stakes intent; must never be auto-handled |
| `sarcasm` | 11 | surface sentiment inverts the literal reading |
| `multi_intent` | 12 | three or more intent signals in one message |
| `novel` | 19 | bottom 5% by nearest-neighbour similarity - retrieval has nothing close to ground a reply in |

### The set is deliberately NOT distribution-matched

| cluster | true share | golden share |
|---|---|---|
| c0 | 6.6% | 8.5% |
| c1 | 9.8% | 11.0% |
| c2 | 11.6% | 9.0% |
| c3 | 12.4% | 8.0% |
| c4 | 9.1% | 11.5% |
| c5 | 9.3% | 13.0% |
| c6 | 11.3% | 8.0% |
| c7 | 10.0% | 11.0% |
| c8 | 10.0% | 9.5% |
| c9 | 9.8% | 10.5% |

**This biases every headline number computed on it.** Accuracy here is not an estimate of production accuracy: rare classes and hard cases are over-represented on purpose, so the set is harder than real traffic in some respects and easier in others. Carried into the report's "what is misleading about my headline number" section.

---

## 2. Labelling process

Two independent labellers from **different model families** - `openai/gpt-oss-120b` (A) and `qwen/qwen3.8-27b` (B).

### Two anti-circularity rules

1. **Labellers were not told the escalation policy.** They received intent definitions and boundary notes but never dispositions. Had they been told "refund_billing is always escalate", the auto/escalate labels would restate gate G1, and later measuring G1 against them would be circular - the gate would score perfectly by construction.
2. **Labellers never saw the brand's actual reply.** They saw only the customer message, exactly what the pipeline sees at inference. Showing the historical reply would leak the answer into the label and import the brand's own mistakes as ground truth.

### Inter-labeller agreement

| | raw agreement | Cohen kappa |
|---|---|---|
| intent (16 strata, 11 classes) | 75.0% | **0.721** |
| auto vs escalate | 57.0% | **0.273** |

113 of 200 rows disagreed on intent or decision and went to adjudication.

**This statistic is itself a finding.** It measures how hard the labelling task is, which a single-labeller process cannot reveal. A modest kappa means the taxonomy's boundaries are genuinely contested in places, and any classifier evaluated against these labels inherits that ceiling.

### Where the labellers disagreed most

| intent pair | n |
|---|---|
| other <-> service_complaint | 5 |
| delivery_delay <-> prime_membership | 5 |
| delivery_delay <-> order_investigation | 3 |
| delivery_delay <-> delivery_failure | 3 |
| other <-> product_digital | 3 |
| refund_billing <-> service_complaint | 3 |
| delivery_failure <-> order_investigation | 2 |
| prime_membership <-> refund_billing | 2 |
| delivery_delay <-> unresolved_followup | 2 |
| other <-> prime_membership | 2 |

These pairs are the taxonomy's weakest boundaries, identified **before** the classifier was built - so Phase 6's confusion matrix can be checked against a prediction rather than explained after the fact.

---

## 3. Review, stated precisely

### Who reviewed, in plain terms

**The adjudicator was the AI assistant working on this repo, not an independent human annotator.** Every adjudication was written against the explicit rubric in `src/threadpilot/escalation.py` and the taxonomy boundary notes, and each carries a written justification in `eval/golden/adjudications.jsonl` naming the rule applied - so the reasoning is auditable line by line rather than taken on trust.

That is a real limitation and is not dressed up as anything else: the same system drafted the labels, designed the taxonomy, wrote the rubric and adjudicated the disagreements, so a consistent misconception propagates through all four unchallenged. Two partial mitigations: the drafts came from two different model families, and the adjudicator disagreed with BOTH of them often (on the 63 same-intent/opposite-decision rows it split 32 auto / 31 escalate, where labeller A had said auto on all 63 and labeller B escalate on all 63).

**The repo owner should spot-check a sample** - `eval/golden/adjudications.jsonl` is human-readable and each entry states its reasoning.

Claiming blanket human review of 200 rows would be easy to write and impossible to verify. The enforced process instead:

| provenance | n |
|---|---|
| `human_adjudicated_disagreement` | 113 |
| `both_labellers_agreed` | 47 |
| `human_confirmed_agreement` | 34 |
| `human_override_of_agreement` | 6 |

* **Every labeller disagreement was adjudicated.** `finalize_golden.py` refuses to emit a golden set while any disagreement lacks an explicit written adjudication.
* **A seeded random sample of 40 AGREED rows was also reviewed.** This is the part that matters: two LLMs from different families can agree and both be wrong, and their agreement then looks like confirmation. Random rather than confidence-ranked, because sampling the least-confident rows would bias the estimate upward and make it useless for the rest.
* **Measured residual label noise: 15.0%** (6 overrides in 40 audited rows). That is the estimated error rate in the 47 rows accepted on agreement without individual inspection.

Total human overrides: 119. 124 rows carry `ambiguity_flag`.

### Final composition

| intent | n |
|---|---|
| `other` | 34 |
| `delivery_delay` | 22 |
| `order_investigation` | 21 |
| `service_complaint` | 20 |
| `delivery_failure` | 20 |
| `refund_billing` | 19 |
| `account_security` | 15 |
| `product_digital` | 15 |
| `unresolved_followup` | 12 |
| `item_condition` | 11 |
| `prime_membership` | 11 |

Decisions: {'escalate': 110, 'auto_handle': 90}

---

## 4. Label noise in the SOURCE data

Distinct from label noise in this set, and worth stating plainly because it caps what any evaluation here can mean.

* **The brand's historical reply is a reference, not ground truth.** Some of what AmazonHelp actually sent is unhelpful, deflecting, or truncated mid-sentence. `reference_reply` is kept for comparison, never treated as a ceiling a good reply must match. A draft that beats the historical reply is not penalised for differing from it.
* **Roughly 16% of reference replies are unusable as precedent** (fragments, bare acknowledgements, bare channel switches) by the classifier in `threadpilot.data.clean`, which itself agrees with an independent rater at only kappa 0.374.
* **The source contains genuine encoding damage.** U+FFFD appears in ~0.004% of texts - flagged via a `mojibake` column, never silently repaired, since guessing the original character would be fabrication.
* **Tweets are truncated by the platform**, so some messages end mid-clause and are genuinely ambiguous to a human reader too.
* **~24% of real traffic is excluded** as non-English (Japanese largest), so these labels describe English traffic only.

---

## 4b. A taxonomy defect this labelling exposed

`other` finished as the **largest class** - larger than any named intent. That is not a residual behaving normally; inspecting it shows two coherent intents the taxonomy lacks:

* **praise / positive feedback** - "Absolutely fantastic customer service ... Bravo", "Fantastic service from Daniela". The taxonomy was derived from clusters of a complaint-dominated corpus, so satisfied customers have nowhere to go. Both labellers repeatedly put these in `service_complaint`, which is defined as *dissatisfaction* - a real error caught during adjudication.
* **pre-purchase / product questions** - "When will amazon restock this item?", a camera whose wifi does not work, checkout and promo queries. These concern products rather than an existing order.

The taxonomy was deliberately frozen before labelling (D12, D14), and re-deriving it now would let it be fitted to the evaluation data. So the defect is reported rather than patched, and it is the highest-value single change in the report's "what I would do next".

**Consequence for the metrics:** `other` is a 17% catch-all, so the majority-class baseline is inflated and per-class scores for `other` measure a bag of unlike things.

---

## 5. Known weaknesses of this golden set

1. **Not distribution-matched** (section 1) - accuracy here does not estimate production accuracy.
2. **Drafted by LLMs.** Reviewed per section 3, but rows marked `both_labellers_agreed` were not individually inspected; their error rate is estimated, not zero.
3. **Both labellers share a provider** (Groq) and a broad pretraining era. Different families reduce correlated error; they do not eliminate it.
4. **n=200 across 11 classes** is roughly 18 per class, so per-class confidence intervals are wide. Every proportion in the report carries a bootstrap 95% CI for this reason.
5. **One author.** The same person designed the taxonomy, wrote the labelling prompt and adjudicated disagreements, so consistent misconceptions propagate through all three unchallenged.

*Generated 2026-09-14 12:22:07.*
