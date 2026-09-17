# ThreadPilot — Project Requirements

> Status: living document. Last substantive update: Phase 0.
> Audience: the Hiver evaluators, and any engineer picking this repo up cold.

---

## 1. The problem, restated in my own words

A customer support team receives a continuous stream of inbound public tweets. A human
agent currently has to read each one and answer three questions before they can act:

1. **What is this person actually asking for?** (intent)
2. **What should we say back?** — and critically, *what have we said before to people
   asking the same thing*, because a support org's value is consistency, not creativity.
3. **Can a machine safely send this, or does a human need to touch it?**

ThreadPilot is a triage agent that answers those three questions for a single brand. For
each incoming support tweet it produces:

- an **intent label** drawn from a taxonomy derived from that brand's real traffic,
- a **draft reply grounded in how this brand has historically resolved similar issues**
  (retrieved real precedent, not invented text),
- an **auto-handle vs. escalate decision with an explicit stated reason**.

The thing being graded is not the agent. It is the **evidence that the agent works**, and
the honesty of that evidence. The assignment brief says *"the proof is worth more than the
system."* Every scope decision in this repo resolves in favour of evaluation rigour over
capability. Where those conflict, capability loses.

## 2. Dataset and brand selection

**Dataset:** *Customer Support on Twitter* (Kaggle, `thoughtvector/customer-support-on-twitter`),
`twcs.csv`, ~2.81M tweets across ~100 brand support accounts. Columns: `tweet_id`,
`author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id`.

**We build for exactly one brand.** One brand is the correct scope because "grounded in how
*the brand* has historically resolved similar issues" is a per-brand claim — a retrieval
index pooled across brands would ground replies in the wrong company's policies, voice and
remedies.

### 2.1 Pre-registered selection criteria

I am writing these criteria down **before** running the EDA, so that brand choice is a
measurement and not a rationalisation. The brand must score well on all five:

| # | Criterion | Why it matters | How measured (Phase 1) |
|---|---|---|---|
| C1 | **Volume** — at least 20k brand replies | The retrieval index needs enough distinct precedent after dedup | row count where `inbound == False` |
| C2 | **Intent diversity** | A brand with one repeated complaint makes classification trivial and the eval meaningless | entropy over KMeans cluster assignment of embedded inbound messages |
| C3 | **In-thread resolution rate** | *The decisive criterion.* A brand whose replies are mostly "please DM us" offers nothing to ground a draft in | share of first brand replies matching channel-deflection patterns (DM, direct message, check your inbox) — we want this **low** |
| C4 | **Thread depth** | Need 2+ turn threads to observe what resolution actually looks like | distribution of reconstructed thread lengths |
| C5 | **Escalation signal present** | Need both plainly-automatable traffic (status, policy, how-to) and plainly-human traffic (refunds, account/security, safety, legal) or the auto/escalate task is degenerate | keyword + cluster inspection for both poles |

### 2.2 Stated hypothesis (recorded before Phase 1 ran)

I expect **AppleSupport** to fail C3 badly despite winning C1 — its public replies are
heavily channel-deflection ("Send us a DM and we'll take a look"). I expect the airlines
(**Delta**, **AmericanAir**, **British_Airways**) and **SpotifyCares** to score well on C3
and C5, because they resolve substantively in public and have a natural
automatable/escalate split. **Current leading candidate: an airline account.**

If the EDA falsifies this hypothesis I will say so in `docs/report.md` rather than quietly
re-writing the hypothesis. A falsified pre-registration is a finding, not an embarrassment.

### 2.3 OUTCOME — brand selected: `AmazonHelp`

*Filled in after Phase 1. The hypothesis above is left exactly as written.*

**The hypothesis was half right.**

| Prediction | Outcome |
|---|---|
| AppleSupport fails C3 on channel deflection | ✅ **Confirmed.** 28.5% bare channel switches; 8th of 12 overall despite being 2nd by volume |
| An airline wins | ❌ **Falsified.** Best airline (AmericanAir) placed 3rd; Delta 10th |

**Selected brand: `AmazonHelp`** — but *not* because it topped the composite score, and the
reason it was chosen matters more than the choice.

The composite score could not separate the top two. `scripts/brand_sensitivity.py` re-ranked
every candidate under four defensible definitions of "usable precedent", holding the scoring
weights fixed, and the winner **flipped**: strict and most-inclusive definitions picked
AmazonHelp, the project-default and rater-aligned definitions picked SpotifyCares. Reporting
"SpotifyCares scored highest" would therefore have been an artifact of one definitional
choice, not a finding.

AmazonHelp was selected on **robustness** instead: it is the only brand in the top 3 under
all four definitions (ranks 1, 2, 2, 1), and it carries 168,814 usable pairs against
SpotifyCares' 43,092 — a ~4x larger retrieval index, which bears directly on a
grounding-based system. See `decision_log.md` D25 and
`eval/results/brand_sensitivity.md`.

**The cost of that choice, stated up front:** AmazonHelp has the *lowest* intent-diversity
proxy among the leading brands (0.626 vs SpotifyCares' 0.731). The classification task will
be correspondingly easier and per-class metrics less interesting. I bought decision
robustness and index size at the price of task difficulty; this belongs in the report's
"what's misleading about my headline number" section, not in a footnote.

**Subsample** (`data/processed/subsample.parquet`, 2.8MB, seed 20260909 — rebuilt locally, not committed; see README):
8,000 pairs across 7,512 threads; 6,429 usable-and-canonical rows form the retrieval index
candidate pool. Reply composition: 44.6% self-contained, 27.0% link referral, 13.8%
fragment, 6.8% diagnostic ask, 3.2% bare channel switch, 2.6% bare acknowledgement, 2.1%
handoff with ask, 0.1% truncated.

### 2.4 How much to trust C3, the criterion that drove all of this

C3 comes from a reply classifier I wrote (`threadpilot.data.clean`). It was validated
against an independent LLM rater that was given the downstream definition of usable
precedent and no hint of the heuristic (`scripts/validate_reply_proxy.py`).

| Version | Raw agreement | Cohen kappa |
|---|---|---|
| v1 (5-way) | 59.2% | **0.201** |
| v2 (8-way) | 69.7% | **0.374** |

v1 was barely above chance and had to be rebuilt; the reasoning is in `decision_log.md`
D21–D24. v2 is *fair, not strong*. Per-class agreement is 0.83–0.92 wherever the definition
is unambiguous (`bare_ack` 0.92, `link_referral` 0.89, `diagnostic_ask` and
`handoff_with_ask` 0.83) and collapses on exactly two classes — `truncated` (0.12) and
`fragment` (0.47). Those two are a **definitional** divergence rather than classifier noise:
my classifier asks "is this a complete, well-formed reply?", the rater asks "does the visible
text contain help?" The sensitivity analysis above exists precisely because that
disagreement could not be resolved by further tuning, and the honest test was whether the
decision depended on it.

## 3. Users of this system

| User | What they need from it | How the design serves them |
|---|---|---|
| **Support agent** (primary) | Not to start from a blank box. A defensible first draft plus the precedent it came from, so they can verify in seconds rather than re-research | draft reply is always shown *next to* the retrieved precedents that justify it |
| **Team lead** | Confidence that "auto-handle" means safe-to-send, and that escalations arrive with a reason attached | decision always carries a stated reason; hard safety gates are deterministic and auditable, not model whim |
| **Hiver evaluators** (the actual audience) | To verify claims cheaply and to find where I am wrong before I do | committed golden set, committed eval results, committed LLM response cache so metrics reproduce with **no API key**, and an explicit "what's misleading about my headline number" section |

The third row is the real customer of this repo. Design decisions that help an evaluator
audit the work beat decisions that make the demo look better.

## 4. Core features, mapped 1:1 to the assignment's three requirements

### R1 — Intent classification
- A taxonomy of ~6-9 intents **derived from the data** (embed, cluster, human-name, merge),
  not guessed a priori. The derivation is documented and the intermediate cluster
  exemplars are committed so the naming step is auditable.
- Classifier emits an intent plus a self-reported confidence.
- Evaluated against 200 hand-reviewed labels: accuracy, macro-F1, per-class P/R/F1,
  confusion matrix, bootstrap 95% CIs (mandatory at this n — a bare point estimate on
  ~25 examples per class would be dishonest).

### R2 — Grounded reply drafting
- Retrieval index of **(customer message, brand reply) pairs** from resolved historical
  threads for the chosen brand only.
- Top-k retrieved pairs are injected into the drafting prompt as precedent, with an
  explicit instruction not to assert anything the precedent does not support.
- **Leakage control:** every thread appearing in the golden set is excluded from the
  retrieval index. Without this the system would retrieve the literal target reply and the
  eval would be meaningless. This is the single most important correctness detail in the
  whole harness.
- Evaluated by LLM-as-judge on groundedness / relevance / tone / actionability / safety,
  plus a hard binary "would a human send this unedited?" — and the judge itself is
  validated against human re-scoring.

### R3 — Auto-handle vs. escalate, with a stated reason
- Hybrid decision: the LLM proposes `auto_handle` or `escalate` **with a reason string**,
  and a deterministic policy layer may override to `escalate` on hard gates
  (always-escalate intents, retrieval similarity below threshold, account-specific/PII
  requests, legal or safety language).
- Justified by **cost asymmetry**: a wrong "escalate" costs an agent thirty seconds; a
  wrong "auto-handle" sends a customer a wrong or unsafe answer in the brand's name. The
  rule layer makes that asymmetry auditable and tunable instead of leaving it to sampling
  temperature.
- Thresholds are tuned on a **separate silver dev set**, never on the golden set, so the
  headline number is not a tuned number. (See `docs/architecture.md` section 6.)
- Reported as a **coverage/risk curve**, not one number: at each auto-handle coverage
  level, what fraction of auto-handled replies are actually acceptable?

## 5. Explicit non-goals — what I chose not to build

Captured now, while the reasoning is fresh, because "problem framing" is graded.

1. **Multi-brand generalisation.** Grounding is a per-brand claim (section 2). Out of scope.
2. **Full 2.81M-row processing.** A committed, seeded subsample of one brand reproduces in
   minutes. Scaling is an engineering exercise that would buy zero evaluation insight.
3. **Fine-tuning.** Zero-cost constraint, and it would trade away the auditability of
   prompt+retrieval for a marginal metric gain I could not honestly attribute.
4. **A vector database.** The index is ~10^4 x 384 floats. Exact brute-force cosine in
   numpy is faster than a FAISS build and has no dependency risk. Adding Chroma would be
   resume-driven engineering.
5. **Multi-turn conversational agent.** The unit of work is *triage of an inbound message*,
   which is a single-turn decision. Threads are reconstructed to *mine precedent*, not to
   hold a live conversation.
6. **Actually sending anything / any live integration.** No side effects.
7. **A polished UI.** CLI + notebook only (see `docs/design.md`). Effort redirected to
   failure analysis and judge validation.
8. **Sentiment / urgency / language detection as separate models.** Urgency is folded into
   the escalation reason where it matters. Extra heads would be extra unvalidated surface.
9. **PII redaction beyond the dataset's own anonymisation.** The source already replaces
   handles with `@<number>`. I detect account-specific requests as an escalation *signal*
   but do not claim a compliance-grade redaction pipeline.
10. **Beating a leaderboard.** There is no leaderboard. The deliverable is a defensible
    measurement of a modest system, including where it fails.

## 6. Definition of "good" for this brand

A trustworthy auto-reply from this system must satisfy **all** of the following. These are
the criteria the LLM judge rubric operationalises, and they were written before the rubric.

1. **Grounded** — every factual or procedural claim traces to a retrieved precedent or to
   the customer's own message. No invented policies, URLs, phone numbers, timeframes, or
   compensation amounts.
2. **On-intent** — answers what was actually asked, not an adjacent easier question.
3. **Non-committal about things it cannot commit to** — never promises a refund, credit,
   compensation, or a specific resolution time unless precedent shows the brand routinely
   does so for that intent.
4. **In voice** — matches the brand's observed register (length, formality, apology
   pattern) as seen in precedent, not generic-chatbot English.
5. **Actionable** — leaves the customer with a next step, even when that step is "a
   specialist will follow up".
6. **Safe to send unedited** — the binary that matters. If an agent would have to rewrite
   it, "auto-handle" was the wrong call regardless of how good the prose was.

**And the system-level definition of good:** on the auto-handled slice, criterion 6 holds
at a high rate, *and* the escalated slice genuinely contains the hard cases. A system that
achieves safety by escalating everything is worthless, which is precisely why the trivial
baseline in Phase 5 is `always-escalate` — it exists to make that failure mode impossible
to hide behind.

## 7. Constraints (from the brief)

- Zero cost: Groq free tier for generation and judging; `all-MiniLM-L6-v2` run locally for
  embeddings. No paid API, no paid infra.
- Everything inside a `.venv`; pinned `requirements.txt`; nothing installed globally.
- Reproducible in **under 15 minutes** on the committed subsample, fixed seed.
- No secrets committed; `.env` gitignored, `.env.example` provided.
- Every borrowed snippet, prompt pattern or dataset transformation cited in `CITATIONS.md`.
