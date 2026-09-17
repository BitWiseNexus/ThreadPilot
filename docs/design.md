# ThreadPilot — Design

> Status: living document. Last substantive update: Phase 0.

---

## 1. There is no UI

**Decided in Phase 0: no Streamlit app, no web frontend. CLI + one EDA notebook only.**

The brief states the proof is worth more than the system. A demo UI would consume time that
belongs to failure analysis and judge validation, and it would not produce a single number
an evaluator could check. If Phases 1-7 land with slack, a minimal Streamlit view is
reconsidered in Phase 8 — but it is explicitly not planned.

Consequently this document does the other job: it defines **how CLI and notebook output are
formatted**, so a reviewer skimming results can read them without reverse-engineering the
code, and how the eval report presents numbers.

## 2. CLI output conventions

Rendered with `rich`. The governing principle: **a reviewer must be able to see the evidence
next to the claim.** A draft reply shown without the precedent it came from is unverifiable,
so the two are never separated.

### 2.1 Single-tweet triage (`python -m threadpilot.cli triage "..."`)

```
┌─ INBOUND ─────────────────────────────────────────────────────────────────┐
│ @115712  2017-10-31 14:02                                                 │
│ my flight got cancelled and nobody has rebooked me, ive been on hold 2hrs  │
└───────────────────────────────────────────────────────────────────────────┘

INTENT     flight_disruption            confidence 0.91
DECISION   ESCALATE                     gate G1 (always-escalate intent)

┌─ PRECEDENT (top 3 of 5 retrieved) ───────────────────────────────────────┐
│ [0.87] "cancelled my flight and no rebooking options offered"             │
│      -> "We're sorry — please DM us your record locator and we'll get     │
│         you on the next available flight."                                │
│ [0.81] "stuck at ORD, flight cancelled, 3 hour hold time"                 │
│      -> "Apologies for the wait. Our team can rebook you — send us your   │
│         confirmation number by DM."                                       │
│ [0.79] ...                                                                │
└───────────────────────────────────────────────────────────────────────────┘

┌─ DRAFT ───────────────────────────────────────────────────────────────────┐
│ We're really sorry about the cancellation and the hold time. Send us your  │
│ confirmation number and we'll look at the next available options for you.  │
└───────────────────────────────────────────────────────────────────────────┘

REASON  Rebooking requires access to the booking record and may involve
        compensation; precedent shows this is always handled by an agent.
```

Rules:

- **Similarity scores are always printed** next to each precedent. A retrieval claim without
  its score is not inspectable.
- **The firing gate is always named** (`G1`...`G5`), not paraphrased. When no gate fires and
  the LLM chose `auto_handle`, that is printed as `gate none (LLM proposal accepted)` so the
  two decision sources are never conflated.
- **Colour carries exactly one meaning: the decision.** Neutral grays for all structure;
  one accent for `AUTO-HANDLE` (green), one for `ESCALATE` (amber). Amber, not red —
  escalation is the safe, expected outcome for a large share of traffic, and red would
  frame correct caution as failure. Confidence and similarity are printed as numbers, never
  colour-coded, so colour never competes for attention with the decision.
- **No spinners or progress animation in piped output.** `--plain` disables all styling for
  clean redirection to a file, because an evaluator will want to diff runs.

### 2.2 Batch runs

Batch output is JSONL, one `TriageResult` per line, plus a `rich` summary table at the end.
JSONL because it diffs cleanly, streams, and survives interruption. Human-readable tables
are a *view* of committed JSON, never the source of a number.

### 2.3 Logging

`INFO` to stderr, results to stdout — so `> results.jsonl` stays clean. Every LLM call logs
model, cache hit/miss, and latency at `DEBUG`. Cache hit rate is printed at the end of every
batch run, because a run that was 100% cache is a replay and the operator must know that.

## 3. Notebook conventions

- Notebooks are for **exploration only**. Every number that reaches the report is produced
  by a script under `scripts/` or `eval/`, never by a notebook cell. This keeps the
  reproducibility promise honest — a notebook with stale cell outputs is not evidence.
- `notebooks/01_eda.ipynb` is committed **with outputs** so the brand-selection evidence is
  visible without running anything.
- Every figure gets an axis-labelled title and an explicit n in the caption.

## 4. How the eval report presents results

**Tables, not walls of text.** Every headline metric appears in a table with its caveat in
an adjacent column — not in a footnote, not in a later paragraph. The caveat travels with
the number, because a number that gets quoted without its caveat has been misquoted, and
the format should make that hard.

Standard shape for every metric table:

| Metric | Trivial | Simple (TF-IDF) | ThreadPilot | 95% CI | Caveat |
|---|---|---|---|---|---|
| Intent macro-F1 | ... | ... | ... | [.., ..] | n=200; `other` class is a catch-all and inflates apparent difficulty |
| Safe-to-send rate (auto-handled) | n/a (0 coverage) | ... | ... | [.., ..] | judged by LLM; see judge validation kappa |
| Auto-handle coverage | 0% | ... | ... | — | capped by `ALWAYS_ESCALATE` design |

Additional presentation rules:

- **Never a bare point estimate.** Every proportion carries a bootstrap 95% CI.
- **The trivial baseline's `n/a`s are shown, not hidden.** Its zero-coverage perfection on
  safety is the most instructive cell in the table.
- **Coverage/risk curve is a figure, not a number** (`eval/results/coverage_risk.png`).
- **Judge scores are always presented next to the judge-validation kappa.** An LLM-judged
  quality number without its agreement statistic is an unvalidated claim.
- **Failure analysis quotes verbatim.** Real tweet text, real draft output, no cleaned-up
  paraphrase.
- Fixed decimal places (3 for F1/proportions, 2 for similarity). No trailing-digit theatre.
