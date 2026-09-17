# ThreadPilot — Architecture

> Status: living document. Last substantive update: Phase 0.

---

## 1. Tech stack

| Layer | Choice | Version / model | Why, and free-tier limits |
|---|---|---|---|
| Language | Python | **3.12.4** | Machine also has 3.14.3, but `torch` / `sentence-transformers` wheel availability on 3.14 is unreliable. 3.12 is the boring, working choice. Pinned in README. |
| Isolation | `venv` | stdlib | `python -m venv .venv`. Nothing global. |
| Generation LLM | **Groq** | `openai/gpt-oss-120b` (131k ctx) | Free tier, verified live 2026-09-09 (`docs/groq_models_2026-09-09.json`). **The originally planned `llama-3.3-70b-versatile` does not exist** — Groq's free tier serves no Llama chat model. Largest general model available; ~0.5s per short call. It is a *reasoning* model (separate `reasoning` channel), which is why token budgets are generous and empty content is an error (D20). |
| Judge LLM | **Groq, different family** | `qwen/qwen3.8-27b` (131k ctx) | Alibaba Qwen vs OpenAI gpt-oss — genuinely different lineages, not just different sizes. Emits clean JSON with no `<think>` leakage (unlike `qwen3.6-27b`, rejected for that). Residual weaknesses, both disclosed: same provider, and the judge is *smaller* than the generator. `JUDGE_MODEL_ALT=gpt-oss-120b` runs the same validation so judges are chosen by agreement-with-human, not by preference. See D19. |
| Embeddings | `sentence-transformers` | `all-MiniLM-L6-v2`, 384-dim, CPU | Runs locally, no API, no cost. Small and fast enough to embed ~30k texts on CPU in a couple of minutes, cached to disk after. |
| Vector search | **numpy brute-force cosine** | — | Index is ~10^4 x 384 float32 (tens of MB). Exact search is a single matmul, sub-millisecond, zero dependency risk, and *exact* rather than approximate — so retrieval quality is never confounded by ANN recall. Revisit only if the index exceeds ~10^6 vectors, which it will not. |
| Dataframes | `pandas` + `pyarrow` | — | Parquet for interim artifacts; the raw CSV is read once, chunked. |
| Classical ML | `scikit-learn` | — | TF-IDF baseline, KMeans for taxonomy derivation, metrics, Cohen's kappa. |
| LLM plumbing | `groq`, `python-dotenv`, `tenacity` | — | `tenacity` for backoff on 429s — non-negotiable on a free tier. |
| CLI output | `rich` | — | Tables and panels so triage output is skimmable by a human reviewer (see `docs/design.md`). |
| Plots | `matplotlib` | — | Confusion matrix, coverage/risk curve. Static PNGs committed under `eval/results/`. |
| Tests | `pytest` | — | Focused on the parts where a silent bug would invalidate the eval: thread reconstruction, golden/index leakage, metric computation. |
| Notebook | `jupyter` | — | Phase 1 EDA only. Notebooks are for exploration; every number that reaches the report is produced by a script. |
| Task runner | `python tasks.py <cmd>` | stdlib | **Not a Makefile** — `make` is not present on this Windows machine and requiring it would break the reproducibility promise for a Windows evaluator. |

## 2. Repository layout

```
ThreadPilot/
├── .env.example              # GROQ_API_KEY placeholder
├── .gitignore                # .venv, .env, data/raw, data/interim, __pycache__
├── README.md                 # the only thing an evaluator must read
├── CITATIONS.md              # every borrowed snippet / prompt pattern / transformation
├── requirements.txt          # pinned, hash-free but exact ==
├── tasks.py                  # python tasks.py setup|data|taxonomy|eval|repro
│
├── data/
│   ├── raw/                  # GITIGNORED. twcs.csv (~500MB) lands here
│   ├── interim/              # GITIGNORED. embeddings cache, parquet intermediates
│   └── processed/            # subsample.parquet GITIGNORED (CC BY-NC-SA, not
│                             # redistributed); subsample_meta.json COMMITTED
│
├── docs/
│   ├── requirements.md
│   ├── architecture.md       # this file
│   ├── phases.md
│   ├── design.md
│   ├── memory.md             # recovery mechanism — read this first if context is lost
│   ├── decision_log.md       # 10-15 non-obvious decisions, captured as made
│   └── report.md             # Phase 8 deliverable
│
├── scripts/
│   ├── download_data.py      # kaggle CLI; detects an already-present CSV and skips
│   ├── build_subsample.py    # seeded, deterministic; rebuilds data/processed/
│   └── build_golden_set.py   # drafts labels for human review; never auto-accepts
│
├── src/threadpilot/
│   ├── config.py             # seeds, paths, model names, thresholds — one source of truth
│   ├── llm.py                # Groq client, retry/backoff, on-disk response cache
│   ├── data/
│   │   ├── load.py           # chunked CSV read, dtype control
│   │   ├── threads.py        # thread reconstruction from in_response_to_tweet_id
│   │   └── clean.py          # normalisation, near-dupe removal
│   ├── taxonomy.py           # intent set + cluster-derivation code
│   ├── retrieval.py          # index build, embed cache, top-k cosine
│   ├── classify.py           # R1
│   ├── draft.py              # R2
│   ├── decide.py             # R3: LLM proposal + deterministic gate layer
│   ├── pipeline.py           # orchestration: classify -> retrieve -> draft -> decide
│   ├── baselines.py          # trivial + simple baselines
│   └── cli.py                # `python -m threadpilot.cli triage "..."`
│
├── eval/
│   ├── golden/
│   │   ├── golden_set.jsonl      # COMMITTED. 200 hand-reviewed examples
│   │   ├── dev_silver.jsonl      # COMMITTED. threshold-tuning set, LLM-labelled
│   │   └── labelling_notes.md    # sampling strategy, ambiguous cases, label noise
│   ├── judge.py              # rubric prompt + parsing
│   ├── metrics.py            # classification metrics, bootstrap CIs, kappa
│   ├── run_eval.py           # produces every number in the report
│   ├── judge_validation.py   # human-vs-judge agreement study
│   └── results/              # COMMITTED. json + md tables + pngs
│
├── cache/
│   └── llm_cache.sqlite      # COMMITTED. lets evaluators reproduce metrics with NO API key
│
├── tests/
└── notebooks/01_eda.ipynb
```

## 3. Data flow

```
 Kaggle twcs.csv  (2.81M rows, ~100 brands)
        |
        |  scripts/download_data.py           [once, ~500MB, gitignored]
        v
 [1] CHUNKED LOAD + DTYPE CONTROL             src/threadpilot/data/load.py
        |
        v
 [2] THREAD RECONSTRUCTION                    data/threads.py
        |    follow in_response_to_tweet_id to build conversations;
        |    keep (first inbound customer msg -> first brand reply) pairs
        v
 [3] BRAND FILTER                             one brand only (Phase 1 choice)
        |
        v
 [4] CLEAN + NEAR-DUPE REMOVAL                data/clean.py
        |    strip URLs/handles-noise, normalise whitespace,
        |    drop near-identical pairs (they would inflate retrieval)
        v
 [5] SEEDED SUBSAMPLE  --> data/processed/     [deterministic; NOT committed,
        |                                       rebuilt locally - CC BY-NC-SA]
        |
        +-----------------------------+------------------------------+
        |                             |                              |
        v                             v                              v
 [6] TAXONOMY DERIVATION      [7] GOLDEN SET (200)          [8] RETRIEVAL INDEX
     embed -> KMeans ->            stratified sample,            (msg -> reply) pairs
     inspect exemplars ->          hand-reviewed labels          embedded w/ MiniLM
     human-name -> merge                 |                              |
        |                                |                              |
        |                                +---- LEAKAGE GUARD -----------+
        |                                      golden thread_ids are
        |                                      EXCLUDED from the index
        v                                                               |
   intent taxonomy  -------------------------------------------+        |
                                                              |        |
                                                              v        v
 [9] PIPELINE  (src/threadpilot/pipeline.py)
        inbound tweet
            -> classify(intent, confidence)          [R1]
            -> retrieve(top-k precedent pairs)        [grounding source]
            -> draft(reply | precedent)               [R2]
            -> decide(auto_handle|escalate + reason)  [R3]
            -> TriageResult
        |
        +--> [10] BASELINES  trivial (majority+canned+always-escalate)
        |                    simple  (TF-IDF clf + per-intent template, no LLM)
        v
 [11] EVAL HARNESS  (eval/run_eval.py)
        classification metrics + bootstrap CIs
        LLM-as-judge on reply quality (different model family)
        judge validation vs human re-scores (kappa)
        coverage/risk curve for auto-handle
        |
        v
 [12] eval/results/*  -->  docs/report.md
```

## 4. How the grounding requirement is actually implemented

The requirement is *"grounded in how the brand has historically resolved similar issues."*
Concretely:

**What is indexed.** One record per historical exchange:

```json
{
  "pair_id": "...",
  "thread_id": "...",
  "customer_msg": "cleaned text of the inbound customer tweet",
  "brand_reply":  "cleaned text of the brand's reply to it",
  "resolved": true,
  "created_at": "..."
}
```

The **embedded / searched field is `customer_msg`**, not the reply. The retrieval question
is "who else asked this?", and the payoff is the `brand_reply` attached to that neighbour —
i.e. *what we said to them*. Embedding the reply instead would retrieve on answer-similarity,
which is the wrong similarity for this task.

**What "resolved" means.** A pair is admitted to the index only if the brand reply is
substantive: it is not a pure channel-deflection ("DM us"), not a bare acknowledgement
("sorry to hear that!"), and clears a minimum length. Deflections are exactly the precedent
we must *not* teach the model to imitate, since a draft that says "please DM us" is a
non-answer that would score well on tone and badly on usefulness. The deflection filter and
its measured effect on index size are reported in Phase 4.

**Retrieval.** Embed the incoming tweet with the same MiniLM model, cosine against the
index matrix, take top-k. `k` starts at 5 and is tuned on the silver dev set only
(decision-logged). Also returned: the **max similarity**, which feeds the escalation gate —
if the closest precedent is far away, the brand has no history for this and a human should
look.

**Drafting.** The k precedents are rendered into the prompt as labelled examples of prior
handling, with an instruction that any specific claim (timeframe, policy, URL, amount) must
appear in the precedent or the customer message. Groundedness is then *measured*, not
assumed — the judge scores it against the same precedent block the drafter saw, which is
the only fair test.

**Leakage guard (the detail that makes or breaks the eval).** Golden-set examples are
sampled first; their `thread_id`s are then subtracted from the index before it is built.
There is a `pytest` test asserting the intersection is empty. Without this the retriever
would surface the exact target reply and every quality metric would be inflated to the
point of fraud.

## 5. How auto-handle vs. escalate is decided

**Chosen design: LLM proposal + deterministic override layer.** Not pure-LLM, not pure-rule.

```
                 +-----------------------------+
 intent,         |  LLM decision call          |
 confidence,     |  -> {decision, reason,      |
 draft,          |      risk_flags[]}          |
 precedent  ---> +-----------------------------+
                              |
                              v
                 +-----------------------------+
                 |  DETERMINISTIC GATES        |   any gate firing
                 |  (can only force ESCALATE,  |   => escalate, with the
                 |   never force auto_handle)  |   gate named as the reason
                 +-----------------------------+
                   G1 intent in ALWAYS_ESCALATE  (refund/billing, account+security,
                                                  legal/safety, complaint-with-harm)
                   G2 max retrieval similarity < TAU_SIM   (no precedent)
                   G3 classifier confidence < TAU_CONF     (unsure what it even is)
                   G4 account-specific request detected    (needs private data)
                   G5 LLM raised any risk_flag
                              |
                              v
                    decision + stated reason
```

**Why this and not pure-LLM.** Three reasons, all about the eval rather than the model:

1. **Cost asymmetry.** A false escalate wastes an agent's half-minute. A false auto-handle
   publishes a wrong answer under the brand's name. The gates are one-directional — they
   can only *add* caution — which encodes that asymmetry structurally.
2. **Auditability.** When the system escalates, a team lead gets a named reason ("no
   comparable precedent", similarity 0.31) rather than a sentence the model generated about
   itself. Self-reported LLM rationales are post-hoc narration and I do not want to present
   them as the mechanism.
3. **Tunability without retraining.** `TAU_SIM` and `TAU_CONF` give a real coverage/risk
   knob, which is what produces the curve in section 6 rather than a single point estimate.

The honest cost: the gates are hand-designed, so they can encode my own blind spots, and
`ALWAYS_ESCALATE` caps achievable auto-handle coverage by construction. Both go in the
"what's misleading" section. A pure-LLM ablation is run in Phase 6 so the rule layer has to
justify its existence with a number rather than an argument.

## 6. Threshold tuning and the honesty of the headline number

Three disjoint data pools:

| Pool | n | Labels | Used for |
|---|---|---|---|
| `dev_silver` | ~100 | LLM-drafted, spot-checked only | **All** threshold/`k`/prompt tuning |
| `golden_set` | 200 | Hand-reviewed by me, every one | Reported metrics. Touched once per config, ideally once total |
| retrieval index | rest of subsample | none needed | Grounding precedent. Disjoint from golden by thread_id |

Tuning on `dev_silver` rather than on the golden set is what lets the headline number be
described as held-out. The cost is that silver labels are noisier, so thresholds are
slightly mis-set relative to a golden-tuned optimum — a trade I take deliberately, because
a slightly-suboptimal honest number beats an optimal tuned-on-test number.

**Reporting form.** The auto-handle result is a **coverage/risk curve**: sweep `TAU_SIM` /
`TAU_CONF`, and at each resulting auto-handle coverage plot the share of auto-handled
replies that are judged safe-to-send-unedited. A single operating point is also named for
headline purposes, with its caveat printed next to it. This framing exists because the
trivial `always-escalate` baseline achieves a *perfect* false-auto-handle rate at zero
coverage — any single-number safety metric is therefore gameable, and the curve is the
minimum honest presentation.

## 7. LLM response cache and the no-API-key reproducibility path

`src/threadpilot/llm.py` caches every completion in `cache/llm_cache.sqlite`, keyed by
`sha256(provider, model, prompt, temperature, max_tokens, schema_version)`.

Two consequences, both deliberate:

- **Free-tier survival.** Re-running the eval after a metrics-code change costs zero API
  calls.
- **Evaluators reproduce without a key.** The cache is **committed**. `python tasks.py eval
  --offline` replays it and regenerates every number and table in the report with no
  `GROQ_API_KEY` present. A cache miss in offline mode is a hard error, never a silent
  fallback.

This is a *recording*, not a fresh run, and the README says so plainly. `--no-cache` forces
live calls; because generation temperature is set to 0 for classification and decisions
(and a fixed low value for drafting), a live re-run should land close to the recording, but
LLM APIs are not bit-reproducible and the report will not pretend otherwise.

## 8. Determinism

- Single seed in `config.py` (`SEED = 20260909`), threaded through subsampling, KMeans,
  golden-set stratification, and bootstrap resampling.
- Temperature 0 for classify and decide; fixed low temperature for drafting.
- Embeddings are deterministic on CPU for a pinned model revision; the model revision is
  pinned, not just the model name.
- Residual nondeterminism: the LLM API itself. Mitigated by the cache, disclosed in the report.
