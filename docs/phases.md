# ThreadPilot — Phases

> Status: living document. Each phase ends with a **checkpoint**: update `docs/memory.md`,
> report status, wait for go-ahead.
> Time estimates are working estimates, not commitments.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Phase 0 — Environment & repo skeleton `[x]` DONE

**Goal:** a working, isolated, reproducible shell for everything that follows.

- [x] Five planning docs exist in skeleton form
- [x] `.venv` on Python 3.12.4; `requirements.txt` pinned (55 packages, frozen from a real resolved install)
- [x] `.gitignore` (+ `docs/` per user request — revisit before submission)
- [x] `.env.example` + `python-dotenv` wiring; `GROQ_API_KEY` **verified live**
- [x] Confirmed available Groq models from the live `/models` endpoint. **The planned
      `llama-3.3-70b-versatile` does not exist** — no Llama chat model remains on the free
      tier. Catalogue committed to `docs/groq_models_2026-09-09.json`; choice is D19
- [x] Repo skeleton per `architecture.md` section 2; `import threadpilot` works
- [x] `scripts/download_data.py` — 3-tier (kagglehub -> CLI -> manual hint), idempotent
- [x] `tasks.py` with 10 commands, plus `check.py` environment self-check
- [x] `CITATIONS.md` — dataset, models, 6 methodology citations, AI-assistance disclosure

**Exit criteria:** MET, except the dataset download which is deferred into Phase 1 because
it needs one user-run command. `import threadpilot` works; a live Groq call succeeds; 7 tests
pass; `tasks.py check` reports no failures.

**Commit point: yes** — "Phase 0: repo skeleton, venv, planning docs".

---

## Phase 1 — Data exploration & brand selection `[x]` DONE

**Goal:** choose the brand by measurement against the criteria pre-registered in
`requirements.md` section 2.1.

- [x] Chunked pass over the full CSV (2,811,774 rows; 108 brand accounts) for C1
- [x] Threads reconstructed from parent pointers; multi-turn rate per brand (C4)
- [x] C3 measured for top 12 brands via an 8-way reply classifier — **and validated against an independent rater (kappa 0.374), after v1 failed at 0.201**
- [x] TF-IDF+SVD+KMeans entropy proxy for intent diversity (C2)
- [x] Both escalation poles measured (C5)
- [x] Candidates scored; **hypothesis half falsified and recorded as such** (AppleSupport
      confirmed; "an airline wins" false). Composite score proved unable to separate the top
      two, so a 4-definition sensitivity analysis decided it on robustness -> `AmazonHelp`
- [x] Seeded subsample committed: 8,000 pairs / 7,512 threads / 2.8MB, with `thread_id` for the Phase 4 leakage guard and duplicate *grouping* rather than deletion
- [ ] `notebooks/01_eda.ipynb` — DEFERRED to Phase 8. Every number is already produced by scripts and committed under `eval/results/`, so the notebook adds presentation, not evidence

**Exit criteria:** MET. Brand chosen with a numbers table *and* a sensitivity analysis
showing why the table alone was insufficient; deterministic subsample committed. Byte-identity
of re-runs is asserted by `tests/test_subsample.py`.

**Checkpoint with user: yes** — brand choice affects everything downstream.
**Commit point: yes.**

---

## Phase 2 — Intent taxonomy design `[x]` DONE

**Goal:** a ~6-9 intent taxonomy **derived from the data**, with an auditable derivation.

- [x] Embedded 8,000 English thread-opening messages (MiniLM, disk-cached)
- [x] KMeans swept k=4..14. **Silhouette was uninformative** (0.051-0.065, spread 0.014),
      so k=10 chosen by interpretability and the sweep reported as evidence the automatic
      criterion did not discriminate (D29). Distinctive terms + near- AND far-from-centroid
      exemplars dumped per cluster
- [x] Named from **exemplars, not term lists** (D30 - term lists would have invented an `amazon_pay_india` intent that does not exist). Clusters 0+1 merged; explicit `other` kept
- [x] `eval/results/taxonomy_clusters.md` + `.json` committed with full per-cluster evidence
- [x] `src/threadpilot/taxonomy.py`: 10 intents + `other`, each with definition, 3 verbatim
      examples, a boundary note naming the sibling it contrasts with, and a disposition.
      `eval/results/taxonomy.md` is generated from the module so the two cannot drift
- [x] Logged as D27-D33, including the deliberate deviation to 10 intents (D31) and the inclusion of `account_security` despite it having no cluster (D32)

**Exit criteria:** MET. Taxonomy frozen before any labelling. Every named intent has a
boundary note that names the sibling it contrasts with, asserted by
`tests/test_taxonomy.py`. `config.ALWAYS_ESCALATE` populated from the taxonomy (3 intents,
~30% of traffic) and the Phase 0 strict-xfail placeholder flipped to a real assertion.
72 tests green.

**Commit point: yes.**

---

## Phase 3 — Golden evaluation set (200, stratified) `[x]` DONE

**Goal:** 200 hand-reviewed examples, built **before** the pipeline, so the pipeline cannot
be built to the test.

- [x] Stratified on Phase 2 CLUSTERS (intent labels do not exist at sampling time -
      the chicken-and-egg is documented, not hidden). Equal allocation across 10 clusters,
      `account_security` oversampled ~8x, plus a deliberate hard-case pool.
      Strata and per-stratum n in `eval/results/golden_strata.json`
- [x] Hard cases sought, not avoided: 22 order-ID, 21 account, 19 novel (bottom 5% by
      nearest-neighbour similarity), 17 very-short, 12 multi-intent, 11 sarcasm
- [x] TWO independent labellers from different model families, then human adjudication.
      **113 disagreements
      adjudicated + 40 agreed rows audited = 153/200 (76.5%)
      individually reviewed.** The other 47 are accepted on agreement, with their error
      rate MEASURED rather than assumed (see below)
- [x] All fields present, plus `provenance`, `reviewer`, `labeller_A_intent`,
      `labeller_B_intent` and `nn_similarity` so every label's origin is traceable
- [x] `dev_silver` 100 rows, single labeller (D5 accepts noisier labels for a tuning set). Thread-disjointness from golden asserted in code and verified: 0 overlap
- [x] `eval/golden/labelling_notes.md` GENERATED from the measured artifacts so it cannot
      drift. Covers sampling, the two anti-circularity rules, inter-labeller agreement,
      who actually reviewed, source-data label noise, and a taxonomy defect found during
      labelling (D40)
- [x] **Overrode 119 of 200.** More usefully, the audit of
      agreed rows gives a measured **~15%
      residual label noise** estimate for the rows nobody inspected

**Exit criteria:** MET. 200 records; golden/dev_silver thread-disjoint (asserted);
override rate and measured residual noise reported.
**Headline finding:** inter-labeller Cohen kappa was 0.721 on
intent but only **0.273 on auto-vs-escalate** - gpt-oss
escalated 34.3% of messages, Qwen 73.8%. The escalation label is intrinsically contested,
which caps what any classifier evaluated against it can claim.

**Checkpoint with user: yes.**
**Commit point: yes** — this is the highest-value artifact in the repo.

---

## Phase 4 — Core pipeline `[x]` DONE

- [x] `retrieval.py`: 6,155 vectors, exact cosine. **Leakage guard excluded 235 rows** across 300 held-out threads (golden AND dev_silver). 5 dedicated leakage tests, incl. one proving the guard actually raises and one proving `build_index` refuses to build unguarded
- [x] Only `usable_precedent` AND `is_canonical` rows admitted: 8,000 -> 6,390 -> 6,155. Composition in `eval/results/index_info.json`
- [x] `classify.py` (R1) — batched (D39), boundary notes in the prompt, dispositions deliberately withheld so intent and decision cannot contaminate each other
- [x] `draft.py` (R2) — precedent-grounded, anti-fabrication instruction, and the model's own decision proposal in the same call (D41). Absent precedent is stated explicitly rather than left as an empty section
- [x] `decide.py` (R3) — six one-directional gates G1-G6. Reasons name the gate AND the number behind it (`G2: no comparable precedent (max similarity 0.31 < 0.45)`)
- [x] `pipeline.py` + `cli.py`. Both ablations (`--no-gates`, `--no-retrieval`) run the same pipeline the headline numbers come from, not a separate code path
- [x] 33 new tests: leakage (5), index contents, search semantics, and gate logic including the one-directional property asserted from both sides

**Exit criteria:** MET. End-to-end CLI verified on live messages; leakage tests green.
**Notable:** on the very first run the model proposed `auto_handle` for "someone has changed
the email on my account and I cannot log in" and gate G1 overrode it - the rule layer earned
its place on its first real input (D42).
**Batching validated** (D44): per-item agreement 80%, but accuracy
vs gold identical at 73.3% batched and
73.3% unbatched - batching relocates errors, does not add
them.
**Reframing for Phase 6** (D45): the classifier's 73.3% sits
at the 75% inter-labeller ceiling, so accuracy must be
reported against that ceiling, not against 100%.

**Commit point: yes.**

---

## Phase 5 — Baselines `[x]` DONE

- [x] **Trivial:** majority intent + canned reply + always-escalate. Measured
      **0.0% false-auto-handle at 0.0% coverage**, 11.0% accuracy, macro-F1 0.018 -
      simultaneously the safest and most useless system. The gameability argument is now
      a number in the results rather than prose (D49)
- [x] **Simple:** TF-IDF + logistic regression, per-intent templates derived from REAL
      brand replies (nearest each cluster centroid), escalate on confidence. No LLM.
      Fitted under TWO training regimes because the choice changes the verdict (D47):
      silver (100 rows) 30.0% acc / F1 0.263; cluster (6,948 rows) 36.0% acc / F1 0.332.
      **Plus a third baseline not in the plan** - `retrieval_1nn`, which copies the nearest
      historical reply verbatim and isolates exactly what generation adds (D46): 36.0% acc,
      78.5% coverage, 56.1% FAH
- [x] All three emit the same `TriageResult` shape as the pipeline, so scoring runs an
      identical path. Asserted by test, along with a test that no baseline calls the LLM

**Exit criteria:** MET for the baselines; the system's own numbers land in Phase 6.
Preliminary picture: the simple baselines fail in the OPPOSITE direction to the trivial one -
53-56% false-auto-handle at 58-82% coverage, i.e. reckless automation. Neither pole is
acceptable, which is what the coverage/risk curve exists to show.

**Commit point: yes.**

---

## Phase 6 — Evaluation harness `[x]` DONE

- [ ] Classification: accuracy, macro-F1, per-class P/R/F1, confusion matrix,
      **bootstrap 95% CIs** (non-negotiable at n=200)
- [ ] Escalation: precision/recall on `escalate`, false-auto-handle rate reported
      separately, **coverage/risk curve**
- [ ] LLM-as-judge: rubric over groundedness / on-intent / non-overcommitment / voice /
      actionability, plus the binary "send unedited?"; judge on a **different model family**
      than the generator; position/order effects controlled
- [ ] **Judge validation (mandatory):** 40 examples re-scored independently by human;
      report Cohen's kappa (quadratic-weighted for the ordinal criteria) + exact-agreement
      % + qualitative notes on *where* judge and human diverge
- [ ] Ablations: (a) no-retrieval drafting, to isolate what grounding actually buys;
      (b) pure-LLM decision with gates off, to make the rule layer earn its place
- [ ] `eval/results/` committed: json + markdown tables + pngs

**Exit criteria:** every number in the report is regenerable by one command, offline.

**Checkpoint with user: yes** — judge validation is graded heavily; I want a second pair of
eyes on the human re-scores.
**Commit point: yes.**

---

## Phase 7 — Failure analysis `[x]` DONE

- [ ] Mine golden-set results for failure modes; cluster them
- [ ] Top 5, each with: real examples (verbatim), frequency, a **hypothesis for the cause**,
      and whether it is fixable by prompt, by retrieval, by taxonomy, or not at all cheaply
- [ ] Include at least one failure of the *eval itself*, not only of the system — e.g. a
      case where the judge is wrong or the golden label is arguable
- [ ] Note which failures the gates caught vs. which slipped through as auto-handle (the
      latter are the dangerous ones and get the most space)

**Exit criteria:** five documented modes with evidence, not speculation.

**Commit point: yes.**

---

## Phase 8 — Report, decision log, README `[x]` DONE

- [ ] `docs/report.md`: problem framing, method, eval design, results tables, judge
      validation, failure analysis
- [ ] **"What's misleading about my headline number"** — mandatory, written to actually
      undermine my own result. Known candidates to interrogate: golden set may not match
      the true message distribution; judge shares a provider with the generator; a
      few-thousand-tweet subsample of one brand may not generalise to 2.81M rows across
      100 brands; historical brand replies are a noisy reference, not ground truth;
      `ALWAYS_ESCALATE` inflates apparent safety; committed cache is a recording;
      n=200 gives wide CIs; I wrote both the system and its labels
- [ ] **"What I'd do next"** — ranked, with the reasoning for the ranking
- [ ] `docs/decision_log.md` — 10-15 non-obvious decisions, captured **as made** throughout,
      not reconstructed here
- [ ] README: quickstart, 15-minute path, offline path, honest limitations up front
- [ ] `CITATIONS.md` final pass

**Exit criteria:** an evaluator can read `report.md` alone and know what was built, how well
it works, and where it breaks.

**Commit point: yes.**

---

## Phase 9 — Reproducibility pass `[x]` DONE

- [x] Fresh tree built by excluding every gitignored path (no index, no subsample, no venv)
- [x] Followed the README only, from a new venv, timing each step
- [x] **FOUND A REAL BUG**: `--offline` crashed on a clean tree (index is gitignored). Fixed by lazy-loading the index; re-verified with `GROQ_API_KEY=""` -> 7 systems in 4s (D55)
- [x] **2m 44s** total against the 15-minute promise
- [x] Measured timings recorded in the README, plus the honest note that 33/153 tests skip on a fresh clone because they need the dataset

**Exit criteria:** MET and documented. 2m 44s measured. The pass earned its place by
finding that the central reproducibility promise did not hold on a clean tree (D55).

**Commit point: yes** — final.

---

## Running decision points I owe the user

| # | Question | When | Status |
|---|---|---|---|
| 1 | LLM provider | Phase 0 | **Resolved:** Groq |
| 2 | Dataset access | Phase 0 | **Resolved:** user supplies `kaggle.json` |
| 3 | Golden set size | Phase 3 | **Resolved:** 200, stratified |
| 4 | UI or not | Phase 8 | **Resolved:** CLI + notebook only |
| 5 | Brand choice | Phase 1 | **Resolved:** `AmazonHelp`, on robustness (D25) |
| 6 | Taxonomy size | Phase 2 | **Resolved:** 10 intents + `other` (D31) |
| 7 | Judge model family | Phase 0 | **Resolved:** `qwen/qwen3.8-27b`, D19 |
