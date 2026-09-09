# ThreadPilot

A support-tweet triage agent for a single brand. For each inbound customer tweet it:

1. **classifies intent** against a taxonomy derived from that brand's real traffic,
2. **drafts a reply grounded in how the brand has historically resolved similar issues**
   (retrieved real precedent, not invented text),
3. **decides auto-handle vs. escalate, with a stated reason.**

Built on the Kaggle [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset (~2.81M tweets).

> **Status: Phase 0 of 9 complete** (environment + planning). The pipeline and evaluation
> harness are not built yet. This README describes what exists today and is updated as
> phases land — see `docs/phases.md` for the plan and `docs/memory.md` for exact current
> state.

---

## What this project is actually optimising for

The brief this was built against says **"the proof is worth more than the system."** So the
deliverable is not a clever agent — it is *credible evidence about how well a modest agent
works, and where it fails.* Concretely, that means:

- A **200-example golden set**, hand-reviewed, built **before** the pipeline so the pipeline
  cannot be built to the test.
- **Two baselines** — a trivial one (majority intent + canned reply + always-escalate) and a
  real one (TF-IDF classifier + template replies, no LLM). The gap is the headline result.
- **LLM-as-judge, validated** against human re-scoring with Cohen's kappa. An LLM-judged
  quality number without an agreement statistic is an unvalidated claim.
- Thresholds tuned on a **separate silver dev set**, never on the golden set.
- A mandatory **"what's misleading about my headline number"** section.
- A **decision log** capturing non-obvious choices as they are made, including their costs.

## Quickstart

Requires **Python 3.12** (3.14 is not supported — see `docs/decision_log.md` D1).

```bash
git clone <this repo> && cd ThreadPilot

py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell
# source .venv/Scripts/activate     # bash

python tasks.py setup               # core deps + editable install
python tasks.py check               # verify the environment
python tasks.py test
```

`python tasks.py` with no arguments lists every command.

### Do I need an API key?

**No — not to reproduce the reported numbers.** LLM responses are cached in
`cache/llm_cache.sqlite`, which is committed, so:

```bash
python tasks.py eval --offline      # regenerates all reported metrics, no key needed
```

A cache miss in offline mode is a hard error, never a silent fallback. **Be clear about what
this is: a replay of a recording, not a fresh run.** For live calls, put a free
[Groq](https://console.groq.com/keys) key in `.env` (copy `.env.example`) and use
`python tasks.py eval --no-cache`.

### Do I need torch?

**No — not to reproduce the reported numbers.** `torch` + `sentence-transformers` are
quarantined in `requirements-embed.txt` because torch is a ~200MB wheel and the single
biggest cost in a fresh setup. The embedding matrix is committed. You only need them to
rebuild embeddings from scratch:

```bash
python tasks.py setup-embed
```

### Getting the raw data (optional)

Only needed to regenerate the subsample from scratch; the seeded subsample is committed.

```bash
python tasks.py data                # ~500MB, one time
```

Requires a Kaggle token: get one at [kaggle.com/settings/api](https://www.kaggle.com/settings/api)
("Generate New Token") and save it as a single line in `~/.kaggle/access_token`. Note that
`kaggle.json` with `KAGGLE_USERNAME`/`KAGGLE_KEY` is Kaggle's **legacy** scheme; it still
works, but the single-token file is current. The download script tries `kagglehub`, then the
`kaggle` CLI, then falls back to printing manual instructions — it will not hard-block you.

## Repository map

| Path | What |
|---|---|
| `docs/` | Planning + write-up: requirements, architecture, phases, design, memory, decision log, report |
| `src/threadpilot/` | The pipeline: `config`, `llm`, `retrieval`, `classify`, `draft`, `decide`, `pipeline`, `baselines`, `cli` |
| `eval/` | Evaluation harness, golden set, judge, metrics, committed results |
| `scripts/` | Data download, subsample builder, golden-set label drafter |
| `tests/` | Invariants that would otherwise fail silently and invalidate results |
| `cache/` | Committed LLM response cache (enables the no-key path) |
| `tasks.py` | Task runner. Not a Makefile — `make` is absent on stock Windows |

## Stack

Python 3.12 · pandas / numpy / pyarrow · scikit-learn · Groq free tier
(`openai/gpt-oss-120b` generating, `qwen/qwen3.8-27b` judging — deliberately different
model lineages) · `all-MiniLM-L6-v2` embeddings run locally on CPU · **numpy brute-force
cosine** rather than a vector DB (the index is ~10^4 × 384; exact search is one matmul).

Zero paid services. Everything runs in a local venv.

## Known limitations

Stated up front rather than buried, and expanded in `docs/report.md` as results land.

- **One brand, sampled.** Grounding is a per-brand claim, so results do not transfer across
  brands, and a few-thousand-tweet subsample may not represent the full 2.81M rows.
- **The judge shares a provider with the generator.** Different model families (Qwen vs
  gpt-oss) reduce self-preference bias; same-provider and same-era pretraining mean it is
  reduced, not eliminated. The judge is also *smaller* than the generator.
- **Historical brand replies are a noisy reference, not ground truth.** What the brand
  actually said is a comparison point, not a ceiling — some of it is bad.
- **n=200** gives wide confidence intervals on per-class metrics. Every proportion is
  reported with a bootstrap 95% CI for this reason.
- **The committed cache is a recording.** LLM APIs are not bit-reproducible.
- **I wrote both the system and its labels**, which is a bias no amount of process removes.

## Citations

Dataset, models, borrowed methodology and AI assistance are all itemised in
[`CITATIONS.md`](CITATIONS.md).
