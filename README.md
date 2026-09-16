# ThreadPilot

A support-tweet triage agent for a single brand. For each inbound customer tweet it:

1. **classifies intent** against a taxonomy derived from that brand's real traffic,
2. **drafts a reply grounded in how the brand has historically resolved similar issues**
   (retrieved real precedent, not invented text),
3. **decides auto-handle vs. escalate, with a stated reason.**

Built on the Kaggle [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset (~2.81M tweets). Brand: **AmazonHelp**.

> This was built against a brief that said **"the proof is worth more than the system."**
> So the deliverable is not a clever agent — it is credible evidence about how well a modest
> agent works, and where it fails. **[`docs/report.md`](docs/report.md) is the main
> deliverable.**

---

## Results

| system | accuracy | macro-F1 | coverage | false auto-handle | send unedited |
|---|---|---|---|---|---|
| `trivial` | 11.0% | 0.018 | 0% | **0%** | 1% |
| `simple_tfidf_silver` | 30.0% | 0.263 | 58% | 53% | 15% |
| `simple_tfidf_cluster` | 36.0% | 0.332 | 82% | 54% | 17% |
| `retrieval_1nn` | 36.0% | 0.332 | 78% | 56% | 48% |
| `pipeline_no_retr` | 80.5% | 0.790 | 0% | 0% | 52% |
| `pipeline_no_gates` | 80.5% | 0.790 | 100% | 55% | 78% |
| **`pipeline`** | **80.5%** | **0.790** | 62% | **41%** | **78%** |

**Classification is a decisive win:** 80.5% [75.0%, 86.5%] against a best-baseline 36.0%,
with non-overlapping confidence intervals.

**Escalation is the honest weakness:** of the 62% of messages it auto-handles, **41% should
have gone to a human.**

### Four findings that matter more than the headline

**1. The judge failed validation (κ = 0.333).** Blind re-scoring of 40 replies showed the LLM
judge is systematically *more generous* on every criterion (+0.45 to +1.07 on a 1–5 scale).
It scores **form** — "mirrors precedent" — where a human scores **substance** — "answers
nothing". It passed a reply to a customer reporting a *break-in with police involved* because
it "matches precedent for security issues". Without this study the report would have
overstated reply quality by ~18 points.

**2. The model's self-assessment is worthless.** It proposed `auto_handle` for **all 200**
messages — never once escalated. The deterministic gates supply *100%* of the escalation
capability.

**3. Both halves of the design are necessary.** Retrieval adds +26 points of reply quality
over the LLM alone (78% vs 52%); generation adds +30 points over copying precedent verbatim
(78% vs 48%). Neither component is close on its own.

**4. Two capable models barely agree on the core label.** Inter-labeller Cohen κ on
auto-vs-escalate was **0.273** — gpt-oss escalated 34% of messages, Qwen 74%. Any classifier
evaluated against escalation labels inherits that ceiling.

**The headline accuracy is inflated in two identified ways**, both quantified in the report:
a 17% `other` grab-bag scored 100% (+4.0 points), and methodological coupling that lets it
exceed the 75.0% inter-labeller ceiling.

---

## Quickstart

Requires **Python 3.12** (3.14 unsupported — `docs/decision_log.md` D1).

```bash
git clone <this repo> && cd ThreadPilot

py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell
# source .venv/Scripts/activate     # bash

python tasks.py setup               # core deps + editable install
python tasks.py check               # environment self-check
python tasks.py test
```

### Reproduce every reported number — no API key needed

```bash
python -m eval.run_eval --offline
```

**Measured on a fresh clone** (Windows, Python 3.12.4), not estimated:

| step | time |
|---|---|
| `venv` + `tasks.py setup` | 119s |
| `tasks.py check` | 1s |
| `tasks.py test` | 12s |
| `eval.run_eval --offline` | 32s |
| **total** | **2m 44s** |

**153 tests total; 120 run on a fresh clone and 33 skip** — the skipped ones need the
retrieval index and subsample, which require the one-time dataset download below. They are
skipped rather than silently passing.

LLM responses are cached in `cache/llm_cache.sqlite`, which is committed. A cache miss in
offline mode is a **hard error**, never a silent fallback. Be clear about what this is: a
replay of a recording, not a fresh run.

### Getting the data (required once, for the full pipeline)

```bash
python tasks.py data                # ~500MB Kaggle download
python tasks.py subsample           # ~20s, deterministic
```

The subsample is **not committed** — it derives from a CC BY-NC-SA 4.0 dataset, so this repo
does not redistribute tweet text. The rebuild is deterministic from `SEED = 20260909`, and
`tests/test_subsample.py` verifies byte-identity by SHA-256, so you can *check* your copy
matches rather than trust it. `data/processed/subsample_meta.json` is committed as the
reference.

Needs a Kaggle token at `~/.kaggle/access_token` (get one at
[kaggle.com/settings/api](https://www.kaggle.com/settings/api)). Note `kaggle.json` is the
**legacy** scheme; the single-token file is current.

### Try one message

```bash
python tasks.py triage "my parcel never arrived and it says delivered"
```

---

## What's where

| path | what |
|---|---|
| **[`docs/report.md`](docs/report.md)** | **The deliverable.** Results, judge validation, failure analysis, what's misleading, what's next |
| [`docs/decision_log.md`](docs/decision_log.md) | 55 non-obvious decisions, captured as made — including the ones that turned out wrong |
| [`docs/requirements.md`](docs/requirements.md) | Problem framing, pre-registered brand criteria, non-goals |
| [`eval/golden/`](eval/golden/) | 200 hand-reviewed examples + labelling notes + every adjudication with its reasoning |
| [`eval/results/`](eval/results/) | Every number, as committed JSON and markdown |
| `src/threadpilot/` | `config`, `llm`, `embeddings`, `retrieval`, `classify`, `draft`, `decide`, `pipeline`, `baselines` |
| `tests/` | 153 tests, focused on invariants that would fail *silently* |

## Stack

Python 3.12 · pandas / numpy / scikit-learn · Groq free tier (`openai/gpt-oss-120b`
generating, `qwen/qwen3.8-27b` judging — deliberately different model lineages) ·
`all-MiniLM-L6-v2` embeddings on CPU · **numpy brute-force cosine**, no vector DB (the index
is ~6k × 384; exact search is one matmul).

Zero paid services.

## Known limitations

Stated here rather than buried; each is quantified in the report.

- **The judge is unvalidated-grade (κ = 0.333)** and systematically generous. Every
  judge-derived number should be read with that attached.
- **The escalation label itself is contested** (inter-labeller κ = 0.273). This caps what any
  escalation metric can claim.
- **~24% of real traffic is out of scope** — non-English (led by Japanese) and mid-thread
  messages were excluded.
- **Golden labels carry ~15% measured residual noise**, estimated by auditing rows where both
  labellers agreed and finding 6 of 40 where both were wrong.
- **The golden set is deliberately not distribution-matched** — hard cases were sought, rare
  intents oversampled. Accuracy here is *not* an estimate of production accuracy.
- **One brand, English, thread openers only.** Results do not transfer.
- **I wrote the system, taxonomy, rubric, labels and adjudications.** Partial mitigations are
  documented; the bias is not removable.
- **Batching shifts ~20% of individual predictions** without changing aggregate accuracy.
  Aggregates are quotable; a specific row's prediction is not stable.

## Citations

Dataset, models, borrowed methodology and AI assistance are itemised in
[`CITATIONS.md`](CITATIONS.md).
