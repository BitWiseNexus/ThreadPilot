# Citations & Attributions

Everything borrowed — data, code, prompt patterns, methodological choices — is listed here.
The assignment grades this explicitly, so the bar I am holding myself to is: **if I did not
derive it myself, it is named here, with a link.** Entries are added as the work happens,
not reconstructed at the end.

Where a citation is more useful next to the code, it also appears as an inline comment; this
file is the index, not the only copy.

---

## 1. Dataset

**Customer Support on Twitter** — Stuart Axelbrooke (`thoughtvector`), Kaggle.
<https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter>

- ~2.81M tweets, ~100 brand support accounts, collected 2017.
- Licence: per the Kaggle dataset page (CC BY-NC-SA 4.0 at time of access — non-commercial,
  which this take-home is).
- Handles are already pseudonymised by the dataset author (`@115712`-style numeric ids). I
  did not perform that anonymisation and do not claim it as compliance-grade
  (see `docs/requirements.md` non-goal 9).
- Accessed 2026-09-09.

### Transformations I applied to it

Each of these is my own work but is listed here because the assignment asks for dataset
transformations to be declared:

| Transformation | Where | Note |
|---|---|---|
| Thread reconstruction from `in_response_to_tweet_id` | `src/threadpilot/data/threads.py` | My own; the dataset ships flat rows with parent pointers, not threads |
| Brand filter to a single support account | `scripts/build_subsample.py` | Brand chosen by pre-registered criteria, `docs/requirements.md` 2.1 |
| Channel-deflection / bare-acknowledgement filter on index admission | `src/threadpilot/retrieval.py` | My own heuristic; rationale and measured effect in `decision_log.md` D8 |
| Near-duplicate removal | `src/threadpilot/data/clean.py` | My own |
| Seeded stratified subsample | `scripts/build_subsample.py` | `SEED = 20260909` |

## 2. Models

**`sentence-transformers/all-MiniLM-L6-v2`** — Reimers & Gurevych / UKPLab.
<https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
Apache 2.0. Run locally on CPU. Used as-is; no fine-tuning.

> Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese
> BERT-Networks.* EMNLP 2019. <https://arxiv.org/abs/1908.10084>

**`openai/gpt-oss-120b`** via **Groq** free tier — generation (classification, drafting,
decisions). Apache 2.0. <https://huggingface.co/openai/gpt-oss-120b>

**`qwen/qwen3.8-27b`** via **Groq** free tier — LLM-as-judge. Alibaba Cloud / Qwen.
<https://huggingface.co/Qwen>

Chosen for being a different model lineage than the generator (`decision_log.md` D7, D19).
The full live model catalogue as of access is committed at
`docs/groq_models_2026-09-09.json` — recorded because Groq's free-tier lineup had changed
from what I expected (no Llama chat models remain), and a reader should be able to see the
menu I actually chose from rather than take my word for it.
<https://console.groq.com/docs/models>

## 3. Libraries

Standard use of documented public APIs, no code copied: `pandas`, `numpy`, `pyarrow`,
`scikit-learn`, `groq`, `kagglehub`, `python-dotenv`, `tenacity`, `rich`, `matplotlib`,
`pytest`, and optionally `torch` + `sentence-transformers`. Versions in `requirements.txt`.

## 4. Methodological patterns I borrowed

These are ideas, not code, but they are not mine and the write-up should say so.

- **LLM-as-a-judge, and its known failure modes** (position bias, verbosity bias,
  self-enhancement/self-preference bias). This is the direct reason the judge runs on a
  different model family than the generator and why option order is controlled.
  > Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.*
  > <https://arxiv.org/abs/2306.05685>

- **Validating an automated judge against human labels with a chance-corrected agreement
  statistic** rather than raw agreement.
  > Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales.*
  > Educational and Psychological Measurement, 20(1), 37-46.
  > Weighted variant: Cohen, J. (1968). *Weighted kappa.* Psychological Bulletin, 70(4).
  > Implementation: `sklearn.metrics.cohen_kappa_score`.

- **Retrieval-augmented generation** as the mechanism for "grounded in how the brand has
  historically resolved similar issues".
  > Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  > Tasks.* NeurIPS 2020. <https://arxiv.org/abs/2005.11401>

- **Selective prediction / coverage-risk curves** — the framing behind reporting
  auto-handle as a curve instead of a single accuracy number (`decision_log.md` D9).
  > El-Yaniv, R. & Wiener, Y. (2010). *On the Foundations of Noise-free Selective
  > Classification.* JMLR 11.
  > Geifman, Y. & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.*
  > <https://arxiv.org/abs/1705.08500>

- **Bootstrap confidence intervals** for small-sample metrics.
  > Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.

- **Pre-registration** of selection criteria and hypotheses before looking at the data
  (`docs/requirements.md` 2.1-2.2) — borrowed from experimental-science practice, not from
  an ML paper.

## 5. Environment facts learned by reading docs, not by recall

Recorded because they materially shaped the code and because "verify, do not remember" is
the habit the assignment rewards.

- Kaggle's current auth is a `KGAT_`-prefixed token at `~/.kaggle/access_token` or the
  `KAGGLE_API_TOKEN` env var; `kaggle.json` (`KAGGLE_USERNAME`/`KAGGLE_KEY`) is now labelled
  **"Legacy API Credentials"**. I had initially assumed the legacy pair was current and was
  corrected by the docs.
  <https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md> ·
  <https://github.com/Kaggle/kagglehub> · <https://www.kaggle.com/docs/api>
- `kagglehub.dataset_download(slug)` reads that token natively and returns a local path;
  cache location is overridable with `KAGGLEHUB_CACHE`.
- Kaggle credential resolution happens at import/first-use, so `load_dotenv()` must run
  *before* importing any kaggle package. Related upstream behaviour:
  <https://github.com/Kaggle/kaggle-cli/issues/882>

## 6. AI assistance in producing this repo

Stated plainly, since the assignment values honesty about method.

- This repo was built in a pair-programming session with Claude Code (Claude Opus 5). Code,
  docs and prompts were drafted collaboratively.
- **Golden-set labels:** the LLM drafted candidate labels; **every one was reviewed and
  corrected by a human** before inclusion. The override rate is reported in
  `eval/golden/labelling_notes.md` and is itself treated as evidence about how far LLM
  labelling can be trusted on this task.
- No text was copied from another submission, tutorial or blog post.
