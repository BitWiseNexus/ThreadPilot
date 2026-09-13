"""Single source of truth for paths, seeds, models and thresholds.

Nothing else in the codebase hardcodes a path, a seed, a model name or a
threshold. Two reasons, both about the eval rather than tidiness:

  * a threshold that appears in two files will eventually disagree with itself,
    and the report would then describe a configuration that was never run;
  * every tunable is printed into eval/results/run_config.json alongside the
    metrics, so a reported number is always traceable to the exact config that
    produced it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
# One seed, threaded through subsampling, KMeans, golden-set stratification
# and bootstrap resampling. Residual nondeterminism is the LLM API itself,
# which is why responses are cached (decision_log.md D10).
SEED = 20260909

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # gitignored: twcs.csv (~500MB)
INTERIM_DIR = DATA_DIR / "interim"    # gitignored: regenerable intermediates
PROCESSED_DIR = DATA_DIR / "processed"  # COMMITTED: seeded subsample

EVAL_DIR = REPO_ROOT / "eval"
GOLDEN_DIR = EVAL_DIR / "golden"
RESULTS_DIR = EVAL_DIR / "results"

CACHE_DIR = REPO_ROOT / "cache"
LLM_CACHE_DB = CACHE_DIR / "llm_cache.sqlite"  # COMMITTED: offline replay

RAW_CSV = RAW_DIR / "twcs.csv"
SUBSAMPLE_PARQUET = PROCESSED_DIR / "subsample.parquet"
GOLDEN_JSONL = GOLDEN_DIR / "golden_set.jsonl"
DEV_SILVER_JSONL = GOLDEN_DIR / "dev_silver.jsonl"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, GOLDEN_DIR, RESULTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Brand (Phase 1)
# --------------------------------------------------------------------------
# Chosen by measurement against criteria pre-registered in requirements.md 2.1,
# then by ROBUSTNESS when the composite score turned out unable to separate the
# top two (see decision_log.md D25 and eval/results/brand_sensitivity.md).
# AmazonHelp is the only brand in the top 3 under all four defensible
# definitions of "usable precedent", and carries ~4x the retrieval index of the
# runner-up. Known cost: the lowest intent-diversity proxy of the leaders.
BRAND = "AmazonHelp"

# Committed subsample size. "A few thousand tweets from one brand" per the
# brief; large enough for a useful retrieval index plus a 200-example golden set
# and a ~100-example silver dev set, small enough to commit and to embed on CPU
# inside the 15-minute reproducibility budget.
SUBSAMPLE_PAIRS = 8000

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# Generation: classification, drafting, decisions.
# NOTE: verified against the live /models endpoint on 2026-09-09 (evidence saved
# at docs/groq_models_2026-09-09.json). Groq's free tier no longer serves ANY
# Llama chat model, so the originally planned llama-3.3-70b-versatile does not
# exist. Do not "fix" this back to a Llama id from memory - re-check the endpoint.
GEN_MODEL = os.environ.get("THREADPILOT_GEN_MODEL", "openai/gpt-oss-120b")

# Judge: a DIFFERENT model family than GEN_MODEL. An LLM judge systematically
# favours text from its own family, so judging gpt-oss output with gpt-oss would
# inflate every quality metric in a way no amount of prompt care fixes.
# gpt-oss (OpenAI) generating / Qwen (Alibaba Cloud) judging gives genuinely
# different lineages, not merely different sizes. See decision_log.md D7, D19.
JUDGE_MODEL = os.environ.get("THREADPILOT_JUDGE_MODEL", "qwen/qwen3.8-27b")

# Alternate judge, used ONLY in the Phase 6 judge-validation study: we measure
# both candidates against human re-scores and report both agreements. Note this
# one shares a family with the generator, so a high score from it is suspect by
# construction - it is a comparison point, not a candidate for headline numbers.
JUDGE_MODEL_ALT = "openai/gpt-oss-120b"

# gpt-oss models are REASONING models: they emit a separate `reasoning` channel
# and only then the answer. Two consequences that cost real debugging time:
#   1. With too small a token budget, content comes back EMPTY while
#      finish_reason is still "stop" - a silent failure. llm.py treats empty
#      content as an error rather than as a valid response.
#   2. The reasoning trace is worth keeping: it is logged and used as evidence
#      in the Phase 7 failure analysis.
GEN_REASONING_EFFORT = "low"   # sufficient for label/decision tasks, and cheaper
MAX_TOKENS_CLASSIFY = 512
MAX_TOKENS_DRAFT = 1024
MAX_TOKENS_DECIDE = 768
MAX_TOKENS_JUDGE = 1024

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
# Pin the revision, not just the name: a silently updated model would change
# every similarity score and therefore every gate decision.
EMBED_REVISION = "main"  # replaced with a commit sha in Phase 4

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Temperatures. 0 wherever the output is a label or a decision, so those are
# as close to reproducible as the API allows. Drafting gets a small nonzero
# value because a reply at temperature 0 tends to be stilted and repetitive.
TEMP_CLASSIFY = 0.0
TEMP_DECIDE = 0.0
TEMP_DRAFT = 0.3
TEMP_JUDGE = 0.0

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
RETRIEVAL_K = 5  # tuned on dev_silver ONLY, never on the golden set (D5)

# --------------------------------------------------------------------------
# Escalation gates (see docs/architecture.md section 5)
# --------------------------------------------------------------------------
# Gates are ONE-DIRECTIONAL: they can only force escalate, never grant
# auto-handle. This encodes the cost asymmetry structurally rather than
# trusting the model to respect it.
TAU_SIM = 0.45   # G2: below this, no comparable precedent exists
TAU_CONF = 0.60  # G3: below this, we do not even know what the message is

# G1: intents that always go to a human regardless of model confidence.
# Sourced from the taxonomy rather than restated here, so the disposition that
# drives the gate is the same one documented next to each intent's definition.
# Two copies would eventually disagree and the report would then describe a
# policy that was never run.
#
# Imported lazily inside the function to keep config import-cycle-free
# (taxonomy imports nothing from config, but that could change).
def _always_escalate() -> tuple[str, ...]:
    from .taxonomy import ALWAYS_ESCALATE as _ae
    return _ae


ALWAYS_ESCALATE: tuple[str, ...] = _always_escalate()


@dataclass(frozen=True)
class RunConfig:
    """Snapshot of every tunable, serialised next to each results file."""

    seed: int = SEED
    gen_model: str = GEN_MODEL
    judge_model: str = JUDGE_MODEL
    judge_model_alt: str = JUDGE_MODEL_ALT
    gen_reasoning_effort: str = GEN_REASONING_EFFORT
    embed_model: str = EMBED_MODEL
    embed_revision: str = EMBED_REVISION
    retrieval_k: int = RETRIEVAL_K
    tau_sim: float = TAU_SIM
    tau_conf: float = TAU_CONF
    always_escalate: tuple[str, ...] = field(default_factory=lambda: ALWAYS_ESCALATE)
    temp_classify: float = TEMP_CLASSIFY
    temp_draft: float = TEMP_DRAFT
    temp_decide: float = TEMP_DECIDE
    temp_judge: float = TEMP_JUDGE

    def to_dict(self) -> dict:
        from dataclasses import asdict

        d = asdict(self)
        d["always_escalate"] = list(d["always_escalate"])
        return d
