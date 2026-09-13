"""Configuration invariants.

These are not box-ticking tests. Each one guards a decision that, if silently
broken, would invalidate reported numbers rather than crash — the dangerous
class of bug in an evaluation harness.
"""

from __future__ import annotations

import re

import pytest

from threadpilot import config


def family(model_id: str) -> str:
    """Coarse model-lineage key.

    Groq ids look like 'openai/gpt-oss-120b' or 'qwen/qwen3.8-27b'. The vendor
    prefix is the lineage signal we care about; falling back to the leading
    alpha run handles unprefixed ids like 'allam-2-7b'.
    """
    if "/" in model_id:
        return model_id.split("/", 1)[0].lower()
    m = re.match(r"[a-z]+", model_id.lower())
    return m.group(0) if m else model_id.lower()


def test_judge_is_a_different_family_than_generator():
    """Guards decision_log.md D7.

    An LLM judge favours text from its own family, so a same-family judge would
    inflate every reply-quality metric in the report. That failure is invisible
    in the output — the numbers just come out flattering — so it gets an
    assertion rather than a comment.
    """
    assert config.JUDGE_MODEL, (
        "JUDGE_MODEL is unset. eval/judge.py must never silently fall back to "
        "the generator; set it explicitly (see D19)."
    )
    gen, judge = family(config.GEN_MODEL), family(config.JUDGE_MODEL)
    assert gen != judge, (
        f"Judge and generator share lineage '{gen}' "
        f"({config.GEN_MODEL} vs {config.JUDGE_MODEL}). This is D7's "
        "self-preference bias; pick a different family."
    )


def test_alt_judge_is_declared_and_distinct_from_primary_judge():
    """The alt judge exists to be compared against the primary on
    agreement-with-human. If it equals the primary, that comparison is vacuous.
    """
    assert config.JUDGE_MODEL_ALT
    assert config.JUDGE_MODEL_ALT != config.JUDGE_MODEL


def test_no_llama_chat_model_is_configured():
    """Guards against a plausible-looking regression.

    Groq's free tier serves no Llama chat model (verified 2026-09-09; evidence
    at docs/groq_models_2026-09-09.json). 'llama-3.3-70b-versatile' is the id a
    reader — human or model — is most likely to "restore" from memory, and it
    would fail only at call time, deep into a run.
    """
    for name, mid in (
        ("GEN_MODEL", config.GEN_MODEL),
        ("JUDGE_MODEL", config.JUDGE_MODEL),
        ("JUDGE_MODEL_ALT", config.JUDGE_MODEL_ALT),
    ):
        assert "llama-3." not in mid.lower(), (
            f"{name}={mid} looks like a Llama chat id, which Groq no longer "
            "serves. Re-check the live /models endpoint."
        )


def test_label_and_decision_temperatures_are_zero():
    """Intent labels and escalate decisions must be as reproducible as the API
    allows; sampling there would add noise to the metrics themselves.
    """
    assert config.TEMP_CLASSIFY == 0.0
    assert config.TEMP_DECIDE == 0.0
    assert config.TEMP_JUDGE == 0.0
    # Drafting is deliberately nonzero (temperature 0 prose is stilted), but
    # should stay low enough that a re-run resembles the recording.
    assert 0.0 < config.TEMP_DRAFT <= 0.5


def test_token_budgets_are_generous_enough_for_a_reasoning_generator():
    """Guards decision_log.md D20.

    gpt-oss spends budget on a reasoning channel before answering; at 120 tokens
    it returned empty content with finish_reason='stop'. Small budgets here
    would produce blank drafts recorded as valid outputs.
    """
    assert config.MAX_TOKENS_CLASSIFY >= 256
    assert config.MAX_TOKENS_DRAFT >= 512
    assert config.MAX_TOKENS_DECIDE >= 256
    assert config.MAX_TOKENS_JUDGE >= 512


def test_seed_is_fixed_and_paths_resolve():
    assert isinstance(config.SEED, int)
    assert config.REPO_ROOT.is_dir()
    assert (config.REPO_ROOT / "pyproject.toml").is_file()
    for d in (config.RAW_DIR, config.INTERIM_DIR, config.PROCESSED_DIR,
              config.GOLDEN_DIR, config.RESULTS_DIR, config.CACHE_DIR):
        assert d.is_dir(), f"{d} should be created on import of config"


def test_gates_are_probabilities_and_thresholds_are_sane():
    assert 0.0 < config.TAU_SIM < 1.0
    assert 0.0 < config.TAU_CONF < 1.0
    assert config.RETRIEVAL_K >= 1


def test_always_escalate_is_populated():
    """Was a strict-xfail through Phase 1, by design: ALWAYS_ESCALATE had to
    stay empty until the taxonomy was derived from data, so the safety gate
    could not be quietly guessed a priori (D12). Phase 2 populated it, the
    strict xfail failed as intended, and it is now a real assertion.

    Detailed invariants live in tests/test_taxonomy.py; this one only guards
    the wiring between config and the taxonomy.
    """
    assert len(config.ALWAYS_ESCALATE) > 0
