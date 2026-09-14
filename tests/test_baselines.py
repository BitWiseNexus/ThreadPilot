"""Baseline tests.

Baselines only mean something if they are (a) genuinely independent of the
system they are compared against, and (b) scored through the identical harness.
Both are asserted here, because a baseline that quietly shares code with the
pipeline, or that is scored on a friendlier path, produces a flattering gap that
is pure artifact.
"""

from __future__ import annotations

import pickle

import pytest

from threadpilot import baselines, config, llm, retrieval, taxonomy
from threadpilot.baselines import (
    CANNED_REPLY, Retrieval1NNBaseline, SimpleBaseline, TrivialBaseline,
)
from threadpilot.pipeline import TriageResult

TEXTS = [
    "my parcel never arrived and it says delivered",
    "I want my money back for order 403-7128266-4201160",
    "how do I set up my new echo dot",
]


@pytest.fixture(scope="module")
def index():
    if not retrieval.VECTORS_PATH.exists():
        pytest.skip("no retrieval index")
    return retrieval.RetrievalIndex.load()


@pytest.fixture(scope="module")
def simple():
    p = config.INTERIM_DIR / "baseline_simple_cluster.pkl"
    if not p.exists():
        pytest.skip("baselines not built; run scripts/build_baselines.py")
    with p.open("rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------
# No baseline may call an LLM
# --------------------------------------------------------------------------
def test_no_baseline_calls_the_llm(monkeypatch, index, simple):
    """If a baseline quietly used the LLM, the reported gap would be measuring
    the same model against itself. Also the practical reason baselines are free
    to refit: they cost no token budget."""
    def boom(*a, **kw):
        raise AssertionError("a baseline called the LLM")

    monkeypatch.setattr(llm, "complete", boom)

    TrivialBaseline("delivery_delay").triage(TEXTS)
    simple.triage(TEXTS)
    Retrieval1NNBaseline.build(index, classifier=simple.pipeline).triage(TEXTS)


# --------------------------------------------------------------------------
# Identical output shape -> identical scoring path
# --------------------------------------------------------------------------
@pytest.mark.parametrize("make", ["trivial", "simple", "nn"])
def test_every_baseline_emits_triage_results(make, index, simple):
    b = {"trivial": lambda: TrivialBaseline("delivery_delay"),
         "simple": lambda: simple,
         "nn": lambda: Retrieval1NNBaseline.build(index, classifier=simple.pipeline)
         }[make]()
    res = b.triage(TEXTS, pair_ids=["a", "b", "c"])
    assert len(res) == len(TEXTS)
    for r in res:
        assert isinstance(r, TriageResult)
        assert r.classification.intent in taxonomy.BY_NAME
        assert r.decision.decision in ("auto_handle", "escalate")
        assert isinstance(r.draft.reply, str)
        r.to_dict()          # must serialise like a pipeline result


# --------------------------------------------------------------------------
# The trivial baseline's defining property
# --------------------------------------------------------------------------
def test_trivial_never_auto_handles_and_that_is_the_point():
    """It scores a PERFECT false-auto-handle rate at ZERO coverage.

    This is the whole reason it exists: any single-number safety metric that
    ranks this first is broken, which is the argument for reporting auto-handle
    as a coverage/risk curve (D9). If this ever auto-handles, the demonstration
    is lost.
    """
    res = TrivialBaseline("delivery_delay").triage(TEXTS)
    assert all(r.decision.decision == "escalate" for r in res)
    assert all(r.draft.reply == CANNED_REPLY for r in res)
    assert len({r.classification.intent for r in res}) == 1


def test_trivial_picks_the_actual_majority():
    b = TrivialBaseline.fit(["a"] * 5 + ["b"] * 2)
    assert b.majority_intent == "a"


# --------------------------------------------------------------------------
# retrieval_1nn must copy, not generate
# --------------------------------------------------------------------------
def test_1nn_returns_a_precedent_reply_verbatim(index, simple):
    """The comparison only isolates generation if the reply is genuinely
    unmodified. Any paraphrasing here would make it a weak generator instead of
    a clean control."""
    b = Retrieval1NNBaseline.build(index, classifier=simple.pipeline)
    for r in b.triage(TEXTS):
        if r.precedents:
            assert r.draft.reply == r.precedents[0].brand_reply


def test_1nn_and_simple_share_intents_so_only_the_reply_differs(index, simple):
    """They are given the same classifier on purpose: the pair isolates the
    REPLY step. If their intents diverged, any quality gap would be confounded
    by classification differences."""
    b = Retrieval1NNBaseline.build(index, classifier=simple.pipeline)
    a_int = [r.classification.intent for r in simple.triage(TEXTS)]
    b_int = [r.classification.intent for r in b.triage(TEXTS)]
    assert a_int == b_int


# --------------------------------------------------------------------------
# Known structural limitation, asserted so it cannot be forgotten
# --------------------------------------------------------------------------
def test_cluster_trained_baseline_cannot_predict_clusterless_intents(simple):
    """`account_security` and `other` have no Phase 2 cluster, so a
    cluster-pseudo-labelled classifier can NEVER predict them - including the
    highest-stakes intent in the taxonomy.

    That is a real structural advantage of the LLM approach (it can label
    intents with no cluster), and it is asserted here so the comparison is
    read with it in mind rather than quietly benefiting the baseline.
    """
    classes = set(simple.pipeline.classes_)
    assert "account_security" not in classes
    assert len(classes) < len(taxonomy.INTENT_NAMES)


def test_templates_cover_every_intent():
    """A missing template would silently fall back to the canned reply and make
    the template baseline look worse than it is."""
    p = config.RESULTS_DIR / "baselines_info.json"
    if not p.exists():
        pytest.skip("baselines not built")
    import json
    info = json.loads(p.read_text(encoding="utf-8"))
    for name in taxonomy.INTENT_NAMES:
        assert name in info["templates"], f"no template for {name}"


def test_templates_are_real_brand_replies_not_hand_written():
    """Hand-written templates would make the baseline a measure of my prose
    rather than of the approach. Most should be drawn from the corpus; only the
    two clusterless intents fall back."""
    p = config.RESULTS_DIR / "baselines_info.json"
    if not p.exists():
        pytest.skip("baselines not built")
    import json
    info = json.loads(p.read_text(encoding="utf-8"))
    fallbacks = set(baselines.FALLBACK_TEMPLATES.values()) | {CANNED_REPLY}
    derived = [v for v in info["templates"].values() if v not in fallbacks]
    assert len(derived) >= 8, "too few templates came from real replies"
