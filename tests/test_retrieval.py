"""Retrieval and leakage-guard tests.

The leakage guard is the single most important correctness property in this
repo. If a golden-set thread reaches the retrieval index, the drafter is handed
the very reply the golden row was built from, the judge scores a near-copy as
excellent, and every reply-quality number becomes fraudulent - with no error
message anywhere. It is tested from several angles for that reason.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from threadpilot import config, retrieval
from threadpilot.retrieval import LeakageError, Precedent, RetrievalIndex

pytestmark = pytest.mark.skipif(
    not retrieval.VECTORS_PATH.exists(),
    reason="no retrieval index; run python scripts/build_index.py",
)


@pytest.fixture(scope="module")
def index() -> RetrievalIndex:
    return RetrievalIndex.load()


@pytest.fixture(scope="module")
def golden_threads() -> set[int]:
    if not config.GOLDEN_JSONL.exists():
        pytest.skip("no golden set yet")
    return {int(json.loads(l)["thread_id"])
            for l in config.GOLDEN_JSONL.open(encoding="utf-8")}


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------
def test_no_golden_thread_is_in_the_index(index, golden_threads):
    """THE test. A failure here invalidates every reported quality metric."""
    overlap = index.thread_ids & golden_threads
    assert not overlap, (
        f"{len(overlap)} golden threads leaked into the retrieval index: "
        f"{sorted(overlap)[:5]}")


def test_dev_silver_threads_are_also_excluded(index):
    """Silver tunes the thresholds golden then reports on. Retrieving a silver
    row's own reply while tuning would optimise against a leaked answer, and the
    tuned thresholds would not transfer."""
    p = config.GOLDEN_DIR / "dev_silver_candidates.jsonl"
    if not p.exists():
        pytest.skip("no dev_silver")
    silver = {int(json.loads(l)["thread_id"]) for l in p.open(encoding="utf-8")}
    assert not (index.thread_ids & silver)


def test_assert_disjoint_from_actually_raises():
    """A guard that cannot fail is not a guard. Verify it detects a real leak
    rather than merely being called."""
    meta = pd.DataFrame({
        "pair_id": ["a-1", "b-2"], "thread_id": [111, 222],
        "customer_msg": ["m1", "m2"], "brand_reply": ["r1", "r2"],
        "reply_kind": ["self_contained"] * 2, "implies_escalation": [False, False],
        "has_url": [False, False], "dup_count": [1, 1],
    })
    vecs = np.eye(2, config.EMBED_DIM, dtype=np.float32)
    idx = RetrievalIndex(vecs, meta, {})

    idx.assert_disjoint_from({999})          # no overlap -> fine
    with pytest.raises(LeakageError) as e:
        idx.assert_disjoint_from({222, 999})
    assert "222" in str(e.value)


def test_build_index_refuses_to_silently_build_unguarded(monkeypatch):
    """Leaking must take a deliberate act, not a forgotten argument.

    build_index(held_out=None) resolves the held-out set itself and raises if it
    finds none, rather than defaulting to an empty set and quietly indexing
    everything.
    """
    monkeypatch.setattr(retrieval, "held_out_thread_ids", lambda: set())
    with pytest.raises(LeakageError, match="no held-out threads"):
        retrieval.build_index(held_out=None, save=False)


def test_index_info_records_what_the_guard_excluded(index):
    """The guard's effect is reported as a number, not left implicit."""
    info = index.info
    assert info["n_excluded_by_leakage_guard"] > 0, (
        "guard excluded nothing - suspicious, since golden rows were drawn "
        "from the same subsample the index is built from")
    assert info["n_indexed"] == len(index)
    assert info["n_indexed"] + info["n_excluded_by_leakage_guard"] \
        == info["n_usable_canonical"]


# --------------------------------------------------------------------------
# What is in the index
# --------------------------------------------------------------------------
def test_only_usable_precedent_kinds_are_indexed(index):
    """Bare acknowledgements, bare channel switches, fragments and truncated
    replies must never become precedent - a draft grounded in 'DM us' would
    score well on tone and badly on usefulness (D8/D21)."""
    from threadpilot.data.clean import ReplyKind
    usable = {k.value for k in ReplyKind if k.usable_as_precedent}
    assert set(index.meta["reply_kind"].unique()) <= usable


def test_vectors_are_unit_normalised(index):
    """search() treats the dot product AS cosine similarity, which is only true
    for unit vectors. Un-normalised rows would silently rank by magnitude."""
    norms = np.linalg.norm(index.vectors[:200], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_vectors_and_meta_stay_aligned(index):
    assert len(index.vectors) == len(index.meta)
    with pytest.raises(ValueError):
        RetrievalIndex(index.vectors[:10], index.meta, {})


# --------------------------------------------------------------------------
# Search behaviour
# --------------------------------------------------------------------------
def test_search_returns_k_sorted_descending(index):
    res = index.search_text("where is my parcel it has not arrived", k=5)
    assert len(res) == 5
    sims = [p.similarity for p in res]
    assert sims == sorted(sims, reverse=True)
    assert all(-1.01 <= s <= 1.01 for s in sims)


def test_search_is_semantic_not_lexical(index):
    """Retrieval must find paraphrases, else grounding only works for customers
    who happen to use the brand's vocabulary."""
    res = index.search_text("my package never showed up", k=5)
    joined = " ".join(p.customer_msg.lower() for p in res)
    assert any(w in joined for w in ("deliver", "parcel", "package", "arrive"))


def test_an_unrelated_message_scores_lower_than_a_related_one(index):
    """Gate G2 escalates on low max-similarity, so similarity must actually
    track relatedness or the gate is noise."""
    related = index.search_text("my parcel has not been delivered yet", k=1)[0]
    unrelated = index.search_text(
        "the mitochondrion is the powerhouse of the cell", k=1)[0]
    assert related.similarity > unrelated.similarity


def test_precedent_render_includes_the_similarity_score(index):
    """design.md: a retrieval claim without its score is not inspectable."""
    p = index.search_text("late delivery", k=1)[0]
    assert "sim" in p.render() and f"{p.similarity:.2f}" in p.render()
