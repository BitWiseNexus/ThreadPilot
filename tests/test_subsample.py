"""Integrity of the locally-built subsample.

The subsample is the artifact every later phase reads, so a defect here is a
defect in every downstream number.

It is NOT committed - it derives from a CC BY-NC-SA 4.0 dataset and is rebuilt
locally instead (decision_log.md D26). That makes these tests more important,
not less: an evaluator running them is checking that *their* rebuild matches the
one the reported numbers came from. The committed subsample_meta.json is the
reference they check against.

Everything here skips cleanly when the subsample has not been built yet, and the
byte-identity rebuild test additionally needs the raw CSV.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from threadpilot import config
from threadpilot.data.clean import ReplyKind

pytestmark = pytest.mark.skipif(
    not config.SUBSAMPLE_PARQUET.exists(),
    reason="subsample not built yet; run python scripts/build_subsample.py",
)

REQUIRED_COLUMNS = {
    "pair_id", "thread_id", "parent_id", "reply_id", "customer_msg",
    "brand_reply", "raw_customer", "raw_reply", "created_at", "reply_kind",
    "usable_precedent", "implies_escalation", "has_url", "has_info_request",
    "n_chars", "mojibake", "dedupe_key", "dup_count", "is_canonical",
    "lang", "is_opener",
}


@pytest.fixture(scope="module")
def sub() -> pd.DataFrame:
    return pd.read_parquet(config.SUBSAMPLE_PARQUET)


@pytest.fixture(scope="module")
def meta() -> dict:
    return json.loads(
        (config.PROCESSED_DIR / "subsample_meta.json").read_text("utf-8"))


def test_schema_is_complete(sub: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(sub.columns)
    assert not missing, f"subsample is missing columns: {sorted(missing)}"


def test_pair_ids_unique(sub: pd.DataFrame) -> None:
    """A duplicated pair would double-count in every metric."""
    assert sub["pair_id"].is_unique


def test_thread_ids_present(sub: pd.DataFrame) -> None:
    """thread_id is what the Phase 4 leakage guard keys on.

    The dataset ships no conversation id, so these are computed by walking parent
    pointers. If they were null the guard would still *pass* while leaking golden
    replies into the retrieval index -- the single worst silent failure available
    to this project.

    An earlier version asserted `nunique() < len(sub)`, on the assumption that
    some conversations contribute several brand replies. That stopped holding
    once the subsample was restricted to thread openers, and the right response
    was a stronger invariant rather than a loosened one: for an opener, the
    customer tweet IS the conversation root, so thread_id must equal parent_id.
    That would catch a broken parent-walk, which the old inequality would not.
    """
    assert sub["thread_id"].notna().all()
    assert (sub["thread_id"] > 0).all()
    if sub["is_opener"].all():
        mismatched = sub[sub["thread_id"] != sub["parent_id"]]
        assert mismatched.empty, (
            f"{len(mismatched)} opener rows whose thread_id != parent_id; "
            "the parent-pointer walk is wrong"
        )
    else:
        assert sub["thread_id"].nunique() <= len(sub)


def test_no_empty_text(sub: pd.DataFrame) -> None:
    assert (sub["customer_msg"].str.len() > 0).all()
    assert (sub["brand_reply"].str.len() > 0).all()


def test_usable_flag_matches_reply_kind(sub: pd.DataFrame) -> None:
    """usable_precedent must be derivable from reply_kind, not drift from it.

    Two sources of truth for index admission would eventually disagree, and the
    report would then describe an index that was never built.
    """
    expected = sub["reply_kind"].map(
        lambda k: ReplyKind(k).usable_as_precedent)
    assert (sub["usable_precedent"] == expected).all()


def test_escalation_flag_matches_reply_kind(sub: pd.DataFrame) -> None:
    expected = sub["reply_kind"].map(lambda k: ReplyKind(k).implies_escalation)
    assert (sub["implies_escalation"] == expected).all()


def test_all_reply_kinds_are_known(sub: pd.DataFrame) -> None:
    known = {k.value for k in ReplyKind}
    assert set(sub["reply_kind"].unique()) <= known


def test_canonical_flag_is_one_per_dedupe_group(sub: pd.DataFrame) -> None:
    """Duplicates are GROUPED, not deleted, so the golden set can still see true
    message frequency while the index uses one row per group."""
    per_group = sub[sub["is_canonical"]].groupby("dedupe_key").size()
    assert (per_group == 1).all(), "a dedupe group has multiple canonical rows"
    # Every group present must have exactly one canonical representative.
    assert sub["dedupe_key"].nunique() == int(sub["is_canonical"].sum())


def test_index_candidate_pool_is_non_trivial(sub: pd.DataFrame) -> None:
    """The retrieval index is usable AND canonical rows. If that pool collapsed,
    grounding would silently degrade to almost nothing."""
    pool = sub["usable_precedent"] & sub["is_canonical"]
    assert pool.sum() >= 3000, f"only {pool.sum()} index candidates"


def test_meta_matches_the_data(sub: pd.DataFrame, meta: dict) -> None:
    """The committed metadata is what the report quotes; it must not go stale."""
    assert meta["brand"] == config.BRAND
    assert meta["seed"] == config.SEED
    assert meta["n_pairs"] == len(sub)
    assert meta["n_threads"] == sub["thread_id"].nunique()
    assert meta["n_usable_precedent"] == int(sub["usable_precedent"].sum())
    assert meta["n_canonical"] == int(sub["is_canonical"].sum())


def test_unit_of_analysis_is_enforced(sub: pd.DataFrame, meta: dict) -> None:
    """The subsample must actually contain what the project claims to triage.

    requirements.md non-goal 5 defines the unit as a single inbound message, and
    Phase 2 clustering showed language outranked intent as a signal. Both filters
    are therefore load-bearing, not cosmetic: without them the taxonomy measures
    chitchat and language ID. Asserted here so a future edit cannot quietly drop
    them and leave the report describing a different dataset.
    """
    assert meta["filters"]["thread_openers_only"] is True
    assert meta["filters"]["english_only"] is True
    assert sub["is_opener"].all(), "non-opener rows leaked into the subsample"
    assert (sub["lang"] == "en").all(), "non-English rows leaked into the subsample"


def test_exclusion_funnel_is_recorded(meta: dict) -> None:
    """Every drop is counted, so a reviewer can disagree with the policy and
    reproduce a different one instead of having to trust the final number."""
    f = meta["funnel"]
    for key in ("brand_replies", "pairs_with_parent", "after_opener_filter",
                "dropped_non_english", "dropped_language_breakdown"):
        assert key in f, f"funnel is missing {key}"
    assert f["after_opener_filter"] <= f["pairs_with_parent"]
    assert f["dropped_non_english"] > 0, (
        "no non-English rows dropped - the language filter is probably not "
        "running, since AmazonHelp demonstrably answers in several languages"
    )


def test_mojibake_is_flagged_and_rare(sub: pd.DataFrame) -> None:
    """Pre-existing U+FFFD damage in the Kaggle source. Flagged, not repaired.
    If this ever spikes it means a decoding regression on our side."""
    rate = float(sub["mojibake"].mean())
    assert rate < 0.02, f"mojibake rate {rate:.3%} is too high to be source damage"


@pytest.mark.skipif(
    not config.RAW_CSV.exists(),
    reason="raw twcs.csv absent; byte-identity rebuild needs it "
           "(python tasks.py data)",
)
def test_rebuild_is_byte_identical(tmp_path) -> None:
    """The reproducibility promise, actually tested rather than asserted.

    Determinism here is not free: it required seeding every shuffle and sorting
    the frame by pair_id before writing, because the trim step's output order
    otherwise depends on how the permutation fell.
    """
    original = config.SUBSAMPLE_PARQUET
    meta_path = config.PROCESSED_DIR / "subsample_meta.json"
    backup = tmp_path / "subsample.parquet.bak"
    meta_backup = tmp_path / "subsample_meta.json.bak"
    shutil.copy2(original, backup)
    # The builder also rewrites the metadata with a fresh `generated` timestamp.
    # Without restoring it, simply running the test suite leaves the git tree
    # dirty, which makes a clean checkout look modified to anyone reviewing.
    shutil.copy2(meta_path, meta_backup)
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    try:
        r = subprocess.run(
            [sys.executable, "scripts/build_subsample.py"],
            cwd=config.REPO_ROOT, capture_output=True, text=True, timeout=900,
        )
        assert r.returncode == 0, f"rebuild failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        after = hashlib.sha256(original.read_bytes()).hexdigest()
        assert before == after, (
            "rebuilding the subsample from the same seed produced different "
            "bytes. Something in the pipeline is order- or hash-dependent."
        )
    finally:
        shutil.copy2(backup, original)
        shutil.copy2(meta_backup, meta_path)
