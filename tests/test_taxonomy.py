"""Taxonomy invariants.

The taxonomy defines what the classifier can say and which intents may never be
automated, so errors here are not local: they change every per-class metric and
the escalation policy simultaneously.
"""

from __future__ import annotations

import pytest

from threadpilot import config
from threadpilot.taxonomy import (
    ALWAYS_ESCALATE, BY_NAME, INTENT_NAMES, INTENTS, Disposition,
    describe, prompt_block,
)


def test_names_are_unique_and_snake_case():
    assert len(INTENT_NAMES) == len(set(INTENT_NAMES))
    for n in INTENT_NAMES:
        assert n.islower() and " " not in n, n


def test_other_bucket_exists():
    """requirements.md asks for an explicit residual class. Forcing every
    message into a named intent would hide the taxonomy's own coverage gaps."""
    assert "other" in BY_NAME


def test_every_named_intent_has_real_examples_and_a_boundary():
    """Boundary notes are what make hand-labelling consistent in Phase 3 and
    what failure analysis refers back to in Phase 7. An intent without one is
    an intent two labellers will disagree about."""
    for i in INTENTS:
        if i.name == "other":
            continue
        assert len(i.examples) >= 3, f"{i.name} has too few examples"
        assert all(len(e) > 20 for e in i.examples), f"{i.name} has a stub example"
        assert len(i.boundary) > 60, f"{i.name} boundary note is too thin"
        # The real requirement is that a boundary names the sibling intent it
        # is contrasted against, so a labeller knows which pair to disambiguate.
        # An earlier version of this test looked for the literal string "vs ",
        # which failed prime_membership ("OVERLAPS HEAVILY with delivery_delay")
        # - a boundary that does its job in different words. Testing the
        # formatting convention rather than the property was the bug.
        others = {n for n in INTENT_NAMES if n != i.name and n != "other"}
        named = [o for o in others if o in i.boundary]
        assert named, (
            f"{i.name} boundary names no sibling intent; a labeller cannot tell "
            f"which pair it disambiguates. Boundary: {i.boundary[:120]}")


def test_dispositions_are_assigned():
    for i in INTENTS:
        assert isinstance(i.disposition, Disposition)


def test_always_escalate_is_derived_not_restated():
    """config must source the gate from the taxonomy. Two copies of a policy
    eventually disagree, and the report would then describe a configuration
    that was never run."""
    assert set(config.ALWAYS_ESCALATE) == set(ALWAYS_ESCALATE)
    assert set(ALWAYS_ESCALATE) == {
        i.name for i in INTENTS if i.disposition is Disposition.ALWAYS_ESCALATE}


def test_always_escalate_is_populated_but_minimal():
    """Flipped from a strict-xfail placeholder once Phase 2 derived the
    taxonomy.

    Both halves matter. Empty means the safety gate does nothing. Too large
    means the system achieves safety by escalating everything, which
    requirements.md section 6 calls out as worthless - a trivial
    always-escalate baseline already scores perfectly on false-auto-handle at
    zero coverage, so a bloated gate would be indistinguishable from it.
    """
    assert len(ALWAYS_ESCALATE) >= 1, "safety gate G1 would be inert"
    named = [i for i in INTENTS if i.name != "other"]
    assert len(ALWAYS_ESCALATE) < len(named) / 2, (
        "G1 covers half the taxonomy; auto-handle coverage is capped so low "
        "that the system cannot be distinguished from always-escalate")


def test_escalation_gate_leaves_meaningful_coverage_available():
    """The share of traffic G1 forecloses must leave room to demonstrate value.

    If ALWAYS_ESCALATE covered most traffic, a high 'safe-to-send' rate would be
    a statement about the gate, not about the system.
    """
    capped = sum(i.approx_share for i in INTENTS if i.name in ALWAYS_ESCALATE)
    assert 0.05 < capped < 0.50, f"G1 forecloses {capped:.0%} of traffic"


def test_highest_stakes_intents_are_gated():
    """account_security and refund_billing are the cases where a wrong
    auto-reply does real harm - compromised accounts and money commitments."""
    assert "account_security" in ALWAYS_ESCALATE
    assert "refund_billing" in ALWAYS_ESCALATE


def test_shares_are_plausible_and_overlap_is_documented():
    """Shares sum slightly above 1.0 by design.

    account_security has no cluster of its own (only ~2.5% of traffic, so KMeans
    scattered it across others), so its share overlaps with the clusters it was
    drawn from. Asserting an exact sum of 1.0 would force a tidy fiction; this
    checks the looseness stays small enough to be the documented overlap rather
    than a bookkeeping error.
    """
    total = sum(i.approx_share for i in INTENTS)
    assert 0.95 <= total <= 1.10, f"shares sum to {total:.3f}"
    scattered = [i for i in INTENTS if i.cluster is None and i.name != "other"]
    assert scattered, "expected at least one intent with no source cluster"
    for i in scattered:
        assert "cluster" in i.__doc__ if i.__doc__ else True


def test_cluster_provenance_is_recorded():
    """Every intent either names the cluster it came from or is explicitly
    cluster-less, so the derivation stays auditable against
    eval/results/taxonomy_clusters.md."""
    clusters = [i.cluster for i in INTENTS if i.cluster is not None]
    assert len(clusters) == len(set(clusters)), "two intents claim one cluster"


def test_prompt_block_carries_boundaries_not_just_names():
    """The classifier prompt must include boundary notes: adjacent-intent
    confusion is the expected failure mode, and boundaries are the information
    that resolves it."""
    block = prompt_block()
    for n in INTENT_NAMES:
        assert n in block
    assert block.count("boundary:") >= len(INTENT_NAMES) - 1


@pytest.mark.parametrize("name", [n for n in INTENT_NAMES])
def test_describe_renders(name: str):
    out = describe(name)
    assert name in out and "%" in out
