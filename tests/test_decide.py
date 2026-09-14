"""Gate-layer tests.

The gates are the auditable floor under a decision that Phase 3 measured at
Cohen kappa 0.273 between two capable models. Their defining property is that
they are ONE-DIRECTIONAL: a gate may force escalate, never grant auto_handle.
Losing that would silently convert the safety layer into a coin flip, so it is
asserted from both directions.

No network: everything here is pure logic over constructed inputs.
"""

from __future__ import annotations

import pytest

from threadpilot import config, taxonomy
from threadpilot.classify import Classification
from threadpilot.decide import Gate, account_specific, decide
from threadpilot.draft import Draft
from threadpilot.retrieval import Precedent


def prec(sim: float) -> Precedent:
    return Precedent(pair_id="p", thread_id=1, customer_msg="c", brand_reply="r",
                     reply_kind="self_contained", similarity=sim,
                     implies_escalation=False)


def good_draft(decision="auto_handle", flags=()) -> Draft:
    return Draft(reply="Sorry about that, could you check your tracking?",
                 decision_proposal=decision, reason="generic safe reply",
                 risk_flags=tuple(flags))


SAFE_INTENT = next(i.name for i in taxonomy.INTENTS
                   if i.name not in taxonomy.ALWAYS_ESCALATE and i.name != "other")
CONFIDENT = Classification(intent=SAFE_INTENT, confidence=0.95)


def run(**kw):
    base = dict(text="where is my parcel", classification=CONFIDENT,
                draft=good_draft(), precedents=[prec(0.80)])
    base.update(kw)
    return decide(**base)


# --------------------------------------------------------------------------
# The one-directional property
# --------------------------------------------------------------------------
def test_clean_case_accepts_the_model_proposal():
    d = run()
    assert d.decision == "auto_handle"
    assert d.gates_fired == ()
    assert not d.overrode_model


def test_no_gate_can_turn_an_escalate_proposal_into_auto_handle():
    """Gates add caution and never remove it. If the model escalates, the
    outcome is escalate regardless of how clean the signals look."""
    d = run(draft=good_draft(decision="escalate"))
    assert d.decision == "escalate"
    assert d.gates_fired == ()          # none needed to fire
    assert not d.overrode_model


@pytest.mark.parametrize("kw,gate", [
    (dict(classification=Classification(intent=taxonomy.ALWAYS_ESCALATE[0],
                                        confidence=0.99)),
     Gate.G1_ALWAYS_ESCALATE_INTENT),
    (dict(precedents=[prec(0.10)]), Gate.G2_NO_PRECEDENT),
    (dict(classification=Classification(intent=SAFE_INTENT, confidence=0.10)),
     Gate.G3_LOW_CONFIDENCE),
    (dict(text="please look at order 123-4567890-1234567"),
     Gate.G4_ACCOUNT_SPECIFIC),
    (dict(draft=good_draft(flags=("E3",))), Gate.G5_MODEL_RISK_FLAG),
    (dict(draft=Draft(reply="", decision_proposal="auto_handle", reason="")),
     Gate.G6_EMPTY_DRAFT),
])
def test_each_gate_forces_escalate_over_an_auto_handle_proposal(kw, gate):
    d = run(**kw)
    assert d.decision == "escalate"
    assert gate in d.gates_fired
    assert d.overrode_model, "overriding the model must be recorded, not silent"


def test_empty_precedent_list_fires_the_no_precedent_gate():
    """The no-retrieval ablation must not accidentally become more permissive:
    with no precedent, max similarity is 0 and G2 fires."""
    d = run(precedents=[])
    assert d.decision == "escalate"
    assert Gate.G2_NO_PRECEDENT in d.gates_fired


# --------------------------------------------------------------------------
# Reasons must be auditable
# --------------------------------------------------------------------------
def test_reason_names_the_gate_and_the_number_behind_it():
    """design.md: a team lead should get a named reason, not a sentence the
    model generated about itself."""
    d = run(precedents=[prec(0.11)])
    assert d.reason.startswith("G2")
    assert "0.11" in d.reason and f"{config.TAU_SIM:.2f}" in d.reason

    d2 = run(classification=Classification(intent=SAFE_INTENT, confidence=0.12))
    assert "0.12" in d2.reason


def test_multiple_gates_are_all_reported():
    d = run(classification=Classification(intent=taxonomy.ALWAYS_ESCALATE[0],
                                          confidence=0.05),
            precedents=[prec(0.05)])
    assert len(d.gates_fired) >= 3
    for g in d.gates_fired:
        assert g.value in d.reason


# --------------------------------------------------------------------------
# The ablation
# --------------------------------------------------------------------------
def test_gates_off_leaves_the_model_proposal_untouched():
    """Phase 6 ablation: the rule layer must justify itself with a number, so
    turning it off has to actually turn it off."""
    d = run(classification=Classification(intent=taxonomy.ALWAYS_ESCALATE[0],
                                          confidence=0.01),
            precedents=[prec(0.01)], use_gates=False)
    assert d.decision == "auto_handle"
    assert d.gates_fired == ()


# --------------------------------------------------------------------------
# G4 must stay narrow
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "please check order 403-7128266-4201160",
    "my order no: 12345678 has not arrived",
    "tracking number: 1Z999AA10123456784",
    "contact me at __email__",
])
def test_g4_fires_on_explicit_identifiers(text):
    assert account_specific(text)


@pytest.mark.parametrize("text", [
    "where is my order",
    "my parcel is late again",
    "I want to return this item",
    "when will my delivery arrive",
])
def test_g4_does_not_fire_on_mere_topic(text):
    """An earlier, looser form of this idea ('needs private data') was the
    measured cause of the Phase 3 labelling divergence: it is true of nearly
    every support message, so it did no work. G4 must key on an explicit
    identifier, not the subject matter."""
    assert not account_specific(text)


def test_thresholds_come_from_config_not_hardcoded():
    """Thresholds are swept to build the coverage/risk curve, so they must be
    parameters rather than constants buried in the gate."""
    strict = run(precedents=[prec(0.50)], tau_sim=0.90)
    loose = run(precedents=[prec(0.50)], tau_sim=0.10)
    assert strict.decision == "escalate"
    assert loose.decision == "auto_handle"
