"""R3: auto-handle vs escalate, with a stated reason.

Design (architecture.md section 5, decision_log.md D6): the LLM *proposes* a
decision while drafting; this module applies deterministic gates on top.

## The gates are one-directional

A gate can only force `escalate`. None can grant `auto_handle`. That encodes the
cost asymmetry structurally rather than trusting a model to respect it: a wrong
escalate costs an agent thirty seconds, a wrong auto-handle publishes a wrong
answer under the brand's name.

## Why a rule layer at all, when the model already proposes

Because the model's proposal is not reproducible enough to be a policy. Phase 3
measured two capable models at **Cohen kappa 0.273** on this exact judgement -
one escalated 34% of messages, the other 74%. A decision that unstable needs an
auditable floor under it. The gates provide that, and a team lead gets a named
reason ("no comparable precedent, similarity 0.31") rather than a sentence the
model generated about itself.

The gates are also what make the coverage/risk curve possible: TAU_SIM and
TAU_CONF are knobs, tuned on dev_silver only (D5).

## Honest cost

The gates are hand-designed and encode my blind spots, and G1 caps achievable
auto-handle coverage by construction (~30% of traffic). Phase 6 runs a gates-off
ablation so the rule layer has to justify itself with a number rather than this
argument.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from . import config, taxonomy
from .classify import Classification
from .draft import Draft
from .retrieval import Precedent


class Gate(str, Enum):
    """Deterministic escalation gates. Each can only force escalate."""

    G1_ALWAYS_ESCALATE_INTENT = "G1"
    G2_NO_PRECEDENT = "G2"
    G3_LOW_CONFIDENCE = "G3"
    G4_ACCOUNT_SPECIFIC = "G4"
    G5_MODEL_RISK_FLAG = "G5"
    G6_EMPTY_DRAFT = "G6"


GATE_REASONS = {
    Gate.G1_ALWAYS_ESCALATE_INTENT:
        "intent is always handled by a person",
    Gate.G2_NO_PRECEDENT:
        "no comparable precedent in brand history",
    Gate.G3_LOW_CONFIDENCE:
        "intent could not be determined confidently",
    Gate.G4_ACCOUNT_SPECIFIC:
        "needs private account or order data",
    Gate.G5_MODEL_RISK_FLAG:
        "model raised a risk flag on its own draft",
    Gate.G6_EMPTY_DRAFT:
        "no usable draft was produced",
}

# G4: the message hands over, or asks about, data only the account holds.
# Deliberately narrow. An earlier, looser version of this idea ("needs private
# data") was the measured cause of the labelling divergence in Phase 3 - it is
# true of almost every support message, so it did no work. This fires only on
# an explicit identifier, not on the mere topic.
ACCOUNT_SPECIFIC_RE = re.compile(
    r"\b\d{3}-\d{7}-\d{7}\b"                       # Amazon order id
    r"|\border\s*(?:id|no|number|#)\s*[:#]?\s*\d"  # "order no 12345"
    r"|\b(?:my|our) (?:card|bank|credit card) (?:end|ending|number)"
    r"|\b(?:tracking|awb|waybill) (?:number|no|id)\s*[:#]?\s*\w"
    r"|__email__|__credit_card__|__phone__",       # dataset's own PII masks
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Decision:
    decision: str                       # "auto_handle" | "escalate"
    reason: str
    gates_fired: tuple[Gate, ...] = ()
    model_proposal: str = ""
    overrode_model: bool = False
    max_similarity: float = 0.0
    confidence: float = 0.0

    @property
    def is_escalate(self) -> bool:
        return self.decision == "escalate"

    @property
    def gate_codes(self) -> tuple[str, ...]:
        return tuple(g.value for g in self.gates_fired)


def account_specific(text: str) -> bool:
    return bool(ACCOUNT_SPECIFIC_RE.search(text or ""))


def decide(
    *,
    text: str,
    classification: Classification,
    draft: Draft,
    precedents: list[Precedent],
    tau_sim: float | None = None,
    tau_conf: float | None = None,
    use_gates: bool = True,
) -> Decision:
    """Combine the model proposal with the deterministic gates.

    `use_gates=False` is the Phase 6 ablation: pure LLM decision, so the rule
    layer's contribution is measured rather than asserted.
    """
    tau_sim = config.TAU_SIM if tau_sim is None else tau_sim
    tau_conf = config.TAU_CONF if tau_conf is None else tau_conf
    max_sim = max((p.similarity for p in precedents), default=0.0)

    fired: list[Gate] = []
    if use_gates:
        if classification.intent in taxonomy.ALWAYS_ESCALATE:
            fired.append(Gate.G1_ALWAYS_ESCALATE_INTENT)
        if max_sim < tau_sim:
            fired.append(Gate.G2_NO_PRECEDENT)
        if classification.confidence < tau_conf:
            fired.append(Gate.G3_LOW_CONFIDENCE)
        if account_specific(text):
            fired.append(Gate.G4_ACCOUNT_SPECIFIC)
        if draft.risk_flags:
            fired.append(Gate.G5_MODEL_RISK_FLAG)
        if not draft.reply.strip():
            fired.append(Gate.G6_EMPTY_DRAFT)

    proposal = draft.decision_proposal or "escalate"

    if fired:
        decision = "escalate"
        detail = "; ".join(GATE_REASONS[g] for g in fired)
        if Gate.G2_NO_PRECEDENT in fired:
            detail += f" (max similarity {max_sim:.2f} < {tau_sim:.2f})"
        if Gate.G3_LOW_CONFIDENCE in fired:
            detail += f" (confidence {classification.confidence:.2f} < {tau_conf:.2f})"
        reason = f"{'+'.join(g.value for g in fired)}: {detail}"
        overrode = proposal == "auto_handle"
    else:
        # No gate fired, so the model's proposal stands - including when it
        # chooses to escalate. Gates add caution; they never remove it.
        decision = proposal
        reason = (draft.reason or "model proposal").strip()
        if decision == "auto_handle":
            reason = f"no gate fired; {reason}"
        overrode = False

    return Decision(
        decision=decision, reason=reason, gates_fired=tuple(fired),
        model_proposal=proposal, overrode_model=overrode,
        max_similarity=max_sim, confidence=classification.confidence,
    )
