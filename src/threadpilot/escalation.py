"""The escalation rubric: what "escalate" means, precisely.

## Why this file exists

The Phase 3 labelling run used a loose rubric whose central test was "does this
need private account data?". Two capable labellers then diverged systematically:
gpt-oss-120b escalated 34.3% of messages, qwen3.8-27b escalated 73.8%, and
Cohen kappa on the decision was **0.312** against 0.702 for intent. Reading the
disagreements showed the cause was not randomness but the rubric: almost every
support message technically touches private data, so the test did no work and
each model fell back on its own disposition.

The fix is to frame the question around the **reply** rather than the message:

    Does a safe, grounded, non-overcommitting reply exist that a competent
    agent would be content to send UNEDITED?

That is answerable. "Please share your order number and we'll look into it" is
such a reply, so needing an order lookup is not by itself grounds to escalate.

## Why sharing this rubric with the pipeline is NOT circular

A previous anti-circularity rule (scripts/label_golden.py) says labellers must
not be told the escalation POLICY. That still holds, and this does not violate
it - the distinction matters:

* **Task definition** (this rubric) is shared deliberately. Grading a system
  against a target it was never told is not rigour, it is a trick question.
* **Mechanism** (gate G1's list of always-escalate intents) is NOT shared. G1
  decides by intent; the rubric decides per message. Their agreement is
  therefore a real measurement rather than an identity.

Concretely: "how long do refunds usually take?" is `refund_billing`, which G1
escalates unconditionally, but under this rubric it is auto-handleable - a
generic timeline answer is safe. G1 will be measurably wrong on cases like that,
which is exactly what the Phase 6 ablation is for.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EscalationReason:
    code: str
    label: str
    test: str


# Ordered roughly by severity. A message escalates if ANY test fires.
REASONS: tuple[EscalationReason, ...] = (
    EscalationReason(
        "E1", "needs private data AND no useful generic reply exists",
        "Answering correctly requires something only the customer's account "
        "holds, AND the best possible public reply would be an empty "
        "placeholder. NOTE: 'please send us your order number so we can look "
        "into it' IS a useful reply, so an order lookup alone does NOT fire "
        "this. This test is deliberately narrow - its loose predecessor fired "
        "on nearly everything and made the label unreproducible.",
    ),
    EscalationReason(
        "E2", "would commit money or a specific remedy",
        "A correct reply would have to promise a refund, credit, compensation, "
        "replacement, or a specific resolution or date. A bot must not make "
        "commitments on the brand's behalf.",
    ),
    EscalationReason(
        "E3", "account compromise, fraud, or privacy",
        "Hacked or locked accounts, unauthorised access or charges by a third "
        "party, or anything exposing personal data. Highest stakes in the "
        "taxonomy: speed to a human matters more than a tidy reply.",
    ),
    EscalationReason(
        "E4", "legal, safety, or harm",
        "Legal threats, regulatory issues, physical safety hazards (damaged "
        "batteries, spoiled food, injury), or harm to a person.",
    ),
    EscalationReason(
        "E5", "anger that an automated reply would worsen",
        "Repeat contact after a failure, an explicit demand for a human, or "
        "sustained hostility with a demand attached. A template reply to "
        "someone already furious about templates is actively harmful.",
    ),
    EscalationReason(
        "E6", "too ambiguous to answer safely",
        "The message is incoherent, truncated mid-thought, or could mean "
        "several materially different things, so any confident reply risks "
        "being confidently wrong.",
    ),
)

BY_CODE = {r.code: r for r in REASONS}

CORE_QUESTION = (
    "Does a safe, grounded, non-overcommitting reply exist that a competent "
    "support agent would be content to send UNEDITED?"
)


def rubric_block() -> str:
    """The rubric as prompts and the adjudication guide both see it."""
    lines = [
        f"CORE QUESTION: {CORE_QUESTION}",
        "",
        "If yes -> auto_handle. If no -> escalate.",
        "",
        "Escalate if ANY of these fire:",
    ]
    for r in REASONS:
        lines.append(f"  {r.code} ({r.label}): {r.test}")
    lines += [
        "",
        "Do NOT escalate merely because an intent sounds sensitive. Two "
        "messages with the same intent can differ: 'how long do refunds "
        "normally take?' is answerable generically, while 'where is my "
        "£200 refund from three weeks ago?' is not.",
    ]
    return "\n".join(lines)


def reason_codes() -> tuple[str, ...]:
    return tuple(r.code for r in REASONS)
