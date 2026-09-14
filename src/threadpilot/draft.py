"""R2: precedent-grounded reply drafting, plus the model's own safety proposal.

## Why drafting and the decision proposal share one call

They could be separate calls. Combining them is justified on two grounds, one
principled and one practical:

* **Principled.** The escalation rubric (`escalation.py`) asks whether *a safe,
  non-overcommitting reply exists that an agent would send unedited*. That
  question is about a specific reply, so asking it in the same breath as writing
  the reply is the faithful form. Asking "should this escalate?" before any reply
  exists forces the model to guess about a draft it has not written.
* **Practical.** Each drafting call carries the retrieved precedents, which
  dominate its token cost. A separate decision call would carry them again, and
  the free tier is 200k tokens/day.

The deterministic gates stay in `decide.py`, completely outside this call, so the
rule layer remains separable for the Phase 6 gates-off ablation. What this module
produces is only the model's *proposal*.

## Anti-fabrication

The prompt states that any specific claim - a timeframe, policy, URL, amount, or
process step - must be supported by the retrieved precedent or by the customer's
own message. Groundedness is then MEASURED by the judge against the same
precedent block the drafter saw, which is the only fair test of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import batching, config, escalation, taxonomy
from .retrieval import Precedent

log = logging.getLogger(__name__)

SYSTEM = """You are drafting replies for Amazon's customer support Twitter account.

For each item you get the customer's message, its intent, and PRECEDENT: real messages from other customers with the reply Amazon actually sent. The precedent is how this brand handles this kind of issue.

## Writing the reply

- Match the voice of the precedent: brief, plain, human. One or two sentences is normal for this brand.
- GROUND every specific claim. Any timeframe, policy, link, amount or process step must appear in the precedent or in the customer's own message. If the precedent does not support a specific, do not state it.
- Never promise a refund, credit, compensation, replacement or a specific resolution date unless the precedent shows this brand routinely does so for this kind of issue.
- Do not invent order details, and do not claim to have looked anything up.
- If the honest reply is to ask for information, ask for exactly what the precedent asks for.
- No greetings padding, no "we value your feedback" filler, no hashtags.

## Then assess your own draft

{rubric}

Return for each item:
  "id": the item number
  "reply": the drafted reply text
  "grounded_in": list of precedent numbers you actually used (may be empty)
  "decision": "auto_handle" or "escalate"
  "reason": under 15 words, why
  "risk_flags": list of any that apply, else []  -- {codes}

Judge each item independently of the others."""


@dataclass(frozen=True)
class Draft:
    reply: str
    decision_proposal: str          # the MODEL's view; gates may override
    reason: str
    risk_flags: tuple[str, ...] = ()
    grounded_in: tuple[int, ...] = ()

    @property
    def proposes_escalate(self) -> bool:
        return self.decision_proposal == "escalate"


FAILED = Draft(
    reply="", decision_proposal="escalate",
    reason="drafting failed; escalating by default",
    risk_flags=("draft_failed",),
)


@dataclass
class DraftInput:
    text: str
    intent: str
    precedents: list[Precedent] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"CUSTOMER MESSAGE: {self.text}",
                 f"INTENT: {self.intent}"]
        d = taxonomy.BY_NAME.get(self.intent)
        if d:
            lines.append(f"INTENT MEANS: {d.definition}")
        if self.precedents:
            lines.append("PRECEDENT:")
            for i, p in enumerate(self.precedents, 1):
                lines.append(f"  ({i}) {p.render()}")
        else:
            # Stated explicitly rather than left as an empty section: a silent
            # absence reads as "nothing relevant", and the model should know it
            # is drafting unsupported so it stays generic and cautious.
            lines.append("PRECEDENT: none found. You have no basis for any "
                         "specific claim - keep the reply general.")
        return "\n".join(lines)


def _parse(d: dict) -> Draft:
    reply = str(d.get("reply", "")).strip()
    if not reply:
        raise ValueError("empty reply")
    dec = str(d.get("decision", "")).strip()
    if dec not in ("auto_handle", "escalate"):
        raise ValueError(f"bad decision {dec!r}")
    flags = d.get("risk_flags") or []
    valid = set(escalation.reason_codes())
    grounded = d.get("grounded_in") or []
    return Draft(
        reply=reply,
        decision_proposal=dec,
        reason=str(d.get("reason", ""))[:160],
        risk_flags=tuple(str(f) for f in flags if str(f) in valid),
        grounded_in=tuple(int(g) for g in grounded
                          if str(g).lstrip("-").isdigit()),
    )


def draft_many(
    items: list[DraftInput],
    *,
    model: str | None = None,
    batch_size: int = 4,
    offline: bool = False,
    stats: batching.BatchStats | None = None,
) -> list[Draft]:
    """Draft replies and collect the model's own decision proposal.

    batch_size is 4 rather than 8: each item carries its precedent block, so
    batches get long, and generation degrades with context length faster than
    classification does.
    """
    results = batching.complete_batch(
        items,
        system=SYSTEM.format(rubric=escalation.rubric_block(),
                             codes=", ".join(escalation.reason_codes())),
        render_item=lambda it: it.render(),
        parse_item=_parse,
        model=model or config.GEN_MODEL,
        temperature=config.TEMP_DRAFT,
        max_tokens_per_item=260,
        max_tokens_floor=config.MAX_TOKENS_DRAFT,
        batch_size=batch_size,
        offline=offline,
        tag="draft",
        stats=stats,
    )
    out = []
    for i, r in enumerate(results):
        if r is None:
            log.warning("draft failed for item %d - escalating by default", i)
            out.append(FAILED)
        else:
            out.append(r)
    return out
