"""R1: intent classification.

Batched (D39) because the taxonomy prompt is ~1,200 tokens and the message it
classifies is under 100 - at one call per item that ratio caps a model at ~140
classifications per day on the free tier.

The prompt carries the boundary notes, not just the intent names. Adjacent-intent
confusion is the *predicted* failure mode here: the two independent labellers in
Phase 3 disagreed most on `other <-> service_complaint`, `delivery_delay <->
prime_membership` and `delivery_delay <-> order_investigation`, and the boundary
notes are exactly the information that resolves those. Phase 6 can then check the
confusion matrix against a prediction made in advance rather than explaining it
afterwards.

Dispositions are deliberately NOT in the prompt: the classifier's job is the
intent, and letting it see which intents are always-escalate would let intent and
decision contaminate each other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import batching, config, taxonomy

log = logging.getLogger(__name__)

SYSTEM = """You classify inbound customer support tweets sent to Amazon's support account.

Assign exactly one intent from this taxonomy. Read the boundary notes - adjacent intents are easy to confuse and the boundaries are what separate them.

{taxonomy}

For each item return:
  "id": the item number
  "intent": one of the intent names above, exactly as written
  "confidence": 0.0-1.0, how sure you are. Be honest - a low number is useful signal, an inflated one is not.
  "ambiguous_with": another intent name if it is a close second, else null

Judge each item independently of the others."""


@dataclass(frozen=True)
class Classification:
    intent: str
    confidence: float
    ambiguous_with: str | None = None

    @property
    def is_confident(self) -> bool:
        return self.confidence >= config.TAU_CONF


UNKNOWN = Classification(intent="other", confidence=0.0, ambiguous_with=None)


def _parse(d: dict) -> Classification:
    intent = str(d.get("intent", "")).strip()
    if intent not in taxonomy.BY_NAME:
        # Do not silently coerce to `other`: that would hide a prompt/taxonomy
        # mismatch inside a legitimate class and quietly inflate `other`.
        raise ValueError(f"unknown intent {intent!r}")
    conf = float(d.get("confidence", 0.0))
    amb = d.get("ambiguous_with")
    return Classification(
        intent=intent,
        confidence=min(max(conf, 0.0), 1.0),
        ambiguous_with=amb if amb in taxonomy.BY_NAME else None,
    )


def classify_many(
    texts: list[str],
    *,
    model: str | None = None,
    batch_size: int = batching.DEFAULT_BATCH_SIZE,
    offline: bool = False,
    stats: batching.BatchStats | None = None,
) -> list[Classification]:
    """Classify texts, returning one Classification each.

    A failed item becomes UNKNOWN (intent `other`, confidence 0.0) rather than
    None, so downstream code cannot silently skip it - and confidence 0.0 means
    gate G3 will escalate it, which is the correct treatment for a message we
    failed to understand.
    """
    results = batching.complete_batch(
        texts,
        system=SYSTEM.format(taxonomy=taxonomy.prompt_block()),
        render_item=lambda t: t,
        parse_item=_parse,
        model=model or config.GEN_MODEL,
        temperature=config.TEMP_CLASSIFY,
        max_tokens_per_item=120,
        max_tokens_floor=config.MAX_TOKENS_CLASSIFY,
        batch_size=batch_size,
        offline=offline,
        tag="classify",
        stats=stats,
    )
    out = []
    for i, r in enumerate(results):
        if r is None:
            log.warning("classification failed for item %d - treating as UNKNOWN "
                        "(confidence 0.0 will trigger gate G3)", i)
            out.append(UNKNOWN)
        else:
            out.append(r)
    return out


def classify_one(text: str, **kw) -> Classification:
    return classify_many([text], **kw)[0]
