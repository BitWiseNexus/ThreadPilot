"""Orchestration: classify -> retrieve -> draft -> decide.

Batched end to end so a 200-row evaluation fits inside a 200k-token/day free
tier. The stage order matters: classification runs first because the intent goes
into the drafting prompt, and retrieval runs before drafting because the
precedent is what the draft is grounded in.

Retrieval is free (local matmul), so it is never batched or cached specially.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

from . import batching, classify, config, decide as decide_mod, draft as draft_mod
from .classify import Classification
from .decide import Decision
from .draft import Draft, DraftInput
from .retrieval import Precedent, RetrievalIndex

log = logging.getLogger(__name__)


@dataclass
class TriageResult:
    text: str
    classification: Classification
    precedents: list[Precedent]
    draft: Draft
    decision: Decision
    pair_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "text": self.text,
            "intent": self.classification.intent,
            "confidence": round(self.classification.confidence, 3),
            "ambiguous_with": self.classification.ambiguous_with,
            "reply": self.draft.reply,
            "decision": self.decision.decision,
            "reason": self.decision.reason,
            "gates_fired": list(self.decision.gate_codes),
            "model_proposal": self.decision.model_proposal,
            "gates_overrode_model": self.decision.overrode_model,
            "max_similarity": round(self.decision.max_similarity, 4),
            "risk_flags": list(self.draft.risk_flags),
            "grounded_in": list(self.draft.grounded_in),
            "precedents": [
                {"pair_id": p.pair_id, "similarity": round(p.similarity, 4),
                 "customer_msg": p.customer_msg, "brand_reply": p.brand_reply,
                 "reply_kind": p.reply_kind}
                for p in self.precedents
            ],
        }


@dataclass
class RunStats:
    n: int = 0
    seconds: float = 0.0
    classify: batching.BatchStats = field(default_factory=batching.BatchStats)
    draft: batching.BatchStats = field(default_factory=batching.BatchStats)

    def to_dict(self) -> dict:
        return {"n": self.n, "seconds": round(self.seconds, 1),
                "classify": self.classify.to_dict(),
                "draft": self.draft.to_dict()}


def triage_many(
    texts: list[str],
    *,
    index: RetrievalIndex,
    k: int = config.RETRIEVAL_K,
    pair_ids: list[str] | None = None,
    use_gates: bool = True,
    use_retrieval: bool = True,
    offline: bool = False,
    classify_batch_size: int = batching.DEFAULT_BATCH_SIZE,
    draft_batch_size: int = 6,
    model: str | None = None,
    stats: RunStats | None = None,
) -> list[TriageResult]:
    """Triage a list of messages.

    `use_retrieval=False` and `use_gates=False` are the two Phase 6 ablations,
    wired here rather than as separate code paths so the ablation runs the same
    pipeline the headline numbers come from.
    """
    stats = stats if stats is not None else RunStats()
    t0 = time.perf_counter()
    stats.n = len(texts)

    log.info("stage 1/3 classify (%d items, batch %d)", len(texts),
             classify_batch_size)
    classifications = classify.classify_many(
        texts, model=model, batch_size=classify_batch_size, offline=offline,
        stats=stats.classify)

    log.info("stage 2/3 retrieve (k=%d, local)", k)
    per_item: list[list[Precedent]] = []
    for t in texts:
        per_item.append(index.search_text(t, k=k) if use_retrieval else [])

    log.info("stage 3/3 draft + decide (batch %d)", draft_batch_size)
    inputs = [DraftInput(text=t, intent=c.intent, precedents=p)
              for t, c, p in zip(texts, classifications, per_item)]
    drafts = draft_mod.draft_many(
        inputs, model=model, batch_size=draft_batch_size, offline=offline,
        stats=stats.draft)

    results = []
    for i, (t, c, p, d) in enumerate(zip(texts, classifications, per_item, drafts)):
        dec = decide_mod.decide(text=t, classification=c, draft=d, precedents=p,
                                use_gates=use_gates)
        results.append(TriageResult(
            text=t, classification=c, precedents=p, draft=d, decision=dec,
            pair_id=pair_ids[i] if pair_ids else None))

    stats.seconds = time.perf_counter() - t0
    log.info("done in %.1fs", stats.seconds)
    return results


def triage_one(text: str, *, index: RetrievalIndex, **kw) -> TriageResult:
    return triage_many([text], index=index, **kw)[0]
