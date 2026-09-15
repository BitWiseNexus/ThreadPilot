"""LLM-as-judge for reply quality.

## Bias controls, and what each one is for

* **Different model family from the generator.** Qwen judges gpt-oss output. LLM
  judges systematically favour text from their own family, and no amount of
  prompt care fixes it (D7). Residual weakness: same provider, same broad
  pretraining era - reduced, not eliminated, and stated as such.
* **Blind to the system.** The judge never learns whether a reply came from the
  pipeline, a template, or a copied precedent. Otherwise "this looks
  template-ish" becomes a prior instead of an observation.
* **Blind to the gold label and to the brand's actual reply.** Seeing either
  would turn the judge into a similarity scorer against a reference that is
  itself often poor (many historical replies are deflections or truncated).
* **Same evidence for every system.** Groundedness is judged against the
  precedent the RETRIEVAL INDEX returns for that message, not against whatever
  precedent the system happened to use. The template and trivial baselines carry
  no precedent of their own; judging them against an empty evidence block would
  score them unfairly rather than differently, and the comparison is supposed to
  be about the reply.

## The rubric

Five ordinal criteria (1-5) for diagnosis, and one binary that is the actual
headline: **would a competent agent send this unedited?** The ordinals explain a
result; `send_unedited` decides it. Ordinals are scored with quadratic-weighted
kappa in the validation study, because on a 1-5 scale a 4-vs-5 disagreement is
not the same as 1-vs-5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from threadpilot import batching, config
from threadpilot.retrieval import Precedent

log = logging.getLogger(__name__)

CRITERIA = ("grounded", "on_intent", "no_overcommit", "voice", "actionable")

SYSTEM = """You are auditing draft replies for a customer support account (Amazon). Score each draft on its own merits.

You see the customer's message, PRECEDENT (real past messages with the reply the brand actually sent), and a DRAFT reply. You do NOT know who or what wrote the draft. Do not speculate about it.

Score each 1-5, where 3 is "acceptable but unremarkable":

- grounded: is every specific claim (timeframe, policy, link, amount, process step) supported by the precedent or by the customer's own message? Invented specifics score 1-2. A reply that makes no specific claims at all is not "ungrounded" - score it 4 if it is honestly general.
- on_intent: does it address what the customer actually asked, rather than an adjacent easier question?
- no_overcommit: does it avoid promising refunds, compensation, replacements or specific resolution dates that the precedent does not show this brand routinely giving? Overpromising scores 1-2.
- voice: does it match the brand's register in the precedent - brief, plain, human? Generic chatbot padding, hashtags or corporate filler score low.
- actionable: does the customer end up with a clear next step, even if that step is "a specialist will follow up"?

Then the one that matters:

- send_unedited: true ONLY if a competent support agent would send this exact text with no changes. If they would rewrite any part of it, false.

Return for each item:
  "id", "grounded", "on_intent", "no_overcommit", "voice", "actionable" (integers 1-5),
  "send_unedited" (boolean), "why" (under 15 words)

Judge each item independently of the others."""


@dataclass(frozen=True)
class JudgeScore:
    grounded: int
    on_intent: int
    no_overcommit: int
    voice: int
    actionable: int
    send_unedited: bool
    why: str = ""

    @property
    def mean_ordinal(self) -> float:
        return sum((self.grounded, self.on_intent, self.no_overcommit,
                    self.voice, self.actionable)) / 5.0

    def to_dict(self) -> dict:
        return {"grounded": self.grounded, "on_intent": self.on_intent,
                "no_overcommit": self.no_overcommit, "voice": self.voice,
                "actionable": self.actionable,
                "send_unedited": self.send_unedited,
                "mean_ordinal": round(self.mean_ordinal, 3), "why": self.why}


@dataclass
class JudgeItem:
    customer_msg: str
    reply: str
    precedents: list[Precedent]

    def render(self, max_prec: int = 3) -> str:
        lines = [f"CUSTOMER: {self.customer_msg}"]
        if self.precedents:
            lines.append("PRECEDENT:")
            for i, p in enumerate(self.precedents[:max_prec], 1):
                lines.append(f"  ({i}) asked: \"{p.customer_msg[:180]}\"")
                lines.append(f"      brand replied: \"{p.brand_reply[:180]}\"")
        else:
            lines.append("PRECEDENT: none available for this message.")
        lines.append(f"DRAFT: {self.reply or '(empty)'}")
        return "\n".join(lines)


def _clamp(v, lo=1, hi=5) -> int:
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        raise ValueError(f"non-numeric score {v!r}")


def _parse(d: dict) -> JudgeScore:
    return JudgeScore(
        grounded=_clamp(d.get("grounded")),
        on_intent=_clamp(d.get("on_intent")),
        no_overcommit=_clamp(d.get("no_overcommit")),
        voice=_clamp(d.get("voice")),
        actionable=_clamp(d.get("actionable")),
        send_unedited=bool(d.get("send_unedited")),
        why=str(d.get("why", ""))[:160],
    )


def judge_many(
    items: list[JudgeItem],
    *,
    model: str | None = None,
    batch_size: int = 4,
    offline: bool = False,
    stats: batching.BatchStats | None = None,
) -> list[JudgeScore | None]:
    """Score replies. None marks a failed item - never a default score.

    Substituting a neutral 3 for a failure would quietly pull every mean toward
    the middle and make failures invisible in the aggregate.
    """
    return batching.complete_batch(
        items,
        system=SYSTEM,
        render_item=lambda it: it.render(),
        parse_item=_parse,
        model=model or config.JUDGE_MODEL,
        temperature=config.TEMP_JUDGE,
        max_tokens_per_item=170,
        max_tokens_floor=config.MAX_TOKENS_JUDGE,
        batch_size=batch_size,
        offline=offline,
        tag="judge",
        stats=stats,
    )
