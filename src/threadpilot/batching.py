"""Batched LLM calls, with per-item fallback.

## Why batch at all

Groq's free tier caps at ~200,000 tokens per day per model. The Phase 3
labelling run burned 198,787 of them on 200 rows, because every call repeated a
~1,200-token system prompt carrying the full taxonomy. The payload that actually
varied - a tweet - was under 100 tokens. Batching N items into one call amortises
the system prompt N-fold and is the difference between the remaining phases
taking ~1.5 days and ~5.

## Why this is the risky kind of optimisation

Batching introduces a failure mode that per-item calls do not have: a model can
return fewer items than it was given, reorder them, or invent an index, and the
result still parses. Silently dropping or misaligning rows would corrupt the
evaluation invisibly - the metrics would simply be computed over the wrong
subset. So this module is deliberately paranoid:

* every item is given an explicit integer id, and the response must return
  EXACTLY that id set - missing, extra or duplicated ids invalidate the batch;
* an invalid batch falls back to per-item calls rather than losing rows, so
  batching can only cost tokens, never data;
* `BatchStats` records how often that happened, and the number is reported
  rather than assumed to be zero.

Batching also risks *cross-contamination*: an item's label could be influenced
by its neighbours in a way an isolated call would not be. That cannot be argued
away, so `scripts/validate_batching.py` measures it directly - batched vs
unbatched on the same rows - and the agreement is reported alongside any result
produced this way.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from . import config, llm

log = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 8


@dataclass
class BatchStats:
    batches: int = 0
    items: int = 0
    invalid_batches: int = 0
    fallback_items: int = 0
    failed_items: int = 0
    reasons: list[str] = field(default_factory=list)

    def report(self) -> str:
        return (f"batches={self.batches} items={self.items} "
                f"invalid_batches={self.invalid_batches} "
                f"fallback_items={self.fallback_items} "
                f"failed_items={self.failed_items}")

    def to_dict(self) -> dict:
        return {"batches": self.batches, "items": self.items,
                "invalid_batches": self.invalid_batches,
                "fallback_items": self.fallback_items,
                "failed_items": self.failed_items,
                "invalid_reasons": self.reasons[:20]}


class BatchInvalid(RuntimeError):
    """The response did not round-trip the item ids."""


def _extract_items(payload: Any) -> list[dict]:
    """Accept the shapes models actually emit for a list of results."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("results", "items", "labels", "output", "data", "answers"):
            v = payload.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # {"1": {...}, "2": {...}} - inject the key as the id
        if payload and all(str(k).isdigit() for k in payload):
            out = []
            for k, v in payload.items():
                if isinstance(v, dict):
                    d = dict(v)
                    d.setdefault("id", int(k))
                    out.append(d)
            return out
    raise BatchInvalid(f"unrecognised batch shape: {type(payload).__name__}")


def complete_batch(
    items: Sequence[Any],
    *,
    system: str,
    render_item: Callable[[Any], str],
    parse_item: Callable[[dict], Any],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens_per_item: int = 220,
    max_tokens_floor: int = 768,
    batch_size: int = DEFAULT_BATCH_SIZE,
    offline: bool = False,
    tag: str = "",
    stats: BatchStats | None = None,
) -> list[Any | None]:
    """Process items in batches, returning one result per item (None on failure).

    `parse_item` receives one result dict and returns the caller's own type, or
    raises to mark that item failed.
    """
    stats = stats if stats is not None else BatchStats()
    model = model or config.GEN_MODEL
    out: list[Any | None] = [None] * len(items)

    for start in range(0, len(items), batch_size):
        chunk = list(items[start:start + batch_size])
        ids = list(range(1, len(chunk) + 1))
        body = "\n\n".join(f"### ITEM {i}\n{render_item(it)}"
                           for i, it in enumerate(chunk, 1))
        instruction = (
            f"\n\nYou are given {len(chunk)} items, numbered 1 to {len(chunk)}.\n"
            f"Return ONLY a JSON object of the form "
            f'{{"results": [ ... ]}} containing EXACTLY {len(chunk)} entries, '
            f'each with an "id" field matching the item number. Judge every '
            f"item independently of the others."
        )
        budget = max(max_tokens_floor, max_tokens_per_item * len(chunk))
        stats.batches += 1
        stats.items += len(chunk)

        try:
            r = llm.complete(
                [{"role": "system", "content": system + instruction},
                 {"role": "user", "content": body}],
                model=model, temperature=temperature, max_tokens=budget,
                json_mode=True, offline=offline, tag=f"{tag}_batch",
            )
            results = _extract_items(r.json())
            got = [int(d["id"]) for d in results if str(d.get("id", "")).lstrip("-").isdigit()]
            if sorted(got) != ids:
                raise BatchInvalid(
                    f"id mismatch: expected {ids}, got {sorted(got)}")
            by_id = {int(d["id"]): d for d in results}
            parsed = [parse_item(by_id[i]) for i in ids]
        except llm.CacheMissInOfflineMode:
            raise
        except llm.DailyQuotaExhausted:
            # NOT a bad batch - the budget is gone and every remaining item
            # would "fail" identically. Swallowing this once recorded 184 empty
            # drafts as legitimate predictions on a 200-row run: a silent
            # corruption of the evaluation, which is exactly what the rest of
            # this module exists to prevent. Stop instead, so the caller can
            # resume from cache when the quota resets.
            raise
        except Exception as exc:  # noqa: BLE001
            stats.invalid_batches += 1
            reason = f"{type(exc).__name__}: {exc}"[:160]
            stats.reasons.append(reason)
            log.warning("batch %d invalid (%s) - falling back to per-item",
                        stats.batches, reason)
            parsed = _per_item_fallback(
                chunk, system=system, render_item=render_item,
                parse_item=parse_item, model=model, temperature=temperature,
                max_tokens=max(max_tokens_floor, max_tokens_per_item * 2),
                offline=offline, tag=tag, stats=stats)

        for off, val in enumerate(parsed):
            out[start + off] = val
    return out


def _per_item_fallback(chunk, *, system, render_item, parse_item, model,
                       temperature, max_tokens, offline, tag, stats
                       ) -> list[Any | None]:
    """One call per item. Costs tokens; never loses rows."""
    res: list[Any | None] = []
    for it in chunk:
        stats.fallback_items += 1
        single = (system + '\n\nReturn ONLY a JSON object for this single item, '
                           'including an "id" field set to 1.')
        try:
            r = llm.complete(
                [{"role": "system", "content": single},
                 {"role": "user", "content": f"### ITEM 1\n{render_item(it)}"}],
                model=model, temperature=temperature, max_tokens=max_tokens,
                json_mode=True, offline=offline, tag=f"{tag}_single",
            )
            payload = r.json()
            d = payload if isinstance(payload, dict) and "id" in payload else None
            if d is None:
                found = _extract_items(payload)
                d = found[0] if found else None
            res.append(parse_item(d) if d is not None else None)
        except llm.CacheMissInOfflineMode:
            raise
        except llm.DailyQuotaExhausted:
            raise           # see complete_batch: never becomes a None result
        except Exception as exc:  # noqa: BLE001
            stats.failed_items += 1
            log.warning("per-item fallback failed: %s", exc)
            res.append(None)
    return res
