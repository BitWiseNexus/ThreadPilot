"""Batching safety tests.

Batching exists to fit inside a 200k-token/day free tier, but it introduces a
failure mode per-item calls do not have: a response can drop, reorder or invent
item ids and still parse cleanly. That corrupts an evaluation *invisibly* - the
metrics are simply computed over the wrong subset - so the id round-trip is
tested rather than trusted.

No network: `llm.complete` is monkeypatched.
"""

from __future__ import annotations

import json

import pytest

from threadpilot import batching, llm
from threadpilot.batching import BatchStats, _extract_items, complete_batch


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def install(monkeypatch, responder):
    """Replace llm.complete with a scripted responder."""
    calls = []

    def fake(messages, **kw):
        calls.append({"messages": messages, **kw})
        return FakeResponse(responder(messages, kw, len(calls)))

    monkeypatch.setattr(llm, "complete", fake)
    monkeypatch.setattr(batching.llm, "complete", fake)
    return calls


ITEMS = [f"message {i}" for i in range(1, 6)]


def render(x):
    return x


def parse(d):
    return d["label"]


def test_happy_path_returns_one_result_per_item(monkeypatch):
    def responder(messages, kw, n):
        body = messages[1]["content"]
        ids = [int(l.split()[-1]) for l in body.splitlines() if l.startswith("### ITEM")]
        return {"results": [{"id": i, "label": f"L{i}"} for i in ids]}

    calls = install(monkeypatch, responder)
    st = BatchStats()
    out = complete_batch(ITEMS, system="sys", render_item=render,
                         parse_item=parse, batch_size=5, stats=st)
    assert out == ["L1", "L2", "L3", "L4", "L5"]
    assert len(calls) == 1, "5 items in one batch should be a single call"
    assert st.invalid_batches == 0 and st.fallback_items == 0


def test_missing_id_triggers_fallback_and_loses_nothing(monkeypatch):
    """The dangerous case: a batch silently returns 4 of 5 results.

    Without validation this would return 4 labels and a None, and the eval would
    quietly be computed over 4 rows. The fallback must recover all 5.
    """
    def responder(messages, kw, n):
        body = messages[1]["content"]
        ids = [int(l.split()[-1]) for l in body.splitlines() if l.startswith("### ITEM")]
        if len(ids) > 1:                      # batch call: drop the last item
            return {"results": [{"id": i, "label": f"L{i}"} for i in ids[:-1]]}
        return {"id": 1, "label": "FALLBACK"}  # per-item call

    install(monkeypatch, responder)
    st = BatchStats()
    out = complete_batch(ITEMS, system="sys", render_item=render,
                         parse_item=parse, batch_size=5, stats=st)
    assert len(out) == 5
    assert all(v is not None for v in out), "fallback must not lose rows"
    assert st.invalid_batches == 1
    assert st.fallback_items == 5


def test_reordered_ids_are_realigned_not_misassigned(monkeypatch):
    """A shuffled response must not shift labels onto the wrong items - that is
    a silent correctness bug, not a crash."""
    def responder(messages, kw, n):
        body = messages[1]["content"]
        ids = [int(l.split()[-1]) for l in body.splitlines() if l.startswith("### ITEM")]
        shuffled = list(reversed(ids))
        return {"results": [{"id": i, "label": f"L{i}"} for i in shuffled]}

    install(monkeypatch, responder)
    out = complete_batch(ITEMS, system="sys", render_item=render,
                         parse_item=parse, batch_size=5, stats=BatchStats())
    assert out == ["L1", "L2", "L3", "L4", "L5"], "results must map back by id"


def test_extra_or_invented_id_invalidates_the_batch(monkeypatch):
    def responder(messages, kw, n):
        body = messages[1]["content"]
        ids = [int(l.split()[-1]) for l in body.splitlines() if l.startswith("### ITEM")]
        if len(ids) > 1:
            extra = [{"id": i, "label": f"L{i}"} for i in ids]
            extra.append({"id": 99, "label": "ghost"})
            return {"results": extra}
        return {"id": 1, "label": "FALLBACK"}

    st = BatchStats()
    install(monkeypatch, responder)
    out = complete_batch(ITEMS, system="sys", render_item=render,
                         parse_item=parse, batch_size=5, stats=st)
    assert st.invalid_batches == 1
    assert len(out) == 5 and all(v is not None for v in out)


def test_batches_split_correctly(monkeypatch):
    def responder(messages, kw, n):
        body = messages[1]["content"]
        ids = [int(l.split()[-1]) for l in body.splitlines() if l.startswith("### ITEM")]
        return {"results": [{"id": i, "label": f"B{n}I{i}"} for i in ids]}

    calls = install(monkeypatch, responder)
    out = complete_batch(ITEMS, system="sys", render_item=render,
                         parse_item=parse, batch_size=2, stats=BatchStats())
    assert len(calls) == 3, "5 items at batch_size 2 -> 3 calls"
    assert len(out) == 5 and all(v is not None for v in out)


@pytest.mark.parametrize("payload,n", [
    ([{"id": 1}, {"id": 2}], 2),
    ({"results": [{"id": 1}]}, 1),
    ({"items": [{"id": 1}, {"id": 2}]}, 2),
    ({"1": {"label": "a"}, "2": {"label": "b"}}, 2),
])
def test_extract_items_accepts_the_shapes_models_actually_emit(payload, n):
    assert len(_extract_items(payload)) == n


def test_extract_items_rejects_garbage():
    with pytest.raises(batching.BatchInvalid):
        _extract_items("not json at all")


def test_offline_cache_miss_propagates(monkeypatch):
    """Offline mode must fail loudly, never fall back to a live call - the
    fallback path would silently turn a replay into a fresh run (D10)."""
    def fake(messages, **kw):
        raise llm.CacheMissInOfflineMode("k", "m")

    monkeypatch.setattr(batching.llm, "complete", fake)
    with pytest.raises(llm.CacheMissInOfflineMode):
        complete_batch(ITEMS, system="s", render_item=render, parse_item=parse,
                       offline=True, stats=BatchStats())
