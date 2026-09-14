"""Retrieval index over historical (customer message -> brand reply) pairs.

This is the mechanism behind "grounded in how the brand has historically
resolved similar issues". Three properties matter more than speed:

## 1. The leakage guard

Golden-set threads are EXCLUDED from the index. Without this the retriever would
surface the exact reply the golden row was built from, the drafter would copy
it, and every reply-quality metric would be inflated to the point of fraud - a
failure that produces beautiful numbers and no error message.

The exclusion is by `thread_id`, not `pair_id`, because one conversation can
contain several brand replies and a sibling reply from the same thread leaks
nearly as much as the row itself. `build_index` refuses to build without being
told the held-out threads, so leaking requires deliberately passing an empty
set rather than merely forgetting an argument.

## 2. What is indexed, and what is searched

Records are (customer_msg, brand_reply) pairs, but the **embedded and searched
field is `customer_msg`**. The retrieval question is "who else asked this?" and
the payoff is the reply attached to that neighbour. Embedding replies instead
would retrieve on answer-similarity, which is the wrong similarity for an
unanswered incoming message (decision_log.md D3).

Only rows that are `usable_precedent` (the 8-way classifier in `data.clean`) and
`is_canonical` (one row per near-duplicate group) are admitted.

## 3. Exact search, deliberately

Brute-force cosine over a ~6k x 384 float32 matrix is a single matmul. Being
exact means retrieval quality is never confounded with approximate-search recall
when attributing a failure in Phase 7 (D2).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, embeddings

log = logging.getLogger(__name__)

INDEX_DIR = config.INTERIM_DIR
VECTORS_PATH = INDEX_DIR / "index_vectors.npy"
META_PATH = INDEX_DIR / "index_meta.parquet"
INFO_PATH = INDEX_DIR / "index_info.json"


class LeakageError(RuntimeError):
    """A held-out thread reached the retrieval index."""


@dataclass(frozen=True)
class Precedent:
    pair_id: str
    thread_id: int
    customer_msg: str
    brand_reply: str
    reply_kind: str
    similarity: float
    implies_escalation: bool

    def render(self, max_chars: int = 240) -> str:
        return (f'[sim {self.similarity:.2f}] customer: "{self.customer_msg[:max_chars]}"\n'
                f'            we replied: "{self.brand_reply[:max_chars]}"')


class RetrievalIndex:
    """In-memory exact-cosine index. Small enough that this is the right shape."""

    def __init__(self, vectors: np.ndarray, meta: pd.DataFrame, info: dict):
        if len(vectors) != len(meta):
            raise ValueError(f"vectors {len(vectors)} != meta {len(meta)}")
        self.vectors = vectors
        self.meta = meta.reset_index(drop=True)
        self.info = info
        self._threads = set(self.meta["thread_id"].tolist())

    def __len__(self) -> int:
        return len(self.meta)

    @property
    def thread_ids(self) -> set[int]:
        return set(self._threads)

    def assert_disjoint_from(self, held_out_thread_ids: set[int]) -> None:
        """Fail loudly if any held-out thread is present.

        Called at build time AND available to tests and the eval harness, because
        this is the one invariant whose violation is invisible in the output.
        """
        overlap = self._threads & set(held_out_thread_ids)
        if overlap:
            raise LeakageError(
                f"{len(overlap)} held-out thread(s) are in the retrieval index, "
                f"e.g. {sorted(overlap)[:5]}. Every reply-quality metric "
                f"computed against this index would be inflated."
            )

    def search(self, query_vec: np.ndarray, k: int = config.RETRIEVAL_K
               ) -> list[Precedent]:
        idx, sims = embeddings.cosine_topk(query_vec, self.vectors, k=k)
        out = []
        for i, s in zip(idx[0], sims[0]):
            r = self.meta.iloc[int(i)]
            out.append(Precedent(
                pair_id=str(r["pair_id"]), thread_id=int(r["thread_id"]),
                customer_msg=str(r["customer_msg"]),
                brand_reply=str(r["brand_reply"]),
                reply_kind=str(r["reply_kind"]),
                similarity=float(s),
                implies_escalation=bool(r["implies_escalation"]),
            ))
        return out

    def search_text(self, text: str, k: int = config.RETRIEVAL_K) -> list[Precedent]:
        vec = embeddings.embed([text], normalize=True, show_progress=False)[0]
        return self.search(vec, k=k)

    def save(self) -> None:
        np.save(VECTORS_PATH, self.vectors)
        self.meta.to_parquet(META_PATH, index=False)
        INFO_PATH.write_text(json.dumps(self.info, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "RetrievalIndex":
        if not VECTORS_PATH.exists():
            raise FileNotFoundError(
                "no retrieval index. Run: python scripts/build_index.py")
        vectors = np.load(VECTORS_PATH)
        meta = pd.read_parquet(META_PATH)
        info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
        return cls(vectors, meta, info)


def held_out_thread_ids() -> set[int]:
    """Threads that must never enter the index: golden AND dev_silver.

    dev_silver is included because thresholds are tuned on it; retrieving a
    silver row's own reply while tuning would optimise against a leaked answer
    and the tuned thresholds would not transfer.
    """
    ids: set[int] = set()
    for path in (config.GOLDEN_JSONL,
                 config.GOLDEN_DIR / "dev_silver_candidates.jsonl"):
        if path.exists():
            for line in path.open(encoding="utf-8"):
                line = line.strip()
                if line:
                    ids.add(int(json.loads(line)["thread_id"]))
    return ids


def build_index(*, held_out: set[int] | None = None, save: bool = True
                ) -> RetrievalIndex:
    """Build the index from the committed subsample.

    `held_out` is required rather than defaulted to empty: leaking should take a
    deliberate act, not a forgotten argument. Pass `set()` explicitly only in
    tests that are studying leakage.
    """
    if held_out is None:
        held_out = held_out_thread_ids()
        if not held_out:
            raise LeakageError(
                "no held-out threads found. Build the golden set first "
                "(scripts/finalize_golden.py), or pass held_out=set() "
                "explicitly if you really intend an unguarded index.")

    if not config.SUBSAMPLE_PARQUET.exists():
        raise FileNotFoundError("missing subsample. Run: python tasks.py subsample")

    df = pd.read_parquet(config.SUBSAMPLE_PARQUET)
    n_all = len(df)

    eligible = df[df["usable_precedent"] & df["is_canonical"]]
    n_eligible = len(eligible)

    kept = eligible[~eligible["thread_id"].isin(held_out)]
    n_excluded = n_eligible - len(kept)
    log.info("index: %d rows -> %d usable+canonical -> %d after leakage guard "
             "(%d excluded)", n_all, n_eligible, len(kept), n_excluded)

    t0 = time.perf_counter()
    vecs = embeddings.embed(kept["customer_msg"].tolist(), normalize=True,
                            show_progress=False)
    cols = ["pair_id", "thread_id", "customer_msg", "brand_reply", "reply_kind",
            "implies_escalation", "has_url", "dup_count"]
    meta = kept[cols].reset_index(drop=True)

    info = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "brand": config.BRAND, "seed": config.SEED,
        "embed_model": config.EMBED_MODEL,
        "embed_revision": embeddings.resolve_revision(),
        "n_subsample": int(n_all),
        "n_usable_canonical": int(n_eligible),
        "n_held_out_threads": len(held_out),
        "n_excluded_by_leakage_guard": int(n_excluded),
        "n_indexed": int(len(meta)),
        "reply_kind_counts": meta["reply_kind"].value_counts().to_dict(),
        "build_seconds": round(time.perf_counter() - t0, 1),
    }

    index = RetrievalIndex(vecs, meta, info)
    index.assert_disjoint_from(held_out)   # belt and braces: verify what we built
    if save:
        index.save()
    return index
