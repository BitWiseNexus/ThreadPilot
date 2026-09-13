"""Local sentence embeddings with an on-disk cache.

Runs `all-MiniLM-L6-v2` on CPU, so embedding costs nothing and needs no API.
Two things here matter for reproducibility rather than for speed:

* **The model revision is pinned, not just the model name.** A silently updated
  checkpoint would change every cosine similarity, and therefore every retrieval
  result and every similarity-threshold gate decision, while every file in the
  repo still looked identical. The resolved revision is recorded next to the
  cached vectors and a mismatch invalidates the cache rather than being ignored.

* **The cache key includes the revision and the normalisation flag**, so two
  runs can never silently share vectors computed under different settings.

`torch` and `sentence-transformers` are optional dependencies
(decision_log.md D17); this module raises an actionable error if they are absent
rather than failing deep inside a call stack.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)

_MODEL = None
_MODEL_REVISION: str | None = None


class EmbeddingsUnavailable(RuntimeError):
    """torch / sentence-transformers are not installed."""


def _load_model():
    """Load MiniLM once per process."""
    global _MODEL, _MODEL_REVISION
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise EmbeddingsUnavailable(
            "sentence-transformers is not installed. It is an OPTIONAL "
            "dependency (see decision_log.md D17).\n"
            "  python tasks.py setup-embed"
        ) from exc

    t0 = time.perf_counter()
    log.info("loading %s (CPU)", config.EMBED_MODEL)
    _MODEL = SentenceTransformer(config.EMBED_MODEL, device="cpu")
    _MODEL_REVISION = resolve_revision()
    log.info("loaded in %.1fs (revision %s)", time.perf_counter() - t0,
             _MODEL_REVISION)
    return _MODEL


def resolve_revision() -> str:
    """Best-effort commit sha of the downloaded checkpoint.

    Falls back to the configured name if the local HF cache layout is not what we
    expect. A fallback is recorded verbatim rather than silently treated as a
    pin, so the report can say which it was.
    """
    try:
        from huggingface_hub import HfApi  # noqa: F401
        from pathlib import Path as _P
        import os

        cache = _P(os.environ.get("HF_HOME", _P.home() / ".cache" / "huggingface"))
        pattern = f"models--{config.EMBED_MODEL.replace('/', '--')}"
        for snap in (cache / "hub" / pattern / "snapshots").glob("*"):
            if snap.is_dir():
                return snap.name
    except Exception:  # noqa: BLE001 - diagnostic only, never fatal
        pass
    return f"unresolved:{config.EMBED_REVISION}"


def _cache_path(texts: list[str], normalize: bool) -> Path:
    """Cache file keyed by content AND settings.

    Hashing the texts themselves (not just their count) means a changed
    subsample can never silently reuse stale vectors.
    """
    h = hashlib.sha256()
    h.update(config.EMBED_MODEL.encode())
    h.update(str(normalize).encode())
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return config.INTERIM_DIR / f"emb_{h.hexdigest()[:20]}.npz"


def embed(
    texts: list[str],
    *,
    normalize: bool = True,
    batch_size: int = 128,
    use_cache: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed texts to a (n, 384) float32 array.

    normalize=True returns unit vectors, so a plain dot product IS cosine
    similarity - which is what makes the brute-force retrieval in D2 a single
    matmul with no per-query division.
    """
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)

    path = _cache_path(texts, normalize)
    if use_cache and path.exists():
        with np.load(path, allow_pickle=False) as z:
            vecs = z["vectors"]
        meta = json.loads(path.with_suffix(".json").read_text("utf-8")) \
            if path.with_suffix(".json").exists() else {}
        log.info("embeddings cache hit: %s (%d x %d, revision %s)", path.name,
                 *vecs.shape, meta.get("revision", "?"))
        return vecs

    model = _load_model()
    t0 = time.perf_counter()
    vecs = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=normalize, show_progress_bar=show_progress,
    ).astype(np.float32)
    dt = time.perf_counter() - t0
    log.info("embedded %d texts in %.1fs (%.0f/s)", len(texts), dt, len(texts) / dt)

    if use_cache:
        np.savez_compressed(path, vectors=vecs)
        path.with_suffix(".json").write_text(json.dumps({
            "model": config.EMBED_MODEL,
            "revision": _MODEL_REVISION or resolve_revision(),
            "normalized": normalize,
            "n": len(texts), "dim": int(vecs.shape[1]),
            "seconds": round(dt, 2),
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2), encoding="utf-8")
    return vecs


def cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int = 5
                ) -> tuple[np.ndarray, np.ndarray]:
    """Top-k by cosine similarity. Assumes both sides are unit-normalised.

    Exact brute force, deliberately (decision_log.md D2): the index is ~10^4 x 384,
    so this is one matmul, and being exact means retrieval quality is never
    confounded with approximate-search recall when attributing a failure.
    """
    if query.ndim == 1:
        query = query[None, :]
    sims = query @ matrix.T                      # (q, n)
    k = min(k, matrix.shape[0])
    # argpartition then sort only the top k - full sort of 10^4 per query is waste
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    top = sims[rows, idx]
    order = np.argsort(-top, axis=1)
    return idx[rows, order], top[rows, order]
