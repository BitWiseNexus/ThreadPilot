"""Groq client with an on-disk response cache.

Three jobs, all of them about making the evaluation trustworthy rather than
about making calls:

1. **Cache every completion** in cache/llm_cache.sqlite, keyed by a hash of
   everything that could change the output. The cache is committed, so an
   evaluator regenerates every reported number with NO API key
   (`--offline`). See decision_log.md D10.
2. **Fail loudly on empty content.** gpt-oss is a reasoning model: given too
   small a token budget it returns empty content with finish_reason="stop",
   having spent the budget on its reasoning channel. Accepting that would
   record blank drafts as valid outputs and the eval would silently measure
   nothing. See D20.
3. **Survive the free tier** via bounded retry with backoff on 429/5xx.

Offline mode never falls back to a live call, and a live call never silently
overwrites a cached response with a different one - both would make "reproduced"
mean nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter,
)

from . import config

log = logging.getLogger(__name__)

# Bump when the response *shape* we parse changes, so old entries are not
# reinterpreted under new assumptions. Not bumped for prompt edits: a prompt
# change already changes the key, because the prompt is part of the key.
SCHEMA_VERSION = 1

_LOCAL = threading.local()


class LLMError(RuntimeError):
    pass


class EmptyContentError(LLMError):
    """Model returned no content. See D20 - never treated as a valid answer."""


class DailyQuotaExhausted(LLMError):
    """The model's per-DAY token budget is gone.

    Distinct from a per-minute limit, and the distinction is worth real time.
    A per-minute limit clears in seconds, so retrying with backoff is correct.
    A per-day limit cannot clear within a run, so the same retry policy burns
    six attempts and ~2 minutes of backoff per call and then fails anyway.
    Measured cost of conflating them: a 200-row labelling pass spent ~5 hours
    almost entirely in backoff against an exhausted daily quota.
    """


class CacheMissInOfflineMode(LLMError):
    def __init__(self, key: str, model: str) -> None:
        super().__init__(
            f"offline mode: no cached response for model={model} key={key[:16]}...\n"
            "This means the committed cache does not cover this call. Either run "
            "without --offline (needs GROQ_API_KEY), or you have changed a prompt "
            "or parameter since the cache was recorded."
        )


@dataclass
class LLMResponse:
    text: str
    model: str
    cached: bool
    reasoning: str = ""          # gpt-oss exposes its reasoning channel; kept
    prompt_tokens: int = 0       # for Phase 7 failure analysis
    completion_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 1

    def json(self) -> Any:
        """Parse the response as JSON, tolerating fenced code blocks.

        Models wrap JSON in ```json fences often enough that stripping them here
        is worth it; anything else is a real parse failure and should raise.
        """
        t = self.text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[-1] if "\n" in t else t
            t = t.rsplit("```", 1)[0] if "```" in t else t
        t = t.strip()
        # Some models emit prose before the object; take the outermost braces.
        if not t.startswith("{") and "{" in t:
            t = t[t.index("{"): t.rindex("}") + 1]
        return json.loads(t)


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
def _conn() -> sqlite3.Connection:
    """One connection per thread; sqlite objects are not thread-shareable."""
    c = getattr(_LOCAL, "conn", None)
    if c is None:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(config.LLM_CACHE_DB, timeout=30)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                   key TEXT PRIMARY KEY,
                   model TEXT NOT NULL,
                   text TEXT NOT NULL,
                   reasoning TEXT,
                   prompt_tokens INTEGER,
                   completion_tokens INTEGER,
                   created_at TEXT,
                   tag TEXT
               )"""
        )
        c.commit()
        _LOCAL.conn = c
    return c


def cache_key(*, model: str, messages: list[dict], temperature: float,
              max_tokens: int, extra: dict | None = None) -> str:
    """Hash everything that could change the output. Anything omitted here is a
    silent source of irreproducibility, so err toward including it."""
    payload = {
        "schema": SCHEMA_VERSION, "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_stats() -> dict[str, int]:
    cur = _conn().execute("SELECT COUNT(*), COUNT(DISTINCT model) FROM responses")
    n, models = cur.fetchone()
    return {"entries": n or 0, "models": models or 0}


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------
_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise LLMError(
                "GROQ_API_KEY is not set. For a no-key run use offline=True "
                "(`python tasks.py eval --offline`), which replays the "
                "committed cache."
            )
        from groq import Groq

        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


def _is_daily_quota(exc: BaseException) -> bool:
    """True for a per-day budget error, which no amount of waiting fixes.

    Groq phrases these as "tokens per day (TPD)" / "requests per day (RPD)".
    Matching on the per-day wording specifically is what separates them from
    the per-minute limits that SHOULD be retried.
    """
    s = f"{exc}".lower()
    return ("per day" in s) or ("tpd" in s) or ("rpd" in s)


def _is_retryable(exc: BaseException) -> bool:
    s = f"{type(exc).__name__}: {exc}".lower()
    if _is_daily_quota(exc):
        return False
    return any(t in s for t in (
        "rate limit", "429", "timeout", "timed out", "connection",
        "500", "502", "503", "504", "overloaded", "apistatuserror",
    ))


class _Retryable(LLMError):
    pass


@dataclass
class _Counter:
    n: int = 0
    hits: int = 0
    misses: int = 0
    retries: int = 0
    tokens_out: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def report(self) -> str:
        rate = (self.hits / self.n * 100) if self.n else 0.0
        return (f"llm calls={self.n} cache_hits={self.hits} ({rate:.0f}%) "
                f"live={self.misses} retries={self.retries} "
                f"out_tokens={self.tokens_out:,} models={self.by_model}")


STATS = _Counter()

# Models whose daily budget is known to be exhausted in THIS process. Once a
# model reports a per-day limit, every subsequent call to it will fail the same
# way, so calling it again only wastes wall-clock. Cleared per process, since
# the quota resets on Groq's clock, not ours.
_EXHAUSTED: set[str] = set()


def exhausted_models() -> set[str]:
    return set(_EXHAUSTED)


def reset_exhausted() -> None:
    _EXHAUSTED.clear()


def complete(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    offline: bool = False,
    use_cache: bool = True,
    json_mode: bool = False,
    reasoning_effort: str | None = None,
    tag: str = "",
) -> LLMResponse:
    """One chat completion, cached.

    offline=True   -> replay only; a miss raises CacheMissInOfflineMode.
    use_cache=False -> always call live, and do not read the cache (still writes).
    """
    model = model or config.GEN_MODEL
    extra: dict[str, Any] = {}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort

    key = cache_key(model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, extra=extra)
    STATS.n += 1
    STATS.by_model[model] = STATS.by_model.get(model, 0) + 1

    if use_cache:
        row = _conn().execute(
            "SELECT text, reasoning, prompt_tokens, completion_tokens "
            "FROM responses WHERE key=?", (key,)
        ).fetchone()
        if row is not None:
            STATS.hits += 1
            return LLMResponse(text=row[0], model=model, cached=True,
                               reasoning=row[1] or "", prompt_tokens=row[2] or 0,
                               completion_tokens=row[3] or 0)

    if offline:
        raise CacheMissInOfflineMode(key, model)

    if model in _EXHAUSTED:
        raise DailyQuotaExhausted(
            f"{model} hit its per-day token budget earlier in this run; "
            f"skipping without calling. Cached responses still work, and "
            f"`--offline` replays them without any API access.")

    STATS.misses += 1
    attempts = 0

    @retry(
        retry=retry_if_exception_type(_Retryable),
        stop=stop_after_attempt(6),
        wait=wait_exponential_jitter(initial=2, max=45),
        reraise=True,
    )
    def _call() -> tuple[str, str, int, int]:
        nonlocal attempts
        attempts += 1
        if attempts > 1:
            STATS.retries += 1
        budget = max_tokens * (2 if attempts > 2 else 1)  # D20: grow on retry
        try:
            r = _client().chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=budget, **extra
            )
        except Exception as exc:  # noqa: BLE001
            if _is_daily_quota(exc):
                _EXHAUSTED.add(model)
                log.error("%s daily quota exhausted - no further calls this run",
                          model)
                raise DailyQuotaExhausted(f"{model}: {exc}") from exc
            if _is_retryable(exc):
                log.warning("retryable LLM error (attempt %d): %s", attempts, exc)
                raise _Retryable(str(exc)) from exc
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        msg = r.choices[0].message
        text = (msg.content or "").strip()
        reasoning = (getattr(msg, "reasoning", None) or "")
        if not text:
            # D20: this is the silent failure we refuse to accept. Retrying is
            # worthwhile because the retry doubles the budget.
            log.warning(
                "empty content from %s (finish=%s, reasoning=%d chars, budget=%d)",
                model, r.choices[0].finish_reason, len(reasoning), budget)
            raise _Retryable("empty content (reasoning consumed the budget)")
        u = r.usage
        return text, reasoning, (u.prompt_tokens or 0), (u.completion_tokens or 0)

    t0 = time.perf_counter()
    try:
        text, reasoning, ptok, ctok = _call()
    except _Retryable as exc:
        raise EmptyContentError(
            f"{model} returned no usable content after {attempts} attempts. "
            f"Raise max_tokens (currently {max_tokens}) or lower "
            f"reasoning_effort. Last: {exc}"
        ) from exc
    dt = time.perf_counter() - t0
    STATS.tokens_out += ctok

    _conn().execute(
        "INSERT OR IGNORE INTO responses "
        "(key, model, text, reasoning, prompt_tokens, completion_tokens, created_at, tag) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (key, model, text, reasoning, ptok, ctok,
         time.strftime("%Y-%m-%dT%H:%M:%S"), tag),
    )
    _conn().commit()

    return LLMResponse(text=text, model=model, cached=False, reasoning=reasoning,
                       prompt_tokens=ptok, completion_tokens=ctok,
                       latency_s=dt, attempts=attempts)
