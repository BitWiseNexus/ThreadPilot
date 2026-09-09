"""Environment self-check: `python tasks.py check`

Exists so that a broken setup announces itself in one command with an actionable
message, instead of surfacing three phases later as a confusing traceback. It is
also the first thing an evaluator runs, so it doubles as a statement of what this
project actually requires — and, importantly, of what it does NOT require:
neither an API key nor torch is needed for the offline reproduction path.

Never prints a secret. Credentials are reported as present/absent with a length,
which is enough to diagnose an empty or truncated value without leaking it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

OK = "[ok]"
WARN = "[--]"
BAD = "[!!]"


class Check:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, msg: str) -> None:
        print(f"{OK} {msg}")

    def warn(self, msg: str, hint: str = "") -> None:
        print(f"{WARN} {msg}" + (f"\n     -> {hint}" if hint else ""))
        self.warnings.append(msg)

    def fail(self, msg: str, hint: str = "") -> None:
        print(f"{BAD} {msg}" + (f"\n     -> {hint}" if hint else ""))
        self.failures.append(msg)


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 58 - len(title)))


def mask(value: str) -> str:
    """Describe a secret without revealing it."""
    if not value:
        return "absent"
    return f"present (len {len(value)}, starts {value[:4]}...)"


def main() -> int:  # noqa: C901 - a flat checklist is the clearest shape here
    c = Check()
    print("ThreadPilot environment check")

    # ---------------------------------------------------------------- python
    section("interpreter")
    v = sys.version_info
    print(f"     {sys.executable}")
    if (v.major, v.minor) == (3, 12):
        c.ok(f"Python {v.major}.{v.minor}.{v.micro}")
    elif (v.major, v.minor) == (3, 13):
        c.warn(f"Python {v.major}.{v.minor} (project targets 3.12)",
               "core deps should work; embedding deps are untested here")
    else:
        c.fail(f"Python {v.major}.{v.minor} unsupported",
               "create the venv with Python 3.12: py -3.12 -m venv .venv")
    if not (Path(sys.prefix) != Path(sys.base_prefix)):
        c.warn("not running inside a virtualenv",
               "activate it: .venv\\Scripts\\Activate.ps1")
    else:
        c.ok("running inside a virtualenv")

    # ------------------------------------------------------------- core deps
    section("core dependencies (required)")
    for mod, why in [
        ("pandas", "data"),
        ("numpy", "vectors + metrics"),
        ("pyarrow", "parquet"),
        ("sklearn", "TF-IDF baseline, KMeans, kappa"),
        ("groq", "LLM calls"),
        ("dotenv", "secrets loading"),
        ("tenacity", "429 backoff"),
        ("kagglehub", "dataset download"),
        ("rich", "CLI output"),
        ("matplotlib", "eval plots"),
        ("pytest", "tests"),
    ]:
        if importlib.util.find_spec(mod) is not None:
            c.ok(f"{mod:<12} ({why})")
        else:
            c.fail(f"{mod:<12} MISSING ({why})", "python tasks.py setup")

    # --------------------------------------------------------- optional deps
    section("embedding dependencies (optional - see D17)")
    missing_embed = [m for m in ("torch", "sentence_transformers")
                     if importlib.util.find_spec(m) is None]
    if not missing_embed:
        c.ok("torch + sentence_transformers present (can rebuild embeddings)")
    else:
        print(f"{WARN} not installed: {', '.join(missing_embed)}")
        print("     -> This is FINE. Only needed to rebuild embeddings from")
        print("        scratch. Reproducing reported numbers does not need them.")
        print("        Install with: python tasks.py setup-embed")

    # ------------------------------------------------------------ own package
    section("threadpilot package")
    try:
        from threadpilot import __version__, config

        c.ok(f"import threadpilot ({__version__})")
        c.ok(f"repo root  {config.REPO_ROOT}")
        c.ok(f"seed       {config.SEED}")
        c.ok(f"gen model  {config.GEN_MODEL}")
        if config.JUDGE_MODEL:
            c.ok(f"judge model {config.JUDGE_MODEL}")
        else:
            c.warn("judge model UNSET",
                   "resolved in Phase 6 against the live /models endpoint; "
                   "must differ in family from the generator (D7)")
    except Exception as exc:  # noqa: BLE001
        c.fail(f"import threadpilot failed: {type(exc).__name__}: {exc}",
               "python tasks.py setup   (runs pip install -e .)")
        print("\nCannot continue without the package importable.")
        return 1

    # ----------------------------------------------------------- credentials
    section("credentials (values never printed)")
    print(f"     GROQ_API_KEY      {mask(config.GROQ_API_KEY)}")
    if not config.GROQ_API_KEY:
        c.warn("GROQ_API_KEY absent",
               "needed for LIVE runs only. `python tasks.py eval --offline` "
               "replays the committed cache and needs no key.")
    else:
        c.ok("GROQ_API_KEY set (not yet verified against the API)")

    import os

    tok_file = Path.home() / ".kaggle" / "access_token"
    legacy = Path.home() / ".kaggle" / "kaggle.json"
    kag = []
    if os.environ.get("KAGGLE_API_TOKEN"):
        kag.append("KAGGLE_API_TOKEN env var")
    if tok_file.exists():
        kag.append(f"{tok_file} ({tok_file.stat().st_size}B)")
    if legacy.exists():
        kag.append(f"{legacy} (legacy scheme)")
    print(f"     Kaggle auth       {', '.join(kag) if kag else 'absent'}")
    if not kag:
        c.warn("no Kaggle credentials",
               "needed only to download raw data. Not needed if "
               "data/processed/ is already populated.")

    # ------------------------------------------------------------------ data
    section("data")
    if config.RAW_CSV.exists():
        mb = config.RAW_CSV.stat().st_size / 1024 / 1024
        (c.ok if mb > 300 else c.warn)(f"raw twcs.csv present ({mb:.0f}MB)")
    else:
        c.warn("raw twcs.csv absent",
               "python tasks.py data   (~500MB, one time). Not needed if the "
               "committed subsample is present.")
    if config.SUBSAMPLE_PARQUET.exists():
        c.ok(f"subsample present ({config.SUBSAMPLE_PARQUET.name})")
    else:
        c.warn("subsample absent", "python tasks.py subsample")
    if config.GOLDEN_JSONL.exists():
        n = sum(1 for _ in config.GOLDEN_JSONL.open(encoding="utf-8"))
        c.ok(f"golden set present ({n} examples)")
    else:
        c.warn("golden set absent (built in Phase 3)")
    if config.LLM_CACHE_DB.exists():
        kb = config.LLM_CACHE_DB.stat().st_size / 1024
        c.ok(f"LLM response cache present ({kb:.0f}KB) — offline replay possible")
    else:
        c.warn("LLM response cache absent (populated on the first live run)")

    # --------------------------------------------------------------- summary
    section("summary")
    if c.failures:
        print(f"{BAD} {len(c.failures)} failure(s), {len(c.warnings)} warning(s)")
        for f in c.failures:
            print(f"     - {f}")
        return 1
    print(f"{OK} no failures, {len(c.warnings)} warning(s)")
    if c.warnings:
        print("     Warnings are expected on a fresh clone and in early phases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
