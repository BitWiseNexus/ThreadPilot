"""Download the Kaggle 'Customer Support on Twitter' dataset into data/raw/.

Idempotent: if data/raw/twcs.csv already exists with a plausible size, this
script does nothing and exits 0.

Three acquisition tiers, tried in order (see docs/decision_log.md D16):

  1. kagglehub          -- pure Python, reads ~/.kaggle/access_token natively
  2. kaggle CLI         -- fallback for anyone on legacy kaggle.json auth
  3. manual detection   -- fallback for an evaluator who downloaded by hand

Tier 3 means this script can never hard-block a reviewer: worst case it prints
the exact URL and the exact destination path and exits non-zero.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force     # re-download even if present
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# --- Load .env BEFORE importing any kaggle library. -------------------------
# The kaggle packages resolve credentials at import/first-use time, so an
# env var set after the import can be ignored. dotenv must therefore run first.
# (Related upstream behaviour: Kaggle/kaggle-cli issue #882, where
# auto-authentication consumes KAGGLE_API_TOKEN before user code runs.)
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    sys.exit(
        "python-dotenv is not installed.\n"
        "Activate the venv and run:  pip install -r requirements.in"
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

DATASET_SLUG = "thoughtvector/customer-support-on-twitter"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_SLUG}"
RAW_DIR = REPO_ROOT / "data" / "raw"
TARGET = RAW_DIR / "twcs.csv"

# The full CSV is ~500MB. Anything much smaller is a truncated or partial file,
# which we would rather catch here than three phases later in a confusing EDA.
MIN_PLAUSIBLE_BYTES = 300 * 1024 * 1024
TOKEN_FILE = Path.home() / ".kaggle" / "access_token"
LEGACY_JSON = Path.home() / ".kaggle" / "kaggle.json"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n:.1f}GB"


def already_present() -> bool:
    if not TARGET.exists():
        return False
    size = TARGET.stat().st_size
    if size < MIN_PLAUSIBLE_BYTES:
        print(
            f"! {TARGET} exists but is only {human(size)}, expected >~500MB.\n"
            f"  Treating as incomplete. Delete it and re-run, or pass --force."
        )
        return False
    print(f"[ok] {TARGET} already present ({human(size)}). Nothing to do.")
    return True


def describe_credentials() -> str:
    """Report which auth mechanism is visible, without ever printing a token."""
    bits = []
    if os.environ.get("KAGGLE_API_TOKEN"):
        bits.append("KAGGLE_API_TOKEN env var (set)")
    if TOKEN_FILE.exists():
        bits.append(f"{TOKEN_FILE} ({TOKEN_FILE.stat().st_size}B)")
    if LEGACY_JSON.exists():
        bits.append(f"{LEGACY_JSON} (legacy)")
    return ", ".join(bits) if bits else "none found"


def place_csv(found: Path) -> None:
    """Move the located CSV to data/raw/twcs.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[..] placing {found.name} -> {TARGET}")
    # Copy rather than move: kagglehub's cache is shared, and stealing a file
    # out of it would force a re-download for any other project on this machine.
    shutil.copy2(found, TARGET)
    print(f"[ok] {TARGET} ({human(TARGET.stat().st_size)})")


def find_csv_under(root: Path) -> Path | None:
    """The archive nests the file as twcs/twcs.csv; be liberal about layout."""
    candidates = sorted(
        (p for p in root.rglob("*.csv") if p.stat().st_size > MIN_PLAUSIBLE_BYTES),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    return candidates[0] if candidates else None


def tier1_kagglehub() -> bool:
    try:
        import kagglehub
    except ImportError:
        print("[--] tier 1 (kagglehub): not installed, skipping")
        return False

    print(f"[..] tier 1 (kagglehub): downloading {DATASET_SLUG}")
    print("     ~500MB. This is a one-time cost; the subsample is committed.")
    try:
        path = Path(kagglehub.dataset_download(DATASET_SLUG))
    except Exception as exc:  # noqa: BLE001 - want the reason surfaced verbatim
        print(f"[!!] tier 1 failed: {type(exc).__name__}: {exc}")
        return False

    found = find_csv_under(path)
    if found is None:
        print(f"[!!] tier 1 downloaded to {path} but no large CSV found inside")
        return False
    place_csv(found)
    return True


def tier2_kaggle_cli() -> bool:
    import subprocess

    if shutil.which("kaggle") is None:
        print("[--] tier 2 (kaggle CLI): not on PATH, skipping")
        return False

    print("[..] tier 2 (kaggle CLI): downloading")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", DATASET_SLUG,
           "-p", str(RAW_DIR), "--unzip"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[!!] tier 2 failed: exit {exc.returncode}")
        return False

    found = find_csv_under(RAW_DIR)
    if found is None:
        return False
    if found != TARGET:
        place_csv(found)
    return True


def tier3_manual_hint() -> None:
    print(
        "\n"
        "=" * 72 + "\n"
        "Could not download automatically. Manual path (takes ~2 minutes):\n\n"
        f"  1. Open {DATASET_URL}\n"
        "  2. Click Download (you may need to sign in)\n"
        "  3. Unzip it and copy the CSV to exactly:\n"
        f"       {TARGET}\n"
        "  4. Re-run this script to verify:\n"
        "       python scripts/download_data.py\n\n"
        f"Credentials detected: {describe_credentials()}\n"
        "If that says 'none found', get a token at\n"
        "  https://www.kaggle.com/settings/api  ->  Generate New Token\n"
        f"and save it as a single line in {TOKEN_FILE}\n"
        + "=" * 72
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if data/raw/twcs.csv exists")
    args = ap.parse_args()

    print(f"repo root : {REPO_ROOT}")
    print(f"target    : {TARGET}")
    print(f"creds     : {describe_credentials()}")

    if not args.force and already_present():
        return 0

    for tier in (tier1_kagglehub, tier2_kaggle_cli):
        if tier():
            break
    else:
        tier3_manual_hint()
        return 1

    size = TARGET.stat().st_size
    if size < MIN_PLAUSIBLE_BYTES:
        print(f"[!!] downloaded file is only {human(size)} — looks truncated")
        return 1

    print("\n[ok] done. Next:  python scripts/build_subsample.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
