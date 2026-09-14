"""Build the retrieval index, with the leakage guard enforced.

Usage:  python scripts/build_index.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import config, retrieval  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                        stream=sys.stderr)
    held = retrieval.held_out_thread_ids()
    print(f"held-out threads (golden + dev_silver): {len(held)}")
    idx = retrieval.build_index(held_out=held)

    info = idx.info
    print(f"\n{'='*60}")
    for k in ("n_subsample", "n_usable_canonical", "n_excluded_by_leakage_guard",
              "n_indexed"):
        print(f"  {k:<32} {info[k]:>6,}")
    print(f"  {'embed_revision':<32} {info['embed_revision'][:16]}")
    print(f"\n  reply kinds indexed:")
    for k, v in sorted(info["reply_kind_counts"].items(), key=lambda x: -x[1]):
        print(f"    {k:<22} {v:>5,}")

    # Report the guard as a positive fact, not a silent success.
    idx.assert_disjoint_from(held)
    print(f"\n  LEAKAGE GUARD: OK - 0 of {len(held)} held-out threads present")

    (config.RESULTS_DIR / "index_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nwrote index ({len(idx):,} vectors) + eval/results/index_info.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
