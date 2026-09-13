"""Render the taxonomy to markdown for the report.

Generated from src/threadpilot/taxonomy.py rather than hand-written, so the
documented taxonomy and the one the classifier actually uses cannot drift apart.
A hand-maintained copy would eventually describe a policy that was never run.

Usage:  python scripts/export_taxonomy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import config  # noqa: E402
from threadpilot.taxonomy import ALWAYS_ESCALATE, INTENTS, Disposition  # noqa: E402

OUT = config.RESULTS_DIR / "taxonomy.md"

BADGE = {
    Disposition.ALWAYS_ESCALATE: "**ALWAYS ESCALATE**",
    Disposition.USUALLY_ESCALATE: "usually escalate",
    Disposition.AUTO_ELIGIBLE: "auto-eligible",
}


def main() -> int:
    named = [i for i in INTENTS if i.name != "other"]
    capped = sum(i.approx_share for i in INTENTS if i.name in ALWAYS_ESCALATE)

    L = [
        f"# Intent taxonomy - {config.BRAND}", "",
        "*Generated from `src/threadpilot/taxonomy.py`. Do not edit by hand.*", "",
        f"{len(named)} named intents plus an explicit `other` bucket, derived "
        "from 8,000 English thread-opening customer messages by embedding "
        "(MiniLM) and KMeans, then named by reading exemplars.", "",
        "## How k was chosen", "",
        "Silhouette score was **uninformative**: 0.051-0.065 across k=4..14, a "
        "spread of 0.014. Short support text has no clean geometric cluster "
        "structure, so the automatic criterion simply rewarded the smallest k. "
        "It is reported in `taxonomy_k_sweep.md` as evidence that it did not "
        "discriminate, not as justification for a choice it did not make. "
        "k=10 was chosen because every cluster at that k is nameable and "
        "distinct, while k=9 collapses complaints about support quality into a "
        "generic service bucket.", "",
        "## Why exemplars, not term lists", "",
        "Cluster 1's top terms were `amazon, amazonindia, amazon pay, india`, "
        "which reads like a payments intent. Its actual messages are "
        "generalised brand rage (\"Such a #PoorService\", \"never using Amazon "
        "again\"). Naming from vocabulary would have produced a fictional "
        "intent, so every cluster was named by reading its nearest- and "
        "farthest-from-centroid messages (`taxonomy_clusters.md`).", "",
        "## Escalation policy (gate G1)", "",
        f"`ALWAYS_ESCALATE = {list(ALWAYS_ESCALATE)}` - about **{capped:.0%}** "
        f"of traffic, leaving ~{1-capped:.0%} eligible for automation.", "",
        "Kept deliberately minimal. Over-stuffing this set would cap auto-handle "
        "coverage by construction and let the system look safe by doing nothing "
        "- the trivial always-escalate baseline already scores perfectly on "
        "false-auto-handle at zero coverage.", "",
        "## Intents", "",
        "| intent | share | disposition | cluster |", "|---|---|---|---|",
    ]
    for i in INTENTS:
        src = str(i.cluster) if i.cluster is not None else "-"
        L.append(f"| `{i.name}` | {i.approx_share:.1%} | {BADGE[i.disposition]} | {src} |")
    L += ["", "*Shares sum slightly above 100%: `account_security` has no cluster "
          "of its own (only ~2.5% of traffic, so KMeans scattered it), and its "
          "share overlaps the clusters it was drawn from. It is included anyway "
          "because it is the highest-stakes intent in the set - a triage system "
          "with no label for \"my account was hacked\" has a hole exactly where "
          "automation is most dangerous.*", "", "---", ""]

    for i in INTENTS:
        L += [f"### `{i.name}`", "",
              f"{BADGE[i.disposition]} &middot; ~{i.approx_share:.1%} of traffic"
              + (f" &middot; cluster {i.cluster}" if i.cluster is not None else ""),
              "", i.definition, "", f"**Boundary.** {i.boundary}", ""]
        if i.name != "other":
            L += ["**Real examples from the data:**", ""]
            L += [f"- *\"{e}\"*" for e in i.examples]
            L += [""]

    L += ["---", "",
          "## Known weaknesses", "",
          "- **`prime_membership` vs `delivery_delay` is the hardest boundary.** "
          "Most observed Prime messages are really complaints about slow "
          "delivery. This is expected to be the main source of confusion in the "
          "Phase 6 matrix and is called out in advance rather than explained "
          "away afterwards.",
          f"- **`service_complaint` is the largest class ({[i.approx_share for i in INTENTS if i.name=='service_complaint'][0]:.0%})** "
          "and is heterogeneous by construction, merged from two clusters of "
          "generalised grievance.",
          "- **`account_security` is rare (~2.5%)**, so the golden set must "
          "oversample it to measure it at all. That makes the golden set "
          "deliberately NOT distribution-matched, which biases any headline "
          "accuracy figure and is reported in Phase 3.",
          "- **English only.** Non-English traffic (Japanese, Spanish, French, "
          "German, Portuguese, Italian) was excluded before clustering because "
          "language, not intent, was otherwise the dominant signal. That is "
          "roughly a quarter of real traffic this taxonomy says nothing about.",
          ""]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT} ({len(named)} named intents + other)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
