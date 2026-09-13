"""Phase 2 step 1: derive candidate intent clusters from real customer messages.

This script does NOT name the intents. It produces the evidence a human names
them from, and commits that evidence so the naming step is reviewable rather
than asserted (decision_log.md D12). Writing a plausible support taxonomy by
hand would have been faster, but the classifier would then be evaluated against
my guess about AmazonHelp's traffic rather than against its actual traffic.

What it does:
  1. embed every customer message in the subsample with MiniLM (cached),
  2. sweep k over a range, scoring each with silhouette and cluster balance,
  3. for the chosen k, dump per-cluster evidence: size, distinctive TF-IDF
     terms, and the 15 messages nearest the centroid.

Note it clusters ALL customer messages, not only those whose reply was usable
precedent. The taxonomy describes *inbound traffic*, which exists regardless of
whether the brand happened to answer well - filtering by reply quality here
would bias the taxonomy toward whatever the brand is good at.

Outputs:
    eval/results/taxonomy_k_sweep.{json,md}   the k selection evidence
    eval/results/taxonomy_clusters.{json,md}  per-cluster exemplars + terms

Usage:
    python scripts/derive_taxonomy.py --k-min 4 --k-max 16
    python scripts/derive_taxonomy.py --k 9        # skip the sweep
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threadpilot import config, embeddings  # noqa: E402

OUT_SWEEP_JSON = config.RESULTS_DIR / "taxonomy_k_sweep.json"
OUT_SWEEP_MD = config.RESULTS_DIR / "taxonomy_k_sweep.md"
OUT_CLUSTERS_JSON = config.RESULTS_DIR / "taxonomy_clusters.json"
OUT_CLUSTERS_MD = config.RESULTS_DIR / "taxonomy_clusters.md"

N_EXEMPLARS = 15
N_TERMS = 12


def to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |",
                      "|" + "|".join("---" for _ in cols) + "|",
                      *["| " + " | ".join("" if pd.isna(v) else str(v) for v in r)
                        + " |" for r in df.itertuples(index=False)]])


def distinctive_terms(texts: list[str], labels: np.ndarray, k: int,
                      n_terms: int = N_TERMS) -> dict[int, list[str]]:
    """Terms that distinguish each cluster from the rest of the corpus.

    Plain within-cluster TF-IDF surfaces generic support vocabulary ("order",
    "please", "help") in every cluster, which tells a human nothing about how
    clusters differ. Scoring each term by (mean weight inside the cluster minus
    mean weight outside) instead surfaces what is *characteristic*, which is the
    question a person naming a cluster is actually asking.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=20_000, stop_words="english",
                          min_df=5, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    out: dict[int, list[str]] = {}
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            out[c] = []
            continue
        inside = np.asarray(X[mask].mean(axis=0)).ravel()
        outside = np.asarray(X[~mask].mean(axis=0)).ravel()
        score = inside - outside
        top = np.argsort(-score)[:n_terms]
        out[c] = [str(t) for t in vocab[top]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k-min", type=int, default=4)
    ap.add_argument("--k-max", type=int, default=16)
    ap.add_argument("--k", type=int, default=None,
                    help="skip the sweep and cluster at this k")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                        stream=sys.stderr)

    if not config.SUBSAMPLE_PARQUET.exists():
        sys.exit("missing subsample. Run: python tasks.py subsample")

    df = pd.read_parquet(config.SUBSAMPLE_PARQUET)
    texts = df["customer_msg"].tolist()
    print(f"clustering {len(texts):,} customer messages from {config.BRAND}")

    t0 = time.perf_counter()
    X = embeddings.embed(texts, normalize=True)
    print(f"embeddings: {X.shape} in {time.perf_counter()-t0:.1f}s")

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    sweep_rows = []
    if args.k is None:
        # Silhouette on a fixed seeded subsample: it is O(n^2) and we only need
        # it to compare k values, not as a reported figure.
        rng = np.random.default_rng(config.SEED)
        sil_idx = rng.permutation(len(X))[:3000]
        for k in range(args.k_min, args.k_max + 1):
            km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10)
            lab = km.fit_predict(X)
            sil = float(silhouette_score(X[sil_idx], lab[sil_idx],
                                         metric="cosine"))
            sizes = np.bincount(lab, minlength=k)
            p = sizes / sizes.sum()
            balance = float(-(p * np.log(p + 1e-12)).sum() / np.log(k))
            sweep_rows.append({
                "k": k, "silhouette": round(sil, 4),
                "inertia": round(float(km.inertia_), 1),
                "balance_entropy": round(balance, 4),
                "smallest_cluster": int(sizes.min()),
                "smallest_pct": round(float(sizes.min() / sizes.sum()), 4),
                "largest_pct": round(float(sizes.max() / sizes.sum()), 4),
            })
            print(f"  k={k:<3} sil={sil:.4f} balance={balance:.3f} "
                  f"smallest={sizes.min():>4} ({sizes.min()/sizes.sum():.1%})")
        sweep = pd.DataFrame(sweep_rows)
        OUT_SWEEP_JSON.write_text(json.dumps({
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "brand": config.BRAND, "seed": config.SEED, "n": len(texts),
            "embed_model": config.EMBED_MODEL,
            "note": "silhouette computed on a seeded 3000-row subsample "
                    "(O(n^2)); used only to compare k, never reported as a "
                    "standalone quality figure",
            "sweep": sweep_rows,
        }, indent=2), encoding="utf-8")
        OUT_SWEEP_MD.write_text(
            f"# Taxonomy: choosing k\n\n{len(texts):,} customer messages, "
            f"{config.EMBED_MODEL}, seed {config.SEED}.\n\n"
            "Silhouette alone tends to favour very small k on short support "
            "text, which would collapse genuinely different intents together. "
            "`balance_entropy` (1.0 = even cluster sizes) and `smallest_pct` are "
            "shown next to it because a k whose smallest cluster is <2% of "
            "traffic gives the golden set too few examples per class to measure "
            "anything.\n\n" + to_md(sweep) + "\n", encoding="utf-8")
        # Pick: best silhouette among k that keep every cluster >=3% of traffic,
        # so no intent is too rare to evaluate at n=200.
        viable = sweep[sweep["smallest_pct"] >= 0.03]
        chosen = int((viable if len(viable) else sweep)
                     .sort_values("silhouette", ascending=False).iloc[0]["k"])
        print(f"\nchosen k={chosen} (best silhouette with every cluster >=3%)")
    else:
        chosen = args.k

    km = KMeans(n_clusters=chosen, random_state=config.SEED, n_init=10)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_
    terms = distinctive_terms(texts, labels, chosen)

    clusters = []
    for c in range(chosen):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        # Nearest-centroid exemplars: the most representative messages, which is
        # what a human should read to name the cluster.
        cen = centroids[c] / (np.linalg.norm(centroids[c]) + 1e-12)
        sims = X[idx] @ cen
        near = idx[np.argsort(-sims)[:N_EXEMPLARS]]
        # A few far members too: cluster edges are where merge/split decisions
        # get made, and reading only the centre makes every cluster look clean.
        far = idx[np.argsort(sims)[:5]]
        clusters.append({
            "cluster": c, "size": int(len(idx)),
            "pct": round(float(len(idx) / len(labels)), 4),
            "distinctive_terms": terms[c],
            "exemplars_near_centroid": [texts[i] for i in near],
            "exemplars_far_from_centroid": [texts[i] for i in far],
            "mean_similarity_to_centroid": round(float(sims.mean()), 4),
        })
    clusters.sort(key=lambda d: -d["size"])

    OUT_CLUSTERS_JSON.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "brand": config.BRAND, "seed": config.SEED, "k": chosen,
        "n": len(texts), "embed_model": config.EMBED_MODEL,
        "clusters": clusters,
    }, indent=2), encoding="utf-8")

    lines = [f"# Taxonomy clusters (k={chosen})", "",
             f"{len(texts):,} customer messages from **{config.BRAND}**, "
             f"embedded with `{config.EMBED_MODEL}`, seed {config.SEED}.", "",
             "Committed so the human naming step in `src/threadpilot/taxonomy.py` "
             "is reviewable rather than asserted. Terms are *distinctive* "
             "(mean TF-IDF inside the cluster minus outside), not merely "
             "frequent - plain frequency surfaces 'order' and 'please' in every "
             "cluster and tells a reader nothing.", "",
             "Far-from-centroid examples are included deliberately: cluster "
             "edges are where merge and split decisions get made, and reading "
             "only the centre makes every cluster look cleaner than it is.", ""]
    for cl in clusters:
        lines += [f"## Cluster {cl['cluster']} - {cl['size']:,} msgs "
                  f"({cl['pct']:.1%}), mean sim {cl['mean_similarity_to_centroid']}",
                  "",
                  f"**Distinctive terms:** {', '.join(cl['distinctive_terms'])}",
                  "", "**Nearest centroid:**", ""]
        lines += [f"- {t[:200]}" for t in cl["exemplars_near_centroid"][:10]]
        lines += ["", "**Far from centroid (cluster edge):**", ""]
        lines += [f"- {t[:200]}" for t in cl["exemplars_far_from_centroid"][:4]]
        lines += [""]
    OUT_CLUSTERS_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nwrote {OUT_CLUSTERS_MD.name} and {OUT_CLUSTERS_JSON.name}")
    print(f"\ncluster sizes: " + ", ".join(
        f"c{c['cluster']}={c['size']}({c['pct']:.0%})" for c in clusters))
    print(f"\nNEXT: read {OUT_CLUSTERS_MD.name} and name the intents by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
