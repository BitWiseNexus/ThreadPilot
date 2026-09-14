"""Baselines. None of these makes an LLM call.

The plan called for two - a trivial one and a simple one. There are three,
because the third turned out to be the most informative comparison available and
it would have been a gap not to run it.

## 1. `trivial` - majority intent, one canned reply, always escalate

Exists to be beaten. Its real job is to make one failure mode impossible to hide
behind: **it scores a PERFECT false-auto-handle rate**, because it never
auto-handles anything. Any single-number safety metric that ranks this system
first is a broken metric, which is the whole argument for reporting auto-handle
as a coverage/risk curve (D9).

## 2. `simple_tfidf` - TF-IDF + linear classifier, per-intent template replies

The classical-ML comparison. Two training regimes are provided because the
choice materially changes the verdict and picking one silently would be a way of
rigging the comparison:

* `silver` - trained on the 100 dev_silver rows, the only genuinely labelled
  data that is not the test set. Realistic, and very data-poor (~9 rows/class).
* `cluster` - trained on ~7,700 rows pseudo-labelled by their Phase 2 cluster.
  Free, far larger, and noisier. Golden and silver threads are excluded, so it
  leaks nothing.

Reporting both answers a question one number cannot: how much of the LLM's
advantage is *reasoning* versus simply *not needing labelled data*.

## 3. `retrieval_1nn` - copy the nearest historical reply verbatim

Not in the original plan, and the sharpest test in the set. It uses the same
retrieval index the real pipeline does, then returns the nearest neighbour's
actual brand reply **unmodified**. No generation at all.

This isolates exactly what the LLM contributes. If copying the closest
historical reply scores as well as a generated one, then the generation step is
decoration and the honest finding is that retrieval was doing the work. A
baseline that can embarrass the system is worth more than one that cannot.

All three emit the same `TriageResult` shape as the pipeline, so the eval
harness scores them through an identical path - no bespoke scoring.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, taxonomy
from .classify import Classification
from .decide import Decision, Gate
from .draft import Draft
from .pipeline import TriageResult
from .retrieval import Precedent, RetrievalIndex

log = logging.getLogger(__name__)

# A deliberately bland, safe canned reply for the trivial baseline. Written to be
# the kind of thing that is never actively wrong and never actually helpful -
# which is the point of a floor.
CANNED_REPLY = ("Sorry for the trouble. Please share a few more details and "
                "we'll look into this for you.")

# Fallbacks for the two intents with no source cluster, so the template baseline
# is not silently missing classes.
FALLBACK_TEMPLATES = {
    "account_security": ("Sorry for the trouble with your account. For security "
                         "we'll need to check a few details with you directly."),
    "other": CANNED_REPLY,
}


def _blank_precedents() -> list[Precedent]:
    return []


def _result(text, intent, conf, reply, decision, reason, precedents, pair_id,
            gates=()) -> TriageResult:
    return TriageResult(
        text=text,
        classification=Classification(intent=intent, confidence=conf),
        precedents=precedents,
        draft=Draft(reply=reply, decision_proposal=decision, reason=reason),
        decision=Decision(
            decision=decision, reason=reason, gates_fired=tuple(gates),
            model_proposal=decision, overrode_model=False,
            max_similarity=max((p.similarity for p in precedents), default=0.0),
            confidence=conf),
        pair_id=pair_id,
    )


# ---------------------------------------------------------------------------
# 1. trivial
# ---------------------------------------------------------------------------
class TrivialBaseline:
    """Majority intent + one canned reply + always escalate."""

    name = "trivial"

    def __init__(self, majority_intent: str):
        self.majority_intent = majority_intent

    @classmethod
    def fit(cls, train_intents: list[str]) -> "TrivialBaseline":
        maj = pd.Series(train_intents).value_counts().idxmax()
        log.info("trivial baseline: majority intent = %s", maj)
        return cls(maj)

    def triage(self, texts: list[str], pair_ids: list[str] | None = None
               ) -> list[TriageResult]:
        return [
            _result(t, self.majority_intent, 1.0, CANNED_REPLY, "escalate",
                    "trivial baseline always escalates", _blank_precedents(),
                    pair_ids[i] if pair_ids else None)
            for i, t in enumerate(texts)
        ]


# ---------------------------------------------------------------------------
# 2. simple TF-IDF
# ---------------------------------------------------------------------------
@dataclass
class SimpleBaseline:
    """TF-IDF + LinearSVC intent classifier, per-intent template replies.

    Escalates when the decision-function margin falls below a threshold, which
    is the classical analogue of gate G3 - and gives this baseline a real
    coverage/risk curve of its own, so the comparison is like for like.
    """

    name: str
    pipeline: object
    templates: dict[str, str]
    margin_threshold: float = 0.0
    always_escalate: tuple[str, ...] = ()

    @classmethod
    def fit(cls, texts: list[str], intents: list[str], templates: dict[str, str],
            *, name: str = "simple_tfidf",
            margin_threshold: float = 0.15) -> "SimpleBaseline":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        # LogisticRegression rather than a calibrated LinearSVC. The first
        # attempt used CalibratedClassifierCV(LinearSVC, cv=3) and it FAILED on
        # the silver pool: 100 rows across 11 classes leaves some class with
        # fewer than 3 examples, so 3-fold calibration is impossible. That is
        # not a nuisance to work around, it is a measurement - the data-poor
        # regime is too poor to calibrate - and it is reported as such.
        # LogisticRegression yields predict_proba natively with no inner CV, so
        # both training regimes fit under identical code and stay comparable.
        clf = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=1, ngram_range=(1, 2),
                            stop_words="english", max_features=50_000),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0),
        )
        clf.fit(texts, intents)
        log.info("%s: trained on %d rows, %d classes", name, len(texts),
                 len(set(intents)))
        return cls(name=name, pipeline=clf, templates=templates,
                   margin_threshold=margin_threshold,
                   always_escalate=taxonomy.ALWAYS_ESCALATE)

    def triage(self, texts: list[str], pair_ids: list[str] | None = None
               ) -> list[TriageResult]:
        probs = self.pipeline.predict_proba(texts)
        classes = list(self.pipeline.classes_)
        out = []
        for i, t in enumerate(texts):
            j = int(np.argmax(probs[i]))
            intent, conf = classes[j], float(probs[i][j])
            reply = self.templates.get(intent, CANNED_REPLY)

            gates = []
            if intent in self.always_escalate:
                gates.append(Gate.G1_ALWAYS_ESCALATE_INTENT)
            if conf < self.margin_threshold:
                gates.append(Gate.G3_LOW_CONFIDENCE)
            decision = "escalate" if gates else "auto_handle"
            reason = ("+".join(g.value for g in gates) + ": template baseline"
                      if gates else f"confident template reply ({conf:.2f})")
            out.append(_result(t, intent, conf, reply, decision, reason,
                               _blank_precedents(),
                               pair_ids[i] if pair_ids else None, gates))
        return out


# ---------------------------------------------------------------------------
# 3. retrieval-only nearest neighbour
# ---------------------------------------------------------------------------
@dataclass
class Retrieval1NNBaseline:
    """Return the nearest historical reply verbatim. No generation.

    The intent is borrowed from the same TF-IDF classifier so the comparison
    isolates the REPLY step: this baseline and `simple_tfidf` differ only in
    where the reply text comes from.
    """

    name: str
    index: RetrievalIndex
    classifier: object | None = None
    tau_sim: float = config.TAU_SIM
    always_escalate: tuple[str, ...] = ()

    @classmethod
    def build(cls, index: RetrievalIndex, classifier=None,
              name: str = "retrieval_1nn") -> "Retrieval1NNBaseline":
        return cls(name=name, index=index, classifier=classifier,
                   always_escalate=taxonomy.ALWAYS_ESCALATE)

    def triage(self, texts: list[str], pair_ids: list[str] | None = None
               ) -> list[TriageResult]:
        if self.classifier is not None:
            probs = self.classifier.predict_proba(texts)
            classes = list(self.classifier.classes_)
        out = []
        for i, t in enumerate(texts):
            prec = self.index.search_text(t, k=config.RETRIEVAL_K)
            top = prec[0] if prec else None
            reply = top.brand_reply if top else CANNED_REPLY
            sim = top.similarity if top else 0.0

            if self.classifier is not None:
                j = int(np.argmax(probs[i]))
                intent, conf = classes[j], float(probs[i][j])
            else:
                intent, conf = "other", 0.0

            gates = []
            if intent in self.always_escalate:
                gates.append(Gate.G1_ALWAYS_ESCALATE_INTENT)
            if sim < self.tau_sim:
                gates.append(Gate.G2_NO_PRECEDENT)
            decision = "escalate" if gates else "auto_handle"
            reason = ("+".join(g.value for g in gates) + ": 1-NN baseline"
                      if gates else f"copied nearest precedent (sim {sim:.2f})")
            out.append(_result(t, intent, conf, reply, decision, reason, prec,
                               pair_ids[i] if pair_ids else None, gates))
        return out


# ---------------------------------------------------------------------------
# template derivation
# ---------------------------------------------------------------------------
def derive_templates(index: RetrievalIndex, cluster_of_intent: dict[str, int],
                     subsample: pd.DataFrame) -> dict[str, str]:
    """One template reply per intent, taken from real brand replies.

    Hand-writing eleven templates would make the baseline a measure of my prose
    rather than of the approach. Instead each template is the actual brand reply
    whose customer message sits nearest its cluster centroid - the most
    representative real reply for that intent.
    """
    from . import embeddings

    templates: dict[str, str] = {}
    idx_by_pair = {p: i for i, p in enumerate(index.meta["pair_id"])}

    for intent, cluster in cluster_of_intent.items():
        rows = subsample[subsample["cluster"] == cluster] if "cluster" in subsample \
            else subsample.iloc[0:0]
        rows = rows[rows["usable_precedent"]]
        cand = [p for p in rows["pair_id"] if p in idx_by_pair]
        if not cand:
            continue
        vecs = index.vectors[[idx_by_pair[p] for p in cand]]
        centroid = vecs.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        best = int(np.argmax(vecs @ centroid))
        row = index.meta.iloc[idx_by_pair[cand[best]]]
        templates[intent] = str(row["brand_reply"])

    for intent, fallback in FALLBACK_TEMPLATES.items():
        templates.setdefault(intent, fallback)
    for i in taxonomy.INTENTS:
        templates.setdefault(i.name, CANNED_REPLY)
    return templates
