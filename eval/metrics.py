"""Metrics, every proportion with a bootstrap confidence interval.

At n=200 across 11 classes - roughly 18 rows per class - a bare point estimate
is not a measurement, it is a rounding of noise. Every proportion reported by
this module therefore carries a bootstrap 95% CI, and per-class figures carry
their support so a reader can see which ones rest on twelve examples.

Two framing decisions are baked in here rather than left to the report:

**Accuracy is reported against the inter-labeller ceiling, not against 100%.**
Two independent labellers agreed on intent 75.0% of the time (Phase 3). A
classifier scoring 73% is at that ceiling; quoting it against an implicit 100%
misrepresents it in both directions. `ceiling_relative_accuracy` makes the
comparison explicit.

**The escalation metric that matters is asymmetric.** A false escalate costs an
agent thirty seconds; a false auto-handle publishes a wrong answer under the
brand's name. They are never averaged into one "accuracy" number here, because
averaging them hides exactly the trade the system exists to make.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_BOOTSTRAP = 2000


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n: int

    def fmt(self, pct: bool = True) -> str:
        if pct:
            return f"{self.point:.1%} [{self.lo:.1%}, {self.hi:.1%}]"
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"

    def to_dict(self) -> dict:
        return {"point": round(self.point, 4), "lo": round(self.lo, 4),
                "hi": round(self.hi, 4), "n": self.n}


def bootstrap_ci(values: list[bool] | np.ndarray, *, n_boot: int = DEFAULT_BOOTSTRAP,
                 seed: int = 20260909, alpha: float = 0.05) -> Interval:
    """Percentile bootstrap CI for the mean of a boolean/0-1 sample."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(float(arr.mean()), float(lo), float(hi), n)


def bootstrap_metric_ci(y_true: list, y_pred: list, fn, *,
                        n_boot: int = DEFAULT_BOOTSTRAP,
                        seed: int = 20260909, alpha: float = 0.05) -> Interval:
    """Bootstrap CI for any metric taking (y_true, y_pred).

    Resamples PAIRS, not each side independently - resampling separately would
    destroy the correspondence and produce a meaningless interval.
    """
    yt, yp = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    n = len(yt)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, size=n)
        try:
            vals.append(fn(list(yt[i]), list(yp[i])))
        except Exception:  # noqa: BLE001 - degenerate resample (a class vanishes)
            continue
    if not vals:
        return Interval(float("nan"), float("nan"), float("nan"), n)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return Interval(float(fn(list(yt), list(yp))), float(lo), float(hi), n)


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------
@dataclass
class ClassificationReport:
    accuracy: Interval
    macro_f1: Interval
    per_class: dict
    confusion: dict
    labels: list[str]
    ceiling: float | None = None

    @property
    def ceiling_relative_accuracy(self) -> float | None:
        """Accuracy as a fraction of the inter-labeller ceiling.

        The honest denominator. Two labellers agreed only `ceiling` of the time,
        so that is the practical maximum a classifier can reach against these
        labels; measuring against 1.0 implies a target nobody achieved.
        """
        if not self.ceiling:
            return None
        return self.accuracy.point / self.ceiling

    def to_dict(self) -> dict:
        d = {"accuracy": self.accuracy.to_dict(),
             "macro_f1": self.macro_f1.to_dict(),
             "per_class": self.per_class,
             "confusion": self.confusion,
             "labels": self.labels}
        if self.ceiling:
            d["inter_labeller_ceiling"] = self.ceiling
            d["ceiling_relative_accuracy"] = round(
                self.ceiling_relative_accuracy, 4)
        return d


def classification_report(y_true: list[str], y_pred: list[str], *,
                          labels: list[str] | None = None,
                          ceiling: float | None = None,
                          n_boot: int = DEFAULT_BOOTSTRAP) -> ClassificationReport:
    from sklearn.metrics import (
        confusion_matrix, f1_score, precision_recall_fscore_support,
    )

    labels = labels or sorted(set(y_true) | set(y_pred))
    acc = bootstrap_ci([t == p for t, p in zip(y_true, y_pred)], n_boot=n_boot)
    mf1 = bootstrap_metric_ci(
        y_true, y_pred,
        lambda a, b: f1_score(a, b, average="macro", zero_division=0,
                              labels=labels),
        n_boot=n_boot)

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = {
        lab: {"precision": round(float(p[i]), 4),
              "recall": round(float(r[i]), 4),
              "f1": round(float(f[i]), 4),
              "support": int(s[i]),
              # Support is carried next to every per-class figure: an F1 based
              # on 11 rows should not be read like one based on 34.
              "reliable": bool(s[i] >= 20)}
        for i, lab in enumerate(labels)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    confusion = {t: {labels[j]: int(cm[i][j]) for j in range(len(labels))}
                 for i, t in enumerate(labels)}
    return ClassificationReport(acc, mf1, per_class, confusion, labels, ceiling)


def top_confusions(report: ClassificationReport, k: int = 8) -> list[dict]:
    """Most frequent off-diagonal cells, for comparison against the confusions
    the two labellers themselves made (predicted in advance, Phase 3)."""
    out = []
    for t, row in report.confusion.items():
        for p, n in row.items():
            if t != p and n > 0:
                out.append({"true": t, "predicted": p, "n": n})
    return sorted(out, key=lambda d: -d["n"])[:k]


# --------------------------------------------------------------------------
# escalation
# --------------------------------------------------------------------------
@dataclass
class DecisionReport:
    coverage: Interval               # share auto-handled
    escalate_recall: Interval        # of gold-escalate, share escalated
    escalate_precision: Interval
    false_auto_handle: Interval      # THE dangerous one
    false_escalate: Interval         # the merely costly one
    n: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {"coverage": self.coverage.to_dict(),
                "escalate_recall": self.escalate_recall.to_dict(),
                "escalate_precision": self.escalate_precision.to_dict(),
                "false_auto_handle_rate": self.false_auto_handle.to_dict(),
                "false_escalate_rate": self.false_escalate.to_dict(),
                "n": self.n, "note": self.note}


def decision_report(gold: list[str], pred: list[str], *,
                    n_boot: int = DEFAULT_BOOTSTRAP) -> DecisionReport:
    """Escalation metrics, kept deliberately un-averaged.

    false_auto_handle is conditioned on the AUTO-HANDLED slice, which is the
    operationally meaningful denominator: "of the replies we sent without a
    human, how many should have had one?" Computing it over all rows would
    dilute it with escalated rows and flatter a cautious system.
    """
    g = np.array(gold)
    p = np.array(pred)
    auto = p == "auto_handle"
    gold_esc = g == "escalate"

    cov = bootstrap_ci(auto.tolist(), n_boot=n_boot)
    esc_rec = bootstrap_ci((p[gold_esc] == "escalate").tolist(), n_boot=n_boot) \
        if gold_esc.any() else Interval(float("nan"),) * 1
    esc_pred = p == "escalate"
    esc_prec = bootstrap_ci((g[esc_pred] == "escalate").tolist(), n_boot=n_boot) \
        if esc_pred.any() else Interval(float("nan"), float("nan"), float("nan"), 0)
    fah = bootstrap_ci((g[auto] == "escalate").tolist(), n_boot=n_boot) \
        if auto.any() else Interval(0.0, 0.0, 0.0, 0)
    fe = bootstrap_ci((g[esc_pred] == "auto_handle").tolist(), n_boot=n_boot) \
        if esc_pred.any() else Interval(0.0, 0.0, 0.0, 0)

    note = ""
    if not auto.any():
        note = ("auto-handled nothing, so false-auto-handle is trivially 0.0 - "
                "the exact reason a single safety number is gameable (D9/D49)")
    return DecisionReport(cov, esc_rec, esc_prec, fah, fe, n=len(g), note=note)


def coverage_risk_curve(gold: list[str], confidences: list[float],
                        similarities: list[float], *,
                        acceptable: list[bool] | None = None,
                        steps: int = 25) -> list[dict]:
    """Sweep a confidence threshold and report coverage against risk.

    A single operating point cannot distinguish a system that automates a lot
    slightly badly from one that automates a little very well. The curve can.
    """
    g = np.array(gold)
    conf = np.array(confidences, dtype=float)
    ok = np.array(acceptable, dtype=bool) if acceptable is not None else None

    out = []
    for tau in np.linspace(0.0, 1.0, steps):
        auto = conf >= tau
        if not auto.any():
            out.append({"tau": round(float(tau), 3), "coverage": 0.0,
                        "false_auto_handle": 0.0, "n_auto": 0,
                        "acceptable_rate": None})
            continue
        fah = float((g[auto] == "escalate").mean())
        row = {"tau": round(float(tau), 3),
               "coverage": round(float(auto.mean()), 4),
               "false_auto_handle": round(fah, 4),
               "n_auto": int(auto.sum())}
        if ok is not None:
            row["acceptable_rate"] = round(float(ok[auto].mean()), 4)
        out.append(row)
    return out


def cohen_kappa(a: list, b: list) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b))


def weighted_kappa(a: list[int], b: list[int]) -> float:
    """Quadratic-weighted kappa, for ORDINAL scales like a 1-5 judge rubric.

    Unweighted kappa treats a 4-vs-5 disagreement as identical to 1-vs-5, which
    badly understates agreement on an ordinal scale.
    """
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b, weights="quadratic"))
