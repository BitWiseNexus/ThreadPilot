"""Metrics tests.

A bug here does not crash anything - it produces a plausible wrong number that
goes straight into the report. These tests use constructed inputs with known
answers so the arithmetic is pinned, and they check the two framing properties
the report depends on: that intervals widen as n shrinks, and that the two
escalation error types are never averaged together.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval import metrics as M


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
def test_bootstrap_point_estimate_is_the_plain_mean():
    ci = M.bootstrap_ci([True] * 7 + [False] * 3, n_boot=500)
    assert ci.point == pytest.approx(0.7)
    assert ci.n == 10


def test_interval_brackets_the_point_estimate():
    ci = M.bootstrap_ci([True] * 60 + [False] * 40, n_boot=1000)
    assert ci.lo <= ci.point <= ci.hi


def test_interval_widens_as_n_shrinks():
    """The whole reason CIs are mandatory here: at ~18 rows per class a point
    estimate is a rounding of noise, and the interval has to show that."""
    small = M.bootstrap_ci([True] * 7 + [False] * 3, n_boot=2000)
    large = M.bootstrap_ci([True] * 700 + [False] * 300, n_boot=2000)
    assert (small.hi - small.lo) > (large.hi - large.lo) * 3


def test_degenerate_samples_do_not_explode():
    assert M.bootstrap_ci([], n_boot=100).n == 0
    allsame = M.bootstrap_ci([True] * 20, n_boot=500)
    assert allsame.point == 1.0 and allsame.lo == 1.0 and allsame.hi == 1.0


def test_bootstrap_is_seeded_and_reproducible():
    """Every reported interval must be identical across runs, or the report
    changes without the system changing."""
    v = [True, False] * 25
    a = M.bootstrap_ci(v, n_boot=500)
    b = M.bootstrap_ci(v, n_boot=500)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_metric_bootstrap_resamples_pairs_not_sides():
    """Resampling y_true and y_pred independently would destroy the
    correspondence between them and produce an interval around nothing.
    A perfect predictor must therefore stay perfect under resampling."""
    y = ["a", "b", "c"] * 20
    ci = M.bootstrap_metric_ci(
        y, y, lambda a, b: float(np.mean([x == z for x, z in zip(a, b)])),
        n_boot=300)
    assert ci.point == 1.0 and ci.lo == 1.0 and ci.hi == 1.0


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------
def test_classification_report_arithmetic():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    r = M.classification_report(y_true, y_pred, n_boot=300)
    assert r.accuracy.point == pytest.approx(0.75)
    assert r.per_class["b"]["recall"] == pytest.approx(1.0)
    assert r.per_class["a"]["recall"] == pytest.approx(0.5)
    assert r.confusion["a"]["b"] == 1


def test_per_class_carries_support_and_a_reliability_flag():
    """An F1 built on 11 rows must not read like one built on 34, so support
    travels with every per-class figure."""
    y = ["a"] * 30 + ["b"] * 5
    r = M.classification_report(y, y, n_boot=200)
    assert r.per_class["a"]["support"] == 30 and r.per_class["a"]["reliable"]
    assert r.per_class["b"]["support"] == 5 and not r.per_class["b"]["reliable"]


def test_accuracy_is_expressed_against_the_inter_labeller_ceiling():
    """Reporting accuracy against an implicit 100% misrepresents it in both
    directions when two labellers only agreed 75% of the time (D45)."""
    y_true = ["a"] * 10
    y_pred = ["a"] * 7 + ["b"] * 3
    r = M.classification_report(y_true, y_pred, ceiling=0.75, n_boot=200)
    assert r.accuracy.point == pytest.approx(0.7)
    assert r.ceiling_relative_accuracy == pytest.approx(0.7 / 0.75)
    assert "ceiling_relative_accuracy" in r.to_dict()


def test_ceiling_is_optional():
    r = M.classification_report(["a"], ["a"], n_boot=100)
    assert r.ceiling_relative_accuracy is None


def test_top_confusions_excludes_the_diagonal():
    y_true = ["a"] * 5 + ["b"] * 5
    y_pred = ["b"] * 5 + ["b"] * 5
    conf = M.top_confusions(M.classification_report(y_true, y_pred, n_boot=100))
    assert conf and conf[0] == {"true": "a", "predicted": "b", "n": 5}
    assert all(c["true"] != c["predicted"] for c in conf)


# --------------------------------------------------------------------------
# decisions - the asymmetry is the point
# --------------------------------------------------------------------------
def test_false_auto_handle_is_conditioned_on_the_auto_handled_slice():
    """'Of the replies we sent without a human, how many needed one?'

    Computing it over ALL rows would dilute it with escalated rows and flatter a
    cautious system - the opposite of what this metric is for.
    """
    gold = ["escalate", "escalate", "auto_handle", "auto_handle"]
    pred = ["auto_handle", "escalate", "auto_handle", "auto_handle"]
    r = M.decision_report(gold, pred, n_boot=300)
    # 3 auto-handled, 1 of which should have escalated
    assert r.false_auto_handle.point == pytest.approx(1 / 3)
    assert r.coverage.point == pytest.approx(0.75)


def test_always_escalate_scores_a_perfect_false_auto_handle_rate():
    """The trivial baseline's defining behaviour, and the reason a single
    safety number is gameable (D9/D49). The note must say so explicitly."""
    gold = ["escalate", "auto_handle"] * 10
    pred = ["escalate"] * 20
    r = M.decision_report(gold, pred, n_boot=200)
    assert r.false_auto_handle.point == 0.0
    assert r.coverage.point == 0.0
    assert "gameable" in r.note


def test_the_two_error_types_are_reported_separately():
    """A false escalate costs thirty seconds; a false auto-handle publishes a
    wrong answer. Averaging them hides the trade the system exists to make."""
    gold = ["escalate"] * 5 + ["auto_handle"] * 5
    pred = ["auto_handle"] * 5 + ["escalate"] * 5
    r = M.decision_report(gold, pred, n_boot=200)
    d = r.to_dict()
    assert d["false_auto_handle_rate"]["point"] == pytest.approx(1.0)
    assert d["false_escalate_rate"]["point"] == pytest.approx(1.0)
    assert not any("combined" in k or "balanced" in k for k in d)


def test_escalate_recall_and_precision():
    gold = ["escalate", "escalate", "escalate", "auto_handle"]
    pred = ["escalate", "escalate", "auto_handle", "escalate"]
    r = M.decision_report(gold, pred, n_boot=200)
    assert r.escalate_recall.point == pytest.approx(2 / 3)
    assert r.escalate_precision.point == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# coverage / risk
# --------------------------------------------------------------------------
def test_coverage_falls_monotonically_as_the_threshold_rises():
    gold = ["auto_handle"] * 50 + ["escalate"] * 50
    conf = list(np.linspace(0, 1, 100))
    curve = M.coverage_risk_curve(gold, conf, conf, steps=11)
    covs = [p["coverage"] for p in curve]
    assert covs == sorted(covs, reverse=True)
    assert curve[-1]["coverage"] <= 0.05


def test_curve_reports_zero_coverage_without_dividing_by_zero():
    curve = M.coverage_risk_curve(["escalate"] * 5, [0.0] * 5, [0.0] * 5, steps=5)
    assert curve[-1]["n_auto"] == 0
    assert curve[-1]["false_auto_handle"] == 0.0


def test_curve_carries_acceptable_rate_when_judge_scores_exist():
    gold = ["auto_handle"] * 10
    conf = [0.9] * 10
    ok = [True] * 7 + [False] * 3
    curve = M.coverage_risk_curve(gold, conf, conf, acceptable=ok, steps=3)
    assert curve[0]["acceptable_rate"] == pytest.approx(0.7)


# --------------------------------------------------------------------------
# kappa
# --------------------------------------------------------------------------
def test_weighted_kappa_treats_ordinal_distance_as_meaningful():
    """On a 1-5 scale a 4-vs-5 disagreement is not a 1-vs-5. Unweighted kappa
    scores them identically and badly understates judge agreement."""
    a = [1, 2, 3, 4, 5] * 4
    near = [1, 2, 3, 4, 4] * 4     # off by one at one position
    far = [5, 4, 3, 2, 1] * 4      # reversed
    assert M.weighted_kappa(a, near) > M.weighted_kappa(a, far)
    assert M.weighted_kappa(a, a) == pytest.approx(1.0)


def test_cohen_kappa_of_perfect_agreement_is_one():
    assert M.cohen_kappa([True, False] * 5, [True, False] * 5) == pytest.approx(1.0)
