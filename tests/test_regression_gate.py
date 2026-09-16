"""Pruebas de la guardia anti-regresión (`fashn_vton.eval.regression`).

Solo lógica: sin GPU, sin pesos y sin modelos.
"""

from __future__ import annotations

import pytest

from fashn_vton.eval.regression import TRACKED_METRICS, compare_metrics

BASELINE = {
    "identity_preservation": 0.9500,
    "outside_change": 20.0,
    "color_fidelity": 0.6000,
    "pattern_fidelity": 0.3000,
    "garment_change": 0.4000,
    "sharpness": 50.0,
    "failure_rate": 0.0,
}


def test_identical_metrics_pass():
    report = compare_metrics(BASELINE, dict(BASELINE))
    assert report.ok
    assert not report.reasons
    assert report.improved == []


def test_small_noise_within_tolerance_passes():
    candidate = dict(BASELINE, identity_preservation=0.9450, outside_change=20.3)  # -0,5 % y +1,5 %
    report = compare_metrics(BASELINE, candidate)
    assert report.ok, report.reasons


def test_improvement_passes_and_is_reported():
    candidate = dict(BASELINE, identity_preservation=0.97, color_fidelity=0.62)
    report = compare_metrics(BASELINE, candidate)
    assert report.ok
    assert "identity_preservation" in report.improved
    assert "color_fidelity" in report.improved


def test_regression_of_higher_is_better_metric_fails():
    candidate = dict(BASELINE, identity_preservation=0.85)  # -10 %
    report = compare_metrics(BASELINE, candidate)
    assert not report.ok
    assert any("identity_preservation" in reason for reason in report.reasons)


def test_regression_of_lower_is_better_metric_fails():
    candidate = dict(BASELINE, outside_change=25.0)  # +25 %
    report = compare_metrics(BASELINE, candidate)
    assert not report.ok
    assert any("outside_change" in reason for reason in report.reasons)


def test_missing_baseline_metric_is_ignored_not_failed():
    baseline = {key: value for key, value in BASELINE.items() if key != "color_fidelity"}
    report = compare_metrics(baseline, dict(BASELINE))
    assert report.ok
    assert "color_fidelity" in report.missing


def test_missing_candidate_metric_fails():
    candidate = {key: value for key, value in BASELINE.items() if key != "sharpness"}
    report = compare_metrics(BASELINE, candidate)
    assert not report.ok
    assert any("sharpness" in reason for reason in report.reasons)


def test_nan_candidate_metric_fails():
    report = compare_metrics(BASELINE, dict(BASELINE, color_fidelity=float("nan")))
    assert not report.ok


def test_candidate_run_failures_are_rejected():
    report = compare_metrics(BASELINE, dict(BASELINE, failure_rate=0.2))
    assert not report.ok
    assert any("fallidas" in reason for reason in report.reasons)


def test_tolerance_is_configurable():
    candidate = dict(BASELINE, identity_preservation=0.90)  # -5 %
    assert not compare_metrics(BASELINE, candidate, max_regression=0.02).ok
    assert compare_metrics(BASELINE, candidate, max_regression=0.10).ok


def test_report_is_serialisable():
    report = compare_metrics(BASELINE, dict(BASELINE, sharpness=10.0))
    payload = report.to_dict()
    assert payload["ok"] is False
    assert payload["reasons"]
    # Una fila por métrica comparada (las ausentes en el baseline se marcan aparte).
    assert len(payload["rows"]) == len(TRACKED_METRICS)
    assert "score" in payload["missing"]
    assert {"name", "baseline", "candidate", "relative", "ok"} <= set(payload["rows"][0])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
