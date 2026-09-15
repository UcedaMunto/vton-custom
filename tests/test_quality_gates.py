"""Pruebas del módulo de métricas de calidad (sin redes preentrenadas)."""

from __future__ import annotations

import numpy as np
import pytest

from fashn_vton.eval.quality_gates import (
    GARMENT_IDS,
    IDENTITY_IDS,
    QualityGateConfig,
    aggregate,
    color_fidelity,
    evaluate,
    passes_quality_gate,
    region_mask,
    sharpness,
    wasserstein_distance,
    weighted_score,
)
from fashn_vton.segmentation.labels import LABELS_TO_IDS


def _person(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((120, 100, 3), dtype=np.uint8)
    image[..., 0] = 120
    image[20:60, 20:80] = (200, 180, 170)  # "torso"
    image += rng.integers(0, 25, image.shape).astype(np.uint8)
    return image


def _class_map() -> np.ndarray:
    class_map = np.zeros((120, 100), dtype=np.uint8)
    class_map[20:60, 20:80] = LABELS_TO_IDS["top"]
    class_map[5:15, 40:60] = LABELS_TO_IDS["face"]
    return class_map


def test_wasserstein_distance_zero_for_identical_pixels():
    pixels = _person().reshape(-1, 3)
    assert wasserstein_distance(pixels, pixels) == pytest.approx(0.0)


def test_wasserstein_distance_grows_with_difference():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = np.full((10, 10, 3), 60, dtype=np.uint8)
    assert wasserstein_distance(a, b) == pytest.approx(60.0, abs=1.0)


def test_color_fidelity_bounds():
    a = _person().reshape(-1, 3)
    assert color_fidelity(a, a) == pytest.approx(1.0)
    assert color_fidelity(np.zeros((100, 3)), np.full((100, 3), 255)) == pytest.approx(0.0, abs=1e-6)


def test_sharpness_grows_with_structure():
    flat = np.full((60, 60, 3), 128, dtype=np.uint8)
    structured = flat.copy()
    structured[::2, ::2] = 0
    assert sharpness(structured) > sharpness(flat)


def test_region_mask_uses_the_class_map():
    class_map = _class_map()
    assert region_mask(class_map, GARMENT_IDS).sum() == 40 * 60
    assert region_mask(class_map, IDENTITY_IDS).sum() == 10 * 20
    assert region_mask(None, GARMENT_IDS) is None


def test_evaluate_identical_images_scores_high():
    person = _person()
    # Referencia de prenda: el mismo contenido del torso (misma distribución de color).
    garment = person[20:60, 20:80].copy()
    metrics = evaluate(person, person.copy(), garment, _class_map())
    assert metrics["failure"] is False
    assert metrics["identity_preservation"] == pytest.approx(1.0)
    assert metrics["outside_change"] == pytest.approx(0.0)
    assert metrics["garment_change"] == pytest.approx(0.0)
    assert metrics["color_fidelity"] > 0.9


def test_color_fidelity_drops_when_the_output_colour_does_not_match():
    person = _person()
    garment = person[20:60, 20:80].copy()
    output = person.copy()
    output[20:60, 20:80] = 5  # la prenda generada no tiene nada que ver con la referencia
    metrics = evaluate(person, output, garment, _class_map())
    assert metrics["color_fidelity"] < 0.5


def test_evaluate_detects_change_inside_the_garment_region():
    person = _person()
    output = person.copy()
    output[20:60, 20:80] = 10
    metrics = evaluate(person, output, person, _class_map())
    assert metrics["garment_change"] == pytest.approx(1.0)
    assert metrics["outside_change"] == pytest.approx(0.0)
    assert metrics["identity_preservation"] == pytest.approx(1.0)


def test_evaluate_without_class_map_returns_nan_regions():
    person = _person()
    metrics = evaluate(person, person.copy(), person, None)
    assert np.isnan(metrics["color_fidelity"])
    assert np.isnan(metrics["identity_preservation"])
    assert not np.isnan(metrics["sharpness"])


def test_evaluate_rejects_size_mismatch():
    with pytest.raises(ValueError):
        evaluate(np.zeros((10, 10, 3), np.uint8), np.zeros((20, 20, 3), np.uint8), np.zeros((10, 10, 3), np.uint8))


def test_gate_passes_on_identical_images():
    person = _person()
    metrics = evaluate(person, person.copy(), person, _class_map())
    config = QualityGateConfig(min_sharpness=0.0, min_color_fidelity=0.5, min_pattern_fidelity=0.0)
    passed, reasons = passes_quality_gate(metrics, config)
    assert passed, reasons


def test_gate_fails_when_identity_is_destroyed():
    person = _person()
    output = np.zeros_like(person)
    metrics = evaluate(person, output, person, _class_map())
    passed, reasons = passes_quality_gate(metrics, QualityGateConfig(min_sharpness=0.0))
    assert not passed
    assert any("identity_preservation" in reason for reason in reasons)


def test_gate_fails_on_failure_flag():
    passed, reasons = passes_quality_gate({"failure": True})
    assert not passed
    assert reasons == ["la corrida falló"]


def test_weighted_score_prefers_better_metrics():
    good = {"color_fidelity": 0.9, "identity_preservation": 0.95, "failure": False}
    bad = {"color_fidelity": 0.2, "identity_preservation": 0.30, "failure": False}
    assert weighted_score(good) > weighted_score(bad)
    assert weighted_score({"failure": True}) == 0.0


def test_aggregate_summarises_runs_and_gate():
    person = _person()
    metrics = evaluate(person, person.copy(), person, _class_map())
    summary = aggregate([metrics, metrics], QualityGateConfig(min_sharpness=0.0))
    assert summary["runs"] == 2
    assert summary["failures"] == 0
    assert summary["failure_rate"] == 0.0
    assert summary["passes_gate"] is True
    assert summary["score"] > 0

    summary_with_failure = aggregate([metrics, {"failure": True}], QualityGateConfig(min_sharpness=0.0))
    assert summary_with_failure["failures"] == 1
    assert summary_with_failure["failure_rate"] == 0.5
    assert summary_with_failure["passes_gate"] is False


def test_aggregate_ignores_nan_values():
    summary = aggregate(
        [
            {"color_fidelity": float("nan"), "identity_preservation": 0.8, "failure": False},
            {"color_fidelity": 0.6, "identity_preservation": 0.8, "failure": False},
        ]
    )
    assert summary["color_fidelity"] == pytest.approx(0.6)
    assert summary["identity_preservation"] == pytest.approx(0.8)
