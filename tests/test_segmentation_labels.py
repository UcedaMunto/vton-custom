"""Pruebas de la tabla de etiquetas FASHN (F1 del plan de implementación).

Los valores están anclados a la evidencia registrada antes de retirar la
dependencia NC: ``plan_modelo_comercial/evidencia_labels_originales.json``.
Si alguien cambia la tabla, estas pruebas lo detectan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fashn_vton.segmentation import labels

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "plan_modelo_comercial" / "evidencia_labels_originales.json"


@pytest.fixture(scope="module")
def evidence() -> dict:
    with open(EVIDENCE, encoding="utf-8") as handle:
        return json.load(handle)


def test_label_table_matches_recorded_evidence(evidence):
    """La tabla local debe ser idéntica a la registrada del paquete original."""
    assert evidence["LABELS_TO_IDS"] == labels.LABELS_TO_IDS
    assert evidence["IDENTITY_LABELS"] == labels.IDENTITY_LABELS
    assert evidence["CATEGORY_TO_BODY_COVERAGE"] == labels.CATEGORY_TO_BODY_COVERAGE
    assert evidence["BODY_COVERAGE_TO_LABELS"] == labels.BODY_COVERAGE_TO_LABELS


def test_label_ids_are_dense_and_unique():
    ids = sorted(labels.LABELS_TO_IDS.values())
    assert ids == list(range(len(ids))), "los ids deben ser 0..N-1 sin huecos"
    assert len(labels.ID_TO_LABEL) == labels.NUM_CLASSES


def test_body_coverage_labels_exist_in_label_table():
    for coverage, names in labels.BODY_COVERAGE_TO_LABELS.items():
        assert coverage in {"upper", "lower", "full"}
        for name in names:
            assert name in labels.LABELS_TO_IDS, f"{name} no está en LABELS_TO_IDS"


def test_identity_labels_are_protected_labels():
    for name in labels.IDENTITY_LABELS:
        assert name in labels.LABELS_TO_IDS


def test_coverage_label_ids_are_derived_correctly():
    for coverage, names in labels.BODY_COVERAGE_TO_LABELS.items():
        expected = [labels.LABELS_TO_IDS[name] for name in names]
        assert labels.COVERAGE_LABEL_IDS[coverage] == expected


def test_every_category_maps_to_a_known_coverage():
    assert set(labels.CATEGORY_TO_BODY_COVERAGE) == {"tops", "bottoms", "one-pieces"}
    for coverage in labels.CATEGORY_TO_BODY_COVERAGE.values():
        assert coverage in labels.BODY_COVERAGE_TO_LABELS
