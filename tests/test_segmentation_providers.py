"""Pruebas de los proveedores de segmentación (contrato y heurístico)."""

from __future__ import annotations

import numpy as np
import pytest

from fashn_vton.segmentation import (
    PROVIDERS,
    CustomHumanParserProvider,
    GroundedSam2Provider,
    NoSegmentationProvider,
    PoseHeuristicProvider,
    ProviderNotAvailable,
    Sam2SegmentationProvider,
    available_providers,
    build_segmentation_provider,
    validate_class_map,
)
from fashn_vton.segmentation.labels import LABELS_TO_IDS
from fashn_vton.segmentation.pose_heuristic import (
    build_class_map_from_keypoints,
    keypoints_from_pose,
)

# Keypoints COCO-18 normalizados de una persona "de pie" de referencia.
STANDING = np.zeros((18, 2), dtype=np.float32)
STANDING[0] = (0.50, 0.10)  # nariz
STANDING[1] = (0.50, 0.16)  # cuello
STANDING[2] = (0.42, 0.18)  # hombro derecho
STANDING[3] = (0.36, 0.30)  # codo derecho
STANDING[4] = (0.32, 0.42)  # muñeca derecha
STANDING[5] = (0.58, 0.18)  # hombro izquierdo
STANDING[6] = (0.64, 0.30)  # codo izquierdo
STANDING[7] = (0.68, 0.42)  # muñeca izquierda
STANDING[8] = (0.45, 0.48)  # cadera derecha
STANDING[9] = (0.45, 0.68)  # rodilla derecha
STANDING[10] = (0.45, 0.88)  # tobillo derecho
STANDING[11] = (0.55, 0.48)  # cadera izquierda
STANDING[12] = (0.55, 0.68)  # rodilla izquierda
STANDING[13] = (0.55, 0.88)  # tobillo izquierdo
STANDING[14] = (0.49, 0.09)  # ojo derecho
STANDING[15] = (0.51, 0.09)  # ojo izquierdo
STANDING[16] = (0.47, 0.10)  # oreja derecha
STANDING[17] = (0.53, 0.10)  # oreja izquierda


def _pose_from(keypoints: np.ndarray, scores: np.ndarray | None = None) -> dict:
    """Construye un dict con el formato que devuelve DWposeDetector."""
    scores = np.full(18, 0.9) if scores is None else scores
    return {"bodies": {"candidate": keypoints.astype(np.float32), "subset": scores.reshape(1, 18)}}


def test_registry_contains_all_planned_providers():
    assert set(PROVIDERS) == {"none", "pose-heuristic", "sam2", "grounded-sam2", "custom-parser"}


def test_build_default_provider_is_none():
    provider = build_segmentation_provider()
    assert isinstance(provider, NoSegmentationProvider)
    assert provider.info.commercial_ok is True
    assert provider.info.requires_weights is False


def test_build_unknown_provider_raises_with_options():
    with pytest.raises(ValueError) as excinfo:
        build_segmentation_provider("no-existe")
    assert "none" in str(excinfo.value)


def test_no_segmentation_returns_none_and_is_available():
    provider = NoSegmentationProvider()
    assert provider.is_available() is True
    assert provider.predict(np.zeros((10, 10, 3), dtype=np.uint8)) is None
    assert provider.extract_person_region(np.zeros((10, 10, 3), dtype=np.uint8), "tops") is None


def test_all_providers_are_commercially_ok():
    """Ningún proveedor registrado puede tener licencia no comercial."""
    for name, info in available_providers().items():
        assert info.get("commercial_ok") is True, f"{name} no es comercial"


def test_stub_providers_are_unavailable_and_fail_with_guidance(tmp_path):
    """El proveedor de la etapa 3 sigue siendo un esqueleto documentado."""
    provider = CustomHumanParserProvider(weights_dir=tmp_path / "custom")
    assert provider.is_available() is False
    with pytest.raises(ProviderNotAvailable) as excinfo:
        provider.predict(np.zeros((8, 8, 3), dtype=np.uint8))
    assert provider.info.name in str(excinfo.value)


def test_grounded_sam2_is_implemented_but_needs_weights(tmp_path):
    """Grounded-SAM 2 está implementado: sin pesos no está disponible y avisa."""
    provider = GroundedSam2Provider(detector_dir=tmp_path / "gd")
    assert provider.is_available() is False
    with pytest.raises(ProviderNotAvailable) as excinfo:
        provider.predict(np.zeros((32, 32, 3), dtype=np.uint8), {"category": "tops"})
    assert "grounded-sam2" in str(excinfo.value)


def test_sam2_is_no_longer_a_stub_but_needs_checkpoint(tmp_path):
    """SAM 2 está implementado: sin checkpoint no está disponible y avisa."""
    provider = Sam2SegmentationProvider(weights_dir=tmp_path / "sam2")
    assert provider.is_available() is False
    # sin pose no hay nada que preguntar: degrada devolviendo None
    assert provider.predict(np.zeros((64, 64, 3), dtype=np.uint8), {}) is None
    # en una imagen diminuta las cajas degeneran y tampoco hay nada que segmentar
    degenerate = _pose_from(np.full((18, 2), 0.5, dtype=np.float32))
    assert provider.predict(np.zeros((8, 8, 3), dtype=np.uint8), {"pose": degenerate}) is None
    # con una pose válida, pero sin checkpoint, el error es accionable
    with pytest.raises(ProviderNotAvailable) as excinfo:
        provider.predict(
            np.zeros((864, 576, 3), dtype=np.uint8),
            {"pose": _pose_from(STANDING), "category": "tops"},
        )
    assert "sam2" in str(excinfo.value)


def test_class_map_validation_rejects_bad_outputs():
    provider = NoSegmentationProvider()
    assert validate_class_map(None, provider) is None

    good = np.zeros((4, 4), dtype=np.uint8)
    assert validate_class_map(good, provider).shape == (4, 4)

    with pytest.raises(Exception):
        validate_class_map(np.zeros((4, 4, 3), dtype=np.uint8), provider)  # 3 dimensiones


def test_pose_heuristic_without_pose_returns_none():
    provider = PoseHeuristicProvider()
    assert provider.predict(np.zeros((32, 32, 3), dtype=np.uint8), {}) is None


def test_pose_heuristic_builds_expected_regions():
    class_map = build_class_map_from_keypoints(STANDING, np.full(18, 0.9), (864, 576), "tops")
    assert class_map.shape == (864, 576)
    assert class_map.dtype == np.uint8

    # La prenda superior debe cubrir el torso/brazos, no las piernas.
    assert (class_map == LABELS_TO_IDS["top"]).sum() > 0
    torso_row = class_map[int(864 * 0.30)]
    assert LABELS_TO_IDS["top"] in set(np.unique(torso_row))

    # Cara y manos protegidas (mismas etiquetas que protege el pipeline).
    assert (class_map == LABELS_TO_IDS["face"]).sum() > 0
    assert (class_map == LABELS_TO_IDS["hands"]).sum() > 0

    # Fuera del cuerpo (esquina superior izquierda) debe ser fondo.
    assert class_map[0, 0] == LABELS_TO_IDS["background"]


def test_pose_heuristic_category_changes_garment_label():
    bottom_map = build_class_map_from_keypoints(STANDING, np.full(18, 0.9), (864, 576), "bottoms")
    one_piece_map = build_class_map_from_keypoints(STANDING, np.full(18, 0.9), (864, 576), "one-pieces")
    assert (bottom_map == LABELS_TO_IDS["pants"]).sum() > 0
    assert (bottom_map == LABELS_TO_IDS["top"]).sum() == 0
    assert (one_piece_map == LABELS_TO_IDS["dress"]).sum() > 0


def test_pose_heuristic_is_deterministic():
    first = build_class_map_from_keypoints(STANDING, np.full(18, 0.9), (864, 576), "tops")
    second = build_class_map_from_keypoints(STANDING, np.full(18, 0.9), (864, 576), "tops")
    assert np.array_equal(first, second)


def test_pose_heuristic_provider_uses_context_pose():
    provider = PoseHeuristicProvider()
    image = np.zeros((864, 576, 3), dtype=np.uint8)
    result = provider.predict(image, {"pose": _pose_from(STANDING), "category": "tops"})
    assert result is not None
    assert (result == LABELS_TO_IDS["top"]).sum() > 0


def test_keypoints_from_pose_handles_invisible_points():
    pose = _pose_from(STANDING, np.where(np.arange(18) < 2, -1.0, 0.9))
    keypoints, scores = keypoints_from_pose(pose)
    assert keypoints.shape == (18, 2)
    assert scores[0] == 0.0 and scores[1] == 0.0
    assert scores[2] > 0.9 - 1e-6


def test_keypoints_from_pose_rejects_incomplete_payload():
    assert keypoints_from_pose({}) == (None, None)
    assert keypoints_from_pose({"bodies": {"candidate": np.zeros((5, 2)), "subset": np.zeros((1, 5))}}) == (
        None,
        None,
    )
