"""Pruebas del proveedor SAM 2 (unitarias + integración opcional con checkpoint)."""

from __future__ import annotations

import numpy as np
import pytest

from fashn_vton.segmentation import Sam2SegmentationProvider, build_segmentation_provider
from fashn_vton.segmentation.labels import LABELS_TO_IDS
from fashn_vton.segmentation.pose_heuristic import keypoints_from_pose, pixel_keypoints  # noqa: F401
from fashn_vton.segmentation.sam2 import (
    CATEGORY_REGIONS,
    class_map_from_masks,
    prompt_boxes_from_keypoints,
)

# Keypoints COCO-18 de una persona de pie (mismas que en test_segmentation_providers).
STANDING = np.zeros((18, 2), dtype=np.float32)
STANDING[0] = (0.50, 0.10)
STANDING[1] = (0.50, 0.16)
STANDING[2] = (0.42, 0.18)
STANDING[3] = (0.36, 0.30)
STANDING[4] = (0.32, 0.42)
STANDING[5] = (0.58, 0.18)
STANDING[6] = (0.64, 0.30)
STANDING[7] = (0.68, 0.42)
STANDING[8] = (0.45, 0.48)
STANDING[9] = (0.45, 0.68)
STANDING[10] = (0.45, 0.88)
STANDING[11] = (0.55, 0.48)
STANDING[12] = (0.55, 0.68)
STANDING[13] = (0.55, 0.88)
STANDING[14] = (0.49, 0.09)
STANDING[15] = (0.51, 0.09)
STANDING[16] = (0.47, 0.10)
STANDING[17] = (0.53, 0.10)


def _pose_from(keypoints: np.ndarray, scores: np.ndarray | None = None) -> dict:
    scores = np.full(18, 0.9) if scores is None else scores
    return {"bodies": {"candidate": keypoints.astype(np.float32), "subset": scores.reshape(1, 18)}}


# --- unitarias (sin checkpoint) ---------------------------------------------


def test_prompt_boxes_from_keypoints_covers_body_regions():
    boxes = prompt_boxes_from_keypoints(STANDING, np.full(18, 0.9), (864, 576))
    assert set(boxes) == {"torso", "arm_right", "arm_left", "leg_right", "leg_left"}
    for name, box in boxes.items():
        assert box.shape == (4,)
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= 576, name
        assert 0 <= y0 < y1 <= 864, name
    # el torso debe quedar por encima de las piernas
    assert boxes["torso"][1] < boxes["leg_right"][1]


def test_prompt_boxes_ignore_invisible_keypoints():
    scores = np.zeros(18)  # nada visible
    assert prompt_boxes_from_keypoints(STANDING, scores, (864, 576)) == {}


def test_prompt_boxes_partial_visibility_keeps_available_regions():
    scores = np.full(18, 0.9)
    scores[[2, 3, 4, 5, 6, 7]] = 0.0  # brazos invisibles
    boxes = prompt_boxes_from_keypoints(STANDING, scores, (864, 576))
    assert "torso" in boxes and "leg_right" in boxes
    assert "arm_right" not in boxes and "arm_left" not in boxes


def test_class_map_from_masks_labels_garment_and_protection():
    height, width = 100, 80
    torso = np.zeros((height, width), dtype=bool)
    torso[20:50, 20:60] = True
    legs = np.zeros((height, width), dtype=bool)
    legs[50:90, 25:55] = True
    masks = {"torso": torso, "leg_right": legs}
    points, _ = pixel_keypoints(STANDING, np.full(18, 0.9), (height, width))

    tops_map = class_map_from_masks(masks, points, (height, width), "tops")
    assert (tops_map == LABELS_TO_IDS["top"]).sum() > 0
    assert (tops_map == LABELS_TO_IDS["legs"]).sum() > 0  # las piernas no son la prenda
    assert tops_map[30, 30] == LABELS_TO_IDS["top"]

    bottoms_map = class_map_from_masks(masks, points, (height, width), "bottoms")
    assert bottoms_map[70, 30] == LABELS_TO_IDS["pants"]
    assert bottoms_map[30, 30] == LABELS_TO_IDS["torso"]  # el torso no es la prenda de abajo

    one_piece = class_map_from_masks(masks, points, (height, width), "one-pieces")
    assert one_piece[30, 30] == LABELS_TO_IDS["dress"]
    assert one_piece[70, 30] == LABELS_TO_IDS["dress"]


def test_class_map_from_masks_paints_protection_labels():
    height, width = 200, 160
    blank = np.zeros((height, width), dtype=bool)
    points, _ = pixel_keypoints(STANDING, np.full(18, 0.9), (height, width))
    class_map = class_map_from_masks({"torso": blank}, points, (height, width), "tops")
    assert (class_map == LABELS_TO_IDS["face"]).sum() > 0
    assert (class_map == LABELS_TO_IDS["hands"]).sum() > 0
    assert (class_map == LABELS_TO_IDS["feet"]).sum() > 0


def test_category_regions_are_known_regions():
    from fashn_vton.segmentation.sam2 import REGIONS

    for category, regions in CATEGORY_REGIONS.items():
        assert regions, category
        for region in regions:
            assert region in REGIONS


def test_provider_is_not_available_without_checkpoint(tmp_path):
    provider = Sam2SegmentationProvider(weights_dir=tmp_path / "sam2")
    assert provider.is_available() is False
    assert provider.checkpoint_path() is None
    with pytest.raises(Exception):
        provider.predict(np.zeros((32, 32, 3), dtype=np.uint8), {"pose": _pose_from(STANDING)})


def test_provider_returns_none_without_pose():
    provider = Sam2SegmentationProvider()
    assert provider.predict(np.zeros((32, 32, 3), dtype=np.uint8), {}) is None


def test_provider_metadata_is_commercial():
    provider = build_segmentation_provider("sam2")
    assert isinstance(provider, Sam2SegmentationProvider)
    assert provider.info.commercial_ok is True
    assert "Apache-2.0" in provider.info.license


def test_checkpoint_is_discovered_by_glob(tmp_path):
    folder = tmp_path / "sam2"
    folder.mkdir()
    (folder / "otro_modelo.pt").write_bytes(b"x")
    provider = Sam2SegmentationProvider(weights_dir=folder)
    assert provider.checkpoint_path() == folder / "otro_modelo.pt"


# --- integración real (requiere sam2 instalado + checkpoint) -----------------

_WEIGHTS = None
try:
    from pathlib import Path

    _WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
except Exception:  # pragma: no cover
    _WEIGHTS = None

_NEEDS_SAM2 = _WEIGHTS is None or not (_WEIGHTS / "sam2").is_dir() or not any((_WEIGHTS / "sam2").glob("*.pt"))


@pytest.mark.integration
@pytest.mark.skipif(_NEEDS_SAM2, reason="Requiere sam2 instalado y weights/sam2/*.pt (no versionado).")
def test_sam2_produces_class_map_on_a_real_image():
    """SAM 2 real sobre una imagen sintética con pose sintética."""
    from PIL import Image

    from fashn_vton.dwpose import DWposeDetector

    person = np.asarray(Image.open(_WEIGHTS.parent / "examples" / "data" / "model.webp").convert("RGB"))
    pose_model = DWposeDetector(checkpoints_dir=str(_WEIGHTS / "dwpose"), device="cpu")
    pose = pose_model(person[..., ::-1])

    provider = build_segmentation_provider("sam2")
    assert provider.is_available() is True
    class_map = provider.predict(person, {"pose": pose, "category": "tops", "role": "person"})

    assert class_map is not None
    assert class_map.shape == person.shape[:2]
    assert class_map.dtype == np.uint8
    assert (class_map == LABELS_TO_IDS["top"]).sum() > 0, "SAM 2 no encontró la región de prenda"
    provider.release()
