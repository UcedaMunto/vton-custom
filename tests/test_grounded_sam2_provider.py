"""Pruebas del proveedor Grounded-SAM 2 (detección con texto + máscara SAM 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fashn_vton.segmentation import GroundedSam2Provider, build_segmentation_provider
from fashn_vton.segmentation.grounded_sam2 import (
    CATEGORY_TO_PROMPTS,
    build_prompt,
    class_map_from_garment_mask,
)
from fashn_vton.segmentation.labels import LABELS_TO_IDS

# --- unitarias (sin checkpoints) --------------------------------------------


def test_build_prompt_formats_grounding_dino_text():
    prompt = build_prompt("tops")
    assert prompt.endswith(".")
    assert "shirt." in prompt
    assert prompt == prompt.lower()
    assert build_prompt("bottoms").count(".") == len(CATEGORY_TO_PROMPTS["bottoms"])
    assert build_prompt("inexistente") == ""


def test_class_map_from_garment_mask_labels_the_garment():
    mask = np.zeros((100, 80), dtype=bool)
    mask[20:60, 10:70] = True
    for category, expected in (("tops", "top"), ("bottoms", "pants"), ("one-pieces", "dress")):
        class_map = class_map_from_garment_mask(mask, (100, 80), category)
        assert (class_map == LABELS_TO_IDS[expected]).sum() == mask.sum()
        assert class_map.dtype == np.uint8


def test_class_map_from_garment_mask_paints_protection_when_points_given():
    mask = np.zeros((200, 160), dtype=bool)
    mask[60:140, 40:120] = True
    points = np.full((18, 2), np.nan, dtype=np.float32)
    points[0] = (80, 20)  # nariz
    points[4] = (30, 90)  # muñeca derecha
    class_map = class_map_from_garment_mask(mask, (200, 160), "tops", points)
    assert (class_map == LABELS_TO_IDS["face"]).sum() > 0
    assert (class_map == LABELS_TO_IDS["hands"]).sum() > 0


def test_unknown_category_returns_no_boxes_without_loading_models(tmp_path):
    provider = GroundedSam2Provider(detector_dir=tmp_path)
    boxes, scores = provider.detect_boxes(np.zeros((32, 32, 3), dtype=np.uint8), "sombreros")
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)


def test_provider_unavailable_without_detector(tmp_path):
    provider = GroundedSam2Provider(detector_dir=tmp_path / "nada")
    assert provider.detector_path() is None
    assert provider.is_available() is False
    with pytest.raises(Exception):
        provider.predict(np.zeros((32, 32, 3), dtype=np.uint8), {"category": "tops"})


def test_detector_path_requires_config_and_weights(tmp_path):
    folder = tmp_path / "gd"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"x")
    assert GroundedSam2Provider(detector_dir=folder).detector_path() is None  # falta config.json
    (folder / "config.json").write_text("{}", encoding="utf-8")
    assert GroundedSam2Provider(detector_dir=folder).detector_path() == folder


def test_metadata_is_commercial():
    provider = build_segmentation_provider("grounded-sam2")
    assert isinstance(provider, GroundedSam2Provider)
    assert provider.info.commercial_ok is True
    assert "Apache-2.0" in provider.info.license


# --- integración real (requiere checkpoints) --------------------------------

ROOT = Path(__file__).resolve().parents[1]
_MISSING = not (ROOT / "weights" / "sam2").is_dir() or not (ROOT / "weights" / "grounding_dino").is_dir()


@pytest.mark.integration
@pytest.mark.skipif(_MISSING, reason="Requiere weights/sam2 y weights/grounding_dino (no versionados).")
def test_grounded_sam2_detects_and_segments_the_example_garment():
    from PIL import Image

    garment = np.asarray(Image.open(ROOT / "examples" / "data" / "garment.webp").convert("RGB"))
    provider = build_segmentation_provider("grounded-sam2")
    assert provider.is_available() is True

    boxes, scores = provider.detect_boxes(garment, "tops")
    assert boxes.shape[0] > 0, "Grounding DINO no detectó ninguna prenda"
    assert boxes.shape[1] == 4

    class_map = provider.predict(garment, {"category": "tops", "role": "garment"})
    assert class_map is not None
    assert class_map.shape == garment.shape[:2]
    assert (class_map == LABELS_TO_IDS["top"]).sum() > 0
    provider.release()
