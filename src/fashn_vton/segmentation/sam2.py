"""SAM 2 segmentation provider (stage 2 of the commercial plan).

SAM 2 (``facebookresearch/sam2``) is **Apache-2.0 for both code and
checkpoints**, so it is the recommended replacement when real masks are needed
(garment isolation for model-worn photos, tighter agnostic masks).

How it works here: the DWPose keypoints that the pipeline already computes are
turned into a few axis-aligned boxes (torso, arms, legs); SAM 2 segments each
box and the resulting masks are written into a FASHN class map (``labels.py``)
with the garment label of the requested category. Face, hair, hands and feet
keep their protection labels, exactly like the heuristic provider.

Verified setup (2026-09-15, this machine)::

    pip install sam2==1.1.0
    python -c "from huggingface_hub import hf_hub_download; \
        hf_hub_download('facebook/sam2.1-hiera-tiny', 'sam2.1_hiera_tiny.pt', local_dir='weights/sam2')"

Checkpoint audited: ``facebook/sam2.1-hiera-tiny`` (156 MB, license
``apache-2.0`` per its HuggingFace model card); its sha256 is recorded in
``licenses/manifest.json``.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import numpy as np

from .base import ProviderInfo, SegmentationProvider
from .labels import LABELS_TO_IDS
from .pose_heuristic import (
    CATEGORY_TO_GARMENT_LABEL,
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    bounding_box,
    keypoints_from_pose,
    paint_protection_labels,
    pixel_keypoints,
)

DEFAULT_WEIGHTS_DIR = "weights/sam2"
DEFAULT_CHECKPOINT = "sam2.1_hiera_tiny.pt"
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT_GLOBS = ("*.pt", "*.pth")

#: Body regions prompted to SAM 2 (COCO-18 indices).
REGIONS: dict[str, list[int]] = {
    "torso": [R_SHOULDER, L_SHOULDER, R_HIP, L_HIP],
    "arm_right": [R_SHOULDER, R_ELBOW, R_WRIST],
    "arm_left": [L_SHOULDER, L_ELBOW, L_WRIST],
    "leg_right": [R_HIP, R_KNEE, R_ANKLE],
    "leg_left": [L_HIP, L_KNEE, L_ANKLE],
}

#: Body label given to each region when it is not the garment region.
BODY_LABEL: dict[str, str] = {
    "torso": "torso",
    "arm_right": "arms",
    "arm_left": "arms",
    "leg_right": "legs",
    "leg_left": "legs",
}

#: Regions that receive the garment label of the category.
CATEGORY_REGIONS: dict[str, list[str]] = {
    "tops": ["torso", "arm_right", "arm_left"],
    "bottoms": ["leg_right", "leg_left"],
    "one-pieces": ["torso", "arm_right", "arm_left", "leg_right", "leg_left"],
}


def prompt_boxes_from_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
    margin: float = 0.05,
) -> dict[str, np.ndarray]:
    """Build the SAM 2 prompt boxes from the COCO-18 keypoints (pure function)."""
    points, _ = pixel_keypoints(keypoints, scores, image_shape)
    boxes: dict[str, np.ndarray] = {}
    for name, indices in REGIONS.items():
        box = bounding_box(points, indices, image_shape, margin=margin)
        if box is not None:
            boxes[name] = box
    return boxes


def class_map_from_masks(
    masks: dict[str, np.ndarray],
    points: np.ndarray,
    image_shape: tuple[int, int],
    category: str,
) -> np.ndarray:
    """Compose a FASHN class map from SAM 2 region masks (pure function)."""
    height, width = image_shape
    class_map = np.zeros((height, width), dtype=np.uint8)
    garment_label = CATEGORY_TO_GARMENT_LABEL.get(category, LABELS_TO_IDS["top"])
    garment_regions = set(CATEGORY_REGIONS.get(category, []))

    for name, mask in masks.items():
        if name not in BODY_LABEL:
            continue
        label = garment_label if name in garment_regions else LABELS_TO_IDS[BODY_LABEL[name]]
        class_map[np.asarray(mask, dtype=bool)] = int(label)

    paint_protection_labels(class_map, points, (height, width))
    return class_map


class Sam2SegmentationProvider(SegmentationProvider):
    """SAM 2 masks (DWPose-prompted boxes) mapped to a FASHN class map."""

    info = ProviderInfo(
        name="sam2",
        license="Apache-2.0 (código y checkpoints de facebookresearch/sam2)",
        requires_weights=True,
        commercial_ok=True,
        weights_dir=DEFAULT_WEIGHTS_DIR,
        notes=(
            "Requiere: pip install 'sam2==1.1.0' y un checkpoint Apache-2.0 en "
            f"{DEFAULT_WEIGHTS_DIR}/ (p. ej. {DEFAULT_CHECKPOINT} de facebook/sam2.1-hiera-tiny).\n"
            "Descarga:\n"
            '  python -c "from huggingface_hub import hf_hub_download; '
            "hf_hub_download('facebook/sam2.1-hiera-tiny', 'sam2.1_hiera_tiny.pt', "
            "local_dir='weights/sam2')\"\n"
            "Después verifica su hash con: python licenses/verify_hashes.py"
        ),
    )

    def __init__(
        self,
        weights_dir: str | Path = DEFAULT_WEIGHTS_DIR,
        checkpoint: str | Path | None = None,
        model_cfg: str = DEFAULT_MODEL_CFG,
        device: str | None = None,
        margin: float = 0.05,
    ) -> None:
        self.weights_dir = Path(weights_dir)
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.model_cfg = model_cfg
        self.device = device
        self.margin = margin
        self._predictor = None
        self._lock = threading.Lock()

    # --- availability ------------------------------------------------------
    def checkpoint_path(self) -> Path | None:
        if self.checkpoint and self.checkpoint.is_file():
            return self.checkpoint
        if not self.weights_dir.is_dir():
            return None
        for pattern in CHECKPOINT_GLOBS:
            candidates = sorted(self.weights_dir.glob(pattern))
            if candidates:
                return candidates[0]
        return None

    def is_available(self) -> bool:
        return importlib.util.find_spec("sam2") is not None and self.checkpoint_path() is not None

    # --- model -------------------------------------------------------------
    def _get_predictor(self):
        """Build (once) the SAM 2 image predictor."""
        if self._predictor is not None:
            return self._predictor
        with self._lock:
            if self._predictor is not None:
                return self._predictor
            self.require_available()
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = build_sam2(self.model_cfg, str(self.checkpoint_path()), device=device)
            self._predictor = SAM2ImagePredictor(model)
            self._device = device
        return self._predictor

    def release(self) -> None:
        """Free the model (useful before handing the GPU to another consumer)."""
        with self._lock:
            self._predictor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # --- inference ---------------------------------------------------------
    def predict_masks(self, image: np.ndarray, boxes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Segment every box; returns ``{region_name: bool mask}``."""
        predictor = self._get_predictor()
        predictor.set_image(np.ascontiguousarray(image))
        masks: dict[str, np.ndarray] = {}
        height, width = image.shape[:2]
        for name, box in boxes.items():
            result, scores, _ = predictor.predict(
                box=np.asarray(box, dtype=np.float32)[None, :], multimask_output=False
            )
            array = np.asarray(result)
            if array.ndim == 4:  # (1, 1, H, W)
                array = array[0]
            if array.ndim == 3:  # (1, H, W)
                array = array[0]
            mask = array.astype(bool)
            if mask.shape != (height, width):
                raise RuntimeError(f"SAM 2 devolvió una máscara {mask.shape} distinta de la imagen {(height, width)}.")
            masks[name] = mask
        return masks

    def predict(self, image: np.ndarray, context: dict | None = None) -> np.ndarray | None:
        context = context or {}
        pose = context.get("pose")
        if not pose:
            return None
        keypoints, scores = keypoints_from_pose(pose)
        if keypoints is None:
            return None
        category = context.get("category", "tops")
        height, width = image.shape[:2]

        boxes = prompt_boxes_from_keypoints(keypoints, scores, (height, width), margin=self.margin)
        if not boxes:
            return None

        masks = self.predict_masks(image, boxes)
        points, _ = pixel_keypoints(keypoints, scores, (height, width))
        return class_map_from_masks(masks, points, (height, width), category)


__all__ = [
    "CATEGORY_REGIONS",
    "BODY_LABEL",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_MODEL_CFG",
    "DEFAULT_WEIGHTS_DIR",
    "REGIONS",
    "Sam2SegmentationProvider",
    "class_map_from_masks",
    "prompt_boxes_from_keypoints",
]
