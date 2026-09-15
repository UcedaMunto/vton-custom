"""Grounding DINO + SAM 2 provider (stage 2b of the commercial plan).

Needed when the garment input is a photo of **another person wearing** the
garment: Grounding DINO locates it from text prompts ("shirt", "jacket", …) and
SAM 2 refines the mask, which is then written into a FASHN class map.

Both components are Apache-2.0 (code and checkpoints); the concrete checkpoints
are audited in ``licenses/manifest.json``:

- ``IDEA-Research/grounding-dino-tiny`` (Apache-2.0, ~659 MB)
- ``facebook/sam2.1-hiera-tiny`` (Apache-2.0, ~156 MB)

Verified setup (2026-09-15, this machine)::

    python -c "from huggingface_hub import snapshot_download; \
        snapshot_download('IDEA-Research/grounding-dino-tiny', local_dir='weights/grounding_dino', \
        allow_patterns=['*.json','*.txt','*.safetensors'])"
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
    keypoints_from_pose,
    paint_protection_labels,
    pixel_keypoints,
)
from .sam2 import Sam2SegmentationProvider

DEFAULT_DETECTOR_DIR = "weights/grounding_dino"
DEFAULT_DETECTOR_REPO = "IDEA-Research/grounding-dino-tiny"

#: Text prompts per category (commercial plan §5), lowercase and dot-separated
#: as Grounding DINO expects.
CATEGORY_TO_PROMPTS: dict[str, list[str]] = {
    "tops": ["shirt", "t-shirt", "blouse", "jacket", "sweater"],
    "bottoms": ["pants", "jeans", "skirt", "shorts"],
    "one-pieces": ["dress", "jumpsuit"],
}

#: Minimal set of model files needed to load a local Grounding DINO checkpoint.
DETECTOR_FILE_GLOBS = ("*.safetensors", "model.safetensors", "config.json", "preprocessor_config.json")


def build_prompt(category: str) -> str:
    """Grounding DINO text prompt for a category ('' when unknown)."""
    words = CATEGORY_TO_PROMPTS.get(category, [])
    return " ".join(f"{word}." for word in words)


def _to_numpy(value) -> np.ndarray:
    """Convert a torch tensor (possibly on GPU) or a list to a float32 ndarray."""
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def class_map_from_garment_mask(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    category: str,
    points: np.ndarray | None = None,
) -> np.ndarray:
    """FASHN class map with the detected garment region (pure function)."""
    height, width = image_shape
    class_map = np.zeros((height, width), dtype=np.uint8)
    label = CATEGORY_TO_GARMENT_LABEL.get(category, LABELS_TO_IDS["top"])
    class_map[np.asarray(mask, dtype=bool)] = int(label)
    if points is not None:
        paint_protection_labels(class_map, points, (height, width))
    return class_map


class GroundedSam2Provider(SegmentationProvider):
    """Grounded-SAM 2: detect the garment with text, refine the mask with SAM 2."""

    info = ProviderInfo(
        name="grounded-sam2",
        license="Apache-2.0 (Grounding DINO + SAM 2, código y checkpoints)",
        requires_weights=True,
        commercial_ok=True,
        weights_dir=f"{DEFAULT_DETECTOR_DIR} + weights/sam2",
        notes=(
            "Requiere los pesos de Grounding DINO y de SAM 2:\n"
            f"  {DEFAULT_DETECTOR_DIR}/ (p. ej. {DEFAULT_DETECTOR_REPO}) y weights/sam2/*.pt\n"
            "Descarga del detector:\n"
            '  python -c "from huggingface_hub import snapshot_download; '
            f"snapshot_download('{DEFAULT_DETECTOR_REPO}', local_dir='{DEFAULT_DETECTOR_DIR}', "
            "allow_patterns=['*.json','*.txt','*.safetensors'])\"\n"
            "Pensado para prendas fotografiadas sobre una persona (prompts de texto por categoría)."
        ),
    )

    def __init__(
        self,
        detector_dir: str | Path = DEFAULT_DETECTOR_DIR,
        sam2_provider: Sam2SegmentationProvider | None = None,
        device: str | None = None,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        detector_repo: str = DEFAULT_DETECTOR_REPO,
    ) -> None:
        self.detector_dir = Path(detector_dir)
        self.detector_repo = detector_repo
        self.sam2 = sam2_provider or Sam2SegmentationProvider(device=device)
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._detector = None
        self._processor = None
        self._lock = threading.Lock()

    # --- availability ------------------------------------------------------
    def detector_path(self) -> Path | None:
        if not self.detector_dir.is_dir():
            return None
        has_weights = any(self.detector_dir.glob("*.safetensors")) or any(self.detector_dir.glob("*.bin"))
        has_config = (self.detector_dir / "config.json").is_file()
        return self.detector_dir if (has_weights and has_config) else None

    def is_available(self) -> bool:
        return (
            importlib.util.find_spec("transformers") is not None
            and self.detector_path() is not None
            and self.sam2.is_available()
        )

    def _load(self):
        if self._processor is not None and self._detector is not None:
            return self._processor, self._detector
        with self._lock:
            if self._processor is not None and self._detector is not None:
                return self._processor, self._detector
            self.require_available()
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = AutoProcessor.from_pretrained(str(self.detector_path()), local_files_only=True)
            self._detector = (
                AutoModelForZeroShotObjectDetection.from_pretrained(str(self.detector_path()), local_files_only=True)
                .to(device)
                .eval()
            )
            self._device = device
        return self._processor, self._detector

    def release(self) -> None:
        with self._lock:
            self._detector = None
            self._processor = None
        self.sam2.release()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # --- inference ---------------------------------------------------------
    def detect_boxes(self, image: np.ndarray, category: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(boxes Nx4 xyxy, scores N)`` for the garment of the category."""
        prompt = build_prompt(category)
        if not prompt:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        import torch
        from PIL import Image

        processor, detector = self._load()
        pil = Image.fromarray(image)
        inputs = processor(images=pil, text=prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = detector(**inputs)

        target_sizes = [pil.size[::-1]]
        try:
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:  # transformers >= 5 renombró box_threshold -> threshold
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )

        result = results[0]
        boxes = _to_numpy(result.get("boxes", [])).reshape(-1, 4)
        scores = _to_numpy(result.get("scores", [])).reshape(-1)
        return boxes, scores

    def predict(self, image: np.ndarray, context: dict | None = None) -> np.ndarray | None:
        context = context or {}
        category = context.get("category", "tops")
        height, width = image.shape[:2]

        boxes, _scores = self.detect_boxes(image, category)
        if boxes.shape[0] == 0:
            return None

        # Descartar detecciones diminutas (logos, detalles) y limitar su número.
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = areas >= 0.01 * height * width
        if not keep.any():
            keep = np.ones_like(areas, dtype=bool)
        selected = {f"garment_{index}": box for index, box in enumerate(boxes[keep][:3])}

        masks = self.sam2.predict_masks(image, selected)
        union = np.zeros((height, width), dtype=bool)
        for mask in masks.values():
            union |= mask

        points = None
        pose = context.get("pose")
        if pose:
            keypoints, key_scores = keypoints_from_pose(pose)
            if keypoints is not None:
                points, _ = pixel_keypoints(keypoints, key_scores, (height, width))
        return class_map_from_garment_mask(union, (height, width), category, points)


__all__ = [
    "CATEGORY_TO_PROMPTS",
    "DEFAULT_DETECTOR_DIR",
    "DEFAULT_DETECTOR_REPO",
    "GroundedSam2Provider",
    "build_prompt",
    "class_map_from_garment_mask",
]
