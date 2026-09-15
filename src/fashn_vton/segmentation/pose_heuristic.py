"""License-clean heuristic segmentation from pose keypoints (EXPERIMENTAL).

Builds a FASHN-compatible class map (see ``labels.py``) from the COCO-18 body
keypoints that DWPose already produces in the pipeline: torso/arms/legs regions
plus the garment-region labels of the requested category.

Why it exists: the upstream pipeline used ``fashn-human-parser`` (non-commercial
weights derived from NVIDIA SegFormer) even when masking was not needed. This
provider offers a *no-new-licence* replacement for the flows that need a mask,
so the fork never has to fall back to the NC dependency.

Limitations (documented on purpose):

- It is **not** a semantic human parser: it cannot distinguish a hat from hair, a
  shirt from a jacket, etc. It only uses body geometry.
- Quality has **not** been validated against the original parser; use
  ``scripts/benchmark_providers.py`` and the commercial gate before enabling it
  for real traffic.
- The supported commercial path is still ``NoSegmentationProvider`` +
  ``segmentation_free=True`` + ``flat-lay`` garments.
"""

from __future__ import annotations

import numpy as np

from .base import ProviderInfo, SegmentationProvider
from .labels import LABELS_TO_IDS

# COCO-18 body keypoint indices as returned by this repo's DWPose detector.
NOSE, NECK = 0, 1
R_SHOULDER, R_ELBOW, R_WRIST = 2, 3, 4
L_SHOULDER, L_ELBOW, L_WRIST = 5, 6, 7
R_HIP, R_KNEE, R_ANKLE = 8, 9, 10
L_HIP, L_KNEE, L_ANKLE = 11, 12, 13
R_EAR, L_EAR = 16, 17

VISIBILITY_THRESHOLD = 0.3

# Garment label per category (interoperability ids from labels.LABELS_TO_IDS).
CATEGORY_TO_GARMENT_LABEL = {
    "tops": LABELS_TO_IDS["top"],
    "bottoms": LABELS_TO_IDS["pants"],
    "one-pieces": LABELS_TO_IDS["dress"],
}


def _scale(length: int, fraction: float, minimum: int = 1) -> int:
    return max(minimum, int(round(length * fraction)))


def pixel_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized keypoints to pixel coordinates.

    Returns ``(points, visible)`` where ``points`` is ``(18, 2)`` in pixels with
    ``nan`` for invisible keypoints and ``visible`` is a boolean ``(18,)`` mask.
    Shared by the heuristic and the SAM 2 providers.
    """
    height, width = image_shape
    points = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2).copy()
    visibility = np.asarray(scores, dtype=np.float32).reshape(-1)
    visible = (visibility[: points.shape[0]] >= VISIBILITY_THRESHOLD) & (points >= 0).all(axis=1)
    pixels = np.full_like(points, np.nan)
    pixels[visible, 0] = points[visible, 0] * width
    pixels[visible, 1] = points[visible, 1] * height
    return pixels, visible


def bounding_box(
    points: np.ndarray,
    indices: list[int],
    image_shape: tuple[int, int],
    margin: float = 0.05,
) -> np.ndarray | None:
    """Axis-aligned box ``[x0, y0, x1, y1]`` around the given keypoint indices.

    Returns ``None`` when no index is visible. ``margin`` grows the box by that
    fraction of the image's smallest side (SAM 2 likes a bit of context).
    """
    height, width = image_shape
    selected = points[[i for i in indices if i < len(points) and not np.isnan(points[i]).any()]]
    if selected.size == 0:
        return None
    pad = margin * min(height, width)
    x0 = max(0.0, float(selected[:, 0].min()) - pad)
    y0 = max(0.0, float(selected[:, 1].min()) - pad)
    x1 = min(float(width), float(selected[:, 0].max()) + pad)
    y1 = min(float(height), float(selected[:, 1].max()) + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def build_class_map_from_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
    category: str = "tops",
) -> np.ndarray:
    """Deterministic class map from COCO-18 keypoints (pure function, testable).

    Args:
        keypoints: ``(18, 2)`` array with ``x``/``y`` in ``[0, 1]`` (normalized).
        scores: ``(18,)`` visibility scores.
        image_shape: ``(height, width)`` of the target image.
        category: ``tops`` / ``bottoms`` / ``one-pieces``.

    Returns:
        ``uint8`` ``HxW`` array with FASHN class ids.
    """
    import cv2

    height, width = image_shape
    class_map = np.zeros((height, width), dtype=np.uint8)

    points = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 18:
        raise ValueError(f"Se esperaban 18 keypoints COCO, se recibieron {points.shape[0]}.")
    visibility = np.asarray(scores, dtype=np.float32).reshape(-1)[:18]

    def px(index: int) -> tuple[int, int] | None:
        if visibility[index] < VISIBILITY_THRESHOLD:
            return None
        x, y = points[index]
        if x < 0 or y < 0:
            return None
        return int(round(float(x) * width)), int(round(float(y) * height))

    limb_thickness = _scale(min(height, width), 0.045, minimum=3)

    def draw_chain(indices: list[int], label: int) -> None:
        drawn = [p for p in (px(i) for i in indices) if p is not None]
        for start, end in zip(drawn, drawn[1:]):
            cv2.line(class_map, start, end, color=int(label), thickness=limb_thickness)
        for point in drawn:
            cv2.circle(class_map, point, limb_thickness // 2, int(label), -1)

    for hip, knee, ankle in ((R_HIP, R_KNEE, R_ANKLE), (L_HIP, L_KNEE, L_ANKLE)):
        draw_chain([hip, knee, ankle], LABELS_TO_IDS["legs"])
    for shoulder, elbow, wrist in ((R_SHOULDER, R_ELBOW, R_WRIST), (L_SHOULDER, L_ELBOW, L_WRIST)):
        draw_chain([shoulder, elbow, wrist], LABELS_TO_IDS["arms"])

    shoulders = [px(i) for i in (R_SHOULDER, L_SHOULDER)]
    hips = [px(i) for i in (R_HIP, L_HIP)]
    if all(p is not None for p in shoulders) and all(p is not None for p in hips):
        polygon = np.array([shoulders[0], shoulders[1], hips[1], hips[0]], dtype=np.int32)
        cv2.fillPoly(class_map, [polygon], int(LABELS_TO_IDS["torso"]))

    # Garment region label for the requested category: this is what the model's
    # preprocessing uses to build both the agnostic person and the garment crop.
    garment_label = CATEGORY_TO_GARMENT_LABEL.get(category)
    if garment_label is not None:
        region = class_map == LABELS_TO_IDS["torso"]
        if category in ("tops", "one-pieces"):
            region = region | (class_map == LABELS_TO_IDS["arms"])
        if category in ("bottoms", "one-pieces"):
            region = region | (class_map == LABELS_TO_IDS["legs"])
        class_map[region] = garment_label

    # Identity/limb protection (same semantics as the agnostic mask: face, hair,
    # hands and feet are preserved).
    points_px, _ = pixel_keypoints(points, visibility, (height, width))
    paint_protection_labels(class_map, points_px, (height, width))

    return class_map


def paint_protection_labels(
    class_map: np.ndarray,
    points: np.ndarray,
    image_shape: tuple[int, int],
) -> None:
    """Paint the protection labels (hair, face, hands, feet) in place.

    Same semantics as the agnostic mask: these regions are preserved (never
    masked). Hair is painted first and face/hands/feet on top so the finer
    regions stay visible. ``points`` are pixel coordinates with ``nan`` for
    invisible keypoints (see :func:`pixel_keypoints`).
    """
    import cv2

    height, width = image_shape
    small_side = min(height, width)

    def _circle(index: int, radius: int, label: int) -> None:
        if index >= len(points):
            return
        x, y = points[index]
        if np.isnan(x) or np.isnan(y):
            return
        cv2.circle(class_map, (int(round(x)), int(round(y))), radius, int(label), -1)

    for index in (R_EAR, L_EAR):
        _circle(index, _scale(height, 0.05, minimum=6), LABELS_TO_IDS["hair"])
    for index in (NOSE, 14, 15):
        _circle(index, _scale(height, 0.028, minimum=4), LABELS_TO_IDS["face"])
    for index in (R_WRIST, L_WRIST):
        _circle(index, _scale(small_side, 0.028, minimum=3), LABELS_TO_IDS["hands"])
    for index in (R_ANKLE, L_ANKLE):
        _circle(index, _scale(small_side, 0.03, minimum=3), LABELS_TO_IDS["feet"])


class PoseHeuristicProvider(SegmentationProvider):
    """Heuristic regions from the DWPose output already computed by the pipeline."""

    info = ProviderInfo(
        name="pose-heuristic",
        license="Apache-2.0 (DWPose) + geometría propia",
        requires_weights=True,  # reutiliza los pesos de DWPose que ya carga el pipeline
        commercial_ok=True,
        experimental=True,
        notes=(
            "Proveedor heurístico basado en los keypoints COCO-18 de DWPose. "
            "No es un parser semántico y su calidad no está validada: úsalo solo en "
            "los flujos que necesitan máscara y valídalo con "
            "scripts/benchmark_providers.py. Si no recibe keypoints devuelve None y "
            "el pipeline degrada con un aviso."
        ),
    )

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
        return build_class_map_from_keypoints(keypoints, scores, (height, width), category)


def keypoints_from_pose(pose: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract the best candidate's 18 COCO keypoints from a DWPose output dict.

    Returns ``(keypoints (18,2) normalized, scores (18,))`` or ``(None, None)``.
    """
    bodies = pose.get("bodies") or {}
    candidate = bodies.get("candidate")
    subset = bodies.get("subset")
    if candidate is None or subset is None:
        return None, None
    candidate = np.asarray(candidate, dtype=np.float32)
    subset = np.asarray(subset, dtype=np.float32)
    if candidate.ndim != 2 or candidate.shape[0] < 18:
        return None, None
    score_row = subset[0] if subset.ndim == 2 else subset.reshape(-1)
    if score_row.shape[0] < 18:
        return None, None
    # DWPose marks invisible keypoints with -1 (both coordinates and score).
    scores = np.where(score_row[:18] < 0, 0.0, score_row[:18])
    return candidate[:18, :2], scores


__all__ = [
    "CATEGORY_TO_GARMENT_LABEL",
    "PoseHeuristicProvider",
    "bounding_box",
    "build_class_map_from_keypoints",
    "keypoints_from_pose",
    "paint_protection_labels",
    "pixel_keypoints",
]
