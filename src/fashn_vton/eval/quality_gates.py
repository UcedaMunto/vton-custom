"""License-clean quality metrics for the commercial benchmark (plan §14).

Everything here is statistics over pixels with permissive libraries
(``scikit-image`` BSD-3-Clause, ``scipy`` BSD-3-Clause, ``numpy`` BSD-3-Clause,
``opencv`` Apache-2.0). **No pretrained networks** are used, precisely to avoid
pulling third-party weights with unreviewed licences (LPIPS/VGG would be the
usual choice and is deliberately not used).

Metrics returned by :func:`evaluate` (all higher-is-better except the distances,
which are documented):

- ``identity_preservation``: SSIM between the person and the output *inside* the
  identity regions (face, hair, hands, feet) -> "¿se conservó la persona?".
- ``outside_change``: mean absolute change outside the garment region (lower is
  better: the product should not touch the rest of the photo).
- ``color_fidelity``: 1 - normalised Wasserstein distance between the colour
  distributions of the garment reference and the generated garment region.
- ``pattern_fidelity``: SSIM between the garment reference (resized to the
  region's bounding box) and the generated region -> crude print/logo proxy.
- ``garment_change``: fraction of the garment region that actually changed
  relative to the input person.
- ``sharpness``: variance of the Laplacian over the output (blur/artefact proxy).
- ``failure``: True when the run could not be measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..segmentation.labels import IDENTITY_LABELS, LABELS_TO_IDS

#: Labels treated as "identity" (must be preserved) for the metric.
IDENTITY_IDS = [LABELS_TO_IDS[label] for label in IDENTITY_LABELS]
#: Labels considered "garment region" for the metric.
GARMENT_IDS = [LABELS_TO_IDS[label] for label in ("top", "dress", "skirt", "pants", "belt", "scarf")]


@dataclass
class QualityGateConfig:
    """Thresholds for :func:`passes_quality_gate` (starting values, to calibrate)."""

    min_identity_preservation: float = 0.60
    max_outside_change: float = 8.0
    min_color_fidelity: float = 0.55
    min_pattern_fidelity: float = 0.20
    min_sharpness: float = 20.0
    max_failure_rate: float = 0.05
    weights: dict = field(default_factory=lambda: {"color_fidelity": 3.0, "identity_preservation": 2.0})


def _to_float_image(image) -> np.ndarray:
    array = np.asarray(image).astype(np.float32)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    return array[..., :3]


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(reference, candidate, channel_axis=-1, data_range=255.0))


def wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Wasserstein distance between two RGB pixel sets (0 = identical)."""
    from scipy.stats import wasserstein_distance as _wd

    values = np.asarray(a).reshape(-1, 3).astype(np.float32)
    other = np.asarray(b).reshape(-1, 3).astype(np.float32)
    if values.size == 0 or other.size == 0:
        return float("nan")
    return float(np.mean([_wd(values[:, channel], other[:, channel]) for channel in range(3)]))


def color_fidelity(garment_reference: np.ndarray, generated_region: np.ndarray) -> float:
    """1 - distance/255 between colour distributions (1 = idéntico, 0 = opuesto)."""
    distance = wasserstein_distance(garment_reference, generated_region)
    if np.isnan(distance):
        return float("nan")
    return float(max(0.0, 1.0 - distance / 255.0))


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian (blur/artefact proxy)."""
    import cv2

    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def region_mask(class_map: np.ndarray | None, ids: list[int]) -> np.ndarray | None:
    """Boolean mask of the class ids in ``class_map`` (None when no class map)."""
    if class_map is None:
        return None
    return np.isin(np.asarray(class_map), ids)


def _crop_to_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return image[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _masked_ssim(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    """SSIM averaged inside ``mask`` (SSIM needs spatial structure, so it is
    computed on the full image and then averaged over the mask)."""
    from skimage.metrics import structural_similarity

    _, ssim_map = structural_similarity(reference, candidate, channel_axis=-1, data_range=255.0, full=True)
    ssim_map = ssim_map if ssim_map.ndim == 2 else ssim_map.mean(axis=-1)
    if not mask.any():
        return float("nan")
    return float(ssim_map[mask].mean())


def _garment_reference_pixels(garment_image: np.ndarray) -> np.ndarray:
    """Pixels of the garment photo that are not a light background."""
    flat = garment_image.reshape(-1, 3)
    background = (flat > 235).all(axis=1)
    selected = flat[~background]
    return selected if selected.size else flat


def evaluate(
    person_image,
    output_image,
    garment_image,
    class_map: np.ndarray | None = None,
    change_threshold: float = 20.0,
) -> dict:
    """Compute the license-clean quality metrics for one generated image.

    ``class_map`` is the FASHN class map of the *person* image (the segmentation
    provider can produce it); when missing, the region metrics are ``nan``.
    """
    person = _to_float_image(person_image)
    output = _to_float_image(output_image)
    garment = _to_float_image(garment_image)
    if person.shape != output.shape:
        raise ValueError(f"person {person.shape} y output {output.shape} deben coincidir.")

    difference = np.abs(output - person).mean(axis=-1)
    garment_mask = region_mask(class_map, GARMENT_IDS)
    identity_mask = region_mask(class_map, IDENTITY_IDS)

    metrics: dict = {"sharpness": sharpness(output)}
    metrics["outside_change"] = (
        float(difference[~garment_mask].mean()) if garment_mask is not None and (~garment_mask).any() else float("nan")
    )
    metrics["identity_preservation"] = (
        _masked_ssim(person, output, identity_mask)
        if identity_mask is not None and identity_mask.any()
        else float("nan")
    )
    metrics["garment_change"] = (
        float((difference[garment_mask] > change_threshold).mean())
        if garment_mask is not None and garment_mask.any()
        else float("nan")
    )

    if garment_mask is not None and garment_mask.any():
        metrics["color_fidelity"] = color_fidelity(_garment_reference_pixels(garment), output[garment_mask])

        generated_crop = _crop_to_mask(output, garment_mask)
        reference_crop = _crop_to_mask(garment, np.ones(garment.shape[:2], dtype=bool))
        if generated_crop is not None and reference_crop is not None:
            import cv2

            resized = cv2.resize(
                reference_crop.astype(np.uint8),
                (generated_crop.shape[1], generated_crop.shape[0]),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            metrics["pattern_fidelity"] = _ssim(resized, generated_crop)
        else:
            metrics["pattern_fidelity"] = float("nan")
    else:
        metrics["color_fidelity"] = float("nan")
        metrics["pattern_fidelity"] = float("nan")

    metrics["failure"] = False
    return metrics


def passes_quality_gate(metrics: dict, config: QualityGateConfig | None = None) -> tuple[bool, list[str]]:
    """Check the thresholds; returns ``(passed, reasons_when_failed)``."""
    config = config or QualityGateConfig()
    reasons: list[str] = []
    if metrics.get("failure"):
        return False, ["la corrida falló"]

    def _check(name: str, threshold: float, minimum: bool) -> None:
        value = metrics.get(name)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            reasons.append(f"{name}: sin dato")
        elif minimum and value < threshold:
            reasons.append(f"{name}={value:.3f} < {threshold:.3f}")
        elif not minimum and value > threshold:
            reasons.append(f"{name}={value:.3f} > {threshold:.3f}")

    _check("identity_preservation", config.min_identity_preservation, True)
    _check("color_fidelity", config.min_color_fidelity, True)
    _check("pattern_fidelity", config.min_pattern_fidelity, True)
    _check("sharpness", config.min_sharpness, True)
    _check("outside_change", config.max_outside_change, False)
    return (not reasons), reasons


def weighted_score(metrics: dict, config: QualityGateConfig | None = None) -> float:
    """Single 0-1 score to rank variants (higher is better)."""
    config = config or QualityGateConfig()
    if metrics.get("failure"):
        return 0.0
    total = 0.0
    weight_sum = 0.0
    for name, weight in config.weights.items():
        value = metrics.get(name)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        total += weight * float(value)
        weight_sum += weight
    return float(total / weight_sum) if weight_sum else 0.0


def aggregate(rows: list[dict], config: QualityGateConfig | None = None) -> dict:
    """Aggregate one metric dict per run into means, failure rate and score."""
    config = config or QualityGateConfig()
    names = (
        "identity_preservation",
        "outside_change",
        "color_fidelity",
        "pattern_fidelity",
        "garment_change",
        "sharpness",
    )
    summary: dict = {"runs": len(rows)}
    for name in names:
        values = [
            float(row[name])
            for row in rows
            if row.get(name) is not None and not (isinstance(row[name], float) and np.isnan(row[name]))
        ]
        summary[name] = float(np.mean(values)) if values else None

    failures = sum(1 for row in rows if row.get("failure"))
    summary["failures"] = failures
    summary["failure_rate"] = failures / len(rows) if rows else 0.0
    scored = [weighted_score(row, config) for row in rows if not row.get("failure")]
    summary["score"] = float(np.mean(scored)) if scored else 0.0
    summary["passes_gate"] = bool(
        summary["failure_rate"] <= config.max_failure_rate
        and (summary["identity_preservation"] or 0) >= config.min_identity_preservation
        and (summary["color_fidelity"] or 0) >= config.min_color_fidelity
    )
    return summary


__all__ = [
    "GARMENT_IDS",
    "IDENTITY_IDS",
    "QualityGateConfig",
    "aggregate",
    "color_fidelity",
    "evaluate",
    "passes_quality_gate",
    "region_mask",
    "sharpness",
    "wasserstein_distance",
    "weighted_score",
]
