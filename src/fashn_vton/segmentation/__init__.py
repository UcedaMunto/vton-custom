"""Segmentation providers for the commercial fork (no non-permissive deps).

Public entry point::

    from fashn_vton.segmentation import build_segmentation_provider

    pipeline = TryOnPipeline(weights_dir="./weights",
                             segmentation_provider=build_segmentation_provider("none"))

Available names: ``none`` (default, commercial MVP), ``pose-heuristic``
(experimental, DWPose-based), ``sam2``, ``grounded-sam2`` and ``custom-parser``
(stubs for stages 2/3 of the commercial plan). See ``README.md`` in this
package for the decision matrix.
"""

from __future__ import annotations

from .base import (
    InvalidClassMapError,
    ProviderInfo,
    ProviderNotAvailable,
    SegmentationProvider,
    SegmentationProviderError,
    validate_class_map,
)
from .custom_parser import CustomHumanParserProvider
from .grounded_sam2 import GroundedSam2Provider
from .labels import (
    BODY_COVERAGE_TO_LABELS,
    CATEGORY_TO_BODY_COVERAGE,
    IDENTITY_LABELS,
    LABELS_TO_IDS,
)
from .none import NoSegmentationProvider
from .pose_heuristic import PoseHeuristicProvider
from .sam2 import Sam2SegmentationProvider

#: Registry used by the API/UI/CLI to build a provider by name.
PROVIDERS: dict[str, type[SegmentationProvider]] = {
    "none": NoSegmentationProvider,
    "pose-heuristic": PoseHeuristicProvider,
    "sam2": Sam2SegmentationProvider,
    "grounded-sam2": GroundedSam2Provider,
    "custom-parser": CustomHumanParserProvider,
}

DEFAULT_PROVIDER = "none"


def build_segmentation_provider(name: str | None = None, **kwargs) -> SegmentationProvider:
    """Instantiate a provider by name (``None`` -> the commercial default).

    Raises:
        ValueError: unknown name (lists the available ones).
    """
    key = (name or DEFAULT_PROVIDER).strip().lower()
    if key not in PROVIDERS:
        raise ValueError(
            f"Proveedor de segmentación desconocido: '{name}'. Disponibles: {', '.join(sorted(PROVIDERS))}."
        )
    return PROVIDERS[key](**kwargs)


def available_providers() -> dict[str, dict]:
    """Describe every registered provider and whether it can run here."""
    report = {}
    for name, cls in PROVIDERS.items():
        try:
            instance = cls()
            report[name] = {
                "license": instance.info.license,
                "commercial_ok": instance.info.commercial_ok,
                "experimental": instance.info.experimental,
                "requires_weights": instance.info.requires_weights,
                "available": instance.is_available(),
                "notes": instance.info.notes,
            }
        except Exception as exc:  # noqa: BLE001 - report, never crash
            report[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return report


__all__ = [
    "BODY_COVERAGE_TO_LABELS",
    "CATEGORY_TO_BODY_COVERAGE",
    "DEFAULT_PROVIDER",
    "IDENTITY_LABELS",
    "InvalidClassMapError",
    "LABELS_TO_IDS",
    "PROVIDERS",
    "ProviderInfo",
    "ProviderNotAvailable",
    "CustomHumanParserProvider",
    "GroundedSam2Provider",
    "NoSegmentationProvider",
    "PoseHeuristicProvider",
    "Sam2SegmentationProvider",
    "SegmentationProvider",
    "SegmentationProviderError",
    "available_providers",
    "build_segmentation_provider",
    "validate_class_map",
]
