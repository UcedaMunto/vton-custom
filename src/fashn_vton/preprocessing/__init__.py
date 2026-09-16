"""Preprocessing utilities."""

from .agnostic import (
    BODY_COVERAGE_TO_FASHN_LABELS,
    FASHN_LABELS_TO_IDS,
    create_clothing_agnostic_image,
    create_garment_image,
)
from .fabric import (
    BACKGROUND_MODES,
    MASK_SOURCES,
    MOSAIC_MODES,
    FabricTransferConfig,
    FabricTransferInfo,
    detect_background_color,
    garment_mask_from_background,
    retexture_garment,
)
from .transforms import AspectPreserveResize, PadToShape, ResizePad

__all__ = [
    # Clothing-agnostic creation
    "create_clothing_agnostic_image",
    "create_garment_image",
    # Constants
    "FASHN_LABELS_TO_IDS",
    "BODY_COVERAGE_TO_FASHN_LABELS",
    # Transforms
    "AspectPreserveResize",
    "ResizePad",
    "PadToShape",
    # Sustitución de tela (color/textura/ángulo)
    "BACKGROUND_MODES",
    "MASK_SOURCES",
    "MOSAIC_MODES",
    "FabricTransferConfig",
    "FabricTransferInfo",
    "detect_background_color",
    "garment_mask_from_background",
    "retexture_garment",
]
