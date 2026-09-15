"""No-op segmentation provider: the commercial MVP path.

Nothing is loaded and no weights are needed, so there is no license exposure at
all. Use it together with ``segmentation_free=True`` and
``garment_photo_type="flat-lay"`` (the supported commercial configuration).
"""

from __future__ import annotations

import numpy as np

from .base import ProviderInfo, SegmentationProvider


class NoSegmentationProvider(SegmentationProvider):
    """Returns no class map; the pipeline degrades with an explicit warning."""

    info = ProviderInfo(
        name="none",
        license="N/A (no carga ningún modelo)",
        requires_weights=False,
        commercial_ok=True,
        notes=(
            "Proveedor sin segmentación. Camino comercial soportado: "
            "segmentation_free=True + garment_photo_type='flat-lay'. "
            "Si se pide máscara (segmentation_free=False o prenda 'model'), el "
            "pipeline avisa y procesa sin máscara en vez de fallar."
        ),
    )

    def predict(self, image: np.ndarray, context: dict | None = None) -> np.ndarray | None:
        return None


__all__ = ["NoSegmentationProvider"]
