"""Company-owned human parser provider (stage 3 of the commercial plan) — stub.

Long-term independence: a parser trained by the company (Detectron2, Apache-2.0)
on own/synthetic/licensed data, so both the architecture *and* the weights are
controlled. The label conversion table is already defined here so that a future
checkpoint only needs ``predict()`` implemented.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import ProviderInfo, SegmentationProvider
from .labels import LABELS_TO_IDS


class CustomHumanParserProvider(SegmentationProvider):
    """Own parser (not trained yet) exposed behind the same provider contract."""

    info = ProviderInfo(
        name="custom-parser",
        license="Apache-2.0 (Detectron2) + pesos/datos propios",
        requires_weights=True,
        commercial_ok=True,
        weights_dir="weights/custom-parser",
        notes=(
            "Etapa 3: parser propio entrenado con datos propios. Cuando exista el "
            "checkpoint, solo hay que rellenar predict() devolviendo el mapa de clases "
            "FASHN usando COMMERCIAL_TO_FASHN como tabla de conversión."
        ),
    )

    #: Company label -> FASHN class id (skeleton from the commercial plan §6).
    COMMERCIAL_TO_FASHN: dict[str, int] = {
        "background": LABELS_TO_IDS["background"],
        "head": LABELS_TO_IDS["face"],
        "hair": LABELS_TO_IDS["hair"],
        "torso": LABELS_TO_IDS["torso"],
        "left_arm": LABELS_TO_IDS["arms"],
        "right_arm": LABELS_TO_IDS["arms"],
        "upper_clothing": LABELS_TO_IDS["top"],
        "lower_clothing": LABELS_TO_IDS["pants"],
        "dress": LABELS_TO_IDS["dress"],
        "left_leg": LABELS_TO_IDS["legs"],
        "right_leg": LABELS_TO_IDS["legs"],
        "shoes": LABELS_TO_IDS["feet"],
    }

    def __init__(self, weights_dir: str | Path = "weights/custom-parser") -> None:
        self.weights_dir = Path(weights_dir)

    def is_available(self) -> bool:
        if not self.weights_dir.is_dir():
            return False
        return any(self.weights_dir.glob("*.pt")) or any(self.weights_dir.glob("*.pth"))

    def predict(self, image: np.ndarray, context: dict | None = None) -> np.ndarray | None:
        self.require_available()
        raise NotImplementedError(
            "CustomHumanParserProvider es un esqueleto: falta entrenar el parser propio "
            "(etapa 3 del plan comercial) y rellenar predict()."
        )


__all__ = ["CustomHumanParserProvider"]
