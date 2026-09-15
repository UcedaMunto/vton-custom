"""Segmentation provider interface for the commercial fork.

The upstream pipeline hard-depended on ``fashn-human-parser`` (non-commercial
weights). This package replaces that coupling with an explicit, pluggable
provider so the commercial path never needs a non-permissive dependency:

- ``NoSegmentationProvider``  -> nothing to load; the commercial MVP path
  (``segmentation_free=True`` + ``flat-lay`` garments).
- ``PoseHeuristicProvider``   -> license-clean heuristic regions from DWPose
  (experimental, no semantic parsing).
- ``Sam2SegmentationProvider``, ``GroundedSam2Provider``,
  ``CustomHumanParserProvider`` -> documented stubs for stages 2 and 3; they
  fail with an actionable message until their weights exist.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class SegmentationProviderError(RuntimeError):
    """Base error for segmentation providers."""


class ProviderNotAvailable(SegmentationProviderError):
    """The provider cannot run (missing package, weights or configuration)."""


class InvalidClassMapError(SegmentationProviderError):
    """A provider returned something that is not a FASHN class map."""


@dataclass(frozen=True)
class ProviderInfo:
    """Machine-readable description of a provider (used by API/UI/gate)."""

    name: str
    license: str
    requires_weights: bool
    commercial_ok: bool = True
    experimental: bool = False
    weights_dir: str | None = None
    notes: str = ""


class SegmentationProvider(ABC):
    """Maps a person/garment image to a FASHN class map.

    ``predict`` returns an ``HxW`` ``uint8`` array whose values are the class
    ids of :mod:`fashn_vton.segmentation.labels`, or ``None`` when the provider
    has no mask to offer for that image (the pipeline then degrades with an
    explicit warning instead of failing).
    """

    info: ProviderInfo

    @property
    def name(self) -> str:
        return self.info.name

    def is_available(self) -> bool:
        """True when the provider can actually run in this environment."""
        return True

    @abstractmethod
    def predict(self, image: np.ndarray, context: dict | None = None) -> np.ndarray | None:
        """Return a class map for ``image`` (RGB ``uint8`` ``HxWx3``) or None.

        ``context`` is an optional hint dictionary provided by the pipeline so a
        provider can reuse work already done: ``{"pose": <DWPose output dict>,
        "category": "tops"|"bottoms"|"one-pieces", "role": "person"|"garment"}``.
        Providers must ignore unknown keys and work (or return ``None``) without
        the hint.
        """

    # --- interface named in the commercial plan (section 11) ----------------
    def extract_person_region(self, image: np.ndarray, category: str, context: dict | None = None) -> np.ndarray | None:
        """Region masks for the person image (clothing-agnostic creation)."""
        return self.predict(image, {"category": category, "role": "person", **(context or {})})

    def extract_garment(self, image: np.ndarray, category: str, context: dict | None = None) -> np.ndarray | None:
        """Region masks for the garment image (garment isolation)."""
        return self.predict(image, {"category": category, "role": "garment", **(context or {})})

    # --- helpers -----------------------------------------------------------
    def describe(self) -> str:
        info = self.info
        flags = []
        if info.experimental:
            flags.append("experimental")
        if not info.commercial_ok:
            flags.append("NO COMERCIAL")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        return f"{info.name} (licencia: {info.license}){suffix}"

    def require_available(self) -> None:
        """Raise :class:`ProviderNotAvailable` with guidance when unusable."""
        if not self.is_available():
            raise ProviderNotAvailable(
                f"El proveedor de segmentación '{self.info.name}' no está disponible.\n{self.info.notes}".rstrip()
            )


def validate_class_map(class_map: np.ndarray | None, provider: SegmentationProvider) -> np.ndarray | None:
    """Validate shape/dtype/range of a provider output (None passes through)."""
    if class_map is None:
        return None
    from .labels import NUM_CLASSES

    array = np.asarray(class_map)
    if array.ndim != 2:
        raise InvalidClassMapError(
            f"'{provider.info.name}' devolvió un mapa de clases con {array.ndim} dimensiones (se esperaba HxW)."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise InvalidClassMapError(f"'{provider.info.name}' devolvió un mapa de clases vacío.")
    minimum, maximum = int(array.min()), int(array.max())
    if minimum < 0 or maximum >= NUM_CLASSES:
        raise InvalidClassMapError(
            f"'{provider.info.name}' devolvió ids de clase fuera de rango [0, {NUM_CLASSES - 1}]: "
            f"min={minimum}, max={maximum}."
        )
    return array.astype(np.uint8, copy=False)


__all__ = [
    "InvalidClassMapError",
    "ProviderInfo",
    "ProviderNotAvailable",
    "SegmentationProvider",
    "SegmentationProviderError",
    "validate_class_map",
]
