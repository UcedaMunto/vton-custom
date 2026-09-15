"""FASHN VTON v1.5 — fork comercial (Apache-2.0).

Sin dependencias de licencia no comercial: la segmentación es un proveedor
enchufable (ver `fashn_vton.segmentation`). Ver `README_COMERCIAL.md`.
"""

__version__ = "1.5.0+commercial.1"

from .pipeline import PipelineOutput, TryOnPipeline
from .tryon_mmdit import TryOnModel

__all__ = [
    "TryOnPipeline",
    "PipelineOutput",
    "TryOnModel",
    "__version__",
]
