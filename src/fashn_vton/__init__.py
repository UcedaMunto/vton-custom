"""FASHN VTON v1.5 - fork comercial (Apache-2.0).

Sin dependencias de licencia no comercial: la segmentacion es un proveedor
enchufable (ver `fashn_vton.segmentation`). Ver `README_COMERCIAL.md`.

IMPORTANTE (2026-09-19): este paquete expone `TryOnPipeline`/`TryOnModel` de forma
**perezosa** (PEP 562). Antes, cualquier `import fashn_vton.<submodulo>` cargaba
torch a traves de este `__init__` (~300 MB de RSS): los procesos que no usan el
modelo (gateway, frontend, worker de preprocesado) morian con OOMKilled o
desperdiciaban memoria. Ver `kubernetes/02_ESTADO_DE_EJECUCION.md`.
"""
from __future__ import annotations

__version__ = "1.5.0+commercial.1"

__all__ = [
    "TryOnPipeline",
    "PipelineOutput",
    "TryOnModel",
    "__version__",
]


def __getattr__(name: str):
    """Carga el pipeline/modelo solo cuando se piden de verdad."""
    if name in {"TryOnPipeline", "PipelineOutput"}:
        from .pipeline import PipelineOutput, TryOnPipeline

        return {"TryOnPipeline": TryOnPipeline, "PipelineOutput": PipelineOutput}[name]
    if name == "TryOnModel":
        from .tryon_mmdit import TryOnModel

        return TryOnModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
