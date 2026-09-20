"""Arnés de entrenamiento (fine-tuning LoRA) del fork comercial de FASHN VTON v1.5.

Este subpaquete **no** participa en la inferencia: el pipeline (`TryOnPipeline`)
sigue siendo el mismo y no importa nada de aquí. Vive aparte por dos motivos:

1. Cumplimiento: los datos de investigación con licencia no comercial
   (VITON-HD/DressCode) están prohibidos en el camino comercial
   (`licenses/manifest.json` → `forbidden_artifacts`). El carril de
   entrenamiento los admite de forma **explícita y trazable**
   (:mod:`fashn_vton.train.nc_policy`), produce candidatos fuera del modelo
   activo y bloquea su promoción.
2. Memoria: sin este módulo, cualquier usuario del pipeline arrastraría código
   de entrenamiento que no usa.

Mapa del arnés:

- :mod:`fashn_vton.train.lora` — adaptadores LoRA sobre el MMDiT (sin `peft`).
- :mod:`fashn_vton.train.data` — CSV de pares y dataset con el preprocesado
  exacto de la inferencia.
- :mod:`fashn_vton.train.trainer` — bucle *rectified flow*, checkpoints y
  exportación del candidato mergeado.
- :mod:`fashn_vton.train.nc_policy` — puertas de entrada (entrenar) y de salida
  (promover) para datos/pesos no comerciales.
- `scripts/prepare_nc_dataset.py` y `scripts/fine_tune.py` — CLI.
- `plan_entrenamiento/GUIA_EJECUCION_ENTRENAMIENTO.md` — guía paso a paso.

Los nombres se exponen de forma perezosa (`__getattr__`) para que importar
`fashn_vton.train` no arrastre `torch`/`onnxruntime` en contextos donde no hace
falta (p. ej. verificar la política NC desde un script ligero).
"""

from __future__ import annotations

_LAZY = {
    "PairRecord": (".data", "PairRecord"),
    "TryOnTrainDataset": (".data", "TryOnTrainDataset"),
    "load_pairs_csv": (".data", "load_pairs_csv"),
    "write_pairs_csv": (".data", "write_pairs_csv"),
    "LoRALinear": (".lora", "LoRALinear"),
    "LoraReport": (".lora", "LoraReport"),
    "inject_lora": (".lora", "inject_lora"),
    "TrainConfig": (".trainer", "TrainConfig"),
    "Trainer": (".trainer", "Trainer"),
    "run_training": (".trainer", "run_training"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    """Resuelve los nombres públicos bajo demanda (evita imports pesados)."""
    try:
        module_name, attribute = _LAZY[name]
    except KeyError as exc:  # pragma: no cover - comportamiento estándar de módulos
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    from importlib import import_module

    return getattr(import_module(module_name, __name__), attribute)


def __dir__() -> list[str]:
    return sorted(__all__)
