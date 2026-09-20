"""Adaptadores LoRA mínimos para el TryOnModel (MMDiT) de FASHN VTON v1.5.

Pieza del arnés de fine-tuning (`plan_entrenamiento/PLAN_FINE_TUNING.md` §5):
inyecta adaptadores de bajo rango sobre las proyecciones de atención/MLP del
MMDiT **sin tocar el pipeline de inferencia**. Los módulos base quedan
congelados y el adaptador se puede mergear en los pesos para producir un
checkpoint estándar (`model.safetensors`), que es lo único que `TryOnPipeline`
necesita saber.

Decisión de diseño (2026-09-17): implementación propia, **sin `peft`**. Motivo:
el MMDiT de este fork no es un `UNet2DConditionModel` de `diffusers`, y añadir
una dependencia nueva obligaría a auditarla en `licenses/manifest.json`; el
principio del fork es minimizar superficie. Es LoRA clásico (Hu et al. 2021)
con `B = 0` en la inicialización, de modo que **el modelo arranca siendo
exactamente el actual** (la razón de ser del plan: partir del estado comercial
congelado y solo aceptar cambios que la guardia anti-regresión apruebe).

Invariante que fijan las pruebas (`tests/test_train_lora.py`):

- Con `rank > 0` recién inyectado, la salida del modelo es idéntica a la del
  modelo base (`delta = 0`).
- Solo los parámetros `lora_a` / `lora_b` reciben gradiente.
- `merge_lora` + `unmerge_lora` es exacto (ida y vuelta sin residuo).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
from safetensors.torch import load_file, save_file
from torch import nn

#: Proyecciones de atención (nombre de hoja). Es donde vive la atención
#: imagen<->prenda, que es lo que controla la transferencia de la prenda.
ATTENTION_TARGETS: tuple[str, ...] = ("img_attn.qkv", "img_attn.proj", "txt_attn.qkv", "txt_attn.proj")

#: Atención + MLP fusionado de los bloques `single` (y del patch mixer).
CORE_TARGETS: tuple[str, ...] = ("qkv", "proj", "linear1", "linear2")

#: Conjunto de hojas que se adaptan por defecto.
DEFAULT_TARGETS: tuple[str, ...] = CORE_TARGETS

LORA_A_SUFFIX = "lora_a"
LORA_B_SUFFIX = "lora_b"


class LoRALinear(nn.Module):
    """`nn.Linear` congelado + rama de bajo rango entrenable.

    `out = base(x) + scaling * B·A·x`, con `B` inicializado a cero (arranque
    idéntico al modelo base) y `A` con la inicialización de Kaiming para que el
    gradiente de `B` sea informativo desde el primer paso.

    La rama LoRA se calcula en la precisión de **sus** parámetros (fp32 por
    defecto), aunque el resto del modelo esté en bf16: con 7-8 M de parámetros
    el coste es despreciable y evita momentos de AdamW en bf16 (inestables).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear espera un nn.Linear, recibió {type(base).__name__}")
        if rank < 1:
            raise ValueError(f"El rango de LoRA debe ser >= 1 (recibido {rank})")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank if self.rank else 0.0
        # `merged` == True significa que delta ya está sumado en `base.weight`.
        self.merged = False

        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def delta_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """`scaling · B·A`, en la precisión pedida (por defecto fp32)."""
        weight = (self.lora_b @ self.lora_a) * self.scaling
        return weight if dtype is None else weight.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if self.merged or self.rank <= 0:
            return output
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            signal = x.to(self.lora_a.dtype)
            delta = nn.functional.linear(nn.functional.linear(signal, self.lora_a), self.lora_b) * self.scaling
        return output + delta.to(output.dtype)

    def merge(self) -> None:
        """Suma `delta` a los pesos base (idempotente)."""
        if self.merged:
            return
        self.base.weight.data += self.delta_weight(self.base.weight.dtype)
        self.merged = True

    def unmerge(self) -> None:
        """Resta `delta` de los pesos base (inverso exacto de `merge`)."""
        if not self.merged:
            return
        self.base.weight.data -= self.delta_weight(self.base.weight.dtype)
        self.merged = False

    def extra_repr(self) -> str:  # pragma: no cover - solo informativo
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, merged={self.merged}"
        )


@dataclass
class LoraReport:
    """Resumen de una inyección de LoRA (para el log y para las pruebas)."""

    rank: int
    alpha: float
    targets: tuple[str, ...]
    module_names: list[str] = field(default_factory=list)
    trainable_parameters: int = 0
    total_parameters: int = 0

    @property
    def replaced(self) -> int:
        return len(self.module_names)

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_parameters / self.total_parameters if self.total_parameters else 0.0

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "targets": list(self.targets),
            "replaced_modules": self.replaced,
            "module_names": list(self.module_names),
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_fraction": self.trainable_fraction,
        }


def _matches_target(name: str, targets: Sequence[str]) -> bool:
    """`name` es una hoja objetivo si coincide exactamente o como sufijo punteado."""
    return any(name == target or name.endswith(f".{target}") for target in targets)


def inject_lora(
    model: nn.Module,
    targets: Sequence[str] = DEFAULT_TARGETS,
    rank: int = 16,
    alpha: float | None = None,
) -> LoraReport:
    """Sustituye cada `nn.Linear` objetivo por un :class:`LoRALinear`.

    El modelo base no se modifica en su comportamiento: la rama nueva nace a
    cero. Requiere que `model` esté **ya** en su dispositivo/precisión final,
    porque los parámetros del adaptador usan los tensores base como referencia.
    """
    if isinstance(targets, str):  # comodidad: un solo objetivo
        targets = (targets,)
    targets = tuple(targets)
    alpha = float(rank if alpha is None else alpha)

    # Congela TODO el modelo base antes de crear los adaptadores: así el
    # optimizador solo ve el adaptador y no se reserva memoria para gradientes
    # de 972 M de parámetros (que en fp32 serían ~3,9 GB extra).
    for parameter in model.parameters():
        parameter.requires_grad = False

    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and _matches_target(child_name, targets):
                qualified = f"{name}.{child_name}" if name else child_name
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                replaced.append(qualified)
    if not replaced:
        raise RuntimeError(
            "No se encontró ningún nn.Linear objetivo; ¿cambió el nombre de los módulos del MMDiT? "
            f"Objetivos: {targets}"
        )

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return LoraReport(
        rank=int(rank),
        alpha=alpha,
        targets=targets,
        module_names=replaced,
        trainable_parameters=trainable,
        total_parameters=total,
    )


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    """Mapa `nombre_calificado -> LoRALinear` de los adaptadores presentes."""
    return {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parámetros entrenables del adaptador (lo único que optimiza AdamW)."""
    return [parameter for module in lora_modules(model).values() for parameter in module.parameters()]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Estado del adaptador en un `state_dict` plano y portable."""
    state: dict[str, torch.Tensor] = {}
    for name, module in lora_modules(model).items():
        state[f"{name}.{LORA_A_SUFFIX}"] = module.lora_a.detach().clone()
        state[f"{name}.{LORA_B_SUFFIX}"] = module.lora_b.detach().clone()
    return state


def save_lora(model: nn.Module, path: str | Path, metadata: dict | None = None) -> Path:
    """Guarda el adaptador (solo los deltas, ~30 MB con r=16) en safetensors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.contiguous() for key, value in lora_state_dict(model).items()}
    meta = {str(key): str(value) for key, value in (metadata or {}).items()}
    save_file(state, str(path), metadata=meta or None)
    return path


def load_lora(model: nn.Module, path: str | Path, strict: bool = True) -> int:
    """Carga un adaptador guardado por :func:`save_lora` sobre un modelo ya inyectado."""
    modules = lora_modules(model)
    if strict and not modules:
        raise KeyError(
            f"El modelo no tiene adaptadores LoRA inyectados: llama a inject_lora() (con el mismo "
            f"rango) antes de cargar '{path}'."
        )
    state = load_file(str(path))
    loaded = 0
    for name, module in modules.items():
        key_a, key_b = f"{name}.{LORA_A_SUFFIX}", f"{name}.{LORA_B_SUFFIX}"
        if key_a not in state or key_b not in state:
            if strict:
                raise KeyError(f"El adaptador de '{path}' no contiene '{key_a}'/'{key_b}'")
            continue
        module.lora_a.data.copy_(state[key_a].to(module.lora_a.dtype))
        module.lora_b.data.copy_(state[key_b].to(module.lora_b.dtype))
        loaded += 1
    if strict and loaded != len(modules):
        raise KeyError(
            f"Adaptador incompleto en '{path}': {loaded} módulos cargados de {len(modules)} en el modelo"
        )
    return loaded


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Activa/desactiva la rama LoRA sin borrarla (útil para A/B en el mismo proceso)."""
    for module in lora_modules(model).values():
        module.rank_scale_backup = getattr(module, "rank_scale_backup", module.scaling)
        module.scaling = float(module.rank_scale_backup) if enabled else 0.0


def merge_lora(model: nn.Module) -> None:
    """Incorpora los deltas a los pesos base (el modelo queda equivalente sin adaptador)."""
    for module in lora_modules(model).values():
        module.merge()


def unmerge_lora(model: nn.Module) -> None:
    """Deshace :func:`merge_lora` (vuelve exactamente a los pesos base)."""
    for module in lora_modules(model).values():
        module.unmerge()


def lora_delta_norm(model: nn.Module) -> float:
    """Norma L2 global del delta (`scaling·B·A`) — señal de deriva respecto al baseline."""
    total = 0.0
    for module in lora_modules(model).values():
        total += float(module.delta_weight().pow(2).sum().item())
    return math.sqrt(total)


def merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """`state_dict` del modelo con los deltas aplicados y con los **nombres originales**.

    Es lo que se guarda como `model.safetensors` del candidato: un checkpoint
    normal, cargable por `TryOnPipeline` sin saber que existió un LoRA. Por eso
    se quitan los parámetros del adaptador y se deshace el prefijo `.base.` que
    introduce `LoRALinear` (con `strict=True` en `load_state_dict`, cualquier
    clave de más sería un error).
    """
    lora_map = lora_modules(model)
    state: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key.endswith((f".{LORA_A_SUFFIX}", f".{LORA_B_SUFFIX}")):
            continue
        if ".base." not in key:
            state[key] = value.contiguous()
            continue
        module_name, _, attribute = key.partition(".base.")
        module = lora_map.get(module_name)
        if module is not None and attribute == "weight" and not module.merged:
            value = value + module.delta_weight(value.dtype)
        state[f"{module_name}.{attribute}"] = value.contiguous()
    return state



def save_merged_checkpoint(model: nn.Module, path: str | Path, metadata: dict | None = None) -> Path:
    """Guarda el modelo completo (base + LoRA mergeado) como `model.safetensors`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"format": "pt", **{str(key): str(value) for key, value in (metadata or {}).items()}}
    save_file(merged_state_dict(model), str(path), metadata=meta)
    return path


def write_adapter_card(
    path: str | Path,
    *,
    report: LoraReport,
    base_weights: str,
    base_sha256: str | None,
    extra: dict | None = None,
) -> Path:
    """Deja la ficha del adaptador junto al checkpoint (trazabilidad del candidato)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "lora",
        **report.to_dict(),
        "base_weights": str(base_weights),
        "base_sha256": base_sha256,
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def iter_lora_named_parameters(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    """Pares `(nombre, parámetro)` del adaptador, para logging selectivo."""
    for name, module in lora_modules(model).items():
        yield f"{name}.{LORA_A_SUFFIX}", module.lora_a
        yield f"{name}.{LORA_B_SUFFIX}", module.lora_b


__all__ = [
    "ATTENTION_TARGETS",
    "CORE_TARGETS",
    "DEFAULT_TARGETS",
    "LoRALinear",
    "LoraReport",
    "inject_lora",
    "iter_lora_named_parameters",
    "load_lora",
    "lora_delta_norm",
    "lora_modules",
    "lora_parameters",
    "lora_state_dict",
    "merge_lora",
    "merged_state_dict",
    "save_lora",
    "save_merged_checkpoint",
    "set_lora_enabled",
    "unmerge_lora",
    "write_adapter_card",
]


