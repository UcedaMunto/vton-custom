"""Bucle de entrenamiento LoRA (*rectified flow*) para FASHN VTON v1.5.

Pieza central del arnés de entrenamiento (`plan_entrenamiento/PLAN_FINE_TUNING.md`):

- Pérdida derivada del muestreo real del modelo (`TryOnPipeline._sample`):
  `x_t = (1−t)·ruido + t·limpio`, objetivo `v = limpio − ruido`, y el modelo debe
  predecir `v` con `MSE`. El tiempo se muestrea con el **mismo** `time_shift`
  (`get_rf_schedule`) que la inferencia, para que el entrenamiento vea la
  distribución de `t` que luego se usará al generar.
- *Conditional dropout* por muestra (`mask`), que es lo que hace útil el CFG en
  la inferencia (`forward_for_cfg`).
- `gradient_checkpointing` propio sobre los bloques del MMDiT: sin él las
  activaciones a 576×864 no caben en 12 GB (cálculo del plan §2).
- Reanudable, con checkpoints periódicos del adaptador (loRA) y, al final, un
  `model.safetensors` **mergeado** que `TryOnPipeline` carga sin saber que hubo
  un LoRA — es lo que consume `scripts/baseline.py verify`.

El módulo no decide política de datos: la puerta NC está en
:mod:`fashn_vton.train.nc_policy` y la aplica `scripts/fine_tune.py`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..utils import load_checkpoint, setup_logger, time_shift
from .data import TryOnTrainDataset, collate_train_batch
from .lora import (
    CORE_TARGETS,
    inject_lora,
    load_lora,
    lora_delta_norm,
    lora_parameters,
    save_lora,
    save_merged_checkpoint,
    write_adapter_card,
)

#: Nombre de consumidor del turno de GPU (mismo lock que IDM-CUSTOM / la UI).
GPU_CONSUMER_NAME = "fashn_fine_tune"

#: Contenedores de bloques del MMDiT sobre los que se aplica checkpointing.
CHECKPOINTED_CONTAINERS = ("x_patch_mixer", "double_blocks", "single_blocks")


@dataclass
class TrainConfig:
    """Configuración de una corrida de fine-tuning (por defecto, la receta del plan)."""

    pairs_csv: str
    output_dir: str = "weights/candidates/lora-exp"
    weights_dir: str = "weights"
    rank: int = 16
    alpha: float | None = None
    targets: tuple[str, ...] = CORE_TARGETS
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 1
    grad_accum: int = 4
    max_steps: int = 1000
    save_every: int = 100
    log_every: int = 10
    max_grad_norm: float = 1.0
    cond_drop_prob: float = 0.1
    time_shift_mu: float = 1.5
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    resolution: tuple[int, int] = (576, 864)
    seed: int = 42
    limit: int | None = None
    num_workers: int = 0
    resume: bool = False
    merge_every: int = 0
    merge_at_end: bool = True
    device: str = "auto"
    ca_mode: str = "mask"
    pose_cache_dir: str | None = None
    augment_flip: bool = False
    save_optimizer: bool = True
    dataset_root: str | None = None
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["targets"] = list(self.targets)
        payload["resolution"] = list(self.resolution)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainConfig":
        data = dict(payload)
        if isinstance(data.get("targets"), list):
            data["targets"] = tuple(data["targets"])
        if isinstance(data.get("resolution"), list):
            data["resolution"] = tuple(data["resolution"])
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path


def resolve_device(device: str = "auto") -> torch.device:
    """`auto` -> cuda si hay GPU disponible, si no cpu."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_amp_dtype(mixed_precision: str, device: torch.device) -> torch.dtype | None:
    """Precisión de autocast (`None` = fp32 puro)."""
    mode = (mixed_precision or "fp32").lower()
    if device.type != "cuda" or mode in {"fp32", "none", "no"}:
        return None
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 no soportado por esta GPU; usa --dtype fp16")
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    raise ValueError(f"mixed_precision='{mixed_precision}' no soportado (bf16|fp16|fp32)")


def enable_gradient_checkpointing(model: nn.Module, containers: Iterable[str] = CHECKPOINTED_CONTAINERS) -> int:
    """Envuelve el `forward` de cada bloque para no guardar activaciones.

    No se usa `torch.utils.checkpoint` sobre el modelo entero porque el
    `forward` de `TryOnModel` recibe condicionamientos grandes y `mask`; se
    aplica bloque a bloque, que es donde está el coste de memoria.
    """
    names = set(containers)
    wrapped = 0
    for name, container in model.named_modules():
        if name.split(".")[-1] not in names or not isinstance(container, nn.ModuleList):
            continue
        for block in container:
            if getattr(block, "_fashn_checkpointed", False):
                continue

            def checkpointed_forward(*args, _original=block.forward, **kwargs):
                if not torch.is_grad_enabled():
                    return _original(*args, **kwargs)
                return torch.utils.checkpoint.checkpoint(_original, *args, use_reentrant=False, **kwargs)

            block.forward = checkpointed_forward
            block._fashn_checkpointed = True
            wrapped += 1
    if wrapped == 0:
        raise RuntimeError("No se encontró ningún bloque que envolver; ¿cambió el MMDiT?")
    return wrapped


def load_base_model(
    weights_dir: str | Path = "weights",
    device: str | torch.device = "auto",
    dtype: torch.dtype | None = None,
    model_factory=None,
) -> nn.Module:
    """Carga el `TryOnModel` **original** (el estado comercial) listo para inyectar LoRA.

    `model_factory` permite inyectar otra arquitectura (lo usan las pruebas, que
    trabajan con un MMDiT diminuto en vez de los 972 M de parámetros).
    """
    from ..tryon_mmdit import TryOnModel

    resolved = resolve_device(device) if isinstance(device, str) else device
    weights_path = Path(weights_dir) / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"No encuentro los pesos base en '{weights_path}'. Descárgalos con "
            "`python scripts/download_weights.py --weights-dir ./weights`."
        )
    model = model_factory() if model_factory is not None else TryOnModel()
    state = load_checkpoint(str(weights_path), device="cpu")
    model.load_state_dict(state)
    model.to(resolved, dtype=dtype or torch.float32)
    return model


def sample_times(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    mu: float = 1.5,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Muestrea `t ~ time_shift(U(0,1))`, el mismo schedule que usa la inferencia."""
    uniform = torch.rand(batch_size, device=device, dtype=torch.float32).clamp(eps, 1.0 - 1e-4)
    return time_shift(mu, 1.0, uniform).to(dtype)


def flow_batch(clean: torch.Tensor, noise: torch.Tensor, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Devuelve `(x_t, v)` con `x_t = (1−t)·ruido + t·limpio` y `v = limpio − ruido`."""
    view = times.view(-1, 1, 1, 1).to(clean.dtype)
    x_t = (1.0 - view) * noise + view * clean
    return x_t, clean - noise


def conditional_mask(batch_size: int, drop_prob: float, device: torch.device) -> torch.Tensor:
    """Máscara de *conditional dropout* (`True` = conservar el condicionamiento)."""
    if drop_prob <= 0:
        return torch.ones(batch_size, device=device, dtype=torch.bool)
    return torch.rand(batch_size, device=device) >= float(drop_prob)


def model_kwargs(batch: dict, mask: torch.Tensor | None = None) -> dict:
    """Argumentos del `TryOnModel.forward` a partir de un batch (menos `x` y `times`)."""
    kwargs = {
        "ca_images": batch["ca_images"],
        "garment_images": batch["garment_images"],
        "person_poses": batch["person_poses"],
        "garment_poses": batch["garment_poses"],
        "garment_categories": batch["garment_categories"],
    }
    if mask is not None:
        kwargs["mask"] = mask
    return kwargs


def flow_loss(model: nn.Module, batch: dict, config: TrainConfig, generator: torch.Generator | None = None) -> tuple:
    """Forward + MSE del objetivo de *rectified flow*. Devuelve `(loss, x_t, v)`."""
    clean = batch["x1"]
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    times = sample_times(clean.shape[0], clean.device, clean.dtype, mu=config.time_shift_mu)
    x_t, velocity = flow_batch(clean, noise, times)
    mask = conditional_mask(clean.shape[0], config.cond_drop_prob, clean.device)
    prediction = model(x_t, times, **model_kwargs(batch, mask))["x"]
    loss = F.mse_loss(prediction.float(), velocity.float())
    return loss, x_t, velocity


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _vram_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated() / (1024**3), 3)


class Trainer:
    """Orquesta una corrida: dataset -> modelo base congelado + LoRA -> bucle -> candidato.

    Todo lo que escribe cae en `config.output_dir` (un directorio de candidato),
    **nunca** en `weights/model.safetensors`: el modelo comercial de partida no
    se toca durante el entrenamiento (`scripts/model_registry.py` es el único
    que promueve, y solo con verificación previa).
    """

    def __init__(
        self,
        config: TrainConfig,
        *,
        pose_fn=None,
        model_factory=None,
        logger=None,
    ):
        self.config = config
        self.logger = logger or setup_logger("trainer", level=logging.INFO)
        self.pose_fn = pose_fn
        self.model_factory = model_factory
        self.device = resolve_device(config.device)
        self.amp_dtype = resolve_amp_dtype(config.mixed_precision, self.device)
        self.compute_dtype = self.amp_dtype or torch.float32
        self.step = 0
        self.micro_step = 0
        self.epoch = 0
        self._iterator = None
        self._seconds: list[float] = []
        self._losses: list[float] = []
        self.out_dir = Path(config.output_dir)
        self.adapter_dir = self.out_dir / "lora"
        self.log_path = self.out_dir / "train_log.jsonl"
        self.report = None  # se rellena en setup()
        self.checkpointed_blocks = 0
        self.dataset = None
        self.loader = None
        self.model = None
        self.optimizer = None
        self.scaler = None

    # ------------------------------------------------------------------ construcción
    def setup(self) -> "Trainer":
        config = self.config
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
            torch.cuda.reset_peak_memory_stats()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        pose_device = f"cuda:{self.device.index or 0}" if self.device.type == "cuda" else "cpu"
        self.dataset = TryOnTrainDataset(
            pairs_csv=config.pairs_csv,
            weights_dir=config.weights_dir,
            resolution=tuple(config.resolution),
            device=pose_device,
            pose_fn=self.pose_fn,
            pose_cache_dir=config.pose_cache_dir,
            ca_mode=config.ca_mode,
            augment_flip=config.augment_flip,
            limit=config.limit,
            logger=self.logger,
        )
        generator = torch.Generator()
        generator.manual_seed(config.seed)
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_train_batch,
            drop_last=False,
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )

        self.logger.info("Cargando pesos base (congelados) de %s …", config.weights_dir)
        self.model = load_base_model(
            config.weights_dir, device=self.device, dtype=self.compute_dtype, model_factory=self.model_factory
        )
        self.report = inject_lora(self.model, targets=config.targets, rank=config.rank, alpha=config.alpha)
        if config.gradient_checkpointing:
            self.checkpointed_blocks = enable_gradient_checkpointing(self.model)
        self.model.train()

        self.optimizer = torch.optim.AdamW(
            lora_parameters(self.model),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if self.amp_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler("cuda")

        self._write_json(
            self.out_dir / "train_config.json",
            {
                **config.to_dict(),
                "lora": self.report.to_dict(),
                "device": str(self.device),
                "amp_dtype": str(self.amp_dtype),
                "gradient_checkpointed_blocks": self.checkpointed_blocks,
                "dataset_pairs": len(self.dataset),
                "started_at": _now(),
            },
        )
        if config.provenance:
            from .nc_policy import write_provenance

            write_provenance(self.out_dir, config.provenance)
        describe = (
            f"LoRA r={self.report.rank} alpha={self.report.alpha} en {self.report.replaced} módulos "
            f"({self.report.trainable_parameters:,} params entrenables de {self.report.total_parameters:,})"
        )
        self.logger.info(describe)
        self.logger.info(
            "Dataset: %s pares · resolución %sx%s · batch %s · acumulación %s · dtype %s",
            len(self.dataset),
            config.resolution[0],
            config.resolution[1],
            config.batch_size,
            config.grad_accum,
            self.compute_dtype,
        )
        return self

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ bucle
    def _autocast(self):
        if self.amp_dtype is None:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _to_device(self, batch: dict) -> dict:
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    moved[key] = value.to(self.device, dtype=self.compute_dtype, non_blocking=True)
                else:
                    moved[key] = value.to(self.device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def _next_batch(self) -> dict:
        while True:
            if self._iterator is None:
                self._iterator = iter(self.loader)
            try:
                return next(self._iterator)
            except StopIteration:
                self._iterator = None
                self.epoch += 1

    def _log(self, payload: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _log_progress(self, mean_loss: float, grad_norm) -> None:
        window_size = max(1, self.config.log_every)
        recent_losses = self._losses[-window_size:]
        recent_seconds = self._seconds[-window_size:]
        seconds_per_step = sum(recent_seconds) / len(recent_seconds)
        payload = {
            "at": _now(),
            "step": self.step,
            "epoch": self.epoch,
            "loss": round(mean_loss, 5),
            "loss_avg": round(sum(recent_losses) / len(recent_losses), 5),
            "loss_min": round(min(self._losses), 5),
            "grad_norm": round(float(grad_norm), 4),
            "delta_norm": round(lora_delta_norm(self.model), 5),
            "seconds_per_step": round(seconds_per_step, 3),
            "eta_minutes": round((self.config.max_steps - self.step) * seconds_per_step / 60.0, 2),
            "peak_vram_gb": _vram_gb(self.device),
        }
        self._log(payload)
        vram = f" · VRAM pico {payload['peak_vram_gb']} GiB" if payload["peak_vram_gb"] else ""
        self.logger.info(
            "paso %s/%s · loss %.4f (media %.4f) · |grad| %.3f · |delta| %.4f · %.2f s/paso · ETA %.1f min%s",
            payload["step"],
            self.config.max_steps,
            payload["loss"],
            payload["loss_avg"],
            payload["grad_norm"],
            payload["delta_norm"],
            payload["seconds_per_step"],
            payload["eta_minutes"],
            vram,
        )

    def run(self) -> dict:
        """Ejecuta `config.max_steps` pasos de optimizador y devuelve un resumen."""
        config = self.config
        started = time.time()
        self._resume_if_needed()
        steps_at_start = self.step
        first_loss = None
        accumulator = max(1, config.grad_accum)
        try:
            while self.step < config.max_steps:
                step_started = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for _ in range(accumulator):
                    batch = self._to_device(self._next_batch())
                    with self._autocast():
                        loss, _, _ = flow_loss(self.model, batch, config)
                    if self.scaler is not None:
                        self.scaler.scale(loss / accumulator).backward()
                    else:
                        (loss / accumulator).backward()
                    loss_sum += float(loss.detach().item())
                    self.micro_step += 1

                params = lora_parameters(self.model)
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.step += 1
                mean_loss = loss_sum / accumulator
                if first_loss is None:
                    first_loss = mean_loss
                self._losses.append(mean_loss)
                self._seconds.append(time.time() - step_started)

                if self.step % max(1, config.log_every) == 0 or self.step == 1:
                    self._log_progress(mean_loss, grad_norm)
                if config.save_every and self.step % config.save_every == 0:
                    self.save_checkpoint(tag=f"step-{self.step:06d}")
                if config.merge_every and self.step % config.merge_every == 0:
                    self.export_merged(tag=f"step-{self.step:06d}")
        except KeyboardInterrupt:
            self.logger.warning("Interrupción manual en el paso %s: se guarda el estado actual.", self.step)
        return self._finalize(started, first_loss, steps_at_start)

    def _finalize(self, started: float, first_loss, steps_at_start: int = 0) -> dict:
        config = self.config
        self.save_checkpoint()
        merged = self.export_merged() if config.merge_at_end else None
        seconds = time.time() - started
        steps_done = max(1, self.step - steps_at_start)
        summary = {
            "at": _now(),
            "steps": self.step,
            "steps_this_run": steps_done,
            "micro_steps": self.micro_step,
            "epochs_seen": self.epoch,
            "loss_first": first_loss,
            "loss_last": self._losses[-1] if self._losses else None,
            "loss_min": min(self._losses) if self._losses else None,
            "seconds": round(seconds, 1),
            "seconds_per_step": round(seconds / steps_done, 3),
            "peak_vram_gb": _vram_gb(self.device),
            "delta_norm": round(lora_delta_norm(self.model), 6) if self.model is not None else None,
            "lora": self.report.to_dict() if self.report else None,
            "adapter": str(self.adapter_dir / "adapter.safetensors"),
            "merged_checkpoint": str(merged) if merged else None,
            "output_dir": str(self.out_dir),
            "provenance": config.provenance,
        }
        self._write_json(self.out_dir / "summary.json", summary)
        if merged:
            self.logger.info("Checkpoint mergeado: %s", merged)
        self.logger.info(
            "Fin: %s pasos en %.1f min (%.2f s/paso). Resumen: %s",
            self.step,
            seconds / 60.0,
            seconds / steps_done,
            self.out_dir / "summary.json",
        )
        return summary

    # ------------------------------------------------------------------ salidas
    def save_checkpoint(self, tag: str | None = None) -> Path:
        """Guarda el adaptador y el estado de la corrida (para `--resume`)."""
        metadata = {
            "step": self.step,
            "rank": self.report.rank if self.report else "",
            "alpha": self.report.alpha if self.report else "",
            "pairs_csv": self.config.pairs_csv,
            "base_weights": str(Path(self.config.weights_dir) / "model.safetensors"),
        }
        path = save_lora(self.model, self.adapter_dir / "adapter.safetensors", metadata=metadata)
        if tag:
            save_lora(self.model, self.adapter_dir / f"adapter-{tag}.safetensors", metadata=metadata)
        if self.config.save_optimizer and self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), self.out_dir / "optimizer.pt")
        self._write_json(
            self.out_dir / "state.json",
            {
                "last_step": self.step,
                "micro_step": self.micro_step,
                "epoch": self.epoch,
                "saved_at": _now(),
                "adapter": str(path),
                "loss_last": self._losses[-1] if self._losses else None,
            },
        )
        return path

    def _link_dwpose(self) -> None:
        """Enlaza `dwpose/` para que el candidato sirva como `--weights-dir` del pipeline."""
        source = (Path(self.config.weights_dir) / "dwpose").resolve()
        link = self.out_dir / "dwpose"
        if not source.is_dir() or link.exists():
            return
        try:
            link.symlink_to(source)
        except OSError as exc:  # pragma: no cover - depende del sistema de ficheros
            self.logger.warning("No pude enlazar dwpose en el candidato (%s).", exc)

    def export_merged(self, tag: str | None = None) -> Path:
        """Escribe un `model.safetensors` estándar (base + LoRA) y deja el candidato usable."""
        config = self.config
        if tag:
            target = self.out_dir / "merged" / f"model-{tag}.safetensors"
        else:
            target = self.out_dir / "model.safetensors"
        metadata = {
            "step": self.step,
            "pairs_csv": config.pairs_csv,
            "license_class": str(config.provenance.get("license_class", "desconocida")),
            "commercial_use": str(config.provenance.get("commercial_use", "")),
        }
        save_merged_checkpoint(self.model, target, metadata=metadata)
        self._link_dwpose()
        if tag is None and self.report is not None:
            write_adapter_card(
                self.out_dir / "adapter_card.json",
                report=self.report,
                base_weights=str(Path(config.weights_dir) / "model.safetensors"),
                base_sha256=config.provenance.get("base_sha256"),
                extra={
                    "step": self.step,
                    "loss_last": self._losses[-1] if self._losses else None,
                    "delta_norm": round(lora_delta_norm(self.model), 6),
                    "provenance": config.provenance,
                },
            )
        return target

    def _resume_if_needed(self) -> None:
        if not self.config.resume:
            return
        state_path = self.out_dir / "state.json"
        if not state_path.is_file():
            self.logger.info("--resume sin estado previo en %s: se empieza de cero.", self.out_dir)
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        adapter = Path(state.get("adapter") or (self.adapter_dir / "adapter.safetensors"))
        if adapter.is_file():
            loaded = load_lora(self.model, adapter)
            self.logger.info(
                "Reanudando desde el paso %s (adaptador cargado en %s módulos)", state.get("last_step"), loaded
            )
        optimizer_path = self.out_dir / "optimizer.pt"
        if self.config.save_optimizer and optimizer_path.is_file():
            try:
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            except (RuntimeError, ValueError) as exc:
                self.logger.warning("No pude restaurar el optimizador (%s); sigo con uno nuevo.", exc)
        self.step = int(state.get("last_step", 0))
        self.micro_step = int(state.get("micro_step", 0))


def run_training(config: TrainConfig, *, pose_fn=None, model_factory=None, logger=None) -> dict:
    """Atajo: construye el `Trainer`, lo prepara y ejecuta la corrida completa."""
    trainer = Trainer(config, pose_fn=pose_fn, model_factory=model_factory, logger=logger).setup()
    return trainer.run()


__all__ = [
    "CHECKPOINTED_CONTAINERS",
    "GPU_CONSUMER_NAME",
    "TrainConfig",
    "Trainer",
    "conditional_mask",
    "enable_gradient_checkpointing",
    "flow_batch",
    "flow_loss",
    "load_base_model",
    "model_kwargs",
    "resolve_amp_dtype",
    "resolve_device",
    "run_training",
    "sample_times",
]






