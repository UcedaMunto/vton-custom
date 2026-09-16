#!/usr/bin/env python3
"""Interfaz web local (Gradio) para probar FASHN VTON v1.5.

Envuelve la API pública del repositorio (`fashn_vton.TryOnPipeline`) sin
modificar su código. Ver `PLAN_INTERFAZ_WEB.md` para el diseño completo.

Decisiones de esta UI:
- **Carga perezosa y cacheada** del pipeline: la primera generación carga los
  modelos (~20 s, ~2 GB de VRAM en bf16) y las siguientes los reutilizan.
- **Peticiones serializadas** (`concurrency_limit=1`): el pipeline no es
  thread-safe y comparte la VRAM de la tarjeta.
- **Progreso por paso real**: se envuelve `tqdm` *dentro del módulo del
  pipeline* con una subclase que notifica, y se restaura siempre en `finally`.
  Si ese hook falla, la UI sigue avisando con tiempo transcurrido.
- **Turno de GPU opcional** (`FASHN_USE_GPU_LOCK=1`): usa el turno único de GPU
  del propio fork (`fashn_vton.utils.gpu_lock`), que comparte formato y ruta
  (`~/.idm_gpu.lock`) con la Pista A de IDM-VTON y con IDM-CUSTOM.

Variables de entorno:
    GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_SHARE
    FASHN_WEIGHTS_DIR, FASHN_DEVICE, FASHN_OUTPUT_DIR, FASHN_PRELOAD
    FASHN_USE_GPU_LOCK, FASHN_GPU_LOCK_PATH
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WEIGHTS_DIR = ROOT / "weights"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "webui"
EXAMPLES_DIR = ROOT / "examples" / "data"

# Puerto distinto de las otras UIs de esta máquina: 7860 = demo de IDM-VTON,
# 7861 = servicio de IDM-CUSTOM, 7863 = esta.
DEFAULT_PORT = 7863

CATEGORIES = ["tops", "bottoms", "one-pieces"]
PHOTO_TYPES = ["model", "flat-lay"]
DEVICE_CHOICES = ["auto", "cuda", "cpu"]

GPU_LOCK_CONSUMER = "fashn_webui"

#: CSS del visor grande: un overlay a pantalla completa que se muestra/oculta con
#: `visible` (Gradio 6.27 no trae componente Modal propio).
MODAL_CSS = """
.vton-modal {
    position: fixed; inset: 0; z-index: 3000;
    background: rgba(8, 8, 10, 0.93);
    padding: 1.5rem; overflow: auto;
    border: none;
}
.vton-modal .vton-image { max-height: 72vh; }
.preview-hint p { font-size: 0.72rem; opacity: 0.65; margin: 0.15rem 0 0 0; }
"""

logger = logging.getLogger("fashn_webui")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def weights_dir() -> Path:
    return Path(os.environ.get("FASHN_WEIGHTS_DIR", str(DEFAULT_WEIGHTS_DIR))).resolve()


def output_dir() -> Path:
    return Path(os.environ.get("FASHN_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))).resolve()


def memory_snapshot() -> str:
    """RAM del proceso y VRAM asignada/reservada, best-effort (nunca lanza)."""
    parts = []
    try:
        import psutil

        parts.append(f"RAM={psutil.Process().memory_info().rss / 1024**3:.2f}GiB")
    except Exception:  # noqa: BLE001 - informativo
        pass
    try:
        import torch

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            parts.append(f"VRAM={alloc:.2f}/{reserved:.2f}GiB")
    except Exception:  # noqa: BLE001
        pass
    return " ".join(parts) if parts else "memoria: n/d"


def _missing_weights(directory: Path) -> list[str]:
    """Archivos de pesos obligatorios (mismos que valida el propio pipeline)."""
    required = [
        directory / "model.safetensors",
        directory / "dwpose" / "yolox_l.onnx",
        directory / "dwpose" / "dw-ll_ucoco_384.onnx",
    ]
    return [str(path) for path in required if not path.exists()]


_PIPELINE_LOCK = threading.Lock()
#: Caché de un solo pipeline: al cambiar de proveedor/device se libera el anterior
#: para no acumular copias del modelo en la VRAM (12 GB compartidos).
_PIPELINE_CACHE: dict = {"key": None, "pipeline": None}


def get_pipeline(device: str = "auto", provider_name: str = "none", log=None):
    """Devuelve el `TryOnPipeline` cacheado para (weights_dir, device, provider).

    `device` es `auto` | `cuda` | `cpu`; `provider_name` es el proveedor de
    segmentación (`none` = camino comercial sin máscaras).
    """
    from fashn_vton import TryOnPipeline  # import diferido: tarda unos segundos
    from fashn_vton.segmentation import build_segmentation_provider

    directory = weights_dir()
    missing = _missing_weights(directory)
    if missing:
        raise RuntimeError(
            "Faltan pesos en "
            f"{directory}:\n  - " + "\n  - ".join(missing) + "\n\nDescárgalos con:\n"
            "  ./run_fashn_vton.sh python scripts/download_weights.py --weights-dir ./weights"
        )

    key = (str(directory), device, provider_name)
    with _PIPELINE_LOCK:
        if _PIPELINE_CACHE["key"] != key:
            if _PIPELINE_CACHE["pipeline"] is not None:
                if log:
                    log("Cambió el proveedor/device: liberando el pipeline anterior (VRAM)…")
                _PIPELINE_CACHE["pipeline"] = None
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001 - informativo
                    pass

            started = time.time()
            provider = build_segmentation_provider(provider_name)
            if log:
                log(f"Cargando modelos desde {directory} (device={device}, proveedor={provider.name})…")
                log(f"  {provider.describe()}")
                if provider.info.experimental:
                    log("  AVISO: proveedor experimental, calidad no validada.")
            pipeline = TryOnPipeline(
                weights_dir=str(directory),
                device=None if device == "auto" else device,
                segmentation_provider=provider,
            )
            _PIPELINE_CACHE["key"] = key
            _PIPELINE_CACHE["pipeline"] = pipeline
            if log:
                log(
                    f"Modelos cargados en {time.time() - started:.1f} s "
                    f"(device={pipeline.device}, dtype={pipeline.inference_dtype}) "
                    f"[{memory_snapshot()}]"
                )
        return _PIPELINE_CACHE["pipeline"]


def _install_progress_hook(counter: dict):
    """Envuelve `tqdm` dentro de `fashn_vton.pipeline` para contar pasos reales.

    Devuelve el objeto original para restaurarlo, o `None` si no se pudo
    (en ese caso la UI sigue funcionando, solo sin número de paso).
    """
    try:
        from tqdm.auto import tqdm as real_tqdm

        import fashn_vton.pipeline as pipeline_module

        class _ProgressTqdm(real_tqdm):
            def update(self, n=1):  # type: ignore[override]
                super().update(n)
                try:
                    counter["step"] = int(self.n)
                    counter["total"] = int(self.total or 0)
                except Exception:  # noqa: BLE001 - informativo
                    pass

        original = pipeline_module.tqdm
        pipeline_module.tqdm = _ProgressTqdm
        return original
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo instalar el hook de progreso: %s", exc)
        return None


def _restore_progress_hook(original) -> None:
    if original is None:
        return
    try:
        import fashn_vton.pipeline as pipeline_module

        pipeline_module.tqdm = original
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo restaurar el hook de progreso: %s", exc)


def _gpu_lock(action: str) -> tuple[bool, str]:
    """Reclama/libera el turno único de GPU del fork (``utils.gpu_lock``).

    Solo se usa si ``FASHN_USE_GPU_LOCK=1``. El lock es el mismo archivo
    (``~/.idm_gpu.lock`` por defecto) que usan la Pista A de IDM-VTON y el
    servicio de IDM-CUSTOM, así que el turno es compartido entre proyectos.
    """
    from fashn_vton.utils.gpu_lock import GpuBusyError, acquire, release, status

    lock_path = os.environ.get("FASHN_GPU_LOCK_PATH") or None
    kwargs = {"lock_path": lock_path} if lock_path else {}
    try:
        if action == "acquire":
            acquire(GPU_LOCK_CONSUMER, **kwargs)
            return True, f"turno concedido a {GPU_LOCK_CONSUMER}"
        if action == "release":
            released = release(GPU_LOCK_CONSUMER, **kwargs)
            return released, "liberado" if released else "no lo tenía"
        info = status(**kwargs)
        return True, f"{info.consumer} (pid={info.pid})" if info else "libre"
    except GpuBusyError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001 - informativo
        return False, f"{type(exc).__name__}: {exc}"


def _rgb(value, default: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int]:
    """Color de Gradio (``#rrggbb``, ``rgb(...)`` o tupla) → ``(r, g, b)`` entero."""
    if value is None:
        return default
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return tuple(int(channel) for channel in value[:3])
    text = str(value).strip().lstrip("#")
    if text.lower().startswith("rgb"):
        digits = [part for part in text[text.find("(") + 1 : text.find(")")].split(",") if part.strip()]
        if len(digits) >= 3:
            return tuple(max(0, min(255, int(float(part)))) for part in digits[:3])
    if len(text) == 6:
        return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))
    if len(text) == 3:
        return tuple(int(digit * 2, 16) for digit in text)
    return default


def build_fabric_config(
    enabled,
    fabric_color,
    background,
    background_color,
    mask_source,
    tolerance,
    front_only,
    scale,
    rotation,
    tilt_x,
    tilt_y,
    offset_x,
    offset_y,
    perspective,
    fabric_angle,
    brightness,
    strength,
    shading,
    detail,
    mosaic,
    repeat_mode="cm",
    repeat_cm=12.0,
    garment_width_cm=50.0,
    repeat_px=120.0,
):
    """Traduce los controles de la UI a :class:`FabricTransferConfig`."""
    from fashn_vton.preprocessing.fabric import FabricTransferConfig

    return FabricTransferConfig(
        enabled=bool(enabled),
        repeat_mode=str(repeat_mode),
        repeat_cm=float(repeat_cm),
        garment_width_cm=float(garment_width_cm),
        repeat_px=float(repeat_px),
        scale=float(scale),
        rotation=float(rotation),
        tilt_x=float(tilt_x),
        tilt_y=float(tilt_y),
        offset_x=float(offset_x),
        offset_y=float(offset_y),
        perspective=float(perspective),
        fabric_angle=float(fabric_angle),
        brightness=float(brightness),
        strength=float(strength),
        shading=float(shading),
        detail=float(detail),
        background_tolerance=float(tolerance),
        front_only=float(front_only),
        background=str(background),
        background_color=_rgb(background_color),
        mosaic=str(mosaic),
        mask_source=str(mask_source),
        flat_color=_rgb(fabric_color),
    )


def _draw_scale_grid(image, garment_width_cm: float, step_cm: float = 10.0):
    """Cuadrícula de referencia (cada ``step_cm`` cm) para juzgar el tamaño real.

    Solo se usa en la **previsualización**: la imagen que entra al try-on nunca
    lleva la cuadrícula.
    """
    from PIL import ImageDraw

    if garment_width_cm <= 0:
        return image
    width, height = image.size
    step = max(6, int(round(width / float(garment_width_cm) * step_cm)))
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    for x in range(0, width, step):
        draw.line([(x, 0), (x, height)], fill=(0, 0, 0, 80), width=1)
    for y in range(0, height, step):
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, 80), width=1)
    return canvas


def apply_fabric_to_garment(garment, fabric_image, config, show_grid: bool = False) -> tuple[object, str]:
    """Aplica la tela a la prenda; devuelve ``(imagen nueva, log)``.

    Es una transformación de **entrada**: la imagen resultante es la que entra al
    pipeline. No toca el modelo ni el camino comercial (`enabled=False` ⇒ intacto).
    """
    import numpy as np
    from PIL import Image

    from fashn_vton.preprocessing.fabric import retexture_garment

    fabric_array = None
    if fabric_image is not None:
        if hasattr(fabric_image, "convert"):
            fabric_array = np.asarray(fabric_image.convert("RGB"))
        else:
            fabric_array = np.asarray(fabric_image)[..., :3]

    result, _mask, info = retexture_garment(np.asarray(garment.convert("RGB")), fabric_array, config)
    if not info.applied and config.enabled and (fabric_array is not None or config.flat_color is not None):
        # Si la máscara automática no encontró prenda (fondo con textura, recorte
        # raro…), se reintenta aplicando la tela a toda la imagen en vez de no hacer
        # nada: es lo que el usuario espera al pulsar «previsualizar».
        from dataclasses import replace

        fallback, _fallback_mask, fallback_info = retexture_garment(
            np.asarray(garment.convert("RGB")),
            fabric_array,
            replace(config, mask_source="tela-completa"),
        )
        if fallback_info.applied:
            result, info = fallback, fallback_info
            info.notes.append("máscara automática vacía: tela aplicada a toda la imagen («tela-completa»)")

    image = Image.fromarray(result)
    lines = [info.describe(), info.describe_repeat()]
    if info.background_uniformity is not None:
        lines.append(f"  fondo: uniformidad={info.background_uniformity} (bajo = liso)")
    for note in info.notes:
        lines.append(f"  aviso: {note}")
    if show_grid and info.applied:
        image = _draw_scale_grid(image, config.garment_width_cm, step_cm=10.0)
        lines.append("  cuadrícula de 10 cm solo en la previsualización (no va al try-on)")
    return image, "\n".join(lines)


def preview_fabric(
    garment_image,
    fabric_image,
    fabric_enabled,
    fabric_color,
    fabric_bg_mode,
    fabric_bg_color,
    fabric_mask_source,
    fabric_tolerance,
    fabric_front_only,
    fabric_scale,
    fabric_rotation,
    fabric_tilt_x,
    fabric_tilt_y,
    fabric_offset_x,
    fabric_offset_y,
    fabric_perspective,
    fabric_angle,
    fabric_brightness,
    fabric_strength,
    fabric_shading,
    fabric_detail,
    fabric_mosaic,
    fabric_repeat_mode="cm",
    fabric_repeat_cm=12.0,
    fabric_garment_width_cm=50.0,
    fabric_repeat_px=120.0,
    fabric_show_grid=False,
):
    """Botón «Previsualizar prenda»: aplica la tela sin lanzar el try-on."""
    import gradio as gr

    if garment_image is None:
        raise gr.Error("Sube una imagen de prenda para previsualizar la tela.")
    if fabric_image is None and _rgb(fabric_color) == (255, 255, 255):
        return garment_image, (
            "Sube una **tela** o elige un **color plano** (distinto del blanco) para ver la previsualización."
        )

    config = build_fabric_config(
        fabric_enabled,
        fabric_color,
        fabric_bg_mode,
        fabric_bg_color,
        fabric_mask_source,
        fabric_tolerance,
        fabric_front_only,
        fabric_scale,
        fabric_rotation,
        fabric_tilt_x,
        fabric_tilt_y,
        fabric_offset_x,
        fabric_offset_y,
        fabric_perspective,
        fabric_angle,
        fabric_brightness,
        fabric_strength,
        fabric_shading,
        fabric_detail,
        fabric_mosaic,
        fabric_repeat_mode,
        fabric_repeat_cm,
        fabric_garment_width_cm,
        fabric_repeat_px,
    )
    # El botón de previsualización aplica SIEMPRE la tela: para eso se pulsa.
    # (El interruptor «Aplicar tela a la prenda» decide si se usa al generar.)
    config.enabled = True
    log_line = "vista previa (no se usa al generar)" if not fabric_enabled else ""

    try:
        result, log_text = apply_fabric_to_garment(
            garment_image, fabric_image, config, show_grid=bool(fabric_show_grid)
        )
    except Exception as exc:  # noqa: BLE001 - se muestra en la UI
        raise gr.Error(f"No se pudo aplicar la tela: {type(exc).__name__}: {exc}") from exc
    if log_line:
        log_text = f"{log_text}\n  ({log_line})"
    return result, log_text


def _save_images(images, seed: int, params_log: str) -> list[Path]:
    """Guarda cada muestra como PNG en `outputs/webui/` (gitignored)."""
    directory = output_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []
    for index, image in enumerate(images):
        path = directory / f"{stamp}_s{seed}_{index:02d}.png"
        image.save(path)
        paths.append(path)
    (directory / f"{stamp}_s{seed}_params.txt").write_text(params_log, encoding="utf-8")
    return paths


def _run_inference(pipeline, person, garment, kwargs, counter):
    """Ejecuta el pipeline en un hilo, con el hook de progreso instalado.

    Devuelve `(resultado, segundos)`. No escribe en el log de la UI para no
    romper la línea de progreso (que se actualiza en el hilo principal).
    """
    original = _install_progress_hook(counter)
    started = time.time()
    try:
        result = pipeline(person_image=person, garment_image=garment, **kwargs)
    finally:
        _restore_progress_hook(original)
    return result, time.time() - started


def generate(
    person_image,
    garment_image,
    category: str,
    photo_type: str,
    num_samples: int,
    num_timesteps: int,
    guidance_scale: float,
    seed: int,
    segmentation_free: bool,
    provider_name: str,
    device_choice: str,
    # --- tela propia (opcional; por defecto desactivada) ---
    fabric_image=None,
    fabric_enabled: bool = False,
    fabric_color="#ffffff",
    fabric_bg_mode: str = "original",
    fabric_bg_color="#ffffff",
    fabric_mask_source: str = "auto",
    fabric_tolerance: float = 30.0,
    fabric_front_only: float = 1.0,
    fabric_scale: float = 1.0,
    fabric_rotation: float = 0.0,
    fabric_tilt_x: float = 0.0,
    fabric_tilt_y: float = 0.0,
    fabric_offset_x: float = 0.0,
    fabric_offset_y: float = 0.0,
    fabric_perspective: float = 0.0,
    fabric_angle: float = 0.0,
    fabric_brightness: float = 1.0,
    fabric_strength: float = 1.0,
    fabric_shading: float = 1.0,
    fabric_detail: float = 0.0,
    fabric_mosaic: str = "repetir",
    fabric_repeat_mode: str = "cm",
    fabric_repeat_cm: float = 12.0,
    fabric_garment_width_cm: float = 50.0,
    fabric_repeat_px: float = 120.0,
):
    """Manejador del botón «Generar try-on» (generador de Gradio).

    Va emitiendo (`imagen(es)`, `log`) mientras el pipeline trabaja en un hilo,
    así la UI muestra el paso real, el tiempo y el uso de memoria.
    """
    import gradio as gr

    lines: list[str] = []
    progress_line: dict = {"index": None}

    def log(message: str) -> None:
        lines.append(message)
        progress_line["index"] = None
        logger.info("%s", message)

    def log_progress(message: str) -> None:
        """Escribe el avance en una sola línea que se reemplaza en cada tick."""
        if progress_line["index"] is None:
            lines.append(message)
            progress_line["index"] = len(lines) - 1
        else:
            lines[progress_line["index"]] = message
        logger.info("%s", message)

    def status() -> str:
        return "\n".join(f"{i + 1}. {m}" for i, m in enumerate(lines))

    if person_image is None or garment_image is None:
        raise gr.Error("Sube una imagen de persona y una imagen de prenda.")

    num_samples = int(num_samples)
    num_timesteps = int(num_timesteps)
    seed = int(seed)

    log(
        f"Petición: categoría={category} · prenda={photo_type} · muestras={num_samples} · "
        f"pasos={num_timesteps} · guidance={guidance_scale} · seed={seed} · "
        f"segmentation_free={segmentation_free} · proveedor={provider_name} · device={device_choice}"
    )
    yield None, status(), None

    # Turno único de GPU (opcional, ver docstring del módulo).
    use_lock = _env_flag("FASHN_USE_GPU_LOCK", False)
    acquired = False
    if use_lock:
        log("Solicitando turno de GPU (infra.gpu_lock)…")
        yield None, status(), None
        acquired, message = _gpu_lock("acquire")
        if not acquired:
            log(f"GPU ocupada: {message}")
            yield None, status(), None
            raise gr.Error(
                "La GPU está en uso por otro consumidor (Pista A o IDM-CUSTOM). "
                "Espera a que termine o desactiva FASHN_USE_GPU_LOCK. "
                f"Detalle: {message}"
            )
        log("Turno de GPU concedido.")

    try:
        try:
            pipeline = get_pipeline(device_choice, provider_name, log=log)
        except Exception as exc:  # noqa: BLE001 - se muestra en la UI
            log(f"ERROR al cargar los modelos: {type(exc).__name__}: {exc}")
            yield None, status(), None
            raise gr.Error(f"No se pudieron cargar los modelos: {exc}") from exc

        yield None, status(), None

        # Tela propia (opcional): se aplica a la prenda ANTES de la inferencia.
        fabric_config = build_fabric_config(
            fabric_enabled,
            fabric_color,
            fabric_bg_mode,
            fabric_bg_color,
            fabric_mask_source,
            fabric_tolerance,
            fabric_front_only,
            fabric_scale,
            fabric_rotation,
            fabric_tilt_x,
            fabric_tilt_y,
            fabric_offset_x,
            fabric_offset_y,
            fabric_perspective,
            fabric_angle,
            fabric_brightness,
            fabric_strength,
            fabric_shading,
            fabric_detail,
            fabric_mosaic,
            fabric_repeat_mode,
            fabric_repeat_cm,
            fabric_garment_width_cm,
            fabric_repeat_px,
        )
        if fabric_config.enabled:
            try:
                garment_image, fabric_log = apply_fabric_to_garment(garment_image, fabric_image, fabric_config)
            except Exception as exc:  # noqa: BLE001 - se muestra en la UI
                log(f"ERROR al aplicar la tela: {type(exc).__name__}: {exc}")
                yield None, status(), None
                raise gr.Error(f"No se pudo aplicar la tela: {exc}") from exc
            for line in fabric_log.splitlines():
                log(line)
            if photo_type != "flat-lay":
                log("Nota: con la tela aplicada conviene «Tipo de foto de prenda = flat-lay» (el fondo se recompone).")
            yield None, status(), None

        counter: dict = {"step": 0, "total": num_timesteps}
        kwargs = dict(
            category=category,
            garment_photo_type=photo_type,
            num_samples=num_samples,
            num_timesteps=num_timesteps,
            guidance_scale=float(guidance_scale),
            seed=seed,
            segmentation_free=bool(segmentation_free),
        )

        started = time.time()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_inference, pipeline, person_image, garment_image, kwargs, counter)
            last_report = ""
            while not future.done():
                time.sleep(0.4)
                elapsed = time.time() - started
                step = counter.get("step", 0)
                total = counter.get("total") or num_timesteps
                report = f"Generando… paso {step}/{total} · {elapsed:.0f} s · {memory_snapshot()}"
                if report != last_report:
                    last_report = report
                    log_progress(report)
                    yield None, status(), None
            error = future.exception()

        if error is not None:
            log(f"ERROR durante la generación: {type(error).__name__}: {error}")
            yield None, status(), None
            raise gr.Error(f"{type(error).__name__}: {error}")

        result, inference_time = future.result()
        images = list(result.images)
        total_time = time.time() - started
        log(f"Inferencia terminada en {inference_time:.1f} s")
        log(
            f"{len(images)} imagen(es) generada(s) en {total_time:.1f} s "
            f"({total_time / max(num_timesteps, 1):.2f} s/paso) · {memory_snapshot()}"
        )

        # Transparencia comercial: qué proveedor se usó y si hubo degradación.
        segmentation = (result.metadata or {}).get("segmentation", {})
        if segmentation:
            log(
                "Segmentación: "
                f"proveedor={segmentation.get('provider')} · "
                f"licencia={segmentation.get('license')} · "
                f"máscara persona={'sí' if segmentation.get('person_mask') else 'no'} · "
                f"máscara prenda={'sí' if segmentation.get('garment_mask') else 'no'}"
            )
            for note in segmentation.get("degraded", []) or []:
                log(f"AVISO (degradado, resultado no óptimo): {note}")

        paths = _save_images(images, seed, status())
        for path in paths:
            log(f"Guardado: {path}")

        yield images, status(), images[0] if images else None
    finally:
        if acquired:
            ok, message = _gpu_lock("release")
            log(f"Turno de GPU liberado={'sí' if ok else 'NO'} ({message or 'ok'})")


EXAMPLES = {
    "persona": EXAMPLES_DIR / "model.webp",
    "prenda": EXAMPLES_DIR / "garment.webp",
}


def build_app():
    """Construye la interfaz (Gradio Blocks)."""
    import gradio as gr

    from fashn_vton.segmentation import available_providers

    example_person = EXAMPLES["persona"] if EXAMPLES["persona"].exists() else None
    example_garment = EXAMPLES["prenda"] if EXAMPLES["prenda"].exists() else None

    providers = available_providers()
    provider_choices = sorted(providers)
    provider_notes = "**Proveedores** (todos con licencia comercial):\n" + "\n".join(
        f"- `{name}` — {info.get('license', 'n/d')}"
        + (" · **experimental**" if info.get("experimental") else "")
        + ("" if info.get("available", False) else " · no disponible en este entorno")
        for name, info in sorted(providers.items())
    )

    with gr.Blocks(title="FASHN VTON v1.5 — pruebas locales", css=MODAL_CSS) as demo:
        gr.Markdown(
            "# FASHN VTON v1.5 — interfaz de pruebas\n"
            "Sube una **foto de persona** y una **prenda**, elige la categoría y genera.\n"
            f"Datos: pesos en `{weights_dir()}` · salidas en `{output_dir()}` · "
            f"primer intento carga los modelos (~20 s), los siguientes reutilizan el pipeline."
        )

        with gr.Row():
            person_input = gr.Image(
                label="Persona (imagen completa)",
                type="pil",
                sources=["upload", "clipboard"],
                height=380,
                value=str(example_person) if example_person else None,
            )
            garment_input = gr.Image(
                label="Prenda (foto de modelo o producto)",
                type="pil",
                sources=["upload", "clipboard"],
                height=380,
                value=str(example_garment) if example_garment else None,
            )
            with gr.Column(scale=2, min_width=200):
                result_preview = gr.Image(
                    label="Resultado final",
                    type="pil",
                    height=380,
                    interactive=False,
                    buttons=["download", "fullscreen"],
                )
                show_result_modal = gr.Button("Ver en grande (modal)", size="sm")
                gr.Markdown(f"Se guarda en `{output_dir()}`", elem_classes=["preview-hint"])

        with gr.Row():
            load_examples = gr.Button("Cargar ejemplos del repo", variant="secondary")
            generate_btn = gr.Button("Generar try-on", variant="primary")

        with gr.Accordion("Parámetros de generación", open=True):
            with gr.Row():
                category = gr.Radio(CATEGORIES, value="tops", label="Categoría")
                photo_type = gr.Radio(
                    PHOTO_TYPES,
                    value="flat-lay",
                    label="Tipo de foto de prenda",
                    info="flat-lay = producto (camino comercial) · model = prenda puesta (requiere proveedor)",
                )
                device_choice = gr.Dropdown(
                    DEVICE_CHOICES, value=os.environ.get("FASHN_DEVICE", "auto"), label="Device"
                )
            with gr.Row():
                num_timesteps = gr.Slider(
                    8,
                    50,
                    value=30,
                    step=1,
                    label="Pasos de difusión",
                    info="20 = rápido · 30 = equilibrado · 50 = máxima calidad",
                )
                guidance_scale = gr.Slider(0.5, 5.0, value=1.5, step=0.1, label="Guidance scale")
                num_samples = gr.Slider(1, 4, value=1, step=1, label="Muestras")
                seed = gr.Number(value=42, precision=0, label="Semilla")
            with gr.Row():
                provider_dropdown = gr.Dropdown(
                    provider_choices,
                    value=os.environ.get("FASHN_PROVIDER", "none"),
                    label="Proveedor de segmentación",
                    info="none = camino comercial (sin máscaras) · pose-heuristic = experimental (DWPose)",
                )
                segmentation_free = gr.Checkbox(
                    value=True,
                    label="Modo sin segmentación (segmentation_free)",
                    info="Activado (recomendado y comercial); desactívalo solo con un proveedor que aporte máscaras",
                )
            gr.Markdown(provider_notes)

        from fashn_vton.preprocessing.fabric import BACKGROUND_MODES, MOSAIC_MODES

        with gr.Accordion("Tela propia: color, textura y ángulo (opcional)", open=False):
            gr.Markdown(
                "Sustituye el **color y el estampado** de la prenda con la foto de una tela. "
                "Se asume que la prenda está sobre fondo **blanco o gris** (foto de producto) y el fondo "
                "del resultado es configurable. Al activarlo, usa «Tipo de foto de prenda = **flat-lay**». "
                "Con la tela desactivada el resultado es exactamente el de antes (mismo sha256)."
            )
            with gr.Row():
                fabric_enabled = gr.Checkbox(
                    value=False,
                    label="Aplicar tela a la prenda",
                    info="Desactivado = comportamiento actual, sin cambios",
                )
                fabric_image = gr.Image(
                    label="Tela (foto del estampado)",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=200,
                )
                fabric_color = gr.ColorPicker(
                    value="#ffffff",
                    label="Color plano (si no subes tela)",
                    info="Sustituye solo el color, sin textura",
                )
            with gr.Row():
                fabric_preview_btn = gr.Button("Previsualizar prenda con la tela", variant="secondary")
                fabric_preview = gr.Image(
                    label="Prenda con la tela aplicada (previsualización)",
                    type="pil",
                    height=260,
                    interactive=False,
                )
                fabric_preview_log = gr.Textbox(label="Log de la tela", lines=3, max_lines=6, autoscroll=True)
            with gr.Row():
                fabric_repeat_mode = gr.Radio(
                    [
                        ("centímetros reales", "cm"),
                        ("píxeles de la imagen", "px"),
                        ("relativo (escala)", "scale"),
                    ],
                    value="cm",
                    label="Cómo se mide el motivo",
                    info="cm = medida real de la tela · px = píxeles de la prenda subida",
                )
                fabric_repeat_cm = gr.Slider(
                    2,
                    60,
                    value=12,
                    step=0.5,
                    label="Tamaño del motivo (cm)",
                    info="Cuánto mide una repetición del estampado en la realidad",
                )
                fabric_garment_width_cm = gr.Slider(
                    20,
                    120,
                    value=50,
                    step=1,
                    label="Ancho real de la prenda (cm)",
                    info="Sirve para calibrar: 50 cm de prenda con motivo de 12 cm ⇒ ~4 repeticiones",
                )
            with gr.Row():
                fabric_repeat_px = gr.Slider(
                    20,
                    400,
                    value=120,
                    step=5,
                    label="Tamaño del motivo (px)",
                    info="Solo en modo «píxeles»",
                )
                fabric_scale = gr.Slider(
                    0.2,
                    4.0,
                    value=1.0,
                    step=0.05,
                    label="Escala relativa",
                    info="Solo en modo «relativo»",
                )
                fabric_rotation = gr.Slider(-180, 180, value=0, step=1, label="Ángulo del estampado (°)")
                fabric_angle = gr.Slider(-180, 180, value=0, step=1, label="Enderezar la foto de la tela (°)")
                fabric_show_grid = gr.Checkbox(
                    value=False,
                    label="Cuadrícula de 10 cm (solo previsualización)",
                    info="Se dibuja únicamente en la previsualización: la imagen que va al try-on nunca la lleva",
                )
            fabric_repeat_info = gr.Markdown("*Motivo y repeticiones se calculan con el ancho de la prenda que subas.*")
            with gr.Row():
                fabric_tilt_x = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Inclinación X (sesgo)")
                fabric_tilt_y = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Inclinación Y (sesgo)")
                fabric_perspective = gr.Slider(0.0, 0.8, value=0.0, step=0.05, label="Profundidad Z (perspectiva)")
            with gr.Row():
                fabric_offset_x = gr.Slider(-0.5, 0.5, value=0.0, step=0.01, label="Desplazamiento X")
                fabric_offset_y = gr.Slider(-0.5, 0.5, value=0.0, step=0.01, label="Desplazamiento Y")
                fabric_mosaic = gr.Radio(list(MOSAIC_MODES), value="repetir", label="Mosaico")
            with gr.Row():
                fabric_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Fuerza de la tela")
                fabric_shading = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="Conservar sombras de la prenda")
                fabric_detail = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Conservar detalles (costuras)")
                fabric_brightness = gr.Slider(0.5, 1.8, value=1.0, step=0.05, label="Brillo de la tela")
            with gr.Row():
                fabric_mask_source = gr.Radio(
                    ["auto", "tela-completa"],
                    value="auto",
                    label="Máscara de la prenda",
                    info="auto = separar del fondo claro · tela-completa = toda la imagen",
                )
                fabric_tolerance = gr.Slider(5, 120, value=30, step=1, label="Tolerancia de fondo")
                fabric_front_only = gr.Slider(
                    0.2,
                    1.0,
                    value=1.0,
                    step=0.05,
                    label="Solo panel frontal (banda central)",
                    info="1.0 = toda la prenda",
                )
            with gr.Row():
                fabric_bg_mode = gr.Radio(list(BACKGROUND_MODES), value="original", label="Fondo del resultado")
                fabric_bg_color = gr.ColorPicker(value="#ffffff", label="Color de fondo")

        gallery = gr.Gallery(label="Resultado", columns=4, height=420, object_fit="contain", preview=True)
        status_box = gr.Textbox(label="Progreso y log de la petición", lines=14, max_lines=24, autoscroll=True)

        # Visor grande (modal): overlay a pantalla completa, oculto por defecto.
        with gr.Column(elem_classes=["vton-modal"], visible=False) as result_modal:
            with gr.Row():
                gr.Markdown("### Resultado en grande")
                close_result_modal = gr.Button("✕ Cerrar", variant="secondary", scale=0)
            modal_image = gr.Image(
                type="pil",
                interactive=False,
                height=700,
                show_label=False,
                buttons=["download", "fullscreen"],
                elem_classes=["vton-image"],
            )

        gr.Markdown(
            "**Notas**: la salida es siempre 576×864 (forma de entrada del modelo). "
            "Camino comercial: `flat-lay` + `segmentation_free` activado + proveedor `none` "
            "(sin dependencias de licencia no comercial). El pipeline no es thread-safe: las "
            "peticiones se atienden de una en una. Si necesitas ceder la GPU a la Pista A o a "
            "IDM-CUSTOM, exporta `FASHN_USE_GPU_LOCK=1` antes de lanzar la UI."
        )

        fabric_inputs = [
            fabric_image,
            fabric_enabled,
            fabric_color,
            fabric_bg_mode,
            fabric_bg_color,
            fabric_mask_source,
            fabric_tolerance,
            fabric_front_only,
            fabric_scale,
            fabric_rotation,
            fabric_tilt_x,
            fabric_tilt_y,
            fabric_offset_x,
            fabric_offset_y,
            fabric_perspective,
            fabric_angle,
            fabric_brightness,
            fabric_strength,
            fabric_shading,
            fabric_detail,
            fabric_mosaic,
            fabric_repeat_mode,
            fabric_repeat_cm,
            fabric_garment_width_cm,
            fabric_repeat_px,
        ]
        inputs = [
            person_input,
            garment_input,
            category,
            photo_type,
            num_samples,
            num_timesteps,
            guidance_scale,
            seed,
            segmentation_free,
            provider_dropdown,
            device_choice,
            *fabric_inputs,
        ]
        outputs = [gallery, status_box, result_preview]

        def _repeat_info(garment_image, mode, repeat_cm, garment_width_cm, repeat_px, scale):
            """Texto en vivo: cuántos píxeles/cm mide el motivo y cuántas repeticiones hay."""
            from fashn_vton.preprocessing.fabric import FabricTransferConfig, repeat_reference_px

            if garment_image is None:
                return "*Sube la prenda para calcular el tamaño del motivo y las repeticiones.*"
            canvas_width = getattr(garment_image, "size", (576, 864))[0] or 576
            config = FabricTransferConfig(
                repeat_mode=mode,
                repeat_cm=float(repeat_cm),
                garment_width_cm=float(garment_width_cm),
                repeat_px=float(repeat_px),
                scale=float(scale),
            )
            reference = repeat_reference_px(config, canvas_width)
            cm_per_repeat = reference / canvas_width * float(garment_width_cm)
            return (
                f"**Motivo**: {reference:.0f} px de ancho (~{cm_per_repeat:.1f} cm sobre "
                f"{float(garment_width_cm):.0f} cm de prenda) · "
                f"**repeticiones a lo ancho**: {canvas_width / reference:.2f}"
            )

        repeat_controls = [
            fabric_repeat_mode,
            fabric_repeat_cm,
            fabric_garment_width_cm,
            fabric_repeat_px,
            fabric_scale,
        ]
        for control in [garment_input, *repeat_controls]:
            control.change(
                _repeat_info,
                inputs=[garment_input, *repeat_controls],
                outputs=[fabric_repeat_info],
                api_name=False,  # informativo de la UI: no hace falta exponerlo
            )

        generate_btn.click(
            fn=generate,
            inputs=inputs,
            outputs=outputs,
            api_name="tryon",
            concurrency_limit=1,
        )

        fabric_preview_btn.click(
            fn=preview_fabric,
            inputs=[garment_input, *fabric_inputs, fabric_show_grid],
            outputs=[fabric_preview, fabric_preview_log],
            api_name="preview_fabric",
        )

        def _open_result_modal(image):
            import gradio as gr  # noqa: F811 - import local por claridad

            return gr.update(visible=True), image

        def _close_result_modal():
            import gradio as gr  # noqa: F811 - import local por claridad

            return gr.update(visible=False)

        show_result_modal.click(
            fn=_open_result_modal,
            inputs=[result_preview],
            outputs=[result_modal, modal_image],
        )
        close_result_modal.click(fn=_close_result_modal, inputs=None, outputs=[result_modal])

        def _load_examples():
            import gradio as gr  # noqa: F811 - import local por claridad

            return (
                gr.update(value=str(example_person) if example_person else None),
                gr.update(value=str(example_garment) if example_garment else None),
            )

        load_examples.click(fn=_load_examples, inputs=None, outputs=[person_input, garment_input])

    return demo


def main() -> None:
    import gradio as gr

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Menos ruido de red en los logs (la UI es local).
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    gr_theme = gr.themes.Soft()

    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("GRADIO_SERVER_PORT", str(DEFAULT_PORT)))
    share = _env_flag("GRADIO_SHARE", False)

    logger.info("Pesos: %s", weights_dir())
    logger.info("Salidas: %s", output_dir())

    if _env_flag("FASHN_PRELOAD", False):
        try:
            get_pipeline(os.environ.get("FASHN_DEVICE", "auto"))
        except Exception as exc:  # noqa: BLE001 - la UI debe arrancar igual
            logger.warning("No se pudo precargar el pipeline: %s", exc)

    app = build_app()
    app.queue()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        allowed_paths=[str(output_dir())],
        # En Gradio 6 el tema se pasa en launch(), no en Blocks().
        theme=gr_theme,
    )


if __name__ == "__main__":
    main()
