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
    yield None, status()

    # Turno único de GPU (opcional, ver docstring del módulo).
    use_lock = _env_flag("FASHN_USE_GPU_LOCK", False)
    acquired = False
    if use_lock:
        log("Solicitando turno de GPU (infra.gpu_lock)…")
        yield None, status()
        acquired, message = _gpu_lock("acquire")
        if not acquired:
            log(f"GPU ocupada: {message}")
            yield None, status()
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
            yield None, status()
            raise gr.Error(f"No se pudieron cargar los modelos: {exc}") from exc

        yield None, status()

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
                    yield None, status()
            error = future.exception()

        if error is not None:
            log(f"ERROR durante la generación: {type(error).__name__}: {error}")
            yield None, status()
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

        yield images, status()
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

    with gr.Blocks(title="FASHN VTON v1.5 — pruebas locales") as demo:
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

        gallery = gr.Gallery(label="Resultado", columns=4, height=420, object_fit="contain", preview=True)
        status_box = gr.Textbox(label="Progreso y log de la petición", lines=14, max_lines=24, autoscroll=True)

        gr.Markdown(
            "**Notas**: la salida es siempre 576×864 (forma de entrada del modelo). "
            "Camino comercial: `flat-lay` + `segmentation_free` activado + proveedor `none` "
            "(sin dependencias de licencia no comercial). El pipeline no es thread-safe: las "
            "peticiones se atienden de una en una. Si necesitas ceder la GPU a la Pista A o a "
            "IDM-CUSTOM, exporta `FASHN_USE_GPU_LOCK=1` antes de lanzar la UI."
        )

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
        ]
        outputs = [gallery, status_box]

        generate_btn.click(
            fn=generate,
            inputs=inputs,
            outputs=outputs,
            api_name="tryon",
            concurrency_limit=1,
        )

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
