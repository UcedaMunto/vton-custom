"""HTTP service of the commercial fork (FastAPI).

Design notes
------------
- **Commercial by construction**: the service refuses a provider marked as non
  commercial, and ``/healthz`` reports the compliance status.
- **Single flight**: the diffusion pipeline is not thread-safe and shares 12 GB
  of VRAM, so requests are serialised with an ``asyncio.Lock`` and (optionally)
  the machine-wide GPU turnstile (``utils.gpu_lock``), the same lock file
  IDM-CUSTOM and the IDM-VTON watchdog use.
- **Auth**: ``X-API-Key`` when ``VTON_API_KEYS`` is configured. Without keys the
  service only accepts loopback clients (development) unless
  ``VTON_ALLOW_ANONYMOUS=1``.
- **Limits**: upload size, samples and diffusion steps are capped by env vars.

Run it with ``./run_api.sh`` (adds the CUDA library path for onnxruntime).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Annotated, Callable

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError

from .. import __version__
from ..compliance import commercial_report
from ..segmentation import DEFAULT_PROVIDER, PROVIDERS, build_segmentation_provider
from ..utils.gpu_lock import DEFAULT_LOCK_PATH, GpuBusyError, acquire, release

logger = logging.getLogger("fashn_vton.api")

CATEGORIES = ("tops", "bottoms", "one-pieces")
PHOTO_TYPES = ("model", "flat-lay")
CONSUMER = "fashn_vton_api"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime configuration (all from environment variables)."""

    weights_dir: str = field(default_factory=lambda: os.environ.get("FASHN_WEIGHTS_DIR", "weights"))
    device: str | None = field(default_factory=lambda: os.environ.get("FASHN_DEVICE") or None)
    default_provider: str = field(default_factory=lambda: os.environ.get("FASHN_PROVIDER", DEFAULT_PROVIDER))
    api_keys: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            key.strip() for key in os.environ.get("VTON_API_KEYS", "").split(",") if key.strip()
        )
    )
    allow_anonymous: bool = field(default_factory=lambda: _env_flag("VTON_ALLOW_ANONYMOUS", False))
    use_gpu_lock: bool = field(default_factory=lambda: _env_flag("VTON_USE_GPU_LOCK", False))
    gpu_lock_path: str = field(default_factory=lambda: os.environ.get("VTON_GPU_LOCK_PATH", str(DEFAULT_LOCK_PATH)))
    max_upload_mb: int = field(default_factory=lambda: int(os.environ.get("VTON_MAX_UPLOAD_MB", "12")))
    max_samples: int = field(default_factory=lambda: int(os.environ.get("VTON_MAX_SAMPLES", "4")))
    max_steps: int = field(default_factory=lambda: int(os.environ.get("VTON_MAX_STEPS", "50")))
    strict_segmentation: bool = field(default_factory=lambda: _env_flag("VTON_STRICT_SEGMENTATION", False))


def _load_image(raw: bytes, settings: Settings, role: str) -> Image.Image:
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"La imagen de {role} supera el máximo de {settings.max_upload_mb} MB.",
        )
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"Imagen de {role} inválida: {exc}") from exc
    if image.width < 64 or image.height < 64:
        raise HTTPException(status_code=422, detail=f"La imagen de {role} es demasiado pequeña (mínimo 64x64).")
    return image.convert("RGB")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application (also used by the tests)."""
    settings = settings or Settings()
    app = FastAPI(
        title="FASHN VTON commercial fork",
        version=__version__,
        description=(
            "Servicio de virtual try-on sin dependencias de licencia no comercial. "
            "Camino comercial: segmentation_free=true + garment_photo_type=flat-lay."
        ),
    )
    app.state.settings = settings
    app.state.pipeline_factory = lambda provider_name, device: _build_pipeline(settings, provider_name, device)
    app.state.lock = asyncio.Lock()

    def _client_is_loopback(request: Request) -> bool:
        host = request.client.host if request.client else ""
        return host in {"127.0.0.1", "::1", "localhost", "testclient"}

    def require_api_key(
        request: Request,
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> None:
        if settings.api_keys:
            if x_api_key is None or x_api_key not in settings.api_keys:
                raise HTTPException(status_code=401, detail="Falta una API key válida (X-API-Key).")
            return
        if settings.allow_anonymous or _client_is_loopback(request):
            return
        raise HTTPException(
            status_code=503,
            detail="El servicio no tiene API keys configuradas (VTON_API_KEYS). Petición rechazada.",
        )

    @app.get("/healthz")
    def healthz() -> dict:
        report = commercial_report(check_environment=True, check_dependencies=False)
        return {
            "status": "ok",
            "package_version": __version__,
            "commercial_ready": report.ok,
            "compliance_violations": report.violations,
            "default_provider": settings.default_provider,
            "providers": sorted(PROVIDERS),
            "device": settings.device or "auto",
            "gpu_lock": settings.use_gpu_lock,
        }

    @app.get("/v1/version")
    def version() -> dict:
        return {
            "package": "fashn-vton (fork comercial)",
            "package_version": __version__,
            "model": "fashn-ai/fashn-vton-1.5 (Apache-2.0)",
            "third_party_notices": "THIRD_PARTY_NOTICES.md",
            "manifest": "licenses/manifest.json",
            "providers": sorted(PROVIDERS),
        }

    @app.post("/v1/tryon", dependencies=[Depends(require_api_key)])
    async def tryon(
        request: Request,
        person: Annotated[UploadFile, File(description="Foto de la persona (imagen completa)")],
        garment: Annotated[UploadFile, File(description="Foto de la prenda")],
        category: Annotated[str, Form()] = "tops",
        garment_photo_type: Annotated[str, Form()] = "flat-lay",
        num_timesteps: Annotated[int, Form()] = 30,
        guidance_scale: Annotated[float, Form()] = 1.5,
        seed: Annotated[int, Form()] = 42,
        num_samples: Annotated[int, Form()] = 1,
        provider: Annotated[str | None, Form()] = None,
    ):
        _validate_request(category, garment_photo_type, num_samples, num_timesteps, provider, settings)
        provider_name = (provider or settings.default_provider).strip().lower()

        started = time.time()
        person_image = _load_image(await person.read(), settings, "persona")
        garment_image = _load_image(await garment.read(), settings, "prenda")

        async with app.state.lock:  # single flight: the pipeline shares the GPU
            try:
                result = await asyncio.to_thread(
                    _run,
                    app,
                    provider_name,
                    person_image,
                    garment_image,
                    dict(
                        category=category,
                        garment_photo_type=garment_photo_type,
                        num_samples=num_samples,
                        num_timesteps=num_timesteps,
                        guidance_scale=guidance_scale,
                        seed=seed,
                    ),
                )
            except GpuBusyError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001 - se devuelve como error JSON
                logger.exception("Fallo generando el try-on")
                raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

        return _build_response(result, provider_name, started, request)

    return app


def _validate_request(
    category: str,
    garment_photo_type: str,
    num_samples: int,
    num_timesteps: int,
    provider: str | None,
    settings: Settings,
) -> None:
    if category not in CATEGORIES:
        raise HTTPException(status_code=422, detail=f"category debe ser una de {list(CATEGORIES)}.")
    if garment_photo_type not in PHOTO_TYPES:
        raise HTTPException(status_code=422, detail=f"garment_photo_type debe ser uno de {list(PHOTO_TYPES)}.")
    if not 1 <= num_samples <= settings.max_samples:
        raise HTTPException(status_code=422, detail=f"num_samples debe estar entre 1 y {settings.max_samples}.")
    if not 1 <= num_timesteps <= settings.max_steps:
        raise HTTPException(status_code=422, detail=f"num_timesteps debe estar entre 1 y {settings.max_steps}.")
    name = (provider or settings.default_provider).strip().lower()
    if name not in PROVIDERS:
        raise HTTPException(status_code=422, detail=f"provider desconocido: {name!r}. Opciones: {sorted(PROVIDERS)}.")
    if not PROVIDERS[name]().info.commercial_ok:
        raise HTTPException(
            status_code=403,
            detail=f"El proveedor '{name}' no tiene licencia comercial: prohibido en el servicio.",
        )


def _to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_response(result, provider_name: str, started: float, request: Request):
    """PNG binario (1 muestra + ?raw=true) o JSON con base64 para varias muestras."""
    images = list(result.images)
    metadata = result.metadata or {}
    segmentation = metadata.get("segmentation", {})
    elapsed = round(time.time() - started, 2)

    if len(images) == 1 and request.query_params.get("raw") in {"1", "true"}:
        return Response(
            content=_to_png(images[0]),
            media_type="image/png",
            headers={
                "X-Elapsed-Seconds": f"{elapsed}",
                "X-Segmentation-Provider": str(segmentation.get("provider", provider_name)),
                "X-Segmentation-Degraded": str(bool(segmentation.get("degraded"))).lower(),
            },
        )

    payload_images = []
    for index, image in enumerate(images):
        png = _to_png(image)
        payload_images.append(
            {
                "index": index,
                "format": "png",
                "width": image.width,
                "height": image.height,
                "base64": base64.b64encode(png).decode("ascii"),
                "sha256": hashlib.sha256(png).hexdigest(),
            }
        )
    return JSONResponse(
        {
            "images": payload_images,
            "metadata": metadata,
            "elapsed_seconds": elapsed,
        }
    )


_PIPELINES: dict[tuple[str, str, str], object] = {}


def _build_pipeline(settings: Settings, provider_name: str, device: str | None):
    """Pipeline cacheado por (weights_dir, device, provider)."""
    from ..pipeline import TryOnPipeline

    key = (settings.weights_dir, str(device), provider_name)
    if key not in _PIPELINES:
        provider = build_segmentation_provider(provider_name)
        logger.info("Cargando pipeline %s (provider=%s)", settings.weights_dir, provider_name)
        _PIPELINES[key] = TryOnPipeline(
            weights_dir=settings.weights_dir,
            device=device,
            segmentation_provider=provider,
            strict_segmentation=settings.strict_segmentation,
        )
    return _PIPELINES[key]


def _run(app: FastAPI, provider_name: str, person: Image.Image, garment: Image.Image, kwargs: dict):
    """Inferencia bloqueante, opcionalmente bajo el turno único de GPU."""
    settings: Settings = app.state.settings
    factory: Callable = app.state.pipeline_factory
    pipeline = factory(provider_name, settings.device)

    if not settings.use_gpu_lock:
        return pipeline(person_image=person, garment_image=garment, **kwargs)

    acquire(CONSUMER, lock_path=settings.gpu_lock_path)
    try:
        return pipeline(person_image=person, garment_image=garment, **kwargs)
    finally:
        release(CONSUMER, lock_path=settings.gpu_lock_path)


app = create_app()
