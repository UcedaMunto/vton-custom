"""Asynchronous API gateway (F6 of ``kubernetes/00_PLAN_ARQUITECTURA.md``).

The existing ``/v1/tryon`` endpoint of this fork is synchronous (~46 s for 30
steps), which cannot be published on a queue-based deployment. This gateway is
the async door in front of Redis Streams:

* ``POST /v1/jobs`` validates the upload, stores it in the **transient** bucket,
  creates the job in Redis and answers ``202`` with a ``job_id``.
* ``GET /v1/jobs/{id}`` returns the state, and a short-lived result URL when the
  job is done.
* ``GET /v1/jobs/{id}/result`` streams the PNG and then **deletes it** (plus the
  rest of the job objects): zero retention on delivery (plan section 11).

Admission control (plan section 7.1) is enforced here: upload size, diffusion
steps, maximum queue depth (``429``) and a per-key hourly quota, because with a
single RTX 3060 the ceiling is ~1.3 images/minute.

Run it with ``uvicorn fashn_vton.gateway:app``.
"""
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from .cluster import (
    KEY_GARMENT_RAW,
    KEY_PERSON_RAW,
    KEY_RESULT,
    STATUS_DONE,
    STATUS_ERROR,
    STREAM_PREPROCESS,
    bus,
    storage,
)


def _package_version() -> str:
    """Version del paquete SIN importar ``fashn_vton`` (cuyo __init__ carga torch).

    El gateway y el frontend no necesitan torch ni el pipeline: importar el
    paquete completo costaba ~300 MB de RSS (el frontend moria con OOMKilled con
    un limite de 256Mi, ver 02_ESTADO_DE_EJECUCION.md).
    """
    try:
        from importlib.metadata import version

        return version("fashn-vton")
    except Exception:  # noqa: BLE001 - el paquete puede no estar instalado como distribuible
        return "unknown"


__version__ = _package_version()

logger = logging.getLogger("fashn_vton.gateway")

CATEGORIES = ("tops", "bottoms", "one-pieces")
PHOTO_TYPES = ("model", "flat-lay")


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, all from environment variables (Kubernetes env)."""

    api_keys: tuple[str, ...] = tuple(k.strip() for k in os.environ.get("VTON_API_KEYS", "").split(",") if k.strip())
    allow_anonymous: bool = os.environ.get("VTON_ALLOW_ANONYMOUS", "0").strip().lower() in {"1", "true", "yes", "on"}
    max_upload_mb: int = int(os.environ.get("VTON_MAX_UPLOAD_MB", "12"))
    max_steps: int = int(os.environ.get("VTON_MAX_STEPS", "50"))
    queue_max_depth: int = int(os.environ.get("VTON_QUEUE_MAX_DEPTH", "20"))
    rate_limit_per_hour: int = int(os.environ.get("VTON_RATE_LIMIT_PER_HOUR", "30"))
    job_ttl_seconds: int = int(os.environ.get("VTON_JOB_TTL_SECONDS", "3600"))
    result_ttl_minutes: int = int(os.environ.get("VTON_RESULT_TTL_MINUTES", "30"))
    result_url_ttl_seconds: int = int(os.environ.get("VTON_RESULT_URL_TTL_SECONDS", "300"))

    @property
    def bucket(self) -> str:
        return storage.bucket_name()


settings = Settings()
app = FastAPI(title="vton-api-gateway", version=__version__, description="Puerta de entrada asincrona del servicio de try-on")

_redis = None
_s3 = None


def redis_client():
    global _redis
    if _redis is None:
        _redis = bus.connect()
    return _redis


def s3_client():
    global _s3
    if _s3 is None:
        _s3 = storage.s3_client()
    return _s3


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> str:
    """Same policy as the synchronous API: keys required, loopback allowed in dev."""
    if settings.api_keys:
        if x_api_key and x_api_key in settings.api_keys:
            return x_api_key
        raise HTTPException(status_code=401, detail="Falta la cabecera X-API-Key o la clave no es valida.")
    if settings.allow_anonymous:
        return "anonymous"
    raise HTTPException(status_code=503, detail="El servicio no tiene VTON_API_KEYS configurado (y no permite anonimos).")


async def _read(upload: UploadFile, role: str) -> bytes:
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=422, detail=f"La imagen de {role} esta vacia.")
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"La imagen de {role} supera el maximo de {settings.max_upload_mb} MB.")
    return raw


@app.get("/healthz")
def healthz() -> dict:
    """Liveness/readiness: also reports the queue depth used for admission."""
    payload = {
        "status": "ok",
        "component": "vton-api-gateway",
        "package_version": __version__,
        "queue_max_depth": settings.queue_max_depth,
        "api_keys_configured": bool(settings.api_keys),
    }
    try:
        payload["redis"] = bool(redis_client().ping())
        payload["queue_depth"] = bus.queue_depth(redis_client(), STREAM_PREPROCESS)
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "degraded"
        payload["redis"] = False
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


@app.post("/v1/jobs", status_code=202)
async def create_job(
    person: Annotated[UploadFile, File()],
    garment: Annotated[UploadFile, File()],
    category: Annotated[str, Form()] = "tops",
    garment_photo_type: Annotated[str, Form()] = "flat-lay",
    num_timesteps: Annotated[int, Form()] = 30,
    seed: Annotated[int, Form()] = 42,
    guidance_scale: Annotated[float, Form()] = 1.5,
    api_key: Annotated[str, Depends(require_api_key)] = "",
) -> JSONResponse:
    if category not in CATEGORIES:
        raise HTTPException(status_code=422, detail=f"category debe ser una de {CATEGORIES}.")
    if garment_photo_type not in PHOTO_TYPES:
        raise HTTPException(status_code=422, detail=f"garment_photo_type debe ser uno de {PHOTO_TYPES}.")
    if not 1 <= num_timesteps <= settings.max_steps:
        raise HTTPException(status_code=422, detail=f"num_timesteps debe estar entre 1 y {settings.max_steps}.")

    person_raw = await _read(person, "persona")
    garment_raw = await _read(garment, "prenda")

    try:
        depth = await run_in_threadpool(bus.queue_depth, redis_client(), STREAM_PREPROCESS)
        if settings.queue_max_depth and depth >= settings.queue_max_depth:
            return JSONResponse(
                status_code=429,
                content={"detail": "Servicio ocupado: la cola esta llena. Prueba de nuevo en unos minutos.", "queue_depth": depth},
                headers={"Retry-After": "120"},
            )
        allowed = await run_in_threadpool(bus.allow_rate, redis_client(), f"api:{api_key}", settings.rate_limit_per_hour, 3600)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Cuota superada ({settings.rate_limit_per_hour} jobs/hora para esta clave)."},
                headers={"Retry-After": "600"},
            )

        job_id = uuid.uuid4().hex[:16]
        prefix = f"{job_id}/"
        await run_in_threadpool(storage.put_bytes, s3_client(), f"{prefix}{KEY_PERSON_RAW}", person_raw, "application/octet-stream")
        await run_in_threadpool(storage.put_bytes, s3_client(), f"{prefix}{KEY_GARMENT_RAW}", garment_raw, "application/octet-stream")
        await run_in_threadpool(
            bus.create_job,
            redis_client(),
            job_id,
            settings.job_ttl_seconds,
            category=category,
            garment_photo_type=garment_photo_type,
            num_timesteps=num_timesteps,
            seed=seed,
            guidance_scale=guidance_scale,
            person_key=f"{prefix}{KEY_PERSON_RAW}",
            garment_key=f"{prefix}{KEY_GARMENT_RAW}",
            requested_by=api_key[:6] + "...",
        )
        await run_in_threadpool(bus.ensure_group, redis_client(), STREAM_PREPROCESS, bus.GROUP_CPU)
        await run_in_threadpool(bus.enqueue, redis_client(), STREAM_PREPROCESS, job_id, person_key=f"{prefix}{KEY_PERSON_RAW}", garment_key=f"{prefix}{KEY_GARMENT_RAW}")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("No se pudo crear el job")
        raise HTTPException(status_code=503, detail=f"Servicio de cola no disponible: {type(exc).__name__}: {exc}") from exc

    logger.info("Job %s creado (category=%s, steps=%s)", job_id, category, num_timesteps)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "pending", "status_url": f"/v1/jobs/{job_id}"})


@app.get("/v1/jobs/{job_id}")
def job_state(job_id: str, api_key: Annotated[str, Depends(require_api_key)] = "") -> dict:
    try:
        job = bus.get_job(redis_client(), job_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Servicio de cola no disponible: {exc}") from exc
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado (expirado o inexistente).")

    payload = {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "progress": int(job.get("progress") or 0),
        "created_at": float(job.get("created_at") or 0),
        "attempts": int(job.get("attempts") or 0),
    }
    if job.get("status") == STATUS_DONE:
        key = job.get("result_key") or f"{job_id}/{KEY_RESULT}"
        payload["width"] = int(job.get("width") or 0)
        payload["height"] = int(job.get("height") or 0)
        payload["elapsed_seconds"] = float(job.get("elapsed_seconds") or 0)
        payload["download_url"] = f"/v1/jobs/{job_id}/result"
        payload["result_url"] = storage.presign_get(s3_client(), key, settings.result_url_ttl_seconds)
        payload["result_expires_in"] = settings.result_url_ttl_seconds
    if job.get("status") == STATUS_ERROR:
        payload["error"] = job.get("error", "error desconocido")
    return payload


@app.get("/v1/jobs/{job_id}/result")
async def download_result(job_id: str, api_key: Annotated[str, Depends(require_api_key)] = "") -> Response:
    """Deliver the result and delete it (zero retention on delivery)."""
    job = await run_in_threadpool(bus.get_job, redis_client(), job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado (expirado o inexistente).")
    if job.get("status") != STATUS_DONE:
        raise HTTPException(status_code=409, detail=f"El job no esta listo (estado={job.get('status')}).")
    key = job.get("result_key") or f"{job_id}/{KEY_RESULT}"
    try:
        data = await run_in_threadpool(storage.get_bytes, s3_client(), key)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=410, detail=f"El resultado ya no esta disponible: {exc}") from exc
    # Zero retention: nothing of this job survives the delivery (plan section 11).
    await run_in_threadpool(_purge, job_id, job)
    return Response(content=data, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.delete("/v1/jobs/{job_id}")
async def purge_job(job_id: str, api_key: Annotated[str, Depends(require_api_key)] = "") -> dict:
    job = await run_in_threadpool(bus.get_job, redis_client(), job_id)
    removed = await run_in_threadpool(storage.delete_prefix, s3_client(), f"{job_id}/")
    await run_in_threadpool(bus.purge_job, redis_client(), job_id)
    if not job and not removed:
        raise HTTPException(status_code=404, detail="Job no encontrado.")
    return {"job_id": job_id, "deleted": True, "objects_removed": removed}


def _purge(job_id: str, job: dict) -> int:
    removed = storage.delete_prefix(s3_client(), f"{job_id}/")
    bus.purge_job(redis_client(), job_id)
    logger.info("Job %s: entregado y borrado (%s objetos)", job_id, removed)
    return removed


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    uvicorn.run("fashn_vton.gateway:app", host="0.0.0.0", port=int(os.environ.get("API_PORT", "8000")))
