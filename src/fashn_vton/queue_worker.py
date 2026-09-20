"""GPU queue worker (F5 of ``kubernetes/00_PLAN_ARQUITECTURA.md``).

Consumes ``vton:jobs:gpu`` through the ``vton-gpu`` consumer group, downloads the
normalised inputs from the transient bucket, runs :class:`TryOnPipeline` on the
local GPU, uploads the PNG result and marks the job ``done``.

Two behaviours come straight from the plan:

* **GPU turnstile** (section 8): the RTX 3060 of ``nodo-gpu-1`` is shared with
  ``IDM-CUSTOM``, the IDM-VTON continuous training and the local UI through
  ``~/.idm_gpu.lock``. The worker takes the turn when ``VTON_USE_GPU_LOCK`` is
  enabled and, if another consumer holds it, it **does not ACK**: the job stays
  pending and comes back through ``XAUTOCLAIM`` after the backoff window, until
  ``VTON_JOB_MAX_WAIT_SECONDS`` is exceeded (then it is marked ``error``).
* **Zero retention** (section 11): when the job reaches a terminal state the
  worker deletes every object of that job (inputs and result) from the bucket.

Run it with ``python -m fashn_vton.queue_worker`` (``--once`` for a single job).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from .cluster import (
    GROUP_GPU,
    KEY_GARMENT,
    KEY_GARMENT_RAW,
    KEY_PERSON,
    KEY_PERSON_RAW,
    KEY_RESULT,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_PROCESSING,
    STREAM_GPU,
    bus,
    storage,
)
from .cluster.health import start_health_server
from .utils.gpu_lock import DEFAULT_LOCK_PATH, GpuBusyError, acquire, release
from .utils.gpu_lock import status as lock_status

LOGGER = logging.getLogger("fashn_vton.queue_worker")

#: Consumer name inside the group (kept stable across restarts of the pod).
CONSUMER = os.environ.get("VTON_GPU_CONSUMER") or "gpu-worker"
#: Name registered in the shared GPU lock file.
LOCK_CONSUMER = "reconstruccion_cluster_gpu_worker"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class GpuWorker:
    """Pulls jobs from Redis, runs the try-on pipeline and stores the result."""

    def __init__(
        self,
        weights_dir: str | None = None,
        device: str | None = None,
        provider: str | None = None,
        use_gpu_lock: bool | None = None,
        lock_path: str | None = None,
        busy_retry_seconds: int | None = None,
        max_wait_seconds: int | None = None,
        stale_seconds: int | None = None,
        poll_ms: int | None = None,
        job_ttl_seconds: int | None = None,
    ) -> None:
        self.weights_dir = weights_dir or os.environ.get("FASHN_WEIGHTS_DIR", "weights")
        self.device = device or os.environ.get("FASHN_DEVICE") or None
        self.provider = provider or os.environ.get("FASHN_PROVIDER", "none")
        self.use_gpu_lock = _env_flag("VTON_USE_GPU_LOCK", False) if use_gpu_lock is None else use_gpu_lock
        self.lock_path = lock_path or os.environ.get("VTON_GPU_LOCK_PATH") or str(DEFAULT_LOCK_PATH)
        self.busy_retry_seconds = busy_retry_seconds or _env_int("VTON_GPU_BUSY_RETRY_SECONDS", 20)
        self.max_wait_seconds = max_wait_seconds or _env_int("VTON_JOB_MAX_WAIT_SECONDS", 600)
        self.stale_seconds = stale_seconds or _env_int("VTON_STALE_JOB_SECONDS", self.busy_retry_seconds)
        self.poll_ms = poll_ms or _env_int("VTON_POLL_MS", 5000)
        self.job_ttl_seconds = job_ttl_seconds or _env_int("VTON_JOB_TTL_SECONDS", 3600)
        self._pipeline = None
        self._redis = None
        self._s3 = None
        self._processed = 0
        self._last_error: str | None = None

    # ------------------------------------------------------------------ clients
    @property
    def redis(self):
        if self._redis is None:
            self._redis = bus.connect()
        return self._redis

    @property
    def s3(self):
        if self._s3 is None:
            self._s3 = storage.s3_client()
        return self._s3

    @property
    def pipeline(self):
        """Build the try-on pipeline lazily (loading SDXL-lite weights costs ~20 s)."""
        if self._pipeline is None:
            from .pipeline import TryOnPipeline
            from .segmentation import build_segmentation_provider

            LOGGER.info("Cargando el pipeline (weights=%s, device=%s, provider=%s)", self.weights_dir, self.device or "auto", self.provider)
            self._pipeline = TryOnPipeline(
                weights_dir=self.weights_dir,
                device=self.device,
                segmentation_provider=build_segmentation_provider(self.provider),
            )
        return self._pipeline

    # --------------------------------------------------------------- gpu turn
    def acquire_gpu(self) -> bool:
        """Take the machine-wide GPU turn; ``False`` when somebody else has it."""
        if not self.use_gpu_lock:
            return True
        try:
            acquire(LOCK_CONSUMER, lock_path=self.lock_path)
            return True
        except GpuBusyError as exc:
            current = lock_status(self.lock_path)
            LOGGER.info("GPU ocupada por '%s': %s", current.consumer if current else "?", exc)
            return False

    def release_gpu(self) -> None:
        if self.use_gpu_lock:
            release(LOCK_CONSUMER, lock_path=self.lock_path)

    # ------------------------------------------------------------------- health
    def health(self) -> dict:
        payload: dict = {
            "status": "ok",
            "component": "gpu-worker",
            "hostname": os.uname().nodename,
            "consumer": CONSUMER,
            "processed": self._processed,
            "gpu_lock": "free",
            "queue_depth": -1,
            "last_error": self._last_error,
        }
        try:
            payload["redis"] = bool(self.redis.ping())
            payload["queue_depth"] = bus.queue_depth(self.redis, STREAM_GPU)
        except Exception as exc:  # noqa: BLE001 - reported in the payload
            payload["redis"] = False
            payload["status"] = "degraded"
            payload["error"] = f"{type(exc).__name__}: {exc}"
        if self.use_gpu_lock:
            current = lock_status(self.lock_path)
            payload["gpu_lock"] = current.consumer if current else "free"
        try:
            import torch

            payload["cuda"] = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - torch is optional for the health check
            payload["cuda"] = "unknown"
        return payload

    # -------------------------------------------------------------------- jobs
    def process(self, job_id: str, fields: dict) -> str:
        """Run one job. Returns ``done``, ``error`` or ``busy`` (not ACKed)."""
        job = bus.get_job(self.redis, job_id) or {}
        created = float(job.get("created_at") or time.time())
        waited = time.time() - created
        attempts = int(job.get("attempts") or 0) + 1
        bus.update_job(self.redis, job_id, ttl_seconds=self.job_ttl_seconds, attempts=attempts)

        if waited > self.max_wait_seconds:
            message = f"Turno de GPU no disponible tras {int(waited)} s (limite {self.max_wait_seconds} s)."
            LOGGER.warning("Job %s: %s", job_id, message)
            self._fail(job_id, message)
            return "error"

        if not self.acquire_gpu():
            LOGGER.info("Job %s: GPU ocupada, reintento en %s s (intento %s, espera %s s)", job_id, self.busy_retry_seconds, attempts, int(waited))
            time.sleep(self.busy_retry_seconds)
            return "busy"

        try:
            bus.update_job(self.redis, job_id, ttl_seconds=self.job_ttl_seconds, status=STATUS_PROCESSING, progress=15, worker=os.uname().nodename)
            import io as _io

            from PIL import Image

            person = Image.open(_io.BytesIO(storage.get_bytes(self.s3, fields.get("person_key") or f"{job_id}/{KEY_PERSON}"))).convert("RGB")
            garment = Image.open(_io.BytesIO(storage.get_bytes(self.s3, fields.get("garment_key") or f"{job_id}/{KEY_GARMENT}"))).convert("RGB")
            params = {
                "category": job.get("category", "tops"),
                "garment_photo_type": job.get("garment_photo_type", "flat-lay"),
                "num_timesteps": int(job.get("num_timesteps") or 30),
                "seed": int(job.get("seed") or 42),
                "guidance_scale": float(job.get("guidance_scale") or 1.5),
            }
            LOGGER.info("Job %s: inferencia %s", job_id, params)
            started = time.time()
            output = self.pipeline(person_image=person, garment_image=garment, **params)
            elapsed = time.time() - started
            image = output.images[0]
            key = f"{job_id}/{KEY_RESULT}"
            storage.put_bytes(self.s3, key, _png(image), content_type="image/png")
            bus.update_job(
                self.redis,
                job_id,
                ttl_seconds=self.job_ttl_seconds,
                status=STATUS_DONE,
                progress=100,
                result_key=key,
                width=image.width,
                height=image.height,
                elapsed_seconds=round(elapsed, 2),
                segmentation=(output.metadata or {}).get("segmentation", ""),
            )
            self._processed += 1
            self._purge_inputs(job_id)
            LOGGER.info("Job %s: done en %.1f s", job_id, elapsed)
            return "done"
        except Exception as exc:  # noqa: BLE001 - every failure is reported in the job hash
            LOGGER.exception("Job %s: fallo", job_id)
            self._fail(job_id, f"{type(exc).__name__}: {exc}")
            return "error"
        finally:
            self.release_gpu()

    def _fail(self, job_id: str, message: str) -> None:
        self._last_error = message
        try:
            bus.update_job(self.redis, job_id, ttl_seconds=self.job_ttl_seconds, status=STATUS_ERROR, error=message)
        finally:
            self._purge_inputs(job_id)

    def _purge_inputs(self, job_id: str) -> None:
        """Zero retention of the **inputs** (plan section 11).

        Only the input objects are deleted here: the result must survive until
        the client downloads it (the gateway deletes it on delivery) or until the
        bucket lifecycle rule expires it. Deleting the whole job prefix here
        would destroy the result before it can be delivered.
        """
        keys = [
            f"{job_id}/{KEY_PERSON}",
            f"{job_id}/{KEY_GARMENT}",
            f"{job_id}/{KEY_PERSON_RAW}",
            f"{job_id}/{KEY_GARMENT_RAW}",
        ]
        try:
            removed = storage.delete_objects(self.s3, keys)
            if removed:
                LOGGER.info("Job %s: %s objetos de entrada borrados del almacen transitorio", job_id, removed)
        except Exception as exc:  # noqa: BLE001 - the bucket TTL is the safety net
            LOGGER.warning("Job %s: no se pudieron borrar las entradas (%s); el ILM del bucket las expira", job_id, exc)

    # -------------------------------------------------------------------- loop
    def run_forever(self, once: bool = False, health_port: int | None = None) -> int:
        bus.ensure_group(self.redis, STREAM_GPU, GROUP_GPU)
        if health_port:
            start_health_server(health_port, self.health)
            LOGGER.info("healthz en :%s", health_port)
        LOGGER.info("Worker GPU escuchando %s (grupo %s, consumidor %s)", STREAM_GPU, GROUP_GPU, CONSUMER)
        while True:
            messages = bus.claim_stale(self.redis, STREAM_GPU, GROUP_GPU, CONSUMER, self.stale_seconds * 1000)
            if not messages:
                messages = bus.read_group(self.redis, STREAM_GPU, GROUP_GPU, CONSUMER, count=1, block_ms=self.poll_ms)
            for message_id, fields in messages:
                job_id = fields.get("job_id")
                if not job_id:
                    bus.ack(self.redis, STREAM_GPU, GROUP_GPU, message_id)
                    continue
                result = self.process(job_id, fields)
                if result == "busy":
                    LOGGER.info("Job %s: sin ACK, volvera por XAUTOCLAIM", job_id)
                else:
                    bus.ack(self.redis, STREAM_GPU, GROUP_GPU, message_id)
                if once:
                    return 0
            time.sleep(self.poll_ms / 1000.0)
        return 0


def _png(image) -> bytes:
    import io as _io

    buffer = _io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="procesa como maximo un job y termina")
    parser.add_argument("--health-port", type=int, default=_env_int("VTON_HEALTH_PORT", 8080))
    parser.add_argument("--no-gpu-lock", action="store_true", help="no usar el turno unico de GPU")
    parser.add_argument("--log-level", default=os.environ.get("VTON_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker = GpuWorker(use_gpu_lock=False if args.no_gpu_lock else None)
    return worker.run_forever(once=args.once, health_port=None if args.once else args.health_port)


if __name__ == "__main__":
    sys.exit(main())
