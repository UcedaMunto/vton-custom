"""CPU preprocessing worker (F6 of ``kubernetes/00_PLAN_ARQUITECTURA.md``).

Consumes ``vton:jobs:preprocess`` through the ``vton-cpu`` consumer group: it
downloads the raw client uploads, validates them for real (not just size),
re-encodes them to clean JPEG (dropping EXIF and capping the longest side) and
hands the *normalised* objects to the GPU queue.

It runs on the nodes without the GPU taint (``nodo-orq`` and ``nodo-cpu-1``) and
never loads the try-on model, so it can scale horizontally without touching the
RTX 3060. It also deletes the raw uploads as soon as it has re-encoded them,
which is the fast path of the zero-retention policy (plan section 11).

Run it with ``python -m fashn_vton.preprocess_worker`` (``--once`` for one job).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from .cluster import (
    GROUP_CPU,
    KEY_GARMENT,
    KEY_GARMENT_RAW,
    KEY_PERSON,
    KEY_PERSON_RAW,
    STATUS_ERROR,
    STATUS_PROCESSING,
    STREAM_GPU,
    STREAM_PREPROCESS,
    bus,
    storage,
)
from .cluster.health import start_health_server
from .cluster.images import normalize_jpeg, open_rgb

LOGGER = logging.getLogger("fashn_vton.preprocess_worker")

CONSUMER = os.environ.get("VTON_CPU_CONSUMER") or "cpu-worker"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class PreprocessWorker:
    """Validates client uploads and enqueues the GPU job."""

    def __init__(
        self,
        max_upload_mb: int | None = None,
        poll_ms: int | None = None,
        job_ttl_seconds: int | None = None,
        max_side: int | None = None,
    ) -> None:
        self.max_upload_mb = max_upload_mb or _env_int("VTON_MAX_UPLOAD_MB", 12)
        self.poll_ms = poll_ms or _env_int("VTON_POLL_MS", 5000)
        self.job_ttl_seconds = job_ttl_seconds or _env_int("VTON_JOB_TTL_SECONDS", 3600)
        self.max_side = max_side or _env_int("VTON_MAX_SIDE", 1600)
        self._redis = None
        self._s3 = None
        self._processed = 0
        self._last_error: str | None = None

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

    def health(self) -> dict:
        payload = {
            "status": "ok",
            "component": "cpu-worker",
            "hostname": os.uname().nodename,
            "consumer": CONSUMER,
            "processed": self._processed,
            "queue_depth": -1,
            "last_error": self._last_error,
        }
        try:
            payload["redis"] = bool(self.redis.ping())
            payload["queue_depth"] = bus.queue_depth(self.redis, STREAM_PREPROCESS)
        except Exception as exc:  # noqa: BLE001
            payload["redis"] = False
            payload["status"] = "degraded"
            payload["error"] = f"{type(exc).__name__}: {exc}"
        return payload

    def process(self, job_id: str, fields: dict) -> str:
        try:
            prefix = f"{job_id}/"
            person_key = fields.get("person_key") or f"{prefix}{KEY_PERSON_RAW}"
            garment_key = fields.get("garment_key") or f"{prefix}{KEY_GARMENT_RAW}"

            person = open_rgb(storage.get_bytes(self.s3, person_key), role="persona", max_mb=self.max_upload_mb)
            garment = open_rgb(storage.get_bytes(self.s3, garment_key), role="prenda", max_mb=self.max_upload_mb)

            person_bytes = normalize_jpeg(person, max_side=self.max_side)
            garment_bytes = normalize_jpeg(garment, max_side=self.max_side)
            storage.put_bytes(self.s3, f"{prefix}{KEY_PERSON}", person_bytes, content_type="image/jpeg")
            storage.put_bytes(self.s3, f"{prefix}{KEY_GARMENT}", garment_bytes, content_type="image/jpeg")

            # Zero retention: the raw upload disappears as soon as it is re-encoded.
            storage.delete_objects(self.s3, [person_key, garment_key])

            bus.update_job(
                self.redis,
                job_id,
                ttl_seconds=self.job_ttl_seconds,
                status=STATUS_PROCESSING,
                progress=10,
                person_size=f"{person.width}x{person.height}",
                garment_size=f"{garment.width}x{garment.height}",
                preprocessed_by=os.uname().nodename,
            )
            bus.enqueue(self.redis, STREAM_GPU, job_id, person_key=f"{prefix}{KEY_PERSON}", garment_key=f"{prefix}{KEY_GARMENT}")
            self._processed += 1
            LOGGER.info("Job %s: validado y encolado para la GPU", job_id)
            return "done"
        except Exception as exc:  # noqa: BLE001 - reported in the job hash
            LOGGER.exception("Job %s: fallo en el preprocesado", job_id)
            self._fail(job_id, f"{type(exc).__name__}: {exc}")
            return "error"

    def _fail(self, job_id: str, message: str) -> None:
        self._last_error = message
        try:
            bus.update_job(self.redis, job_id, ttl_seconds=self.job_ttl_seconds, status=STATUS_ERROR, error=message)
        finally:
            try:
                storage.delete_prefix(self.s3, f"{job_id}/")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Job %s: no se pudieron borrar los objetos (%s)", job_id, exc)

    def run_forever(self, once: bool = False, health_port: int | None = None) -> int:
        bus.ensure_group(self.redis, STREAM_PREPROCESS, GROUP_CPU)
        if health_port:
            start_health_server(health_port, self.health)
            LOGGER.info("healthz en :%s", health_port)
        LOGGER.info("Worker CPU escuchando %s (grupo %s, consumidor %s)", STREAM_PREPROCESS, GROUP_CPU, CONSUMER)
        while True:
            messages = bus.claim_stale(self.redis, STREAM_PREPROCESS, GROUP_CPU, CONSUMER, 60000)
            if not messages:
                messages = bus.read_group(self.redis, STREAM_PREPROCESS, GROUP_CPU, CONSUMER, count=1, block_ms=self.poll_ms)
            for message_id, fields in messages:
                job_id = fields.get("job_id")
                if job_id:
                    self.process(job_id, fields)
                bus.ack(self.redis, STREAM_PREPROCESS, GROUP_CPU, message_id)
                if once:
                    return 0
            time.sleep(self.poll_ms / 1000.0)
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="procesa como maximo un job y termina")
    parser.add_argument("--health-port", type=int, default=_env_int("VTON_HEALTH_PORT", 8080))
    parser.add_argument("--log-level", default=os.environ.get("VTON_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker = PreprocessWorker()
    return worker.run_forever(once=args.once, health_port=None if args.once else args.health_port)


if __name__ == "__main__":
    sys.exit(main())
