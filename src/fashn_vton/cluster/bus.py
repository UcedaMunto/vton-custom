"""Redis Streams bus and per-job state helpers.

The job state lives in a Redis hash (``vton:job:<id>``) with a TTL; the work
queues are Redis Streams consumed through consumer groups, so Redis delivers
each job to exactly one worker of the group. Both facts are what makes the
distribution work: see ``kubernetes/00_PLAN_ARQUITECTURA.md`` sections 7 and 11.

Every Redis import is lazy: the core package must stay importable (and testable)
without the cluster dependencies installed.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from . import (  # noqa: F401 - re-exported so callers can import a single module
    GROUP_CPU,
    GROUP_GPU,
    JOB_KEY_PREFIX,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STREAM_GPU,
    STREAM_PREPROCESS,
    VALID_STATUSES,
)


def redis_url() -> str:
    """Render the Redis URL, injecting ``REDIS_PASSWORD`` when it is set.

    Kubernetes passes the URL without credentials plus the password in its own
    Secret, so the password never appears in the Deployment/StatefulSet spec.
    """
    url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    password = os.environ.get("REDIS_PASSWORD")
    if password and "@" not in url.split("//", 1)[-1]:
        scheme, rest = url.split("//", 1)
        return f"{scheme}//:{password}@{rest}"
    return url


def connect(url: str | None = None):
    """Return a ``redis.Redis`` client (decode_responses=True).

    ``socket_timeout`` se fija **explicitamente** (60 s por defecto): si se deja a
    ``None``, cualquier timeout heredado del socket por una operacion anterior se
    aplica a la siguiente lectura y aparecen ``TimeoutError: Timeout reading from
    socket`` en mitad del bucle de los workers (nos paso en el cluster, ver
    02_ESTADO_DE_EJECUCION.md). El valor es mayor que cualquier operacion normal y
    el bucle reintenta sin bloquear.
    """
    import redis  # lazy: only the cluster processes need it

    return redis.Redis.from_url(
        url or redis_url(),
        decode_responses=True,
        socket_timeout=float(os.environ.get("VTON_REDIS_SOCKET_TIMEOUT", "60")),
        socket_connect_timeout=5.0,
        retry_on_timeout=True,
        health_check_interval=30,
    )


def job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


def create_job(client, job_id: str, ttl_seconds: int = 3600, **fields: Any) -> None:
    """Create the job hash in ``pending`` state with a TTL (zero retention)."""
    now = str(time.time())
    payload: dict[str, str] = {
        "job_id": job_id,
        "status": STATUS_PENDING,
        "progress": "0",
        "created_at": now,
        "updated_at": now,
    }
    for key, value in fields.items():
        payload[key] = value if isinstance(value, str) else json.dumps(value)
    client.hset(job_key(job_id), mapping=payload)
    if ttl_seconds:
        client.expire(job_key(job_id), int(ttl_seconds))


def update_job(client, job_id: str, ttl_seconds: int = 0, **fields: Any) -> None:
    """Patch fields of an existing job (adds/refreshes ``updated_at``)."""
    payload: dict[str, str] = {"updated_at": str(time.time())}
    for key, value in fields.items():
        payload[key] = value if isinstance(value, str) else json.dumps(value)
    client.hset(job_key(job_id), mapping=payload)
    if ttl_seconds:
        client.expire(job_key(job_id), int(ttl_seconds))


def get_job(client, job_id: str) -> dict[str, str] | None:
    """Return the job hash, or ``None`` when it expired/was purged."""
    data = client.hgetall(job_key(job_id))
    return data or None


def enqueue(client, stream: str, job_id: str, maxlen: int | None = None, **extra: Any) -> str:
    """Append one job to a stream and return the message id.

    ``maxlen`` (approximate trimming) keeps the stream bounded: Redis Streams
    otherwise grow forever because acknowledged entries are *not* removed, and that
    length is also what the gateway uses for admission control.
    """
    fields = {"job_id": job_id}
    for key, value in extra.items():
        fields[key] = value if isinstance(value, str) else json.dumps(value)
    limit = int(maxlen if maxlen is not None else os.environ.get("VTON_STREAM_MAXLEN", "1000"))
    if limit > 0:
        return client.xadd(stream, fields, maxlen=limit, approximate=True)
    return client.xadd(stream, fields)


def ensure_group(client, stream: str, group: str) -> None:
    """Create the consumer group if it does not exist (idempotent)."""
    try:
        client.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception as exc:  # redis raises ResponseError('BUSYGROUP ...')
        if "BUSYGROUP" not in str(exc):
            raise


def read_group(client, stream: str, group: str, consumer: str, count: int = 1, block_ms: int = 0) -> list[tuple[str, dict]]:
    """Read up to ``count`` new messages for ``consumer``.

    Non-blocking by default (``block_ms=0``): el bucle del worker duerme entre
    intentos. Se evita a proposito ``XREADGROUP ... BLOCK`` porque en redis-py 8
    la lectura bloqueante lanza ``TimeoutError`` en cuanto el socket tiene un
    timeout (nos paso en el cluster, ver 02_ESTADO_DE_EJECUCION.md).
    """
    options: dict = {"count": count}
    if block_ms:
        options["block"] = block_ms
    response = client.xreadgroup(group, consumer, {stream: ">"}, **options)
    if not response:
        return []
    return [(message_id, fields) for _stream, messages in response for message_id, fields in messages]


def ack(client, stream: str, group: str, message_id: str) -> None:
    client.xack(stream, group, message_id)


def claim_stale(client, stream: str, group: str, consumer: str, min_idle_ms: int, count: int = 10) -> list[tuple[str, dict]]:
    """Take over messages whose consumer died (``XAUTOCLAIM``).

    This is what makes a job survive the death of the worker that was running
    it mid-inference (plan section 8.2).
    """
    result = client.xautoclaim(stream, group, consumer, min_idle_ms, start_id="0-0", count=count)
    messages = result[1] if len(result) > 1 else []
    return [(message_id, fields) for message_id, fields in messages if fields]


def queue_depth(client, stream: str) -> int:
    """Pending work in the stream, used for admission control (plan 7.1).

    ``XLEN`` is **not** the backlog: Redis Streams keep acknowledged entries until
    they are trimmed, so the length only grows. The backlog is, per consumer
    group, ``lag`` (not delivered yet) plus ``pending`` (delivered, not acked).
    """
    try:
        groups = client.xinfo_groups(stream)
    except Exception:
        try:
            return int(client.xlen(stream))
        except Exception:
            return 0
    return sum(int(g.get("lag") or 0) + int(g.get("pending") or 0) for g in groups or [])


def purge_job(client, job_id: str) -> None:
    client.delete(job_key(job_id))


def allow_rate(client, bucket: str, limit: int, window_seconds: int) -> bool:
    """Fixed-window rate limit; returns False when ``limit`` is exceeded."""
    if limit <= 0:
        return True
    key = f"vton:rate:{bucket}"
    current = client.incr(key)
    if current == 1:
        client.expire(key, int(window_seconds))
    return int(current) <= int(limit)
