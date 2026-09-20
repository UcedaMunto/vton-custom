"""Pruebas de los componentes del despliegue distribuido (sin Redis ni MinIO reales).

Cubren la logica que el plan de Kubernetes marca como delicada: el turno unico de
GPU (no hacer ACK si esta ocupada), el limite de espera por job, el borrado por
retencion cero, la admision/cuotas y la sesion del frontend.
"""
from __future__ import annotations

import io
import json
import time

import pytest
from PIL import Image

from fashn_vton import queue_worker, web
from fashn_vton.cluster import bus, images
from fashn_vton.cluster.health import start_health_server
from fashn_vton.utils.gpu_lock import GpuBusyError


class FakeRedis:
    """Minimo cliente compatible con lo que usa ``cluster.bus``."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict] = {}
        self.streams: dict[str, list] = {}
        self.counters: dict[str, int] = {}
        self.expires: list[tuple[str, int]] = []
        self.xack_calls: list[tuple] = []
        self.xadd_kwargs: list[dict] = []
        self.groups_info: list[dict] = [{"lag": 0, "pending": 0}]
        self.groups_error = False

    def hset(self, key, mapping=None, **kwargs):
        self.hashes.setdefault(key, {}).update(mapping or {})
        return True

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def expire(self, key, ttl):
        self.expires.append((key, ttl))
        return True

    def delete(self, key):
        return 1 if self.hashes.pop(key, None) is not None else 0

    def xadd(self, stream, fields, **kwargs):
        self.streams.setdefault(stream, []).append(fields)
        self.xadd_kwargs.append(kwargs)
        return f"{len(self.streams[stream])}-0"

    def xinfo_groups(self, stream):
        if self.groups_error:
            raise RuntimeError("no such key")
        return list(self.groups_info)

    def xlen(self, stream):
        return len(self.streams.get(stream, []))

    def incr(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    def ping(self):
        return True

    def xack(self, *args):
        self.xack_calls.append(args)


def _jpeg(size=(200, 300)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 120, 90)).save(buffer, format="JPEG")
    return buffer.getvalue()


# --------------------------------------------------------------------- bus

def test_redis_url_injects_password(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://redis.vton.svc.cluster.local:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "secreto")
    assert bus.redis_url() == "redis://:secreto@redis.vton.svc.cluster.local:6379/0"
    # Si la URL ya trae credenciales no se toca.
    monkeypatch.setenv("REDIS_URL", "redis://:otra@host:6379/0")
    assert bus.redis_url() == "redis://:otra@host:6379/0"


def test_job_hash_roundtrip_and_json_fields():
    client = FakeRedis()
    bus.create_job(client, "abc", 3600, category="tops", num_timesteps=30, meta={"a": 1})
    job = bus.get_job(client, "abc")
    assert job["status"] == bus.STATUS_PENDING
    assert job["category"] == "tops"
    assert json.loads(job["num_timesteps"]) == 30  # los enteros se guardan como JSON
    assert json.loads(job["meta"]) == {"a": 1}
    assert client.expires == [("vton:job:abc", 3600)]  # retencion cero por TTL
    bus.update_job(client, "abc", status=bus.STATUS_DONE, progress=100)
    assert bus.get_job(client, "abc")["status"] == bus.STATUS_DONE
    bus.purge_job(client, "abc")
    assert bus.get_job(client, "abc") is None


def test_enqueue_accepts_maxlen_and_trims():
    client = FakeRedis()
    bus.enqueue(client, bus.STREAM_PREPROCESS, "j1", person_key="j1/person.jpg")
    assert client.streams[bus.STREAM_PREPROCESS][0]["job_id"] == "j1"
    # El stream se recorta con MAXLEN aproximado: nunca crece sin limite.
    assert client.xadd_kwargs[0] == {"maxlen": 1000, "approximate": True}


def test_queue_depth_uses_group_backlog_not_xlen():
    client = FakeRedis()
    bus.enqueue(client, bus.STREAM_PREPROCESS, "j1")
    bus.enqueue(client, bus.STREAM_PREPROCESS, "j2")
    # XLEN seria 2, pero con todos los mensajes entregados y ACKeados la cola esta vacia.
    client.groups_info = [{"lag": 0, "pending": 0}]
    assert bus.queue_depth(client, bus.STREAM_PREPROCESS) == 0
    client.groups_info = [{"lag": 3, "pending": 2}]
    assert bus.queue_depth(client, bus.STREAM_PREPROCESS) == 5
    client.groups_error = True  # sin grupos todavia -> se usa XLEN como respaldo
    assert bus.queue_depth(client, bus.STREAM_PREPROCESS) == 2


def test_allow_rate_limits_by_window():
    client = FakeRedis()
    assert all(bus.allow_rate(client, "ip:1.2.3.4", 2, 600) for _ in range(2))
    assert bus.allow_rate(client, "ip:1.2.3.4", 2, 600) is False
    assert bus.allow_rate(client, "ip:1.2.3.4", 0, 600) is True  # 0 = sin limite


# ------------------------------------------------------------ health server

def test_health_server_serves_json():
    server = start_health_server(0, lambda: {"status": "ok", "component": "test"}, host="127.0.0.1")
    try:
        import urllib.request

        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as response:
            payload = json.loads(response.read().decode())
        assert payload == {"status": "ok", "component": "test"}
    finally:
        server.shutdown()


# ------------------------------------------------------------------- images

def test_open_rgb_rejects_garbage_and_tiny_images():
    with pytest.raises(ValueError):
        images.open_rgb(b"no soy una imagen", "persona")
    tiny = io.BytesIO()
    Image.new("RGB", (10, 10)).save(tiny, format="PNG")
    with pytest.raises(ValueError):
        images.open_rgb(tiny.getvalue(), "persona")


def test_normalize_jpeg_drops_metadata_and_caps_size():
    big = Image.new("RGB", (4000, 2000), (10, 20, 30))
    data = images.normalize_jpeg(big, max_side=1600)
    result = Image.open(io.BytesIO(data))
    assert max(result.size) == 1600
    assert result.format == "JPEG"
    assert b"Exif" not in data[:64]


# ----------------------------------------------------------------- workjobs
class FakePipeline:
    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        from fashn_vton.pipeline import PipelineOutput

        return PipelineOutput(images=[Image.new("RGB", (576, 864), (1, 2, 3))], metadata={"segmentation": "none"})


def _worker(monkeypatch, *, use_lock=False, busy=False, max_wait=600):
    worker = queue_worker.GpuWorker(weights_dir="/tmp", use_gpu_lock=use_lock, max_wait_seconds=max_wait, busy_retry_seconds=0)
    client = FakeRedis()
    worker._redis = client
    worker._s3 = object()  # no se usa: storage.* esta monkeypatcheado
    worker._pipeline = FakePipeline()
    deleted: list[str] = []
    monkeypatch.setattr(queue_worker.storage, "get_bytes", lambda *a, **k: _jpeg())
    monkeypatch.setattr(queue_worker.storage, "put_bytes", lambda *a, **k: "key")
    monkeypatch.setattr(queue_worker.storage, "delete_objects", lambda *a, **k: deleted.append(list(a[1])) or len(a[1]))
    if busy:
        def _raise(*args, **kwargs):
            raise GpuBusyError("GPU en uso por 'pista_a'")

        monkeypatch.setattr(queue_worker, "acquire", _raise)
    return worker, client, deleted


def test_gpu_worker_marks_done_and_purges(monkeypatch):
    worker, client, deleted = _worker(monkeypatch)
    bus.create_job(client, "j1", 3600, category="tops", num_timesteps=8, seed=1)
    assert worker.process("j1", {"job_id": "j1"}) == "done"
    job = bus.get_job(client, "j1")
    assert job["status"] == bus.STATUS_DONE and int(job["progress"]) == 100
    assert job["result_key"] == "j1/result.png"
    purged = [key for batch in deleted for key in batch]
    assert "j1/person.jpg" in purged and "j1/garment.jpg" in purged  # retencion cero de entradas
    assert "j1/result.png" not in purged  # el resultado sobrevive hasta que el cliente lo descarga


def test_gpu_worker_busy_does_not_touch_the_job(monkeypatch):
    worker, client, deleted = _worker(monkeypatch, use_lock=True, busy=True)
    bus.create_job(client, "j2", 3600)
    assert worker.process("j2", {"job_id": "j2"}) == "busy"
    job = bus.get_job(client, "j2")
    assert job["status"] == bus.STATUS_PENDING  # sigue pendiente: se reintentara
    assert "error" not in job
    assert deleted == []


def test_gpu_worker_gives_up_after_max_wait(monkeypatch):
    worker, client, deleted = _worker(monkeypatch, use_lock=True, busy=True, max_wait=1)
    bus.create_job(client, "j3", 3600)
    bus.update_job(client, "j3", created_at=time.time() - 3600)
    assert worker.process("j3", {"job_id": "j3"}) == "error"
    job = bus.get_job(client, "j3")
    assert job["status"] == bus.STATUS_ERROR and "Turno de GPU" in job["error"]
    assert [key for batch in deleted for key in batch]  # al rendirse tambien limpia las entradas


def test_worker_health_reports_queue_and_lock(monkeypatch, tmp_path):
    worker, client, _ = _worker(monkeypatch)
    payload = worker.health()
    assert payload["status"] == "ok" and payload["redis"] is True
    assert payload["queue_depth"] == 0


# ---------------------------------------------------------------------- web

def test_session_cookie_signature(monkeypatch):
    monkeypatch.setattr(web, "SESSION_SECRET", "secreto-de-prueba")
    token = web._session_token()
    assert web._session_ok(token) is True
    assert web._session_ok(token.replace(token[-1], "0" if token[-1] != "0" else "1")) is False
    assert web._session_ok("1.deadbeef") is False
    monkeypatch.setattr(web, "SESSION_SECRET", "")
    assert web._session_ok(None) is True  # sin secreto no se exige sesion


def test_multipart_body_is_well_formed():
    body, content_type = web._multipart({"category": "tops"}, {"person": ("p.jpg", b"abc", "image/jpeg")})
    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="category"' in body and b"tops" in body
    assert b'filename="p.jpg"' in body and b"abc" in body
    assert body.rstrip().endswith(b"--")
