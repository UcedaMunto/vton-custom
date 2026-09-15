"""Test local del configurado `gpu_lock` del fork (compatible con IDM-CUSTOM)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from fashn_vton.utils import gpu_lock


def test_acquire_status_release_roundtrip(tmp_path):
    lock = tmp_path / "gpu.lock"
    assert gpu_lock.status(lock) is None

    info = gpu_lock.acquire("reconstruccion", lock_path=lock)
    assert info.consumer == "reconstruccion"
    assert info.pid == os.getpid()
    assert gpu_lock.status(lock).consumer == "reconstruccion"

    assert gpu_lock.release("reconstruccion", lock_path=lock) is True
    assert gpu_lock.status(lock) is None


def test_second_consumer_is_blocked(tmp_path):
    lock = tmp_path / "gpu.lock"
    gpu_lock.acquire("pista_a", lock_path=lock)
    with pytest.raises(gpu_lock.GpuBusyError) as excinfo:
        gpu_lock.acquire("servicio", lock_path=lock)
    assert "pista_a" in str(excinfo.value)


def test_same_consumer_is_idempotent(tmp_path):
    lock = tmp_path / "gpu.lock"
    first = gpu_lock.acquire("servicio", lock_path=lock)
    second = gpu_lock.acquire("servicio", lock_path=lock)
    assert second.consumer == first.consumer
    assert second.acquired_at >= first.acquired_at


def test_stale_lock_of_dead_pid_is_ignored(tmp_path):
    lock = tmp_path / "gpu.lock"
    gpu_lock.acquire("muerto", lock_path=lock, pid=999_999_999)
    assert gpu_lock.status(lock) is None
    # y otro consumidor puede tomarlo
    gpu_lock.acquire("servicio", lock_path=lock)


def test_release_by_other_consumer_does_nothing(tmp_path):
    lock = tmp_path / "gpu.lock"
    gpu_lock.acquire("pista_a", lock_path=lock)
    assert gpu_lock.release("servicio", lock_path=lock) is False
    assert gpu_lock.status(lock).consumer == "pista_a"


def test_lock_format_is_compatible_with_idm_custom(tmp_path):
    """El JSON debe tener las mismas claves que `infra.gpu_lock` de IDM-CUSTOM."""
    import json

    lock = tmp_path / "gpu.lock"
    gpu_lock.acquire("servicio", lock_path=lock)
    data = json.loads(lock.read_text(encoding="utf-8"))
    assert set(data) == {"consumer", "pid", "acquired_at"}


def test_interop_with_idm_custom_cli_is_optional(tmp_path):
    """Si IDM-CUSTOM está en su ruta habitual, su CLI debe ver el mismo lock."""
    idm_src = tmp_path.parent / "no-existe"
    idm_src = os.environ.get("FASHN_GPU_LOCK_IDM_SRC", "/home/uceda/Documents/IDM-CUSTOM/src")
    if not (os.path.isdir(idm_src) and os.path.isfile(os.path.join(idm_src, "infra", "gpu_lock.py"))):
        pytest.skip("IDM-CUSTOM no está disponible en esta máquina")

    lock = tmp_path / "gpu.lock"
    gpu_lock.acquire("servicio", lock_path=lock)
    env = dict(os.environ, PYTHONPATH=idm_src)
    result = subprocess.run(
        [sys.executable, "-m", "infra.gpu_lock", "--lock-path", str(lock), "status"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "servicio" in (result.stdout + result.stderr)
