"""Single-consumer GPU turnstile (file lock).

On this machine the RTX 3060 is shared by three consumers (the IDM-VTON
research watchdog, the IDM-CUSTOM service and this fork). The lock file uses the
same JSON format and default path as IDM-CUSTOM's ``infra.gpu_lock``
(``~/.idm_gpu.lock``) so the two projects can take turns **without** sharing
code: ``{"consumer": str, "pid": int, "acquired_at": float}``.

Usage::

    from fashn_vton.utils.gpu_lock import acquire, release, status, GpuBusyError

    acquire("servicio")           # raises GpuBusyError if somebody else holds it
    try:
        ...
    finally:
        release("servicio")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_LOCK_PATH = Path.home() / ".idm_gpu.lock"


class GpuBusyError(RuntimeError):
    """Another live consumer already holds the GPU turn."""


@dataclass(frozen=True)
class LockInfo:
    consumer: str
    pid: int
    acquired_at: float


def _read_lock(lock_path: Path) -> LockInfo | None:
    if not lock_path.exists():
        return None
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        return LockInfo(consumer=data["consumer"], pid=int(data["pid"]), acquired_at=float(data["acquired_at"]))
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user -> treat as alive
    return True


def status(lock_path: Path | str = DEFAULT_LOCK_PATH) -> LockInfo | None:
    """Active turn, or ``None`` when free (stale locks of dead PIDs count as free)."""
    path = Path(lock_path)
    info = _read_lock(path)
    if info is None or not _pid_is_alive(info.pid):
        return None
    return info


def acquire(
    consumer: str,
    lock_path: Path | str = DEFAULT_LOCK_PATH,
    pid: int | None = None,
) -> LockInfo:
    """Take the GPU turn for ``consumer``.

    Idempotent for the same consumer (renews the lock). Raises
    :class:`GpuBusyError` when another live consumer holds it.
    """
    path = Path(lock_path)
    current = status(path)
    if current is not None and current.consumer != consumer:
        raise GpuBusyError(
            f"GPU en uso por '{current.consumer}' (pid={current.pid}). "
            f"Libera el turno antes de iniciar '{consumer}'."
        )
    info = LockInfo(consumer=consumer, pid=int(pid if pid is not None else os.getpid()), acquired_at=time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(asdict(info)), encoding="utf-8")
    os.replace(temporary, path)
    return info


def release(consumer: str, lock_path: Path | str = DEFAULT_LOCK_PATH) -> bool:
    """Release the turn if ``consumer`` holds it. Returns True when released."""
    path = Path(lock_path)
    info = _read_lock(path)
    if info is None or info.consumer != consumer:
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def acquire_or_raise_if_busy_message(consumer: str, lock_path: Path | str = DEFAULT_LOCK_PATH) -> str:
    """Acquire and return a human message; raise :class:`GpuBusyError` if busy."""
    info = acquire(consumer, lock_path)
    return f"Turno de GPU concedido a '{info.consumer}' (pid={info.pid})."


__all__ = [
    "DEFAULT_LOCK_PATH",
    "GpuBusyError",
    "LockInfo",
    "acquire",
    "acquire_or_raise_if_busy_message",
    "release",
    "status",
]
