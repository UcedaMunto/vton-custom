#!/usr/bin/env python3
"""Registro de pesos del modelo: congelar, promover y **volver atrás**.

Pieza operativa del plan de fine-tuning (`plan_entrenamiento/PLAN_FINE_TUNING.md`):
deja constancia con hash de cada estado del modelo y permite restaurar cualquier
estado anterior con un comando, sin depender de la memoria de nadie.

    snapshot   Copia los pesos activos a `weights/registry/<tag>/` y los registra
    list       Lista los estados registrados (hash, tamaño, activo)
    current    Qué pesos hay activos ahora y con qué tag de registro coinciden
    promote    Promueve un candidato (entrenado) a pesos activos, guardando antes
               los que había (auto-backup) por si hay que revertir
    rollback   Restaura un estado registrado a `weights/model.safetensors`

Todos los comandos verifican el sha256 y se niegan a operar si no coincide
(`--force` para saltarse la comprobación, bajo tu responsabilidad).

Ejemplos:
    python scripts/model_registry.py snapshot --tag baseline-2026-09-15
    python scripts/model_registry.py list
    python scripts/model_registry.py promote --candidate weights/candidates/lora-v1 --tag lora-v1
    python scripts/model_registry.py rollback --tag baseline-2026-09-15
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashn_vton.compliance import sha256_of_file  # noqa: E402

REGISTRY_DIR = ROOT / "model_registry"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"
SNAPSHOTS_DIR = ROOT / "weights" / "registry"
WEIGHTS_DIR = ROOT / "weights"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load() -> dict:
    if REGISTRY_FILE.is_file():
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {"updated_at": _now(), "active": None, "entries": [], "history": []}


def _save(registry: dict) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    registry["updated_at"] = _now()
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _entry(registry: dict, tag: str) -> dict | None:
    for entry in registry["entries"]:
        if entry["tag"] == tag:
            return entry
    return None


def _record(registry: dict, *, tag: str, source: str, sha256: str, path: Path, notes: str = "") -> None:
    entry = _entry(registry, tag)
    payload = {
        "tag": tag,
        "sha256": sha256,
        "bytes": path.stat().st_size if path.is_file() else None,
        "snapshot_path": str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
        "updated_at": _now(),
        "source": source,
        "notes": notes,
    }
    if entry:
        entry.update(payload)
    else:
        payload["created_at"] = payload["updated_at"]
        registry["entries"].append(payload)
    registry["active"] = tag


def _history(registry: dict, action: str, **extra) -> None:
    registry["history"].append({"at": _now(), "action": action, **extra})


def _active_weights(args) -> Path:
    return Path(args.weights_dir) / "model.safetensors"


def _copy_verified(src: Path, dst: Path, expected_sha: str | None, force: bool) -> str:
    """Copia verificando el origen (y el destino) con sha256."""
    if not src.is_file():
        raise SystemExit(f"ERROR: no existe {src}")
    digest = sha256_of_file(src)
    if expected_sha and digest != expected_sha and not force:
        raise SystemExit(f"ERROR: el sha256 de {src} no coincide con el registrado ({digest} != {expected_sha})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    written = sha256_of_file(dst)
    if written != digest:
        raise SystemExit(f"ERROR: la copia en {dst} no coincide byte a byte ({written} != {digest})")
    return written


def cmd_snapshot(args) -> int:
    registry = _load()
    src = _active_weights(args)
    target = SNAPSHOTS_DIR / args.tag / "model.safetensors"
    if target.exists() and not args.force:
        print(f"ERROR: ya existe el snapshot {target} (usa --force para sobrescribir)")
        return 2

    print(f"Congelando pesos activos: {src}")
    digest = _copy_verified(src, target, expected_sha=None, force=args.force)
    _record(registry, tag=args.tag, source="snapshot", sha256=digest, path=target, notes=args.notes)
    _history(registry, "snapshot", tag=args.tag, sha256=digest)
    _save(registry)
    print(f"OK: {args.tag} → {target.relative_to(ROOT)}")
    print(f"    sha256: {digest}")
    return 0


def cmd_list(args) -> int:
    registry = _load()
    entries = registry.get("entries", [])
    if not entries:
        print("Registro vacío. Crea el primero con:")
        print("  python scripts/model_registry.py snapshot --tag baseline")
        return 0
    print(f"Registro: {REGISTRY_FILE.relative_to(ROOT)}\n")
    print(f"  {'tag':<28} {'activo':<7} {'GB':>6}  {'sha256':<14} creado")
    for entry in entries:
        size = (entry.get("bytes") or 0) / 1024**3
        active = "sí" if entry["tag"] == registry.get("active") else ""
        print(
            f"  {entry['tag']:<28} {active:<7} {size:>6.2f}  {entry['sha256'][:12]}… "
            f"{entry.get('created_at', entry.get('updated_at', ''))}"
        )
    if args.history:
        print("\nHistorial:")
        for row in registry.get("history", []):
            detail = " ".join(f"{key}={value}" for key, value in row.items() if key not in {"at", "action"})
            print(f"  {row['at']}  {row['action']:<9} {detail}")
    return 0


def cmd_current(args) -> int:
    registry = _load()
    src = _active_weights(args)
    if not src.is_file():
        print(f"ERROR: no existe {src}")
        return 2
    digest = sha256_of_file(src)
    matches = [entry["tag"] for entry in registry.get("entries", []) if entry["sha256"] == digest]
    print(f"Pesos activos : {src}")
    print(f"sha256        : {digest}")
    print(f"Tag registrado: {', '.join(matches) if matches else 'NINGUNO (no registrado)'}")
    return 0


def cmd_promote(args) -> int:
    registry = _load()
    candidate = Path(args.candidate) / "model.safetensors"
    if not candidate.is_file():
        print(f"ERROR: no existe {candidate}")
        return 2

    active = _active_weights(args)
    digest = sha256_of_file(candidate)
    if any(entry["sha256"] == digest for entry in registry.get("entries", [])) and not args.force:
        print(f"ERROR: ese checkpoint ya está registrado con el mismo sha256 ({digest[:12]}…)")
        return 2

    backup_tag = f"auto-antes-de-{args.tag}"
    backup_path = SNAPSHOTS_DIR / backup_tag / "model.safetensors"
    print(f"1) Copia de seguridad del estado actual → {backup_tag}")
    backup_digest = _copy_verified(active, backup_path, expected_sha=None, force=True)
    _record(registry, tag=backup_tag, source="auto-backup", sha256=backup_digest, path=backup_path)

    print(f"2) Promoviendo {candidate} → {active}")
    written = _copy_verified(candidate, active, expected_sha=digest, force=args.force)
    snapshot = SNAPSHOTS_DIR / args.tag / "model.safetensors"
    _copy_verified(active, snapshot, expected_sha=written, force=True)
    _record(registry, tag=args.tag, source="promote", sha256=written, path=snapshot, notes=args.notes)
    _history(registry, "promote", tag=args.tag, sha256=written, backup=backup_tag)
    _save(registry)

    print(f"OK: activo = {args.tag} ({written[:12]}…). Para revertir:")
    print(f"    python scripts/model_registry.py rollback --tag {backup_tag}")
    return 0


def cmd_rollback(args) -> int:
    registry = _load()
    entry = _entry(registry, args.tag)
    if entry is None:
        print(f"ERROR: no hay ningún estado registrado con el tag '{args.tag}'")
        return 2

    stored = ROOT / entry["snapshot_path"]
    active = _active_weights(args)
    print(f"Restaurando {args.tag} ({entry['sha256'][:12]}…) → {active}")
    written = _copy_verified(stored, active, expected_sha=entry["sha256"], force=args.force)
    _record(registry, tag=args.tag, source="rollback", sha256=written, path=stored, notes=entry.get("notes", ""))
    _history(registry, "rollback", tag=args.tag, sha256=written)
    _save(registry)
    print("OK: pesos restaurados. Comprueba con:")
    print("    python scripts/baseline.py verify --name baseline-2026-09-15")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--weights-dir", default=str(WEIGHTS_DIR), help="directorio de pesos activos")
    common.add_argument("--force", action="store_true", help="saltarse las comprobaciones de hash")

    snapshot = sub.add_parser("snapshot", parents=[common], help="congelar los pesos activos")
    snapshot.add_argument("--tag", required=True)
    snapshot.add_argument("--notes", default="")

    listing = sub.add_parser("list", parents=[common], help="listar estados registrados")
    listing.add_argument("--history", action="store_true", help="incluir el historial de acciones")

    sub.add_parser("current", parents=[common], help="qué pesos hay activos")

    promote = sub.add_parser("promote", parents=[common], help="promover un candidato entrenado")
    promote.add_argument("--candidate", required=True, help="directorio con model.safetensors")
    promote.add_argument("--tag", required=True)
    promote.add_argument("--notes", default="")

    rollback = sub.add_parser("rollback", parents=[common], help="volver a un estado registrado")
    rollback.add_argument("--tag", required=True)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    commands = {
        "snapshot": cmd_snapshot,
        "list": cmd_list,
        "current": cmd_current,
        "promote": cmd_promote,
        "rollback": cmd_rollback,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
