#!/usr/bin/env python3
"""Verificador de licencias y hashes de terceros (fork comercial FASHN VTON v1.5).

Solo librería estándar: se puede ejecutar antes de instalar el paquete o dentro
de una imagen de producción sin dependencias.

Uso:
    python licenses/verify_hashes.py                    # verifica (hashes + artefactos prohibidos)
    python licenses/verify_hashes.py --capture          # imprime sha256 de los archivos hallados
    python licenses/verify_hashes.py --root ./weights   # raíz alternativa para local_path
    python licenses/verify_hashes.py --json             # salida JSON

Códigos de salida:
    0 = todo correcto (puede haber MISSING/UNVERIFIED, que se avisan)
    1 = MISMATCH (hash distinto) o error de manifiesto
    2 = artefacto PROHIBIDO presente (bloqueante comercial)
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_DEFAULT = HERE / "manifest.json"
ROOT_DEFAULT = HERE.parent
_CHUNK = 1024 * 1024
_SKIP_PARTS = {".git", "__pycache__", ".pytest_cache"}


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def verify(manifest: dict, root: Path) -> list[dict]:
    results = []
    for artifact in manifest.get("artifacts", []):
        name = artifact.get("name", "<sin nombre>")
        expected = artifact.get("sha256")
        relative = artifact.get("local_path")
        if not relative:
            results.append({"name": name, "status": "NO_PATH", "detail": "librería (sin archivo)"})
            continue
        path = root / relative
        if not path.is_file():
            results.append({"name": name, "status": "MISSING", "detail": str(path)})
            continue
        actual = sha256_of_file(path)
        if not expected:
            results.append({"name": name, "status": "UNVERIFIED", "detail": actual})
        elif actual.lower() == str(expected).lower():
            results.append({"name": name, "status": "OK", "detail": actual})
        else:
            results.append(
                {"name": name, "status": "MISMATCH", "detail": f"esperado={expected} obtenido={actual}"}
            )
    return results


def capture(manifest: dict, root: Path) -> dict[str, str]:
    captured = {}
    for artifact in manifest.get("artifacts", []):
        relative = artifact.get("local_path")
        if not relative:
            continue
        path = root / relative
        if path.is_file():
            captured[artifact.get("name")] = sha256_of_file(path)
    return captured


def scan_forbidden(manifest: dict, root: Path) -> list[str]:
    """Artefactos prohibidos presentes bajo `root` (bloqueante comercial)."""
    found = []
    if not root.exists():
        return found
    for path in root.rglob("*"):
        if any(part in _SKIP_PARTS for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        for artifact in manifest.get("forbidden_artifacts", []):
            for pattern in artifact.get("match", []):
                candidates = {pattern, pattern.replace("**/", ""), f"**/{pattern.lstrip('*/')}"}
                if any(fnmatch.fnmatch(relative, candidate) for candidate in candidates) or any(
                    fnmatch.fnmatch(path.name.lower(), candidate.lstrip("*").lower())
                    for candidate in candidates
                ):
                    found.append(f"{relative} (prohibido: {artifact.get('name')})")
                    break
    return sorted(set(found))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verifica licencias/hashes de terceros (gate comercial).")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--capture", action="store_true", help="Imprime sha256 de los archivos hallados.")
    parser.add_argument("--json", action="store_true", help="Salida JSON.")
    parser.add_argument("--no-forbidden-scan", action="store_true", help="No buscar artefactos prohibidos.")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)

    if args.capture:
        print(json.dumps(capture(manifest, args.root), indent=2, ensure_ascii=False))
        return 0

    results = verify(manifest, args.root)
    forbidden = [] if args.no_forbidden_scan else scan_forbidden(manifest, args.root)

    if args.json:
        print(json.dumps({"artifacts": results, "forbidden_present": forbidden}, indent=2, ensure_ascii=False))
    else:
        print(f"Manifiesto: {args.manifest}")
        print(f"Raíz:       {args.root}\n")
        for row in results:
            print(f"  [{row['status']:>10}] {row['name']}: {row['detail']}")
        print(f"\nArtefactos prohibidos registrados: {len(manifest.get('forbidden_artifacts', []))}")
        if forbidden:
            print("  ¡PROHIBIDOS PRESENTES!")
            for item in forbidden:
                print(f"    - {item}")
        else:
            print("  Ninguno presente bajo la raíz analizada.")

    if forbidden:
        return 2
    failed = [row for row in results if row["status"] == "MISMATCH"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
