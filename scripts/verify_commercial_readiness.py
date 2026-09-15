#!/usr/bin/env python3
"""Gate de preparación comercial del fork FASHN VTON v1.5.

Comprueba, en un solo comando, todo lo que el plan comercial exige antes de
publicar o desplegar:

1. Cumplimiento de fuentes y entorno (``fashn_vton.compliance``): imports,
   dependencias de ``pyproject.toml``, distribuciones instaladas, artefactos
   prohibidos presentes en el árbol.
2. Hashes de los pesos descargados contra ``licenses/manifest.json``.
3. Existencia de los archivos de licencias obligatorios.
4. Estado de los proveedores de segmentación (licencia comercial y
   disponibilidad real en este entorno).

Uso:
    python scripts/verify_commercial_readiness.py
    python scripts/verify_commercial_readiness.py --json
    python scripts/verify_commercial_readiness.py --root /ruta/al/repo

Códigos de salida:
    0 = listo (puede haber avisos, p. ej. pesos no descargados)
    1 = fallo de cumplimiento (bloqueante)
    2 = artefacto prohibido presente (bloqueante)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fashn_vton import compliance  # noqa: E402
from fashn_vton.segmentation import available_providers  # noqa: E402

REQUIRED_LICENSE_FILES = [
    "licenses/manifest.json",
    "licenses/verify_hashes.py",
    "licenses/Apache-2.0.txt",
    "licenses/README.md",
    "THIRD_PARTY_NOTICES.md",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Gate de preparación comercial (FASHN VTON fork).")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true", help="Salida JSON.")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    report = compliance.commercial_report(root)

    missing_files = [name for name in REQUIRED_LICENSE_FILES if not (root / name).is_file()]
    providers = available_providers()
    non_commercial = [name for name, info in providers.items() if not info.get("commercial_ok", False)]
    forbidden_present = bool(report.checked.get("forbidden_files", {}).get("violations"))

    payload = {
        "root": str(root),
        "ok": report.ok and not missing_files,
        "violations": report.violations,
        "warnings": report.warnings,
        "missing_license_files": missing_files,
        "hashes": report.checked.get("hashes", []),
        "providers": providers,
        "providers_non_commercial": non_commercial,
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Gate comercial · root={root}\n")
        print(f"1) Cumplimiento de código/entorno: {'OK' if report.ok else 'FALLO'}")
        for violation in report.violations:
            print(f"   - VIOLACIÓN: {violation}")
        for warning in report.warnings:
            print(f"   - aviso: {warning}")

        print("\n2) Archivos de licencias:")
        for name in REQUIRED_LICENSE_FILES:
            status = "OK" if (root / name).is_file() else "FALTA"
            print(f"   [{status:>5}] {name}")

        print("\n3) Hashes de pesos:")
        for row in report.checked.get("hashes", []):
            if row["status"] != "NO_PATH":
                print(f"   [{row['status']:>10}] {row['name']}")

        print("\n4) Proveedores de segmentación:")
        for name, info in providers.items():
            flags = []
            if not info.get("commercial_ok", False):
                flags.append("NO COMERCIAL")
            if info.get("experimental"):
                flags.append("experimental")
            if not info.get("available", False):
                flags.append("no disponible")
            suffix = f" ({', '.join(flags)})" if flags else ""
            print(f"   {name:>16}: {info.get('license')}{suffix}")

        print(
            f"\nRESULTADO: {'LISTO PARA COMERCIAL' if payload['ok'] else 'NO LISTO'} "
            f"| proveedores no comerciales: {non_commercial or 'ninguno'}"
        )

    if forbidden_present:
        return 2
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
