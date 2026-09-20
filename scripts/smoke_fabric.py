#!/usr/bin/env python3
"""Prueba de humo de la «tela propia» contra la UI en marcha (API de Gradio).

Comprueba las dos promesas que se rompieron antes y que ahora cubre la regla
`webui._should_apply_fabric`:

1. La **previsualización** (`/preview_fabric`) aplica la tela siempre, incluso si
   la casilla «Generar con la tela aplicada» está apagada.
2. El **try-on** (`/tryon`) usa la prenda con la tela cuando hay tela configurada:
   el log dice `tela: aplicada`, guarda `..._prenda_con_tela.png` y el PNG del
   resultado **difiere** de la referencia original. Con la casilla desactivada,
   avisa en mayúsculas y el resultado vuelve a ser exactamente la referencia
   (`sha256 a4d618bf…`, el invariante del camino comercial).

Uso:
    ./run_fashn_vton.sh python scripts/smoke_fabric.py
    ./run_fashn_vton.sh python scripts/smoke_fabric.py --url http://127.0.0.1:7863/ --steps 8

Código de salida: 0 = todo OK · 1 = alguna comprobación falló · 2 = error previo.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FABRIC_PATH = ROOT / "outputs" / "smoke_fabric" / "tela_cuadros.png"
PERSON = ROOT / "examples" / "data" / "model.webp"
GARMENT = ROOT / "examples" / "data" / "garment.webp"

#: Salida del camino comercial con los pesos originales (plan comercial, fila 2).
REFERENCE_SHA256 = "a4d618bf4d50910374caf3770ee580d9a7df4b7c9a65d4a51831033ea631c53e"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_fabric(path: Path) -> Path:
    """Tela sintética de cuadros (se genera aquí: no depende de ficheros externos)."""
    size = 256
    tile = np.zeros((size, size, 3), np.uint8)
    tile[..., 0] = 180
    tile[..., 1] = 40
    tile[..., 2] = 40
    tile[::32, :] = (250, 250, 250)
    tile[:, ::32] = (250, 250, 250)
    tile[::64, :] = (40, 60, 160)
    tile[:, ::64] = (40, 60, 160)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tile).save(path)
    return path


def saved_pngs(log_text: str) -> list[str]:
    """Rutas de los PNG que la UI dice haber guardado (líneas `Guardado: …`)."""
    paths: list[str] = []
    for line in log_text.splitlines():
        if "Guardado:" not in line:
            continue
        candidate = line.split("Guardado:", 1)[1].strip()
        if Path(candidate).is_file():
            paths.append(candidate)
    return paths


def fabric_args(fabric_image, enabled: bool) -> list:
    """Argumentos de tela en el orden exacto de la firma de la UI (sin la cuadrícula)."""
    return [
        fabric_image,  # fabric_image
        enabled,  # fabric_enabled
        "#ffffff",  # fabric_color
        "blanco",  # fabric_bg_mode
        "#ffffff",  # fabric_bg_color
        "auto",  # fabric_mask_source
        30.0,  # fabric_tolerance
        1.0,  # fabric_front_only
        1.0,  # fabric_scale
        0.0,  # fabric_rotation
        0.0,  # fabric_tilt_x
        0.0,  # fabric_tilt_y
        0.0,  # fabric_offset_x
        0.0,  # fabric_offset_y
        0.0,  # fabric_perspective
        0.0,  # fabric_angle
        1.0,  # fabric_brightness
        1.0,  # fabric_strength
        1.0,  # fabric_shading
        0.0,  # fabric_detail
        "repetir",  # fabric_mosaic
        "cm",  # fabric_repeat_mode
        12.0,  # fabric_repeat_cm
        50.0,  # fabric_garment_width_cm
        120.0,  # fabric_repeat_px
    ]


def _indent(text: str, prefix: str = "     ") -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines() if line.strip())


def _result_and_garment(log_text: str) -> tuple[str | None, str | None]:
    saved = saved_pngs(log_text)
    garment = next((path for path in saved if Path(path).stem.endswith("prenda_con_tela")), None)
    result = next((path for path in saved if not Path(path).stem.endswith("prenda_con_tela")), None)
    return result, garment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:7863/", help="URL de la UI (por defecto: %(default)s)")
    parser.add_argument("--steps", type=int, default=8, help="pasos de difusión (8 = rápido)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--person", type=Path, default=PERSON)
    parser.add_argument("--garment", type=Path, default=GARMENT)
    args = parser.parse_args(argv)

    for path in (args.person, args.garment):
        if not path.is_file():
            print(f"ERROR: no existe {path}", file=sys.stderr)
            return 2

    try:
        from gradio_client import Client, handle_file
    except ImportError:
        print("ERROR: falta gradio_client (pip install gradio_client)", file=sys.stderr)
        return 2

    fabric = make_fabric(FABRIC_PATH)
    print(f"tela sintética: {fabric}")
    try:
        client = Client(args.url)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: no se pudo conectar a la UI ({exc}). ¿Está corriendo ./run_web.sh?", file=sys.stderr)
        return 2

    failures = 0

    # 1) Previsualización con la casilla APAGADA: debe aplicar la tela igualmente.
    _image, log_preview = client.predict(
        handle_file(args.garment),
        *fabric_args(handle_file(fabric), False),
        False,
        api_name="/preview_fabric",
    )
    ok_preview = "tela: aplicada" in log_preview
    print(f"\n1) Previsualización con la casilla APAGADA: {'OK' if ok_preview else 'FALLO'}")
    print(_indent(log_preview))
    failures += 0 if ok_preview else 1

    # 2) Try-on CON tela: el log lo dice, guarda la prenda usada y el resultado cambia.
    _gallery, log_with, _preview = client.predict(
        handle_file(args.person),
        handle_file(args.garment),
        "tops",
        "flat-lay",
        1,
        args.steps,
        1.5,
        args.seed,
        True,
        "none",
        "auto",
        *fabric_args(handle_file(fabric), True),
        api_name="/tryon",
    )
    result_with, garment_png = _result_and_garment(log_with)
    digest_with = sha256(result_with) if result_with else "?"
    ok_applied = "tela: aplicada" in log_with and garment_png is not None and digest_with != REFERENCE_SHA256
    print(f"\n2) Try-on CON tela: {'OK' if ok_applied else 'FALLO'}")
    print(_indent("\n".join(line for line in log_with.splitlines() if "tela" in line.lower() or "Guardado" in line)))
    print(_indent(f"resultado  : {digest_with[:16]}… (referencia {REFERENCE_SHA256[:16]}…)"))
    print(_indent(f"prenda usada: {Path(garment_png).name if garment_png else '-'}"))
    failures += 0 if ok_applied else 1

    # 3) Try-on SIN tela (casilla desactivada): avisa y vuelve a la referencia.
    _gallery2, log_without, _preview2 = client.predict(
        handle_file(args.person),
        handle_file(args.garment),
        "tops",
        "flat-lay",
        1,
        args.steps,
        1.5,
        args.seed,
        True,
        "none",
        "auto",
        *fabric_args(handle_file(fabric), False),
        api_name="/tryon",
    )
    result_without, _garment2 = _result_and_garment(log_without)
    digest_without = sha256(result_without) if result_without else "?"
    warned = "AVISO: hay tela/color configurado" in log_without
    ok_original = warned and digest_without == REFERENCE_SHA256
    print(f"\n3) Try-on SIN tela (casilla apagada): {'OK' if ok_original else 'FALLO'}")
    print(_indent("\n".join(line for line in log_without.splitlines() if "AVISO" in line)))
    print(_indent(f"resultado  : {digest_without[:16]}… (¿es la referencia? {digest_without == REFERENCE_SHA256})"))
    failures += 0 if ok_original else 1

    ok_different = digest_with != digest_without
    print(f"\n4) Las dos generaciones difieren entre sí: {'OK' if ok_different else 'FALLO'}")
    failures += 0 if ok_different else 1

    print(f"\nResultado: {'TODO OK' if failures == 0 else f'{failures} comprobación(es) fallida(s)'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
