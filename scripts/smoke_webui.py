#!/usr/bin/env python3
"""Prueba de humo de la UI web (Gradio) contra el servidor en marcha.

No carga modelos por su cuenta: usa el servidor de `./run_web.sh` y comprueba
las tres promesas del fork comercial:

    1. El endpoint /tryon responde y el log dice qué proveedor y licencia se usó.
    2. El **camino comercial** (flat-lay + segmentation_free + proveedor none)
       reproduce byte a byte la imagen de referencia documentada en
       plan_modelo_comercial/PLAN_IMPLEMENTACION_COMERCIAL.md (fila 2). Se compara
       el PNG guardado por la UI, no la copia WebP que sirve la galería.
    3. Con una prenda tipo `model` y proveedor `none` la UI avisa de la
       degradación; con `sam2` (Apache-2.0) ya hay máscara y no hay aviso.

Uso:
    # Camino comercial exacto de la referencia (8 pasos, semilla 42)
    python scripts/smoke_webui.py

    # Comparar proveedores y ver el aviso de degradación en el log de la UI
    python scripts/smoke_webui.py --providers none sam2 --garment-photo-type model

    # Otro servidor / otra configuración
    python scripts/smoke_webui.py --url http://127.0.0.1:7863/ --steps 30

Requiere la UI levantada (`./run_web.sh`) y `gradio_client` instalado.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSON = REPO_ROOT / "examples" / "data" / "model.webp"
DEFAULT_GARMENT = REPO_ROOT / "examples" / "data" / "garment.webp"

#: Imagen de referencia del camino comercial pre-fork (ver PLAN_IMPLEMENTACION_COMERCIAL.md).
COMMERCIAL_REFERENCE = {
    "sha256": "a4d618bf4d50910374caf3770ee580d9a7df4b7c9a65d4a51831033ea631c53e",
    "category": "tops",
    "garment_photo_type": "flat-lay",
    "num_timesteps": 8,
    "guidance_scale": 1.5,
    "seed": 42,
    "segmentation_free": True,
    "provider": "none",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reference_run(settings: dict) -> bool:
    """¿Esta corrida usa exactamente la configuración de la referencia documentada?"""
    keys = ("category", "garment_photo_type", "num_timesteps", "guidance_scale", "seed", "provider")
    return all(settings[key] == COMMERCIAL_REFERENCE[key] for key in keys)


def _run(client, person: Path, garment: Path, settings: dict) -> tuple[str, str]:
    """Una petición a /tryon; devuelve (ruta de la imagen, log de la UI).

    El endpoint puede devolver más de dos salidas (p. ej. la previsualización), así
    que se leen por posición en lugar de desempaquetar.
    """
    from gradio_client import handle_file

    response = client.predict(
        handle_file(str(person)),
        handle_file(str(garment)),
        settings["category"],
        settings["garment_photo_type"],
        settings["num_samples"],
        settings["num_timesteps"],
        settings["guidance_scale"],
        settings["seed"],
        settings["segmentation_free"],
        settings["provider"],
        settings["device"],
        api_name="/tryon",
    )
    outputs = list(response) if isinstance(response, (tuple, list)) else [response]
    gallery, log_text = outputs[0], outputs[1]
    images = gallery if isinstance(gallery, list) else [gallery]
    first = images[0]
    image_path = first.get("image") if isinstance(first, dict) else first
    return str(image_path), str(log_text)


def _saved_png(log_text: str) -> str | None:
    """Ruta del PNG que guarda la UI (línea `Guardado: …` del log).

    La galería de Gradio entrega una **copia re-codificada a WebP** en /tmp, así que
    para comparar hashes hay que usar el PNG original guardado en outputs/.
    """
    for line in log_text.splitlines():
        line = line.strip()
        # Las líneas del log van numeradas ("7. Guardado: …"): buscar por contenido.
        if "Guardado:" in line:
            candidate = line.split("Guardado:", 1)[1].strip()
            if Path(candidate).is_file():
                return candidate
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:7863/", help="URL de la UI (por defecto: %(default)s)")
    parser.add_argument("--providers", nargs="+", default=["none"], help="proveedores a probar (por defecto: none)")
    parser.add_argument("--person", type=Path, default=DEFAULT_PERSON, help="imagen de la persona")
    parser.add_argument("--garment", type=Path, default=DEFAULT_GARMENT, help="imagen de la prenda")
    parser.add_argument("--category", default="tops", choices=["tops", "bottoms", "one-pieces"])
    parser.add_argument(
        "--garment-photo-type",
        default="flat-lay",
        choices=["model", "flat-lay"],
        help="tipo de foto de la prenda",
    )
    parser.add_argument("--steps", type=int, default=8, help="pasos de difusión")
    parser.add_argument("--guidance", type=float, default=1.5, help="guidance scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--no-segmentation-free",
        action="store_true",
        help="desactiva segmentation_free (por defecto va activado, como el camino comercial)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    for path in (args.person, args.garment):
        if not path.is_file():
            print(f"ERROR: no existe la imagen {path}", file=sys.stderr)
            return 2

    try:
        from gradio_client import Client
    except ImportError:
        print("ERROR: falta gradio_client (pip install gradio_client)", file=sys.stderr)
        return 2

    print(f"Conectando a {args.url} …")
    try:
        client = Client(args.url)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: no se pudo conectar a la UI ({exc}). ¿Está corriendo ./run_web.sh?", file=sys.stderr)
        return 2

    failures = 0
    for provider in args.providers:
        settings = {
            "category": args.category,
            "garment_photo_type": args.garment_photo_type,
            "num_samples": args.samples,
            "num_timesteps": args.steps,
            "guidance_scale": args.guidance,
            "seed": args.seed,
            "segmentation_free": not args.no_segmentation_free,
            "provider": provider,
            "device": args.device,
        }
        print(f"\n===== proveedor={provider} · prenda={settings['garment_photo_type']} =====")
        try:
            image_path, log_text = _run(client, args.person, args.garment, settings)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR en la petición: {exc}")
            failures += 1
            continue

        for line in log_text.splitlines():
            print(f"  {line}")

        if "Segmentación: proveedor=" not in log_text:
            print("  FALLO: el log no reporta el proveedor de segmentación")
            failures += 1

        # La galería entrega un WebP re-codificado: para el hash usamos el PNG guardado.
        artifact = _saved_png(log_text)
        if artifact is None:
            artifact = image_path
            print("  AVISO: no se encontró la línea 'Guardado:' ; se compara la copia WebP de la galería")

        if _is_reference_run(settings):
            digest = _sha256(artifact)
            same = digest == COMMERCIAL_REFERENCE["sha256"]
            print(f"  archivo: {artifact}")
            print(f"  sha256:  {digest}")
            print(f"  {'OK' if same else 'FALLO'}: idéntico a la referencia pre-fork = {same}")
            failures += 0 if same else 1
        else:
            print(f"  archivo: {artifact}")
            print(f"  sha256:  {_sha256(artifact)} (sin referencia para esta configuración, informativo)")

        if settings["garment_photo_type"] == "model":
            degraded = "AVISO (degradado" in log_text
            expected = provider in {"none", "pose-heuristic"}
            ok = degraded is expected
            print(f"  {'OK' if ok else 'FALLO'}: aviso de degradación esperado={expected} · presente={degraded}")
            failures += 0 if ok else 1

    print(f"\nResultado: {'TODO OK' if failures == 0 else f'{failures} comprobación(es) fallida(s)'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
