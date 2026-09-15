#!/usr/bin/env python3
"""Benchmark de proveedores de segmentación del fork (plan comercial §14).

Compara, sobre los mismos pares persona/prenda, cada proveedor registrado:

    A = none (camino comercial)         B = pose-heuristic (experimental)
    C = sam2                            D = grounded-sam2        E = custom-parser

Mide por corrida: segundos, pico de VRAM (torch.cuda), pico de RSS (psutil) y
estado (ok / degradado / no disponible / error), y guarda la imagen resultante.

IMPORTANTE (licencias): úsalo con **imágenes propias o con licencia comercial**.
El plan exige un benchmark de 100 tops + 100 bottoms + 100 one-pieces antes de
producción; este script solo aporta el arnés. El paquete de ejemplos del repo
(Apache-2.0) sirve para una prueba de humo.

Uso:
    # Prueba de humo con los ejemplos del repo, todos los proveedores
    python scripts/benchmark_providers.py --examples --providers none,pose-heuristic

    # Con tu propio CSV: person_path,garment_path,category[,garment_photo_type]
    python scripts/benchmark_providers.py --pairs pares.csv --providers none --num-timesteps 30

Salidas (en --output-dir, por defecto outputs/benchmark):
    <provider>/<pair>_<n>.png     imágenes generadas
    benchmark.csv                 una fila por corrida
    benchmark_summary.json        medias por proveedor
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fashn_vton.segmentation import PROVIDERS, available_providers, build_segmentation_provider  # noqa: E402


def _memory() -> tuple[float | None, float | None]:
    """(RSS del proceso en GiB, VRAM reservada en GiB) best-effort."""
    rss = vram = None
    try:
        import psutil

        rss = psutil.Process().memory_info().rss / 1024**3
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_reserved() / 1024**3
    except Exception:  # noqa: BLE001
        pass
    return rss, vram


def _reset_vram_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


#: Métricas de calidad (plan §14) que se añaden al CSV y al resumen.
METRIC_FIELDS = (
    "identity_preservation",
    "outside_change",
    "color_fidelity",
    "pattern_fidelity",
    "garment_change",
    "sharpness",
)


def load_pairs(args) -> list[dict]:
    if args.examples or not args.pairs:
        return [
            {
                "name": "examples",
                "person": str(REPO_ROOT / "examples" / "data" / "model.webp"),
                "garment": str(REPO_ROOT / "examples" / "data" / "garment.webp"),
                "category": "tops",
                "garment_photo_type": "flat-lay",
            }
        ]
    pairs = []
    with open(args.pairs, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pairs.append(
                {
                    "name": Path(row["person"]).stem,
                    "person": row["person"],
                    "garment": row["garment"],
                    "category": row.get("category") or "tops",
                    "garment_photo_type": row.get("garment_photo_type") or "flat-lay",
                }
            )
    if args.limit:
        pairs = pairs[: args.limit]
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark de proveedores de segmentación (plan §14).")
    parser.add_argument("--pairs", type=Path, default=None, help="CSV: person,garment[,category[,garment_photo_type]]")
    parser.add_argument("--examples", action="store_true", help="Usar el par de ejemplos del repo (prueba de humo)")
    parser.add_argument("--providers", type=str, default="none,pose-heuristic")
    parser.add_argument("--weights-dir", type=str, default=str(REPO_ROOT / "weights"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "benchmark")
    parser.add_argument("--num-timesteps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Calcular métricas de calidad por corrida (plan §14) y veredicto del gate",
    )
    parser.add_argument(
        "--metrics-provider",
        type=str,
        default="pose-heuristic",
        help="Proveedor usado SOLO para definir las regiones de las métricas (constante entre corridas)",
    )
    args = parser.parse_args()

    from PIL import Image

    from fashn_vton import TryOnPipeline

    providers = [name.strip().lower() for name in args.providers.split(",") if name.strip()]
    unknown = [name for name in providers if name not in PROVIDERS]
    if unknown:
        parser.error(f"proveedores desconocidos: {unknown}. Disponibles: {sorted(PROVIDERS)}")

    pairs = load_pairs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = available_providers()

    metrics_provider = None
    if args.metrics:
        try:
            from fashn_vton.eval import aggregate, evaluate, passes_quality_gate
        except ModuleNotFoundError as exc:  # scikit-image/scipy
            print(
                f"ERROR: faltan dependencias para las métricas ({exc.name}).\n"
                "  Instala el extra:  pip install -e '.[metrics]'"
            )
            return 2

        metrics_provider = build_segmentation_provider(args.metrics_provider)
        if not metrics_provider.is_available():
            print(
                f"[aviso] el proveedor de métricas '{args.metrics_provider}' no está disponible: "
                "las métricas de región saldrán vacías (se usarán solo las globales)."
            )
            metrics_provider = build_segmentation_provider("none")

    rows: list[dict] = []
    for name in providers:
        if not report.get(name, {}).get("available", False):
            rows.append(
                {
                    "provider": name,
                    "pair": "-",
                    "status": "unavailable",
                    "seconds": None,
                    "vram_gib": None,
                    "rss_gib": None,
                    "error": (report.get(name, {}).get("notes") or "no disponible").splitlines()[0],
                }
            )
            print(f"[{name}] no disponible en este entorno (se registra y se continúa)")
            continue

        pipeline = TryOnPipeline(
            weights_dir=args.weights_dir,
            device=args.device,
            segmentation_provider=build_segmentation_provider(name),
        )
        provider_dir = args.output_dir / name
        provider_dir.mkdir(parents=True, exist_ok=True)

        for pair in pairs:
            person = Image.open(pair["person"]).convert("RGB")
            garment = Image.open(pair["garment"]).convert("RGB")
            _reset_vram_peak()
            started = time.time()
            status, error = "ok", None
            metrics: dict = {}
            try:
                class_map = None
                if metrics_provider is not None:
                    person_np = np.asarray(person)
                    pose = pipeline.pose_model(person_np[..., ::-1])
                    class_map = metrics_provider.predict(
                        person_np,
                        {"pose": pose, "category": pair["category"], "role": "person"},
                    )

                result = pipeline(
                    person_image=person,
                    garment_image=garment,
                    category=pair["category"],
                    garment_photo_type=pair["garment_photo_type"],
                    num_timesteps=args.num_timesteps,
                    seed=args.seed,
                )
                degraded = (result.metadata or {}).get("segmentation", {}).get("degraded") or []
                status = "degraded" if degraded else "ok"
                for index, image in enumerate(result.images):
                    image.save(provider_dir / f"{pair['name']}_{index:02d}.png")

                if metrics_provider is not None and result.images:
                    output = result.images[0]
                    person_resized = np.asarray(person.resize(output.size, Image.LANCZOS))
                    garment_ref = np.asarray(garment.resize(output.size, Image.LANCZOS))
                    class_map_resized = None
                    if class_map is not None:
                        class_map_resized = np.asarray(Image.fromarray(class_map).resize(output.size, Image.NEAREST))
                    metrics = evaluate(person_resized, np.asarray(output), garment_ref, class_map_resized)
            except Exception as exc:  # noqa: BLE001 - se registra y se sigue
                status, error = "error", f"{type(exc).__name__}: {exc}"
                metrics = {"failure": True}
            seconds = round(time.time() - started, 2)
            rss, vram = _memory()
            row = {
                "provider": name,
                "pair": pair["name"],
                "status": status,
                "seconds": seconds,
                "vram_gib": round(vram, 2) if vram else None,
                "rss_gib": round(rss, 2) if rss else None,
                "error": error,
            }
            for field_name in METRIC_FIELDS:
                value = metrics.get(field_name)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    row[field_name] = None
                else:
                    row[field_name] = round(float(value), 4)
            rows.append(row)
            print(
                f"[{name}] {pair['name']}: {status} {seconds}s vram={row['vram_gib']}GiB"
                + (f" error={error}" if error else "")
                + (f" color={row.get('color_fidelity')}" if args.metrics else "")
            )
        del pipeline

    csv_path = args.output_dir / "benchmark.csv"
    fieldnames = ["provider", "pair", "status", "seconds", "vram_gib", "rss_gib", *METRIC_FIELDS, "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict] = {}
    for name in providers:
        provider_rows = [row for row in rows if row["provider"] == name and row["status"] in {"ok", "degraded"}]
        seconds = [row["seconds"] for row in provider_rows if row["seconds"] is not None]
        summary[name] = {
            "runs": len(provider_rows),
            "errors": sum(1 for row in rows if row["provider"] == name and row["status"] == "error"),
            "degraded": sum(1 for row in rows if row["provider"] == name and row["status"] == "degraded"),
            "seconds_mean": round(sum(seconds) / len(seconds), 2) if seconds else None,
            "vram_gib_max": max(
                (row["vram_gib"] for row in provider_rows if row["vram_gib"] is not None), default=None
            ),
            "rss_gib_max": max((row["rss_gib"] for row in provider_rows if row["rss_gib"] is not None), default=None),
            "available": report.get(name, {}).get("available", False),
        }
        if args.metrics:
            metric_rows = [
                {**{field: row.get(field) for field in METRIC_FIELDS}, "failure": row["status"] == "error"}
                for row in rows
                if row["provider"] == name
            ]
            summary[name]["quality"] = aggregate(metric_rows)
            summary[name]["passes_gate"] = bool(summary[name]["quality"].get("passes_gate"))

    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nCSV:     {csv_path}")
    print(f"Resumen: {args.output_dir / 'benchmark_summary.json'}")
    for name, data in summary.items():
        print(f"  {name:>16}: {data}")
        if args.metrics and "quality" in data:
            quality = data["quality"]
            ok, reasons = passes_quality_gate({**quality, "failure": False})
            print(
                f"                   gate={'PASA' if data['passes_gate'] else 'NO PASA'} "
                f"score={quality['score']:.3f} color={quality['color_fidelity']} "
                f"identidad={quality['identity_preservation']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
