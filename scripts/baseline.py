#!/usr/bin/env python3
"""Baseline inmutable del modelo comercial: capturar y verificar.

Protege «lo que ya funciona»: guarda una foto fija del modelo (pesos, salidas,
métricas y entorno) y, después de cualquier fine-tuning, dice con números si el
candidato mejora o **empeora**. Si empeora, no se promueve (ver
`plan_entrenamiento/PLAN_FINE_TUNING.md`).

    capture  Genera un conjunto fijo (pares × semillas) con el camino comercial y
             guarda PNG + sha256 + métricas + huella del entorno en
             `baselines/<name>/manifest.json` (el JSON se commitea; los PNG quedan
             en disco para revisión humana).
    verify   Vuelve a generar (con otro `--weights-dir` si se quiere), compara
             hashes, similitud contra las imágenes de referencia y métricas, y
             aplica la guardia anti-regresión (`fashn_vton.eval.compare_metrics`).

Código de salida de `verify`: 0 = no empeora · 1 = regresión · 2 = error.

Ejemplos:
    # 1) Congelar el estado bueno (antes de entrenar nada)
    ./run_fashn_vton.sh python scripts/baseline.py capture --name baseline-2026-09-15

    # 2) Con tu catálogo (CSV: person,garment[,category[,garment_photo_type]])
    ./run_fashn_vton.sh python scripts/baseline.py capture --name baseline-propio \\
        --pairs pares.csv --seeds 42,7,1234

    # 3) ¿Empeora el candidato? (fine-tuning, LoRA mergeado, otra versión de torch)
    ./run_fashn_vton.sh python scripts/baseline.py verify --name baseline-2026-09-15 \\
        --weights-dir weights/candidates/lora-v1

La generación usa SIEMPRE el camino comercial (`flat-lay` + `segmentation_free` +
proveedor `none`). `--metrics-provider` se usa **solo** para el mapa de clases con
el que se miden las regiones: no interviene en la generación.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashn_vton import TryOnPipeline  # noqa: E402
from fashn_vton.compliance import sha256_of_file  # noqa: E402
from fashn_vton.eval import aggregate, compare_metrics, evaluate  # noqa: E402
from fashn_vton.segmentation import build_segmentation_provider  # noqa: E402

DEFAULT_PERSON = ROOT / "examples" / "data" / "model.webp"
DEFAULT_GARMENT = ROOT / "examples" / "data" / "garment.webp"
BASELINES_DIR = ROOT / "baselines"
CATEGORIES = ("tops", "bottoms", "one-pieces")
PHOTO_TYPES = ("model", "flat-lay")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pairs_from_args(args) -> list[dict]:
    """Lista de pares a generar (por defecto, el ejemplo Apache-2.0 del repo)."""
    if not args.pairs:
        return [
            {
                "name": "ejemplo_tops",
                "person": str(DEFAULT_PERSON),
                "garment": str(DEFAULT_GARMENT),
                "category": "tops",
                "garment_photo_type": "flat-lay",
            }
        ]

    pairs: list[dict] = []
    with open(args.pairs, encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            person = row.get("person") or row.get("person_path")
            garment = row.get("garment") or row.get("garment_path")
            if not person or not garment:
                raise SystemExit(f"ERROR: fila {index + 2} del CSV sin 'person'/'garment': {row}")
            category = (row.get("category") or "tops").strip()
            photo_type = (row.get("garment_photo_type") or "flat-lay").strip()
            if category not in CATEGORIES or photo_type not in PHOTO_TYPES:
                raise SystemExit(f"ERROR: fila {index + 2}: category/garment_photo_type inválidos: {row}")
            pairs.append(
                {
                    "name": row.get("name") or f"par{index:03d}",
                    "person": person,
                    "garment": garment,
                    "category": category,
                    "garment_photo_type": photo_type,
                }
            )
    if not pairs:
        raise SystemExit("ERROR: el CSV no tiene pares.")
    return pairs


def _environment(pipeline) -> dict:
    """Huella del entorno: sin esto, un cambio de resultado no es interpretable."""
    import torch

    import fashn_vton

    commit = None
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
            or None
        )
    except Exception:  # noqa: BLE001 - informativo
        pass

    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "device": str(pipeline.device),
        "dtype": str(pipeline.inference_dtype),
        "package_version": getattr(fashn_vton, "__version__", None),
        "git_commit": commit,
    }


def _load_metrics_provider(name: str):
    """Proveedor usado SOLO para el mapa de clases de las métricas (no genera)."""
    provider = build_segmentation_provider(name)
    if provider.name == "none" or not provider.is_available():
        print(f"[aviso] proveedor de métricas '{name}' no disponible: las métricas de región saldrán nan")
        return None
    return provider


def _run_pair(pipeline, pair: dict, seed: int, steps: int, guidance: float):
    """Una generación del camino comercial (flat-lay + segmentation_free)."""
    person = Image.open(pair["person"]).convert("RGB")
    garment = Image.open(pair["garment"]).convert("RGB")
    started = time.time()
    result = pipeline(
        person_image=person,
        garment_image=garment,
        category=pair["category"],
        garment_photo_type=pair["garment_photo_type"],
        num_samples=1,
        num_timesteps=steps,
        guidance_scale=guidance,
        seed=seed,
        segmentation_free=True,
    )
    seconds = round(time.time() - started, 2)
    return person, garment, result.images[0], result, seconds


def _measure(pipeline, metrics_provider, person, garment, output, category: str) -> dict:
    """Métricas de calidad de una salida (mismo protocolo que benchmark_providers)."""
    class_map = None
    if metrics_provider is not None:
        person_np = np.asarray(person)
        pose = pipeline.pose_model(person_np[..., ::-1])
        class_map = metrics_provider.predict(person_np, {"pose": pose, "category": category, "role": "person"})

    person_resized = np.asarray(person.resize(output.size, Image.LANCZOS))
    garment_ref = np.asarray(garment.resize(output.size, Image.LANCZOS))
    class_map_resized = None
    if class_map is not None:
        class_map_resized = np.asarray(Image.fromarray(class_map).resize(output.size, Image.NEAREST))
    return evaluate(person_resized, np.asarray(output), garment_ref, class_map_resized)


def cmd_capture(args) -> int:
    pairs = _pairs_from_args(args)
    seeds = [int(value) for value in str(args.seeds).split(",") if value.strip()]
    out_dir = Path(args.output_dir) / args.name
    images_dir = out_dir / "images"
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists() and not args.force:
        print(f"ERROR: ya existe {manifest_path}. Usa otro --name o --force para sobrescribir.")
        return 2
    weights_file = Path(args.weights_dir) / "model.safetensors"
    if not weights_file.is_file():
        print(f"ERROR: no existe {weights_file}")
        return 2

    images_dir.mkdir(parents=True, exist_ok=True)
    metrics_provider = _load_metrics_provider(args.metrics_provider)
    pipeline = TryOnPipeline(
        weights_dir=args.weights_dir,
        device=None if args.device == "auto" else args.device,
        segmentation_provider=build_segmentation_provider("none"),
    )
    print(
        f"Baseline '{args.name}': {len(pairs)} par(es) × {len(seeds)} semilla(s) = {len(pairs) * len(seeds)} imagen(es)"
    )
    print(f"Pesos: {weights_file} ({sha256_of_file(weights_file)[:12]}…)")

    runs: list[dict] = []
    for pair in pairs:
        for seed in seeds:
            print(f"  · {pair['name']} seed={seed} …", end=" ", flush=True)
            person, garment, output, result, seconds = _run_pair(pipeline, pair, seed, args.steps, args.guidance)
            relative = f"images/{pair['name']}_s{seed}.png"
            path = out_dir / relative
            output.save(path)
            metrics = _measure(pipeline, metrics_provider, person, garment, output, pair["category"])
            runs.append(
                {
                    "pair": pair["name"],
                    "seed": seed,
                    "category": pair["category"],
                    "garment_photo_type": pair["garment_photo_type"],
                    "person": pair["person"],
                    "garment": pair["garment"],
                    "image": relative,
                    "sha256": sha256_of_file(path),
                    "seconds": seconds,
                    "degraded": (result.metadata or {}).get("segmentation", {}).get("degraded") or [],
                    "metrics": metrics,
                }
            )
            print(f"{seconds} s · sha256 {runs[-1]['sha256'][:12]}…")

    summary = aggregate([run["metrics"] for run in runs])
    manifest = {
        "name": args.name,
        "created_at": _now(),
        "generation": {
            "provider": "none",
            "segmentation_free": True,
            "num_timesteps": args.steps,
            "guidance_scale": args.guidance,
            "num_samples": 1,
            "seeds": seeds,
            "pairs_source": str(args.pairs) if args.pairs else "examples/data (Apache-2.0)",
        },
        "metrics_provider": args.metrics_provider,
        "weights": {
            "dir": str(Path(args.weights_dir).resolve()),
            "file": "model.safetensors",
            "sha256": sha256_of_file(weights_file),
            "bytes": weights_file.stat().st_size,
        },
        "environment": _environment(pipeline),
        "runs": runs,
        "summary": summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nManifest: {manifest_path}")
    print(
        f"Resumen : score={summary.get('score'):.4f} identity={summary.get('identity_preservation')} "
        f"outside={summary.get('outside_change')} color={summary.get('color_fidelity')} "
        f"sharpness={summary.get('sharpness')} gate={'OK' if summary.get('passes_gate') else 'NO'}"
    )
    return 0


def _resolve(path: str) -> Path:
    """Ruta absoluta, o relativa a la raíz del repo si no existe tal cual."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return ROOT / path


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(reference, candidate, channel_axis=-1, data_range=255.0))


def cmd_verify(args) -> int:
    baseline_dir = Path(args.output_dir) / args.name
    manifest_path = baseline_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: no existe el baseline {manifest_path}")
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_file = Path(args.weights_dir) / "model.safetensors"
    if not weights_file.is_file():
        print(f"ERROR: no existe {weights_file}")
        return 2

    candidate_hash = sha256_of_file(weights_file)
    same_weights = candidate_hash == manifest["weights"]["sha256"]
    generation = manifest["generation"]

    print(f"Baseline : {args.name} ({manifest['created_at']})")
    print(f"  pesos  : {manifest['weights']['sha256'][:12]}…")
    print(f"  actual : {candidate_hash[:12]}…  ({'IDÉNTICOS' if same_weights else 'DISTINTOS (candidato nuevo)'})")

    metrics_provider = _load_metrics_provider(manifest.get("metrics_provider", args.metrics_provider))
    pipeline = TryOnPipeline(
        weights_dir=args.weights_dir,
        device=None if args.device == "auto" else args.device,
        segmentation_provider=build_segmentation_provider("none"),
    )

    rows: list[dict] = []
    print("\n  par                    seed  sha256    ssim_base   mae   identity  outside  color  sharpness")
    for run in manifest["runs"]:
        person = Image.open(_resolve(run["person"])).convert("RGB")
        garment = Image.open(_resolve(run["garment"])).convert("RGB")
        started = time.time()
        result = pipeline(
            person_image=person,
            garment_image=garment,
            category=run["category"],
            garment_photo_type=run["garment_photo_type"],
            num_samples=1,
            num_timesteps=generation["num_timesteps"],
            guidance_scale=generation["guidance_scale"],
            seed=run["seed"],
            segmentation_free=True,
        )
        output = result.images[0]
        seconds = round(time.time() - started, 2)

        candidate_path = baseline_dir / "candidate" / Path(run["image"]).name
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(candidate_path)
        digest = sha256_of_file(candidate_path)

        ssim_vs_base = mae = None
        reference_path = baseline_dir / run["image"]
        if reference_path.is_file():
            reference = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.float32)
            current = np.asarray(output.convert("RGB"), dtype=np.float32)
            ssim_vs_base = round(_ssim(reference, current), 4)
            mae = round(float(np.abs(reference - current).mean()), 2)

        metrics = _measure(pipeline, metrics_provider, person, garment, output, run["category"])
        rows.append(
            {
                "pair": run["pair"],
                "seed": run["seed"],
                "sha256": digest,
                "sha256_matches_baseline": digest == run["sha256"],
                "ssim_vs_baseline": ssim_vs_base,
                "mae_vs_baseline": mae,
                "seconds": seconds,
                "metrics": metrics,
                "degraded": (result.metadata or {}).get("segmentation", {}).get("degraded") or [],
            }
        )
        ssim_text = "-" if ssim_vs_base is None else f"{ssim_vs_base:.4f}"
        mae_text = "-" if mae is None else f"{mae:.2f}"
        print(
            f"  {run['pair']:<22} {run['seed']:>4}  {digest[:8]}…  {ssim_text:>8}  {mae_text:>5}  "
            f"{_fmt(metrics.get('identity_preservation')):>8}  {_fmt(metrics.get('outside_change'), 2):>7}  "
            f"{_fmt(metrics.get('color_fidelity'), 2):>5}  {_fmt(metrics.get('sharpness'), 2):>9}"
        )

    candidate_summary = aggregate([row["metrics"] for row in rows])
    report = compare_metrics(manifest["summary"], candidate_summary, max_regression=args.max_regression)

    reasons = list(report.reasons)
    ssim_values = [row["ssim_vs_baseline"] for row in rows if row["ssim_vs_baseline"] is not None]
    mean_ssim = round(sum(ssim_values) / len(ssim_values), 4) if ssim_values else None
    if args.min_ssim is not None and mean_ssim is not None and mean_ssim < args.min_ssim:
        reasons.append(f"similitud media {mean_ssim} < {args.min_ssim} (desvío catastrófico)")

    print(
        f"\nCandidato: score={_fmt(candidate_summary.get('score'))} "
        f"identity={_fmt(candidate_summary.get('identity_preservation'))} "
        f"outside={_fmt(candidate_summary.get('outside_change'), 2)} "
        f"color={_fmt(candidate_summary.get('color_fidelity'), 2)} "
        f"ssim_media={mean_ssim}"
    )
    for reason in reasons:
        print(f"  - {reason}")
    if report.improved:
        print(f"  mejora en: {', '.join(report.improved)}")
    print(f"\nDECISIÓN: {'PROMOVER (no empeora)' if not reasons else 'NO PROMOVER (regresión)'}")

    verification = {
        "verified_at": _now(),
        "baseline": args.name,
        "weights": {
            "dir": str(Path(args.weights_dir).resolve()),
            "sha256": candidate_hash,
            "same_as_baseline": same_weights,
        },
        "environment": _environment(pipeline),
        "runs": rows,
        "candidate_summary": candidate_summary,
        "baseline_summary": manifest["summary"],
        "mean_ssim_vs_baseline": mean_ssim,
        "regression": report.to_dict(),
        "decision": "promote" if not reasons else "reject",
        "reasons": reasons,
    }
    reports_dir = baseline_dir / "verifications"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Informe: {report_path}")

    return 0 if not reasons else 1


def _fmt(value, digits: int = 4) -> str:
    """Valor numérico con formato, o '-' si falta o es nan."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}f}" if math.isfinite(number) else "-"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--name", required=True, help="nombre del baseline (p. ej. baseline-2026-09-15)")
    common.add_argument("--output-dir", type=Path, default=BASELINES_DIR, help="dónde guardar los baselines")
    common.add_argument("--weights-dir", default="weights", help="directorio de pesos a evaluar")
    common.add_argument(
        "--metrics-provider",
        default="pose-heuristic",
        choices=["none", "pose-heuristic", "sam2", "grounded-sam2"],
        help="proveedor SOLO para el mapa de clases de las métricas (no genera)",
    )
    common.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    capture = sub.add_parser("capture", parents=[common], help="congelar el estado bueno")
    capture.add_argument("--pairs", type=Path, default=None, help="CSV: person,garment[,category[,garment_photo_type]]")
    capture.add_argument("--seeds", default="42,1234", help="semillas separadas por comas")
    capture.add_argument("--steps", type=int, default=8, help="pasos de difusión")
    capture.add_argument("--guidance", type=float, default=1.5)
    capture.add_argument("--force", action="store_true", help="sobrescribir un baseline existente")

    verify = sub.add_parser("verify", parents=[common], help="comparar un candidato contra el baseline")
    verify.add_argument("--max-regression", type=float, default=0.02, help="regresión relativa tolerada (0.02 = 2 %%)")
    verify.add_argument(
        "--min-ssim",
        type=float,
        default=None,
        help="guarda opcional: falla si la similitud media con el baseline baja de este valor",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "capture":
        return cmd_capture(args)
    return cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
