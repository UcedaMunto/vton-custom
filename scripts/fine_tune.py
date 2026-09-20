#!/usr/bin/env python3
"""Fine-tuning LoRA de FASHN VTON v1.5 sobre un CSV de pares (datos de investigación).

Es la puerta de entrada al arnés: valida política de datos, bloquea cualquier
escritura fuera de `weights/candidates/` (el modelo comercial activo no se toca),
se turna la GPU y lanza `fashn_vton.train.trainer`.

Ejemplos:

    # 0) Comprobar datos y plan, sin cargar pesos ni tocar la GPU
    python scripts/fine_tune.py --pairs datos/nc/dresscode/pairs.csv --check-only

    # 1) Prueba de humo (2 pares, 2 pasos) — mide tiempo y VRAM reales
    export FASHN_ALLOW_NC_TRAINING=1
    ./run_fashn_vton.sh python scripts/fine_tune.py --pairs datos/nc/dresscode/pairs.csv \\
        --limit 2 --steps 2 --out weights/candidates/smoke --dtype bf16 --no-merge

    # 2) Corrida real (2.000 pasos) desde el modelo comercial congelado
    ./run_fashn_vton.sh python scripts/fine_tune.py --pairs datos/nc/dresscode/pairs.csv \\
        --split train --steps 2000 --rank 16 --out weights/candidates/lora-dresscode-v1

    # 3) Reanudar tras una caída
    ./run_fashn_vton.sh python scripts/fine_tune.py --pairs datos/nc/dresscode/pairs.csv \\
        --steps 4000 --resume --out weights/candidates/lora-dresscode-v1

Después (verificación, NO promoción automática):

    ./run_fashn_vton.sh python scripts/baseline.py verify --name baseline-2026-09-15 \\
        --weights-dir weights/candidates/lora-dresscode-v1

Recordatorio legal: si el CSV declara una licencia no comercial, el candidato
resultante **no puede ser el modelo comercial**; `scripts/model_registry.py
promote` lo bloquea (ver `fashn_vton.train.nc_policy`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashn_vton.compliance import sha256_of_file  # noqa: E402
from fashn_vton.train.data import license_classes, load_pairs_csv  # noqa: E402
from fashn_vton.train.nc_policy import (  # noqa: E402
    NC_TRAINING_ENV_FLAG,
    NcTrainingNotAllowed,
    assert_nc_training_allowed,
    build_provenance,
    is_nc_license,
    provenance_banner,
)
from fashn_vton.train.trainer import GPU_CONSUMER_NAME, TrainConfig, Trainer  # noqa: E402
from fashn_vton.utils.gpu_lock import GpuBusyError, acquire, release  # noqa: E402

CANDIDATES_DIR = ROOT / "weights" / "candidates"
DEFAULT_POSE_CACHE = ROOT / "datos" / "nc" / "pose_cache"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", required=True, help="CSV de pares (ver fashn_vton.train.data)")
    parser.add_argument("--weights-dir", default="weights", help="pesos base (el modelo comercial congelado)")
    parser.add_argument("--out", default=None, help="directorio del candidato (debe estar en weights/candidates/)")
    parser.add_argument("--tag", default=None, help="atajo: --out weights/candidates/<tag>")
    parser.add_argument("--rank", type=int, default=16, help="rango de LoRA")
    parser.add_argument("--alpha", type=float, default=None, help="alpha de LoRA (por defecto = rank)")
    parser.add_argument(
        "--targets",
        default="qkv,proj,linear1,linear2",
        help="hojas de nn.Linear a adaptar (qkv/proj = atención; linear1/linear2 = MLP fusionado)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--steps", type=int, default=1000, help="pasos de optimizador")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4, help="micro-pasos por paso de optimizador")
    parser.add_argument("--save-every", type=int, default=100, help="0 = no guardar intermedios")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--merge-every", type=int, default=0, help="0 = solo al final")
    parser.add_argument("--no-merge", action="store_true", help="no exportar model.safetensors mergeado")
    parser.add_argument("--resolution", default="576x864", help="ANCHOxALTO (múltiplos de 12)")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--cond-drop", type=float, default=0.1, help="probabilidad de tirar el condicionamiento (CFG)")
    parser.add_argument("--time-shift", type=float, default=1.5, help="mu del schedule de rectified flow")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="usar solo los primeros N pares (smoke test)")
    parser.add_argument("--split", default=None, help="filtrar por split del CSV (train/test)")
    parser.add_argument("--num-workers", type=int, default=0, help="workers del DataLoader (0 = en proceso)")
    parser.add_argument("--ca-mode", default="mask", choices=("mask", "none"))
    parser.add_argument("--pose-cache", default=str(DEFAULT_POSE_CACHE), help="caché de renders DWPose")
    parser.add_argument("--no-pose-cache", action="store_true")
    parser.add_argument("--augment-flip", action="store_true", help="volteo horizontal aleatorio (~0,5)")
    parser.add_argument("--no-grad-checkpointing", action="store_true", help="NO recomendado en 12 GB")
    parser.add_argument("--no-save-optimizer", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continuar desde <out>/state.json")
    parser.add_argument("--dataset-root", default=None, help="raíz del dataset (para el guardia NC)")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--allow-nc", action="store_true", help="opt-in explícito para datos no comerciales")
    parser.add_argument("--no-gpu-lock", action="store_true", help="no tomar el turno de GPU (si ya lo tienes)")
    parser.add_argument("--gpu-lock-path", default=None)
    parser.add_argument("--check-only", action="store_true", help="valida datos/licencias/plan y sale")
    return parser


def _resolve_output_dir(args) -> Path:
    """El candidato siempre va a `weights/candidates/`; nunca sobre el modelo activo."""
    if args.out:
        candidate = Path(args.out)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
    elif args.tag:
        candidate = CANDIDATES_DIR / args.tag
    else:
        raise SystemExit("ERROR: indica --tag <nombre> o --out weights/candidates/<nombre>")
    resolved = candidate.resolve()
    inside = True
    try:
        resolved.relative_to(CANDIDATES_DIR.resolve())
    except ValueError:
        inside = False
    if not inside:
        raise SystemExit(
            f"ERROR: el candidato debe vivir dentro de {CANDIDATES_DIR} (recibido '{resolved}').\n"
            "Motivo: el entrenamiento NUNCA escribe el modelo comercial activo "
            "(weights/model.safetensors); promover es un acto aparte y verificado "
            "(`scripts/model_registry.py promote`)."
        )
    if resolved == (ROOT / "weights").resolve():
        raise SystemExit("ERROR: --out no puede ser el propio directorio de pesos.")
    return resolved


def _parse_resolution(spec: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in spec.lower().replace(" ", "").split("x"))
    except ValueError as exc:
        raise SystemExit(f"ERROR: --resolution inválida ('{spec}'); usa ANCHOxALTO (p. ej. 576x864)") from exc
    return width, height


def _base_weights_info(weights_dir: Path) -> dict:
    model_path = weights_dir / "model.safetensors"
    if not model_path.is_file():
        raise SystemExit(f"ERROR: no encuentro los pesos base en {model_path}")
    return {
        "path": str(model_path.resolve()),
        "sha256": sha256_of_file(model_path),
        "bytes": model_path.stat().st_size,
    }


def _registered_active() -> tuple[str | None, str | None]:
    """`(tag, sha256)` del estado activo del registro de pesos, si existe."""
    registry_path = ROOT / "model_registry" / "registry.json"
    if not registry_path.is_file():
        return None, None
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None
    active = registry.get("active")
    for entry in registry.get("entries", []):
        if entry.get("tag") == active:
            return active, entry.get("sha256")
    return active, None


def _build_config(args, *, output_dir: Path, pairs_csv: Path, provenance: dict) -> TrainConfig:
    targets = tuple(part.strip() for part in args.targets.split(",") if part.strip())
    pose_cache = None if args.no_pose_cache else args.pose_cache
    return TrainConfig(
        pairs_csv=str(pairs_csv),
        output_dir=str(output_dir),
        weights_dir=str(Path(args.weights_dir).resolve()),
        rank=args.rank,
        alpha=args.alpha,
        targets=targets,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.steps,
        save_every=args.save_every,
        log_every=args.log_every,
        max_grad_norm=args.max_grad_norm,
        cond_drop_prob=args.cond_drop,
        time_shift_mu=args.time_shift,
        mixed_precision=args.dtype,
        gradient_checkpointing=not args.no_grad_checkpointing,
        resolution=_parse_resolution(args.resolution),
        seed=args.seed,
        limit=args.limit,
        num_workers=args.num_workers,
        resume=args.resume,
        merge_every=args.merge_every,
        merge_at_end=not args.no_merge,
        device=args.device,
        ca_mode=args.ca_mode,
        pose_cache_dir=pose_cache,
        augment_flip=args.augment_flip,
        save_optimizer=not args.no_save_optimizer,
        dataset_root=args.dataset_root,
        provenance=provenance,
    )


def _print_plan(config: TrainConfig, *, pairs: int, classes: list[str], output_dir: Path) -> None:
    width, height = config.resolution
    commercial = bool(config.provenance.get("commercial_use", False))
    print("=" * 78)
    print("Plan de entrenamiento")
    print("=" * 78)
    print(f"  pesos base        : {config.weights_dir}/model.safetensors")
    print(f"  candidato         : {output_dir}")
    print(f"  pares             : {pairs} (licencias: {', '.join(classes) or 'desconocida'})")
    if config.limit:
        print(f"                      · --limit {config.limit}: el entrenamiento usará solo {config.limit} pares")
    print(f"  LoRA              : r={config.rank} alpha={config.alpha or config.rank} targets={list(config.targets)}")
    print(f"  pasos             : {config.max_steps} x {config.grad_accum} micro-pasos (batch {config.batch_size})")
    print(f"  resolución        : {width}x{height} · dtype {config.mixed_precision}")
    print(f"  lr / wd           : {config.learning_rate} / {config.weight_decay} · clip {config.max_grad_norm}")
    print(f"  cond. dropout     : {config.cond_drop_prob} · time_shift mu={config.time_shift_mu}")
    print(f"  checkpoint        : cada {config.save_every or '—'} pasos · merge {'sí' if config.merge_at_end else 'no'}")
    print(f"  grad checkpointing: {'sí' if config.gradient_checkpointing else 'NO (riesgo de OOM en 12 GB)'}")
    print(f"  ca_mode / caché   : {config.ca_mode} / {config.pose_cache_dir or 'sin caché'}")
    print(f"  guardia NC        : {'comercial' if commercial else 'NO COMERCIAL (candidato no promovible)'}")
    print("=" * 78)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    pairs_csv = Path(args.pairs).expanduser().resolve()
    records = load_pairs_csv(pairs_csv, split=args.split)
    classes = license_classes(records)
    nc_classes = [name for name in classes if is_nc_license(name)]
    primary = nc_classes[0] if nc_classes else (classes[0] if classes else "desconocida")

    try:
        guard = assert_nc_training_allowed(
            license_class=primary,
            dataset_root=args.dataset_root,
            pairs_csv=pairs_csv,
            allow_nc=args.allow_nc,
        )
    except NcTrainingNotAllowed as exc:
        print(f"\n{exc}\n")
        return 3

    output_dir = _resolve_output_dir(args)
    weights_dir = Path(args.weights_dir).resolve()
    base = _base_weights_info(weights_dir)

    active_tag, active_sha = _registered_active()
    if active_sha and active_sha.lower() != base["sha256"].lower():
        print(
            f"AVISO: los pesos base ({base['sha256'][:12]}…) no coinciden con el estado registrado "
            f"activo '{active_tag}' ({active_sha[:12]}…). El plan recomienda partir del baseline "
            "congelado; si es intencionado, ignora este aviso."
        )

    provenance = build_provenance(
        dataset=Path(args.dataset_root).name if args.dataset_root else pairs_csv.parent.name,
        license_class=primary,
        commercial_use=not nc_classes,
        dataset_root=args.dataset_root,
        pairs_csv=pairs_csv,
        license_note=f"clases presentes en el CSV: {', '.join(classes) or 'desconocida'}",
        pairs=len(records),
        extra={
            "guard": guard,
            "base_weights": base["path"],
            "base_sha256": base["sha256"],
            "base_registered_tag": active_tag,
            "method": "lora",
            "split_filter": args.split,
        },
    )

    config = _build_config(args, output_dir=output_dir, pairs_csv=pairs_csv, provenance=provenance)
    _print_plan(config, pairs=len(records), classes=classes, output_dir=output_dir)
    print(f"  {provenance_banner(provenance)}")

    if args.check_only:
        print("\n--check-only: datos, licencias y plan validados. No se tocó la GPU ni los pesos.")
        print(f"Para entrenar de verdad: export {NC_TRAINING_ENV_FLAG}=1 y repite sin --check-only.")
        return 0

    device = args.device
    lock_path = Path(args.gpu_lock_path) if args.gpu_lock_path else None
    print(f"Dispositivo solicitado: {device}")
    if not args.no_gpu_lock:
        try:
            if lock_path is None:
                acquire(GPU_CONSUMER_NAME)
            else:
                acquire(GPU_CONSUMER_NAME, lock_path=lock_path)
        except GpuBusyError as exc:
            print(f"ERROR: {exc}\nLibera la GPU (la UI, IDM-CUSTOM, otro entrenamiento) o usa --no-gpu-lock.")
            return 4
        print(f"Turno de GPU concedido a '{GPU_CONSUMER_NAME}'.")

    try:
        summary = Trainer(config).setup().run()
    finally:
        if not args.no_gpu_lock:
            if lock_path is None:
                release(GPU_CONSUMER_NAME)
            else:
                release(GPU_CONSUMER_NAME, lock_path=lock_path)

    print(json.dumps({key: summary[key] for key in ("steps", "seconds", "seconds_per_step", "peak_vram_gb",
                                                   "loss_first", "loss_last", "delta_norm", "output_dir")},
                     indent=2, ensure_ascii=False))
    print(
        "\nSiguientes pasos:\n"
        f"  1) Verificar que no empeora (exit 0 = no empeora):\n"
        f"     ./run_fashn_vton.sh python scripts/baseline.py verify --name baseline-2026-09-15 \\\n"
        f"         --weights-dir {summary['output_dir']}\n"
        "  2) Mirar las imágenes del informe y decidir.\n"
        "  3) Si fuese un candidato COMERCIAL y ganase: `scripts/model_registry.py promote`.\n"
        "     (Los candidatos entrenados con datos NC están bloqueados a propósito.)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())



