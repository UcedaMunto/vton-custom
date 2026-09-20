#!/usr/bin/env python3
"""Convierte los datasets locales de investigación (licencia NO comercial) a CSV de pares.

Genera lo que consume `scripts/fine_tune.py`: un `pairs.csv` con el esquema de
`fashn_vton.train.data` (persona, prenda, objetivo, categoría, máscara
agnóstica, licencia) más la `provenance.json` y un `NOTICE.md` del dataset.

**No copia imágenes**: el CSV apunta a los ficheros originales (los datasets
pesan decenas de GB y además deben quedar fuera del repo, ver
`fashn_vton.train.nc_policy`).

Layouts soportados (los que hay en esta máquina):

1. `dresscode` — `/home/uceda/Documents/IDM-VTON/dataset/DATA_DIR`:
   `<cat>/{images,label_maps,keypoints,skeletons,dense}` +
   `<cat>/{train_pairs.txt,test_pairs_paired.txt}`. La etiqueta de la prenda
   sale del `label_maps/<id>_4.png` (esquema ATR: 4=top, 5=falda, 6=pantalón,
   7=vestido).
2. `prep` — `/home/uceda/Documents/IDM-VTON/dataset/DATA_DIR_PREP` (preparado
   estilo IDM-VTON): `{train,test}/{image,cloth,image-densepose,agnostic-mask}` +
   `{train,test}_pairs.txt` + `vitonhd_*_tagged.json`. La máscara es binaria
   (`agnostic-mask/<id>_mask.png`).

Ejemplos:

    python scripts/prepare_nc_dataset.py dresscode \\
        --root /home/uceda/Documents/IDM-VTON/dataset/DATA_DIR \\
        --out datos/nc/dresscode --verify 20

    python scripts/prepare_nc_dataset.py prep \\
        --root /home/uceda/Documents/IDM-VTON/dataset/DATA_DIR_PREP \\
        --out datos/nc/dresscode-upper-prep --pairs-file train_pairs_clean.txt

    python scripts/prepare_nc_dataset.py dresscode --info     # solo resume el dataset
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashn_vton.train.data import (  # noqa: E402
    AGNOSTIC_KIND_LABELMAP,
    AGNOSTIC_KIND_MASK,
    PairRecord,
    check_files,
    dataset_summary,
    write_pairs_csv,
)
from fashn_vton.train.nc_policy import (  # noqa: E402
    build_provenance,
    is_nc_license,
    write_provenance,
)

DEFAULT_ROOT = Path("/home/uceda/Documents/IDM-VTON/dataset")

#: category de DressCode -> (categoría del modelo, ids de etiqueta de la prenda)
DRESSCODE_CATEGORIES: dict[str, tuple[str, str]] = {
    "upper_body": ("tops", "4"),
    "lower_body": ("bottoms", "5,6"),
    "dresses": ("one-pieces", "7"),
}

#: Tagged json de IDM-VTON -> categoría del modelo.
TAGGED_CATEGORIES = {
    "TOPS": "tops",
    "UPPER_BODY": "tops",
    "BOTTOMS": "bottoms",
    "LOWER_BODY": "bottoms",
    "DRESSES": "one-pieces",
    "DRESS": "one-pieces",
    "ONE-PIECES": "one-pieces",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

SPLIT_FILES = {
    "train": ("train_pairs.txt", "train_pairs_clean.txt"),
    "test": ("test_pairs_paired.txt", "test_pairs.txt"),
}


def _read_pairs_file(path: Path) -> list[tuple[str, str]]:
    """Lee un `*_pairs.txt` (modelo <tab|espacio> prenda) ignorando la 3ª columna."""
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.replace("\t", " ").split()
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def _resolve(directory: Path, stem: str, extensions: tuple[str, ...] = IMAGE_EXTENSIONS) -> Path | None:
    for extension in extensions:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _pairs_file_for_split(pairs_file: str | None, split: str) -> str | None:
    """¿`--pairs-file` aplica a este split?

    `train_pairs_clean.txt` solo existe (y solo tiene sentido) para `train`; si el
    nombre del fichero menciona un split, se usa solo para ese.
    """
    if not pairs_file:
        return None
    name = Path(pairs_file).name.lower()
    mentioned = [token for token in ("train", "test") if token in name]
    if mentioned and split not in mentioned:
        return None
    return pairs_file


def _pick_split_file(base: Path, split: str, root: Path | None = None) -> Path | None:
    """Busca el fichero de pares del split dentro del split y, si no, en la raíz.

    Los dos layouts reales se comportan distinto: DressCode los tiene dentro de
    la carpeta de categoría (`<cat>/train_pairs.txt`) y el preparado de IDM-VTON
    los tiene en la raíz (`<root>/train_pairs.txt`).
    """
    for name in SPLIT_FILES[split]:
        for directory in (base, root):
            if directory is None:
                continue
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def collect_dresscode(root: Path, splits: tuple[str, ...], categories: dict[str, tuple[str, str]]) -> tuple[list, dict]:
    """Recorre el layout DressCode` y devuelve `(máscaras, estadísticas)`."""
    records: list[PairRecord] = []
    stats: Counter = Counter()
    for folder, (category, labels) in categories.items():
        base = root / folder
        if not base.is_dir():
            stats[f"falta-carpeta:{folder}"] += 1
            continue
        for split in splits:
            pairs_file = _pick_split_file(base, split)
            if pairs_file is None:
                stats[f"falta-pares:{folder}/{split}"] += 1
                continue
            for model_name, garment_name in _read_pairs_file(pairs_file):
                person = _resolve(base / "images", Path(model_name).stem)
                garment = _resolve(base / "images", Path(garment_name).stem)
                stem = Path(model_name).stem
                mask_stem = f"{stem.rsplit('_', 1)[0]}_4" if "_" in stem else f"{stem}_4"
                mask = _resolve(base / "label_maps", mask_stem, (".png", ".jpg"))
                if person is None or garment is None or mask is None:
                    stats[f"incompleto:{folder}/{split}"] += 1
                    continue
                records.append(
                    PairRecord(
                        person=str(person),
                        garment=str(garment),
                        target=str(person),
                        category=category,
                        garment_photo_type="flat-lay",
                        agnostic=str(mask),
                        agnostic_kind=AGNOSTIC_KIND_LABELMAP,
                        agnostic_labels=labels,
                        license_class="nc-dresscode",
                        source=f"dresscode/{folder}/{split}",
                        split=split,
                        pair_id=f"{folder}_{split}_{stem}",
                    )
                )
                stats[f"ok:{folder}/{split}"] += 1
    return records, dict(stats)


def _tagged_categories(path: Path) -> dict[str, str]:
    """`file_name -> categoría` a partir del tagged json (si existe)."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("data", payload if isinstance(payload, list) else [])
    mapping: dict[str, str] = {}
    for entry in entries:
        name = str(entry.get("file_name", ""))
        category = str(entry.get("category_name", "")).upper()
        if name and category in TAGGED_CATEGORIES:
            mapping[name] = TAGGED_CATEGORIES[category]
    return mapping


def collect_prep(
    root: Path,
    splits: tuple[str, ...],
    default_category: str,
    pairs_file: str | None = None,
) -> tuple[list, dict]:
    """Recorre el layout preparado estilo IDM-VTON (`image/`, `cloth/`, `agnostic-mask/`)."""
    records: list[PairRecord] = []
    stats: Counter = Counter()
    for split in splits:
        base = root / split
        if not base.is_dir():
            stats[f"falta-carpeta:{split}"] += 1
            continue
        selected = None
        requested = _pairs_file_for_split(pairs_file, split)
        if requested:
            for directory in (base, root):
                candidate = directory / requested
                if candidate.is_file():
                    selected = candidate
                    break
            if selected is None:
                # `--pairs-file` puede existir solo en un split (p. ej. el filtrado
                # `train_pairs_clean.txt` no existe en test): se cae al fichero por defecto.
                stats[f"pares-file-ausente:{split}"] += 1
        if selected is None:
            selected = _pick_split_file(base, split, root)
        if selected is None or not selected.is_file():
            stats[f"falta-pares:{split}"] += 1
            continue
        tagged = _tagged_categories(base / f"vitonhd_{split}_tagged.json")
        for image_name, cloth_name in _read_pairs_file(selected):
            person = _resolve(base / "image", Path(image_name).stem)
            garment = _resolve(base / "cloth", Path(cloth_name).stem)
            mask = _resolve(base / "agnostic-mask", f"{Path(image_name).stem}_mask", (".png", ".jpg"))
            if person is None or garment is None or mask is None:
                stats[f"incompleto:{split}"] += 1
                continue
            category = tagged.get(image_name.split("/")[-1], default_category)
            records.append(
                PairRecord(
                    person=str(person),
                    garment=str(garment),
                    target=str(person),
                    category=category,
                    garment_photo_type="flat-lay",
                    agnostic=str(mask),
                    agnostic_kind=AGNOSTIC_KIND_MASK,
                    agnostic_labels="",
                    license_class="nc-dresscode",
                    source=f"idm-vton-prep/{split}",
                    split=split,
                    pair_id=f"prep_{split}_{Path(image_name).stem}",
                )
            )
            stats[f"ok:{split}"] += 1
    return records, dict(stats)


NOTICE_TEMPLATE = """# Dataset de investigación (licencia NO comercial): {name}

- **Origen**: `{root}`
- **Layout**: `{layout}`
- **Clase de licencia declarada**: `{license_class}`
- **Pares**: {pairs} ({splits})
- **CSV de pares**: `pairs.csv` (rutas absolutas a los ficheros originales; no se copia nada)
- **Generado**: {created_at} por `scripts/prepare_nc_dataset.py`

## Aviso

Este dataset (VITON-HD / DressCode y derivados) está restringido a investigación
y **no puede usarse para entrenar ni evaluar el producto comercial** (ver
`licenses/manifest.json` → `forbidden_artifacts` en el repo del fork). Los pesos
entrenados con él quedan marcados como no comerciales
(`provenance.json` → `commercial_use: false`) y `scripts/model_registry.py` se
niega a promoverlos como modelo activo salvo `--allow-nc` explícito.

El material aquí referenciado vive **fuera** del repositorio a propósito: no debe
entrar en artefactos publicables (imagen Docker, wheels, releases).
"""


def _write_outputs(
    out_dir: Path,
    records: list,
    *,
    name: str,
    layout: str,
    root: Path,
    license_class: str,
    license_note: str,
    stats: dict,
    dataset: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_pairs_csv(out_dir / "pairs.csv", records)
    summary = dataset_summary(records)
    provenance = build_provenance(
        dataset=dataset,
        license_class=license_class,
        commercial_use=not is_nc_license(license_class),
        dataset_root=root,
        pairs_csv=csv_path,
        license_note=license_note,
        pairs=len(records),
        extra={"layout": layout, "stats": stats, "summary": summary, "name": name},
    )
    provenance_path = write_provenance(out_dir, provenance)
    notice_path = out_dir / "NOTICE.md"
    notice_path.write_text(
        NOTICE_TEMPLATE.format(
            name=name,
            root=root,
            layout=layout,
            license_class=license_class,
            pairs=len(records),
            splits=json.dumps(summary["splits"], ensure_ascii=False),
            created_at=provenance["created_at"],
        ),
        encoding="utf-8",
    )
    return {"csv": str(csv_path), "provenance": str(provenance_path), "notice": str(notice_path), "summary": summary}


def _resolve_out(args) -> Path:
    out = Path(args.out)
    return out if out.is_absolute() else (ROOT / out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="layout", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, required=True, help="raíz del dataset original")
    common.add_argument("--out", default="datos/nc/dataset", help="directorio de salida (relativo al repo)")
    common.add_argument("--name", default=None, help="nombre del dataset (por defecto, el del layout)")
    common.add_argument("--splits", default="train,test", help="splits a incluir (train,test)")
    common.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="tope de pares por split (submuestreo aleatorio reproducible con --seed)",
    )
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--license-class", default="nc-dresscode")
    common.add_argument("--license-note", default="")
    common.add_argument("--force-license", action="store_true", help="aceptar una clase de licencia comercial")
    common.add_argument("--verify", type=int, default=0, help="comprueba la existencia de N pares")
    common.add_argument("--info", action="store_true", help="solo resume el dataset, sin escribir nada")

    dresscode = sub.add_parser("dresscode", parents=[common], help="layout DressCode (label_maps)")
    dresscode.add_argument(
        "--categories",
        default=",".join(
            f"{folder}={category}:{labels.replace(',', ';')}"
            for folder, (category, labels) in DRESSCODE_CATEGORIES.items()
        ),
        help="mapa categoria=modelo_cat:ids_etiqueta (usa ';' entre ids; ver DRESSCODE_CATEGORIES)",
    )

    prep = sub.add_parser("prep", parents=[common], help="layout preparado estilo IDM-VTON (agnostic-mask)")
    prep.add_argument("--category", default="tops", choices=("tops", "bottoms", "one-pieces"))
    prep.add_argument("--pairs-file", default=None, help="nombre del fichero de pares (p. ej. train_pairs_clean.txt)")
    return parser


def _parse_categories(spec: str) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        folder, _, rest = chunk.partition("=")
        category, _, labels = rest.partition(":")
        if not (folder and category):
            raise SystemExit(f"Entrada de --categories inválida: '{chunk}' (esperado carpeta=categoria:ids)")
        mapping[folder.strip()] = (category.strip(), labels.replace(";", ",").strip())
    return mapping


def _subsample(records: list, max_pairs: int | None, seed: int) -> list:
    """Tope por split con barajado reproducible (para pruebas rápidas)."""
    if not max_pairs:
        return records
    by_split: dict[str, list] = {}
    for record in records:
        by_split.setdefault(record.split, []).append(record)
    rng = random.Random(seed)
    selected: list = []
    for split, items in by_split.items():
        rng.shuffle(items)
        selected.extend(items[:max_pairs])
    print(f"  · submuestreo: {len(selected)} pares de {len(records)} (máx {max_pairs} por split)")
    return selected


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: no existe la raíz del dataset: {root}")
        return 2

    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    if args.layout == "dresscode":
        categories = _parse_categories(args.categories)
        records, stats = collect_dresscode(root, splits, categories)
        default_name, dataset = "dresscode", "DressCode (VITON-HD/DressCode: licencia de investigación)"
    else:
        records, stats = collect_prep(root, splits, args.category, args.pairs_file)
        default_name, dataset = "idm-vton-prep", "DressCode upper_body preparado estilo IDM-VTON"

    print(f"Dataset: {root}")
    print(f"Layout : {args.layout}")
    print(f"Pares  : {len(records)}")
    for key, value in sorted(stats.items()):
        print(f"  · {key}: {value}")
    if not records:
        print("ERROR: no se encontró ningún par. Revisa --root/--categories/--splits.")
        return 2

    if args.info:
        print(json.dumps(dataset_summary(records), indent=2, ensure_ascii=False))
        return 0

    if not is_nc_license(args.license_class) and not args.force_license:
        print(
            f"ERROR: '{args.license_class}' no es una licencia no comercial, pero este dataset "
            "sí lo es. Si de verdad quieres declararlo así (y asumir el riesgo), usa --force-license."
        )
        return 3

    records = _subsample(records, args.max_pairs, args.seed)
    if args.verify:
        missing = check_files(records, limit=args.verify)
        if missing:
            print(f"ERROR: faltan {len(missing)} ficheros de los primeros {args.verify} pares:")
            for path in missing[:10]:
                print(f"  - {path}")
            return 4
        print(f"  · verificación: los primeros {min(args.verify, len(records))} pares existen")

    out_dir = _resolve_out(args)
    outputs = _write_outputs(
        out_dir,
        records,
        name=args.name or default_name,
        layout=args.layout,
        root=root,
        license_class=args.license_class,
        license_note=args.license_note,
        stats=stats,
        dataset=dataset,
    )
    print("\nEscrito:")
    for key, value in outputs.items():
        print(f"  · {key}: {value}")
    print(
        "\nSiguiente paso (recuerda: datos NO comerciales, el entrenamiento exige opt-in):\n"
        f"  export FASHN_ALLOW_NC_TRAINING=1\n"
        f"  ./run_fashn_vton.sh python scripts/fine_tune.py --pairs {outputs['csv']} --limit 2 --steps 2 "
        "--out weights/candidates/smoke"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())



