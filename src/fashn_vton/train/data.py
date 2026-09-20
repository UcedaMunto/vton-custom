"""Dataset de trípletas para fine-tuning de FASHN VTON v1.5.

Construye **exactamente** el mismo condicionamiento que
:meth:`fashn_vton.TryOnPipeline.__call__` (pre-resize a 864, creación de la
imagen agnóstica, render de pose DWPose y `ResizePad` a 576×864), más el
objetivo `x1` (la foto real de la persona con la prenda puesta) que el bucle de
*rectified flow* necesita. La idea es que entrenamiento e inferencia no se
desalineen: es la lección repetida en la Pista A (ver
`IDM-VTON/configuracion-v14-entrenamiento/04_RECETA_ENTRENAMIENTO_COMERCIAL.md`).

Formato de entrada: un CSV de pares (una fila por muestra) con las columnas

    pair_id,person,garment,target,category,garment_photo_type,
    agnostic,agnostic_kind,agnostic_labels,license_class,source,split

`agnostic` es la máscara/etiquetado que permite borrar la prenda de la foto de
la persona (la imagen "agnóstica" que el modelo aprende a rellenar);
`agnostic_kind` es `mask` (PNG binario) o `labelmap` (PNG de etiquetas);
`agnostic_labels` (solo para `labelmap`) son los ids separados por comas.
`license_class` es obligatorio en la práctica: la política NC
(:mod:`fashn_vton.train.nc_policy`) se apoya en él.

Este módulo no genera pares: los datasets de investigación locales (DressCode,
preparado de IDM-VTON) se convierten con `scripts/prepare_nc_dataset.py`.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..preprocessing import (
    AspectPreserveResize,
    ResizePad,
    create_clothing_agnostic_image,
)
from ..segmentation import CATEGORY_TO_BODY_COVERAGE
from ..segmentation.labels import LABELS_TO_IDS
from ..utils import get_dummy_dw_keypoints, normalize_uint8_to_neg1_1, numpy_to_torch, setup_logger

#: Etiqueta de categoría que espera el modelo (`TryOnPipeline.CATEGORY_TO_LABEL`).
CATEGORY_IDS: dict[str, int] = {"tops": 1, "bottoms": 2, "one-pieces": 3}

#: Etiqueta FASHN con la que se marca la prenda en el mapa de clases sintético.
CATEGORY_TO_GARMENT_LABEL: dict[str, str] = {"tops": "top", "bottoms": "pants", "one-pieces": "dress"}

#: Tipos de fichero admitidos en la columna `agnostic`.
AGNOSTIC_KIND_MASK = "mask"
AGNOSTIC_KIND_LABELMAP = "labelmap"
AGNOSTIC_KINDS = (AGNOSTIC_KIND_MASK, AGNOSTIC_KIND_LABELMAP)

#: Modos de construcción de la imagen agnóstica.
CA_MODES = ("mask", "none")

PAIRS_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "person",
    "garment",
    "target",
    "category",
    "garment_photo_type",
    "agnostic",
    "agnostic_kind",
    "agnostic_labels",
    "license_class",
    "source",
    "split",
)

REQUIRED_COLUMNS: tuple[str, ...] = ("person", "garment", "target", "category")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class PairsCsvError(ValueError):
    """El CSV de pares está mal formado o incompleto."""


@dataclass
class PairRecord:
    """Una muestra de entrenamiento (un par persona/prenda + su objetivo)."""

    person: str
    garment: str
    target: str
    category: str
    garment_photo_type: str = "flat-lay"
    agnostic: str = ""
    agnostic_kind: str = AGNOSTIC_KIND_MASK
    agnostic_labels: str = ""
    license_class: str = ""
    source: str = ""
    split: str = "train"
    pair_id: str = ""

    def __post_init__(self) -> None:
        if not self.pair_id:
            stem = Path(self.person).stem
            self.pair_id = f"{stem}_{Path(self.garment).stem}"
        if self.category not in CATEGORY_IDS:
            raise PairsCsvError(
                f"category='{self.category}' no soportada (usa: {', '.join(CATEGORY_IDS)}); "
                f"pair_id='{self.pair_id}'"
            )
        if self.agnostic_kind not in AGNOSTIC_KINDS:
            raise PairsCsvError(
                f"agnostic_kind='{self.agnostic_kind}' no soportada (usa: {', '.join(AGNOSTIC_KINDS)}); "
                f"pair_id='{self.pair_id}'"
            )

    @property
    def garment_category_id(self) -> int:
        return CATEGORY_IDS[self.category]

    @property
    def agnostic_label_ids(self) -> tuple[int, ...]:
        """Ids de etiqueta que cuentan como prenda dentro de `agnostic` (labelmap)."""
        values = [chunk.strip() for chunk in (self.agnostic_labels or "").replace(";", ",").split(",")]
        ids: list[int] = []
        for value in values:
            if not value:
                continue
            try:
                ids.append(int(value))
            except ValueError as exc:
                raise PairsCsvError(
                    f"agnostic_labels='{self.agnostic_labels}' contiene un id no numérico "
                    f"('{value}') en pair_id='{self.pair_id}'"
                ) from exc
        return tuple(ids)

    def to_row(self) -> dict:
        return {key: ("" if value is None else value) for key, value in asdict(self).items()}


def _normalize_row(row: dict, index: int, origin: str) -> PairRecord:
    def value(*names: str) -> str:
        for name in names:
            raw = row.get(name)
            if raw is not None and str(raw).strip():
                return str(raw).strip()
        return ""

    person = value("person", "person_path", "image", "image_path")
    garment = value("garment", "garment_path", "cloth", "cloth_path")
    target = value("target", "target_path", "label")
    category = value("category", "category_name").lower()
    if not (person and garment and target):
        raise PairsCsvError(
            f"{origin}: fila {index + 2} sin person/garment/target (¿faltan columnas obligatorias "
            f"{REQUIRED_COLUMNS}?)"
        )
    # Normaliza variantes habituales de la categoría ("upper_body", "TOPS", "dress"...).
    aliases = {
        "top": "tops",
        "upper": "tops",
        "upper_body": "tops",
        "bottom": "bottoms",
        "lower": "bottoms",
        "lower_body": "bottoms",
        "one-piece": "one-pieces",
        "onepieces": "one-pieces",
        "dress": "one-pieces",
        "dresses": "one-pieces",
        "full": "one-pieces",
    }
    category = aliases.get(category, category)

    return PairRecord(
        person=person,
        garment=garment,
        target=target,
        category=category,
        garment_photo_type=value("garment_photo_type", "photo_type") or "flat-lay",
        agnostic=value("agnostic", "agnostic_path", "mask"),
        agnostic_kind=value("agnostic_kind", "mask_kind") or AGNOSTIC_KIND_MASK,
        agnostic_labels=value("agnostic_labels", "mask_labels"),
        license_class=value("license_class", "license"),
        source=value("source"),
        split=value("split") or "train",
        pair_id=value("pair_id", "name", "id"),
    )


def load_pairs_csv(path: str | Path, split: str | None = None) -> list[PairRecord]:
    """Lee y valida el CSV de pares (opcionalmente filtrando por `split`)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el CSV de pares: {path}")
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PairsCsvError(f"{path}: el CSV está vacío")
        records = [_normalize_row(row, index, str(path)) for index, row in enumerate(reader)]
    if split:
        records = [record for record in records if record.split == split]
    if not records:
        raise PairsCsvError(f"{path}: no quedó ningún par" + (f" en el split '{split}'" if split else ""))
    return records


def write_pairs_csv(path: str | Path, records: Iterable[PairRecord]) -> Path:
    """Escribe el CSV de pares con el esquema canónico de :data:`PAIRS_COLUMNS`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_row() for record in records]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PAIRS_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def license_classes(records: Sequence[PairRecord]) -> list[str]:
    """Clases de licencia presentes (para la política NC y la `provenance.json`)."""
    return sorted({(record.license_class or "desconocida").strip() for record in records})


def pre_resize_scale_factor(width: int, height: int, max_dim: int, allow_upsampling: bool = False) -> float:
    """Factor de escala del pre-resize del pipeline (`fit` sobre el lado mayor)."""
    scale = min(max_dim / max(width, 1), max_dim / max(height, 1))
    if not allow_upsampling and scale > 1.0:
        return 1.0
    return scale


def resize_mask_like(mask: np.ndarray, scale: float) -> np.ndarray:
    """Reescala una máscara con la misma aritmética entera que `AspectPreserveResize`."""
    if scale == 1.0:
        return mask
    height, width = mask.shape[:2]
    new_size = (max(1, int(scale * width)), max(1, int(scale * height)))
    interpolation = cv2.INTER_NEAREST if mask.dtype != np.uint8 or mask.ndim == 2 else cv2.INTER_AREA
    return cv2.resize(mask, new_size, interpolation=interpolation)


PoseFn = Callable[[np.ndarray], dict]


def build_pose_fn(weights_dir: str | Path = "weights", device: str = "cpu") -> PoseFn:
    """Construye el detector DWPose real del pipeline (BGR uint8 -> pose dict)."""
    from ..dwpose import DWposeDetector

    checkpoints = Path(weights_dir) / "dwpose"
    if not checkpoints.is_dir():
        raise FileNotFoundError(
            f"No encuentro los pesos de DWPose en '{checkpoints}'. Descárgalos con "
            "`python scripts/download_weights.py --weights-dir ./weights`."
        )
    # `Wholebody` espera un id numérico ('cuda:0'), no 'cuda' (ver wholebody.py).
    resolved_device = f"{device}:0" if device.startswith("cuda") and ":" not in device else device
    detector = DWposeDetector(checkpoints_dir=str(checkpoints), device=resolved_device)

    def pose_fn(bgr: np.ndarray) -> dict:
        return detector(bgr)

    return pose_fn


class PoseCache:
    """Caché en disco de los renders de pose (evita repetir DWPose cada época)."""

    def __init__(self, directory: str | Path | None):
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def key(self, path: str | Path, height: int, width: int, role: str = "") -> str:
        source = Path(path)
        try:
            stat = source.stat()
            fingerprint = f"{source.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{height}x{width}|{role}"
        except OSError:
            fingerprint = f"{source}|{height}x{width}|{role}"
        return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()

    def load(self, key: str) -> np.ndarray | None:
        if not self.directory:
            return None
        candidate = self.directory / f"{key}.npy"
        if not candidate.is_file():
            return None
        try:
            return np.load(candidate)
        except (OSError, ValueError):
            return None

    def save(self, key: str, canvas: np.ndarray) -> None:
        if not self.directory:
            return
        try:
            np.save(self.directory / f"{key}.npy", canvas)
        except OSError:
            pass


class TryOnTrainDataset(Dataset):
    """Pares de entrenamiento ya preprocesados como los ve el modelo.

    Devuelve un dict con:

    - `x1` (3, H, W): objetivo, la foto real de la persona con la prenda.
    - `ca_images` (3, H, W): foto de la persona con la prenda borrada (agnóstica).
    - `person_poses` (1, H, W), `garment_poses` (1, H, W): renders de pose (1 canal).
    - `garment_images` (3, H, W): la prenda (modo `flat-lay`, sin enmascarar).
    - `garment_categories` (): id de categoría 1/2/3.
    - `pair_id`, `license_class`: metadatos (no se apilan como tensores).
    """

    def __init__(
        self,
        pairs_csv: str | Path,
        weights_dir: str | Path = "weights",
        resolution: tuple[int, int] = (576, 864),
        device: str = "cpu",
        pose_fn: PoseFn | None = None,
        pose_cache_dir: str | Path | None = None,
        ca_mode: str = "mask",
        mask_threshold: int = 127,
        augment_flip: bool = False,
        limit: int | None = None,
        split: str | None = None,
        logger: logging.Logger | None = None,
    ):
        if ca_mode not in CA_MODES:
            raise ValueError(f"ca_mode='{ca_mode}' no soportado (usa: {', '.join(CA_MODES)})")
        width, height = resolution
        if width % 12 or height % 12:
            raise ValueError(
                f"resolution={resolution} debe ser múltiplo del patch del modelo (12 px en H y W)"
            )
        self.pairs_csv = Path(pairs_csv)
        self.pairs = load_pairs_csv(self.pairs_csv, split=split)
        if limit:
            self.pairs = self.pairs[:limit]
        self.weights_dir = Path(weights_dir)
        self.resolution = (int(width), int(height))
        self.device = device
        self.ca_mode = ca_mode
        self.mask_threshold = int(mask_threshold)
        self.augment_flip = bool(augment_flip)
        self.logger = logger or setup_logger("TryOnTrainDataset", level=logging.INFO)

        self.max_dim = max(self.resolution)
        self.pre_resize = AspectPreserveResize(target_size=(self.max_dim, self.max_dim), mode="fit", backend="pil")
        self.resize_pad = ResizePad(self.resolution, backend="opencv")
        self.pose_cache = PoseCache(pose_cache_dir)
        self._pose_fn = pose_fn
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ utilidades
    @property
    def pose_fn(self) -> PoseFn:
        """Detector de pose perezoso (solo se construye si de verdad hace falta)."""
        if self._pose_fn is None:
            self.logger.info("Construyendo DWPose (device=%s) para el dataset…", self.device)
            self._pose_fn = build_pose_fn(self.weights_dir, device=self.device)
        return self._pose_fn

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.logger.warning(message)

    def __len__(self) -> int:
        return len(self.pairs)

    def _pre_resized(self, path: str) -> tuple[np.ndarray, float]:
        """Carga y aplica el pre-resize del pipeline ('fit' al lado mayor, sin sobremuestreo)."""
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            scale = pre_resize_scale_factor(width, height, self.max_dim, allow_upsampling=False)
            resized = self.pre_resize(rgb, allow_upsampling=False) if scale != 1.0 else rgb
            return np.array(resized), scale

    def _agnostic_np(self, record: PairRecord, person_np: np.ndarray, scale: float) -> np.ndarray:
        """Construye la imagen agnóstica con el mismo código que el pipeline."""
        if self.ca_mode == "none" or not record.agnostic:
            self._warn_once(
                "ca-none",
                "ca_mode='none': la imagen agnóstica es la propia foto de la persona "
                "(sin borrar la prenda). Sirve para pruebas rápidas, NO para entrenar "
                "transferencia real: el modelo puede limitarse a copiar la entrada.",
            )
            return person_np

        mask_path = Path(record.agnostic)
        if not mask_path.is_file():
            raise FileNotFoundError(f"Falta la máscara agnóstica '{mask_path}' (pair_id='{record.pair_id}')")
        with Image.open(mask_path) as mask_image:
            mask = np.array(mask_image)
        mask = resize_mask_like(mask, scale)
        height, width = person_np.shape[:2]
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        if record.agnostic_kind == AGNOSTIC_KIND_LABELMAP:
            label_ids = record.agnostic_label_ids
            if not label_ids:
                raise PairsCsvError(
                    f"pair_id='{record.pair_id}': agnostic_kind='labelmap' exige agnostic_labels "
                    "con los ids de la prenda (p. ej. '4' para upper_body de DressCode)"
                )
            binary = np.isin(mask, list(label_ids))
        else:
            binary = mask > self.mask_threshold

        if not binary.any():
            self._warn_once(
                f"mask-empty-{record.category}",
                f"La máscara agnóstica de '{record.pair_id}' no marcó ningún píxel "
                f"(kind={record.agnostic_kind}): la imagen agnóstica queda igual a la foto. "
                "Revisa agnostic_labels/agnostic_kind en el CSV.",
            )
        label_id = LABELS_TO_IDS[CATEGORY_TO_GARMENT_LABEL[record.category]]
        class_map = np.where(binary, label_id, 0).astype(np.uint8)
        return create_clothing_agnostic_image(
            img_np=person_np,
            seg_pred=class_map,
            labels_to_segment_indices=[label_id],
            body_coverage=CATEGORY_TO_BODY_COVERAGE[record.category],
            mask_value=127,
            logger=self.logger,
        )

    def _pose_canvas(self, path: str, image_np: np.ndarray, *, role: str, flat_lay: bool = False) -> np.ndarray:
        """Render de pose en escala de grises (1 canal), como el pipeline."""
        from ..dwpose import draw_pose  # import local: arrastra onnxruntime

        height, width = image_np.shape[:2]
        cache_key = self.pose_cache.key(path, height, width, role=role)
        cached = self.pose_cache.load(cache_key)
        if cached is not None and cached.shape[:2] == (height, width):
            return cached
        pose = get_dummy_dw_keypoints() if flat_lay else self.pose_fn(image_np[..., ::-1])
        canvas = draw_pose(pose, height, width, grayscale=True)
        self.pose_cache.save(cache_key, canvas)
        return canvas

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """uint8 HxW o HxWx3 -> tensor normalizado; las poses ganan su canal (1, H, W)."""
        tensor = normalize_uint8_to_neg1_1(numpy_to_torch(np.ascontiguousarray(array)))
        return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor

    def __getitem__(self, index: int) -> dict:
        record = self.pairs[index]
        person_np, scale = self._pre_resized(record.person)
        target_np = person_np.copy()
        ca_np = self._agnostic_np(record, person_np.copy(), scale)
        person_pose_np = self._pose_canvas(record.person, person_np, role="person")

        garment_np, _garment_scale = self._pre_resized(record.garment)
        is_flat_lay = record.garment_photo_type == "flat-lay"
        if not is_flat_lay:
            self._warn_once(
                "garment-model-type",
                f"garment_photo_type='{record.garment_photo_type}' requiere el proveedor de "
                "segmentación para aislar la prenda; el arnés de entrenamiento solo soporta "
                "'flat-lay' (la prenda se usa tal cual, como el camino comercial).",
            )
        garment_pose_np = self._pose_canvas(record.garment, garment_np, role="garment", flat_lay=is_flat_lay)

        if self.augment_flip and np.random.rand() < 0.5:
            person_np = np.flip(person_np, axis=1)
            target_np = np.flip(target_np, axis=1)
            ca_np = np.flip(ca_np, axis=1)
            person_pose_np = np.flip(person_pose_np, axis=1)
            garment_np = np.flip(garment_np, axis=1)
            garment_pose_np = np.flip(garment_pose_np, axis=1)

        ca_np = self.resize_pad(ca_np)
        target_np = self.resize_pad(target_np)
        garment_np = self.resize_pad(garment_np)
        person_pose_np = self.resize_pad(person_pose_np, interpolation=cv2.INTER_NEAREST_EXACT)
        garment_pose_np = self.resize_pad(garment_pose_np, interpolation=cv2.INTER_NEAREST_EXACT)

        return {
            "x1": self._to_tensor(target_np),
            "ca_images": self._to_tensor(ca_np),
            "person_poses": self._to_tensor(person_pose_np),
            "garment_images": self._to_tensor(garment_np),
            "garment_poses": self._to_tensor(garment_pose_np),
            "garment_categories": torch.tensor(record.garment_category_id, dtype=torch.long),
            "pair_id": record.pair_id,
            "license_class": record.license_class,
        }


#: Claves que sí se apilan como tensores al hacer batch.
TENSOR_KEYS = ("x1", "ca_images", "person_poses", "garment_images", "garment_poses", "garment_categories")


def collate_train_batch(batch: Sequence[dict]) -> dict:
    """`collate_fn` que apila tensores y conserva los metadatos como listas."""
    collated = {key: torch.stack([item[key] for item in batch]) for key in TENSOR_KEYS}
    collated["pair_id"] = [item["pair_id"] for item in batch]
    collated["license_class"] = [item["license_class"] for item in batch]
    return collated


def check_files(records: Sequence[PairRecord], limit: int | None = None) -> list[str]:
    """Devuelve los ficheros referenciados que no existen (revisa hasta `limit` pares)."""
    missing: list[str] = []
    for record in records[: limit or len(records)]:
        for candidate in (record.person, record.garment, record.target, record.agnostic):
            if candidate and not Path(candidate).is_file():
                missing.append(candidate)
    return missing


def dataset_summary(records: Sequence[PairRecord]) -> dict:
    """Resumen del dataset (categorías, splits, licencias, fuentes) para logs/informes."""

    def counts(values: Iterable[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in values:
            key = value or "(vacío)"
            result[key] = result.get(key, 0) + 1
        return dict(sorted(result.items()))

    return {
        "pairs": len(records),
        "categories": counts(record.category for record in records),
        "splits": counts(record.split for record in records),
        "license_classes": counts(record.license_class for record in records),
        "sources": counts(record.source for record in records),
        "garment_photo_types": counts(record.garment_photo_type for record in records),
        "agnostic_kinds": counts(record.agnostic_kind for record in records),
    }


__all__ = [
    "AGNOSTIC_KIND_LABELMAP",
    "AGNOSTIC_KIND_MASK",
    "AGNOSTIC_KINDS",
    "CA_MODES",
    "CATEGORY_IDS",
    "CATEGORY_TO_GARMENT_LABEL",
    "IMAGE_EXTENSIONS",
    "PAIRS_COLUMNS",
    "REQUIRED_COLUMNS",
    "TENSOR_KEYS",
    "PairRecord",
    "PairsCsvError",
    "PoseCache",
    "PoseFn",
    "TryOnTrainDataset",
    "build_pose_fn",
    "check_files",
    "collate_train_batch",
    "dataset_summary",
    "license_classes",
    "load_pairs_csv",
    "pre_resize_scale_factor",
    "resize_mask_like",
    "write_pairs_csv",
]




