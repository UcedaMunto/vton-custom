"""Pruebas del dataset de entrenamiento (`fashn_vton.train.data`).

Sin GPU, sin pesos y sin el dataset real: se fabrican imágenes diminutas
(90x60), una máscara y un CSV, y se comprueba que el dataset produce los
tensores que el MMDiT espera (3 canales para las imágenes, 1 para las poses),
que respeta el enmascarado agnóstico y que valida el CSV.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from fashn_vton.train.data import (
    CATEGORY_IDS,
    PairRecord,
    PairsCsvError,
    PoseCache,
    TryOnTrainDataset,
    check_files,
    collate_train_batch,
    dataset_summary,
    load_pairs_csv,
    pre_resize_scale_factor,
    resize_mask_like,
    write_pairs_csv,
)
from fashn_vton.utils import get_dummy_dw_keypoints

RESOLUTION = (48, 24)  # (W, H): múltiplos de 12, y H != W para probar el padding


def _write_png(path, array):
    Image.fromarray(array).save(path)


def make_dataset_dir(tmp_path, *, kind="mask", n_pairs=1):
    """Crea imágenes, máscaras y CSV mínimos; devuelve la ruta del CSV.

    Las imágenes son 48x96 (HxW): la misma relación de aspecto que
    `RESOLUTION=(48,24)`, así el pre-resize y el padding del pipeline no meten
    relleno y las aserciones sobre regiones concretas son estables.
    """
    root = tmp_path / "mini"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    person = np.dstack(
        [np.full((48, 96), 200, np.uint8), np.full((48, 96), 30, np.uint8), np.zeros((48, 96), np.uint8)]
    )
    garment = np.dstack(
        [np.zeros((48, 96), np.uint8), np.full((48, 96), 60, np.uint8), np.full((48, 96), 200, np.uint8)]
    )
    mask = np.zeros((48, 96), np.uint8)
    mask[:, :48] = 255 if kind == "mask" else 4

    records = []
    for index in range(n_pairs):
        person_path = root / "images" / f"p{index}.png"
        garment_path = root / "images" / f"g{index}.png"
        mask_path = root / "masks" / f"p{index}{'_mask' if kind == 'mask' else '_label'}.png"
        _write_png(person_path, person)
        _write_png(garment_path, garment)
        _write_png(mask_path, mask)
        records.append(
            PairRecord(
                person=str(person_path),
                garment=str(garment_path),
                target=str(person_path),
                category="tops",
                agnostic=str(mask_path),
                agnostic_kind="mask" if kind == "mask" else "labelmap",
                agnostic_labels="" if kind == "mask" else "4",
                license_class="nc-dresscode",
                source="unit-test",
                split="train" if index % 2 == 0 else "test",
                pair_id=f"pair{index}",
            )
        )
    return write_pairs_csv(root / "pairs.csv", records)


def counting_pose_fn(counter: list):
    """`pose_fn` falso: cuenta llamadas y devuelve el esqueleto vacío."""

    def pose_fn(_bgr: np.ndarray) -> dict:
        counter.append(1)
        return get_dummy_dw_keypoints()

    return pose_fn


def test_dataset_shapes_dtypes_and_ranges(tmp_path):
    pairs_csv = make_dataset_dir(tmp_path)
    dataset = TryOnTrainDataset(
        pairs_csv, resolution=RESOLUTION, pose_fn=counting_pose_fn([]), pose_cache_dir=tmp_path / "pose_cache"
    )
    assert len(dataset) == 1
    item = dataset[0]
    height, width = RESOLUTION[1], RESOLUTION[0]

    assert item["x1"].shape == (3, height, width)
    assert item["ca_images"].shape == (3, height, width)
    assert item["garment_images"].shape == (3, height, width)
    assert item["person_poses"].shape == (1, height, width)
    assert item["garment_poses"].shape == (1, height, width)
    for key in ("x1", "ca_images", "garment_images", "person_poses", "garment_poses"):
        assert item[key].dtype == torch.float32
        assert float(item[key].min()) >= -1.0001
        assert float(item[key].max()) <= 1.0001
    assert int(item["garment_categories"]) == CATEGORY_IDS["tops"]
    assert item["pair_id"] == "pair0"
    assert item["license_class"] == "nc-dresscode"


def test_agnostic_image_erases_the_garment_region(tmp_path):
    pairs_csv = make_dataset_dir(tmp_path)
    dataset = TryOnTrainDataset(pairs_csv, resolution=RESOLUTION, pose_fn=counting_pose_fn([]))
    item = dataset[0]
    # La máscara cubría la mitad izquierda: allí la imagen agnóstica es gris (127)
    # y el objetivo conserva el rojo de la foto original.
    masked_ca = item["ca_images"][:, :, :16]
    masked_target = item["x1"][:, :, :16]
    assert float(masked_ca.abs().max()) < 0.05  # gris 127 -> ~-0,004
    assert float((masked_target - masked_ca).abs().mean()) > 0.1
    # Fuera de la máscara, agnóstica y objetivo coinciden.
    torch.testing.assert_close(item["ca_images"][:, :, 30:], item["x1"][:, :, 30:], atol=1e-6, rtol=0)


def test_labelmap_masks_behave_like_binary_masks(tmp_path):
    mask_csv = make_dataset_dir(tmp_path / "a", kind="mask")
    label_csv = make_dataset_dir(tmp_path / "b", kind="labelmap")
    as_mask = TryOnTrainDataset(mask_csv, resolution=RESOLUTION, pose_fn=counting_pose_fn([]))[0]
    as_label = TryOnTrainDataset(label_csv, resolution=RESOLUTION, pose_fn=counting_pose_fn([]))[0]
    torch.testing.assert_close(as_mask["ca_images"], as_label["ca_images"], atol=1e-6, rtol=0)


def test_ca_mode_none_keeps_the_person_photo(tmp_path):
    pairs_csv = make_dataset_dir(tmp_path)
    dataset = TryOnTrainDataset(pairs_csv, resolution=RESOLUTION, pose_fn=counting_pose_fn([]), ca_mode="none")
    item = dataset[0]
    torch.testing.assert_close(item["ca_images"], item["x1"], atol=0, rtol=0)


def test_pose_cache_avoids_recomputing(tmp_path):
    pairs_csv = make_dataset_dir(tmp_path)
    calls: list = []
    dataset = TryOnTrainDataset(
        pairs_csv,
        resolution=RESOLUTION,
        pose_fn=counting_pose_fn(calls),
        pose_cache_dir=tmp_path / "pose_cache",
    )
    dataset[0]
    assert len(calls) == 1  # solo la persona; la prenda va con pose vacía (flat-lay)
    dataset[0]
    assert len(calls) == 1  # la segunda vez sale de la caché


def test_pose_cache_key_changes_with_role_and_size(tmp_path):
    cache = PoseCache(tmp_path / "cache")
    file = tmp_path / "x.jpg"
    file.write_bytes(b"0")
    assert cache.key(file, 10, 10, role="person") != cache.key(file, 10, 10, role="garment")
    assert cache.key(file, 10, 10) != cache.key(file, 20, 10)
    assert cache.load("no-existe") is None


def test_geometric_helpers_match_pipeline_arithmetic():
    assert pre_resize_scale_factor(768, 1024, 864) == pytest.approx(864 / 1024)
    assert pre_resize_scale_factor(300, 400, 864) == 1.0  # no se sobremuestrea
    mask = np.zeros((90, 60), np.uint8)
    mask[:45, :30] = 1
    resized = resize_mask_like(mask, 0.5)
    assert resized.shape == (45, 30)
    assert set(np.unique(resized)) <= {0, 1}


# --------------------------------------------------------------------- CSV


def test_csv_roundtrip_and_summary(tmp_path):
    pairs_csv = make_dataset_dir(tmp_path, n_pairs=4)
    records = load_pairs_csv(pairs_csv)
    assert len(records) == 4
    assert load_pairs_csv(pairs_csv, split="test")[0].pair_id == "pair1"
    summary = dataset_summary(records)
    assert summary["pairs"] == 4
    assert summary["categories"] == {"tops": 4}
    assert summary["splits"] == {"test": 2, "train": 2}
    assert summary["license_classes"] == {"nc-dresscode": 4}
    assert check_files(records) == []


def test_csv_missing_required_columns_raises(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("person,garment\n/a.png,/b.png\n", encoding="utf-8")
    with pytest.raises(PairsCsvError) as excinfo:
        load_pairs_csv(path)
    assert "target" in str(excinfo.value)


def test_csv_unknown_category_raises(tmp_path):
    with pytest.raises(PairsCsvError):
        PairRecord(person="a", garment="b", target="c", category="calcetines")


def test_csv_unknown_agnostic_kind_raises():
    with pytest.raises(PairsCsvError):
        PairRecord(person="a", garment="b", target="c", category="tops", agnostic_kind="sprite")


def test_csv_invalid_label_ids_raises():
    record = PairRecord(
        person="a", garment="b", target="c", category="tops", agnostic_kind="labelmap", agnostic_labels="4,x"
    )
    with pytest.raises(PairsCsvError):
        _ = record.agnostic_label_ids


def test_csv_aliases_are_normalized(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text(
        "person_path,garment_path,target_path,category_name,license\n/a.png,/b.png,/a.png,upper_body,nc-dresscode\n",
        encoding="utf-8",
    )
    record = load_pairs_csv(path)[0]
    assert record.category == "tops"
    assert record.license_class == "nc-dresscode"
    assert record.garment_photo_type == "flat-lay"


def test_missing_asset_raises_at_getitem(tmp_path):
    pairs_csv = make_dataset_dir(tmp_path)
    records = load_pairs_csv(pairs_csv)
    records[0].person = str(tmp_path / "no-existe.png")
    write_pairs_csv(pairs_csv, records)
    dataset = TryOnTrainDataset(pairs_csv, resolution=RESOLUTION, pose_fn=counting_pose_fn([]))
    assert check_files(records) == [str(tmp_path / "no-existe.png")]
    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_limit_and_split_filter_are_applied(tmp_path):
    pairs_csv = make_dataset_dir(tmp_path, n_pairs=4)
    limited = TryOnTrainDataset(
        pairs_csv, resolution=RESOLUTION, pose_fn=counting_pose_fn([]), limit=2
    )
    assert len(limited) == 2
    only_test = TryOnTrainDataset(
        pairs_csv, resolution=RESOLUTION, pose_fn=counting_pose_fn([]), split="test"
    )
    assert len(only_test) == 2
    assert only_test[0]["pair_id"] == "pair1"


def test_resolution_must_be_multiple_of_patch(tmp_path):
    pairs_csv = make_dataset_dir(tmp_path)
    with pytest.raises(ValueError):
        TryOnTrainDataset(pairs_csv, resolution=(50, 24), pose_fn=counting_pose_fn([]))
    with pytest.raises(ValueError):
        TryOnTrainDataset(pairs_csv, resolution=RESOLUTION, pose_fn=counting_pose_fn([]), ca_mode="inventado")


def test_collate_stacks_tensors_and_keeps_metadata(tmp_path):
    pairs_csv = make_dataset_dir(tmp_path, n_pairs=2)
    dataset = TryOnTrainDataset(pairs_csv, resolution=RESOLUTION, pose_fn=counting_pose_fn([]))
    batch = collate_train_batch([dataset[0], dataset[1]])
    assert batch["x1"].shape == (2, 3, RESOLUTION[1], RESOLUTION[0])
    assert batch["garment_categories"].shape == (2,)
    assert batch["pair_id"] == ["pair0", "pair1"]
    assert batch["license_class"] == ["nc-dresscode", "nc-dresscode"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


