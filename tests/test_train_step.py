"""Prueba de un paso real de entrenamiento (`fashn_vton.train.trainer`).

Se usa un MMDiT diminuto (24x48) y pares sintéticos, así que **corre en CPU, sin
GPU y sin los pesos de 2 GB**: lo que valida es el arnés (pérdida de rectified
flow, `gradient_checkpointing`, checkpoints, `--resume` y que el candidato
mergeado se pueda cargar como un `TryOnModel` normal).

El `gradient_checkpointing` es el punto delicado en 12 GB (sin él las
activaciones de 576x864 no caben), por eso se prueba de verdad y no solo se
confía en que funcione.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from fashn_vton.train.data import PairRecord, write_pairs_csv
from fashn_vton.train.trainer import (
    TrainConfig,
    Trainer,
    conditional_mask,
    enable_gradient_checkpointing,
    flow_batch,
    flow_loss,
    load_base_model,
    sample_times,
)
from fashn_vton.tryon_mmdit import TryOnModel
from fashn_vton.utils import get_dummy_dw_keypoints

INPUT_SHAPE = (24, 48)
RESOLUTION = (48, 24)


def tiny_model(seed: int = 0) -> TryOnModel:
    torch.manual_seed(seed)
    return TryOnModel(
        input_shape=INPUT_SHAPE,
        hidden_size=64,
        n_heads=4,
        double_blocks_depth=1,
        single_blocks_depth=1,
        mlp_ratio=2,
        channels_in=3,
        patch_size=12,
        axes_dim=(4, 4, 8),
        qkv_bias=True,
        n_classes=3,
        use_patch_mixer=True,
        patch_mixer_depth=1,
    )


def fake_pose_fn(_bgr: np.ndarray) -> dict:
    return get_dummy_dw_keypoints()


def make_assets(tmp_path, n_pairs: int = 2) -> tuple[Path, Path]:
    """Pesos base diminutos + CSV de pares; devuelve `(pairs_csv, weights_dir)`."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_file(tiny_model().state_dict(), str(weights_dir / "model.safetensors"))

    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    person = np.dstack(
        [np.full((90, 60), 180, np.uint8), np.full((90, 60), 40, np.uint8), np.full((90, 60), 40, np.uint8)]
    )
    garment = np.dstack(
        [np.full((90, 60), 20, np.uint8), np.full((90, 60), 20, np.uint8), np.full((90, 60), 180, np.uint8)]
    )
    mask = np.zeros((90, 60), np.uint8)
    mask[:, :30] = 255

    records = []
    for index in range(n_pairs):
        person_path = images / f"p{index}.png"
        garment_path = images / f"g{index}.png"
        mask_path = masks / f"p{index}_mask.png"
        Image.fromarray(person).save(person_path)
        Image.fromarray(garment).save(garment_path)
        Image.fromarray(mask).save(mask_path)
        records.append(
            PairRecord(
                person=str(person_path),
                garment=str(garment_path),
                target=str(person_path),
                category="tops",
                agnostic=str(mask_path),
                license_class="nc-dresscode",
                source="unit-test",
                pair_id=f"pair{index}",
            )
        )
    return write_pairs_csv(tmp_path / "pairs.csv", records), weights_dir


def make_config(tmp_path, pairs_csv: Path, weights_dir: Path, **overrides) -> TrainConfig:
    payload = {
        "pairs_csv": str(pairs_csv),
        "output_dir": str(tmp_path / "candidates" / "tiny"),
        "weights_dir": str(weights_dir),
        "rank": 4,
        "max_steps": 2,
        "grad_accum": 2,
        "save_every": 1,
        "log_every": 1,
        "resolution": RESOLUTION,
        "mixed_precision": "fp32",
        "device": "cpu",
        "num_workers": 0,
        "merge_at_end": True,
        "pose_cache_dir": str(tmp_path / "pose_cache"),
        "provenance": {"dataset": "unit-test", "license_class": "nc-dresscode", "commercial_use": False},
    }
    payload.update(overrides)
    return TrainConfig(**payload)


# --------------------------------------------------------------------- rectified flow


def test_flow_batch_endpoints_and_velocity():
    clean = torch.rand(2, 3, 4, 4)
    noise = torch.rand(2, 3, 4, 4)
    x_at_one, velocity = flow_batch(clean, noise, torch.ones(2))
    torch.testing.assert_close(x_at_one, clean, atol=1e-6, rtol=0)
    torch.testing.assert_close(velocity, clean - noise, atol=0, rtol=0)
    x_at_zero, _ = flow_batch(clean, noise, torch.zeros(2))
    torch.testing.assert_close(x_at_zero, noise, atol=1e-6, rtol=0)


def test_sample_times_stay_inside_the_unit_interval():
    times = sample_times(512, torch.device("cpu"), torch.float32, mu=1.5)
    assert float(times.min()) > 0.0
    assert float(times.max()) < 1.0
    assert times.shape == (512,)


def test_conditional_mask_extremes():
    device = torch.device("cpu")
    assert bool(conditional_mask(4, 0.0, device).all())
    assert not bool(conditional_mask(64, 1.0, device).any())


def test_flow_loss_is_finite_and_backpropagates(tmp_path):
    pairs_csv, weights_dir = make_assets(tmp_path)
    model = load_base_model(weights_dir, device="cpu", dtype=torch.float32, model_factory=tiny_model)
    config = make_config(tmp_path, pairs_csv, weights_dir)
    batch = {
        "x1": torch.rand(1, 3, *INPUT_SHAPE),
        "ca_images": torch.rand(1, 3, *INPUT_SHAPE),
        "person_poses": torch.rand(1, 1, *INPUT_SHAPE),
        "garment_images": torch.rand(1, 3, *INPUT_SHAPE),
        "garment_poses": torch.rand(1, 1, *INPUT_SHAPE),
        "garment_categories": torch.tensor([1]),
    }
    from fashn_vton.train.lora import inject_lora, lora_parameters

    inject_lora(model, rank=4)
    loss, _, _ = flow_loss(model, batch, config)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in lora_parameters(model))


def test_gradient_checkpointing_requires_known_containers():
    class Empty(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError):
        enable_gradient_checkpointing(Empty())


# --------------------------------------------------------------------- corrida completa


def test_tiny_end_to_end_training_writes_a_usable_candidate(tmp_path):
    pairs_csv, weights_dir = make_assets(tmp_path)
    config = make_config(tmp_path, pairs_csv, weights_dir)
    trainer = Trainer(config, pose_fn=fake_pose_fn, model_factory=tiny_model).setup()
    assert trainer.report.replaced > 0
    assert trainer.checkpointed_blocks == 3  # patch mixer + double + single
    summary = trainer.run()

    out_dir = Path(summary["output_dir"])
    assert summary["steps"] == 2
    assert summary["micro_steps"] == 4
    assert summary["loss_first"] is not None
    assert np.isfinite(summary["loss_last"])
    assert (out_dir / "lora" / "adapter.safetensors").is_file()
    assert (out_dir / "state.json").is_file()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "train_config.json").is_file()
    assert (out_dir / "provenance.json").is_file()
    log_lines = [json.loads(line) for line in (out_dir / "train_log.jsonl").read_text().splitlines()]
    assert log_lines and log_lines[-1]["step"] == 2

    provenance = json.loads((out_dir / "provenance.json").read_text())
    assert provenance["commercial_use"] is False

    # El candidato es un checkpoint normal: se carga en un TryOnModel limpio.
    merged = Path(summary["merged_checkpoint"])
    assert merged.is_file()
    tiny_model(seed=99).load_state_dict(load_file(str(merged)))

    # El adaptador tiene deltas no nulos (de verdad entrenó).
    assert summary["delta_norm"] > 0
    state = json.loads((out_dir / "state.json").read_text())
    assert state["last_step"] == 2


def test_resume_continues_from_the_saved_step(tmp_path):
    pairs_csv, weights_dir = make_assets(tmp_path)
    first = make_config(tmp_path, pairs_csv, weights_dir)
    Trainer(first, pose_fn=fake_pose_fn, model_factory=tiny_model).setup().run()

    second = make_config(tmp_path, pairs_csv, weights_dir, max_steps=4, resume=True)
    trainer = Trainer(second, pose_fn=fake_pose_fn, model_factory=tiny_model).setup()
    summary = trainer.run()
    assert summary["steps"] == 4
    assert summary["micro_steps"] == 8


def test_training_without_gradient_checkpointing_also_works(tmp_path):
    pairs_csv, weights_dir = make_assets(tmp_path)
    config = make_config(tmp_path, pairs_csv, weights_dir, gradient_checkpointing=False, max_steps=1, grad_accum=1)
    summary = Trainer(config, pose_fn=fake_pose_fn, model_factory=tiny_model).setup().run()
    assert summary["steps"] == 1
    assert summary["delta_norm"] > 0


def test_checkpoint_written_when_merge_is_disabled(tmp_path):
    pairs_csv, weights_dir = make_assets(tmp_path)
    config = make_config(tmp_path, pairs_csv, weights_dir, merge_at_end=False, max_steps=1, grad_accum=1)
    summary = Trainer(config, pose_fn=fake_pose_fn, model_factory=tiny_model).setup().run()
    assert summary["merged_checkpoint"] is None
    assert not (Path(summary["output_dir"]) / "model.safetensors").exists()
    assert Path(summary["adapter"]).is_file()


def test_config_roundtrip(tmp_path):
    config = TrainConfig(pairs_csv="x.csv", targets=("qkv",), resolution=(576, 864))
    path = config.save(tmp_path / "config.json")
    reloaded = TrainConfig.load(path)
    assert reloaded.pairs_csv == "x.csv"
    assert reloaded.targets == ("qkv",)
    assert reloaded.resolution == (576, 864)
    # Claves desconocidas en el JSON se ignoran (compatibilidad hacia delante).
    payload = json.loads(path.read_text())
    payload["campo_futuro"] = 1
    path.write_text(json.dumps(payload))
    assert TrainConfig.load(path).pairs_csv == "x.csv"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


