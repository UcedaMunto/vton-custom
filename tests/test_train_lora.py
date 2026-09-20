"""Pruebas del LoRA propio del arnés de entrenamiento (`fashn_vton.train.lora`).

Solo lógica y un MMDiT **diminuto** (24x48 px, 1 bloque double + 1 single):
sin GPU, sin pesos descargados y en menos de un segundo. Fijan los invariantes
en los que se apoya el plan de fine-tuning:

1. Inyectar LoRA no cambia la salida (el punto de partida es el modelo actual).
2. Solo el adaptador recibe gradiente (la base queda congelada).
3. `merge`/`unmerge` son exactos.
4. El `state_dict` mergeado se carga con `strict=True` en un `TryOnModel` limpio
   (es lo que hará `TryOnPipeline` con el candidato).
"""

from __future__ import annotations

import pytest
import torch

from fashn_vton.train.lora import (
    LoRALinear,
    inject_lora,
    load_lora,
    lora_delta_norm,
    lora_modules,
    lora_parameters,
    merge_lora,
    merged_state_dict,
    save_lora,
    save_merged_checkpoint,
    unmerge_lora,
)
from fashn_vton.tryon_mmdit import TryOnModel

INPUT_SHAPE = (24, 48)
PATCH_SIZE = 12


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
        patch_size=PATCH_SIZE,
        axes_dim=(4, 4, 8),
        qkv_bias=True,
        n_classes=3,
        use_patch_mixer=True,
        patch_mixer_depth=1,
    )


def tiny_cond(batch_size: int = 1) -> dict:
    return {
        "ca_images": torch.rand(batch_size, 3, *INPUT_SHAPE),
        "person_poses": torch.rand(batch_size, 1, *INPUT_SHAPE),
        "garment_images": torch.rand(batch_size, 3, *INPUT_SHAPE),
        "garment_poses": torch.rand(batch_size, 1, *INPUT_SHAPE),
        "garment_categories": torch.tensor([1] * batch_size, dtype=torch.long),
    }


def forward_output(model: TryOnModel, cond: dict, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 3, *INPUT_SHAPE, generator=generator)
    times = torch.tensor([0.5])
    return model(x, times, **cond)["x"]


# --------------------------------------------------------------------- inyección


def test_injection_is_identity_at_start():
    model = tiny_model()
    cond = tiny_cond()
    with torch.no_grad():
        before = forward_output(model, cond)

    report = inject_lora(model, rank=4)
    assert report.replaced > 0
    assert report.trainable_parameters > 0
    assert report.trainable_parameters < report.total_parameters

    with torch.no_grad():
        after = forward_output(model, cond)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert lora_delta_norm(model) == 0.0


def test_targets_are_attention_and_mlp_but_never_convolutions():
    model = tiny_model()
    report = inject_lora(model, rank=2)
    names = report.module_names
    assert "double_blocks.0.img_attn.qkv" in names
    assert "double_blocks.0.txt_attn.proj" in names
    assert "single_blocks.0.linear1" in names
    assert "x_patch_mixer.0.linear2" in names
    # `x_embedder.proj` es una Conv2d (patch embed) y no debe tocarse.
    assert all("x_embedder" not in name for name in names)
    assert all("garment_embedder" not in name for name in names)


def test_unknown_targets_raise():
    model = tiny_model()
    with pytest.raises(RuntimeError):
        inject_lora(model, targets=("no_existe",), rank=2)


def test_invalid_rank_raises():
    model = tiny_model()
    with pytest.raises(ValueError):
        inject_lora(model, rank=0)


def test_lora_linear_rejects_non_linear():
    with pytest.raises(TypeError):
        LoRALinear(torch.nn.Conv2d(3, 3, 1), rank=2, alpha=2)


# --------------------------------------------------------------------- gradientes


def test_only_adapter_receives_gradients():
    model = tiny_model()
    inject_lora(model, rank=4)
    output = forward_output(model, tiny_cond())
    output.pow(2).mean().backward()

    base_with_grad = [
        name
        for name, parameter in model.named_parameters()
        if not name.endswith(("lora_a", "lora_b")) and parameter.grad is not None
    ]
    assert base_with_grad == []

    adapters = lora_modules(model)
    assert len(adapters) > 0
    assert all(module.lora_b.grad is not None for module in adapters.values())
    # `lora_b` nace a cero, así que su gradiente puede ser cero, pero no None.
    assert any(module.lora_b.grad.abs().sum() > 0 for module in adapters.values())


def test_one_step_changes_output_and_delta_norm():
    model = tiny_model()
    inject_lora(model, rank=4)
    with torch.no_grad():
        before = forward_output(model, tiny_cond())

    optimizer = torch.optim.SGD(lora_parameters(model), lr=0.5)
    output = forward_output(model, tiny_cond())
    loss = (output - torch.ones_like(output)).pow(2).mean()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        after = forward_output(model, tiny_cond())
    assert not torch.allclose(before, after)
    assert lora_delta_norm(model) > 0.0


# --------------------------------------------------------------------- merge


def test_merge_and_unmerge_are_exact():
    model = tiny_model()
    inject_lora(model, rank=3)
    for module in lora_modules(model).values():
        with torch.no_grad():
            module.lora_b.normal_(0, 0.02)
    base_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    cond = tiny_cond()
    with torch.no_grad():
        reference = forward_output(model, cond)
    merge_lora(model)
    with torch.no_grad():
        torch.testing.assert_close(forward_output(model, cond), reference, atol=1e-6, rtol=1e-5)
    unmerge_lora(model)
    with torch.no_grad():
        torch.testing.assert_close(forward_output(model, cond), reference, atol=1e-6, rtol=1e-5)

    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, base_state[key], atol=1e-5, rtol=1e-4)


def test_merged_state_dict_has_original_names_and_loads_strictly():
    model = tiny_model()
    inject_lora(model, rank=3)
    for module in lora_modules(model).values():
        with torch.no_grad():
            module.lora_b.normal_(0, 0.05)

    state = merged_state_dict(model)
    assert not any(key.endswith(("lora_a", "lora_b")) for key in state)
    assert not any(".base." in key for key in state)
    assert set(state) == set(tiny_model(seed=2).state_dict())

    # Un `TryOnModel` limpio (misma base, sin adaptadores) carga el candidato
    # con strict=True, exactamente como lo hará `TryOnPipeline`.
    plain = tiny_model()
    plain.load_state_dict(state)
    cond = tiny_cond()
    with torch.no_grad():
        torch.testing.assert_close(forward_output(plain, cond), forward_output(model, cond), atol=1e-6, rtol=1e-5)


def test_merged_state_dict_ignores_double_counting_after_merge():
    model = tiny_model()
    inject_lora(model, rank=2)
    for module in lora_modules(model).values():
        with torch.no_grad():
            module.lora_b.normal_(0, 0.05)
    with torch.no_grad():
        reference = merged_state_dict(model)
    merge_lora(model)
    merged_state = merged_state_dict(model)
    for key, value in reference.items():
        torch.testing.assert_close(value, merged_state[key], atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------- ficheros


def test_save_and_load_adapter_roundtrip(tmp_path):
    model = tiny_model()
    inject_lora(model, rank=4)
    for module in lora_modules(model).values():
        with torch.no_grad():
            module.lora_b.normal_(0, 0.1)
    path = save_lora(model, tmp_path / "lora" / "adapter.safetensors", metadata={"step": 7})
    assert path.is_file()

    reloaded = tiny_model()  # misma base (misma semilla) + adaptador cargado
    inject_lora(reloaded, rank=4)
    assert load_lora(reloaded, path) == len(lora_modules(model))

    cond = tiny_cond()
    with torch.no_grad():
        torch.testing.assert_close(
            forward_output(reloaded, cond), forward_output(model, cond), atol=1e-7, rtol=1e-6
        )


def test_load_adapter_against_model_without_adapters_raises(tmp_path):
    model = tiny_model()
    inject_lora(model, rank=2)
    path = save_lora(model, tmp_path / "adapter.safetensors")
    with pytest.raises(KeyError):
        load_lora(tiny_model(seed=2), path)


def test_save_merged_checkpoint_is_a_normal_checkpoint(tmp_path):
    from safetensors.torch import load_file

    model = tiny_model()
    inject_lora(model, rank=2)
    for module in lora_modules(model).values():
        with torch.no_grad():
            module.lora_b.normal_(0, 0.01)
    path = save_merged_checkpoint(model, tmp_path / "model.safetensors", metadata={"step": 3})
    tiny_model(seed=3).load_state_dict(load_file(str(path)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

