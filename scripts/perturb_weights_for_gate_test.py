"""Genera un candidato DELIBERADAMENTE empeorado para probar la guardia anti-regresión.

Perturba la capa final del MMDiT (la que proyecta a píxeles) con ruido, y guarda el
checkpoint en `weights/candidates/perturbado/`. No es un fine-tuning: es un
"peor modelo" controlado para demostrar que `baseline.py verify` lo rechaza.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[1]  # raíz del repo (este script vive en scripts/)
SRC = REPO / "weights" / "model.safetensors"
DST_DIR = REPO / "weights" / "candidates" / "perturbado"

state = load_file(str(SRC), device="cpu")
keys = [key for key in state if key.startswith("final_layer")]
print("claves final_layer:", keys)
assert keys, "no se encontró la capa final del modelo"

torch.manual_seed(0)
for key in keys:
    tensor = state[key]
    if tensor.is_floating_point() and tensor.ndim >= 2:
        noisy = tensor.float() + torch.randn_like(tensor.float()) * 0.08
        state[key] = noisy.to(tensor.dtype).contiguous()

DST_DIR.mkdir(parents=True, exist_ok=True)
save_file(state, str(DST_DIR / "model.safetensors"), metadata={"comment": "pesos perturbados a proposito (prueba del gate)"})
print("guardado:", DST_DIR / "model.safetensors", (DST_DIR / "model.safetensors").stat().st_size, "bytes")
