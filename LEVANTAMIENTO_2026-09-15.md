# Levantamiento de FASHN VTON v1.5 en esta máquina (2026-09-15)

> **Actualización (2026-09-15, más tarde el mismo día): fork comercial aplicado.**
> Este documento describe el levantamiento del clon original. Después se aplicó
> el plan `plan_modelo_comercial/propuestas_fashn_vton_comercial.md`:
> `fashn-human-parser` (licencia no comercial de NVIDIA SegFormer) **ya no está
> instalado ni es una dependencia**; la segmentación es un proveedor enchufable
> y el camino comercial (`flat-lay` + `segmentation_free`) produce exactamente
> el mismo resultado que antes. Ver
> [`plan_modelo_comercial/PLAN_IMPLEMENTACION_COMERCIAL.md`](plan_modelo_comercial/PLAN_IMPLEMENTACION_COMERCIAL.md)
> y [`README_COMERCIAL.md`](README_COMERCIAL.md). El punto 1 de «Pendiente» de
> más abajo (sobre el parser) ya no aplica.

Nota de operación escrita al dejar el proyecto **funcionando** en
`/home/uceda/Documents/fashn-vton-1.5`. No forma parte del repositorio original
(clon de `fashn-AI/fashn-vton-1.5`, Apache-2.0, commit `7c0f10a`); describe qué
se instaló, cómo se ejecuta y los detalles que el README del repo no cubre.

## Estado

- **Funciona de punta a punta**: `examples/basic_inference.py` genera
  `outputs/output_00.png` (576x864 RGB) a partir de `examples/data/model.webp`
  + `examples/data/garment.webp`.
- Entorno conda dedicado: **`fashn-vton`** (Python 3.11.16), separado de
  `idm-custom` para no contaminar aquel proyecto.
- Pesos: `weights/` (2,2 GB, gitignored).
- GPU: RTX 3060 (12 GB). El modelo de try-on corre en **bfloat16** en GPU;
  la detección de pose (DWPose/ONNX) corre en GPU **solo** si se usa el
  lanzador `run_fashn_vton.sh` (ver "Problema conocido" abajo).

## Cómo ejecutar

```bash
cd /home/uceda/Documents/fashn-vton-1.5

# CLI del README (recomendado: usa el lanzador, que arregla LD_LIBRARY_PATH)
./run_fashn_vton.sh python examples/basic_inference.py \
    --weights-dir ./weights \
    --person-image examples/data/model.webp \
    --garment-image examples/data/garment.webp \
    --category tops

# Alternativas
./run_fashn_vton.sh            # ayuda + variables del entorno
./run_fashn_vton.sh python     # REPL con el entorno listo

# Sin el lanzador (la pose cae a CPU silenciosamente, ver más abajo):
/home/uceda/miniconda3/envs/fashn-vton/bin/python examples/basic_inference.py ...
```

Opciones útiles del CLI: `--category tops|bottoms|one-pieces`,
`--garment-photo-type model|flat-lay`, `--num-timesteps 20|30|50`,
`--num-samples N`, `--seed`, `--guidance-scale`, `--no-segmentation-free`,
`--device cuda|cpu`, `--output-dir`.

API Python:

```python
from fashn_vton import TryOnPipeline
from PIL import Image
pipe = TryOnPipeline(weights_dir="./weights")     # usa GPU si está disponible
res = pipe(person_image=Image.open("person.jpg").convert("RGB"),
           garment_image=Image.open("garment.jpg").convert("RGB"),
           category="tops")                        # -> res.images (lista de PIL)
res.images[0].save("out.png")
```

## Qué se instaló (reproducible)

```bash
conda create -n fashn-vton python=3.11 -y
conda activate fashn-vton
cd /home/uceda/Documents/fashn-vton-1.5
pip install -e .                      # trae torch, torchvision, onnxruntime-gpu, etc.

# Pesos (~2,2 GB): model.safetensors + dwpose/{yolox_l.onnx, dw-ll_ucoco_384.onnx}
python scripts/download_weights.py --weights-dir ./weights
# (los pesos del human parser, ~244 MB, se descargan solos en la primera
#  inferencia a la caché de HuggingFace: ~/.cache/huggingface)
```

Versiones resultantes (verificadas): torch 2.14.0+cu130, torchvision 0.29.0,
onnxruntime-gpu 1.30.0, transformers 5.17.0, safetensors 0.8.0,
huggingface_hub 1.31.0, opencv-python 5.0.0.93, numpy 2.4.6, pillow 12.3.0,
matplotlib 3.11.2, fashn-human-parser 0.1.1. Conjunto CUDA 13 (`nvidia-*`,
`cuda-toolkit 13.0.3`). No se necesitó `HF_TOKEN`.

Consumo en disco: entorno ≈ **6,4 GB**, pesos **2,2 GB**, ~1,5 GB en la caché de
pip (los wheels CUDA ya estaban cacheados de otra instalación) y ~244 MB del
human parser en la caché de HuggingFace.

## Problema conocido (resuelto con el lanzador): pose en CPU

`onnxruntime-gpu` no encuentra por sí solo las librerías CUDA que PyTorch instala
como paquetes `nvidia-*` dentro de `site-packages`:

```
Failed to load library .../libonnxruntime_providers_cuda.so with error:
libcublasLt.so.13: cannot open shared object file
→ providers_efectivos ['CPUExecutionProvider']
```

Síntoma visible: los avisos
`No registered plugin EP device found for 'CUDAExecutionProvider' with device_id=0`
en cada corrida, y la detección de pose corriendo en CPU (el resto del pipeline,
el modelo de difusión, sí usa la GPU siempre).

Solución (aplicada en `run_fashn_vton.sh`, sin tocar el código del repo):

```bash
export LD_LIBRARY_PATH=$(find <env>/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | paste -sd:)
```

Con eso, `onnxruntime` crea el `CUDAExecutionProvider` correctamente
(comprobado con `InferenceSession(...).get_providers()`); el aviso
"plugin EP device" sigue apareciendo pero es ruido de ORT 1.30 (el EP sí se usa).

## Verificación hecha (2026-09-15)

| Prueba | Comando | Resultado |
|---|---|---|
| Instalación | `pip install -e .` | `EXIT=0`, paquete `fashn-vton 1.5.0` editable |
| Pesos | `hf_hub_download` (mismo repo que el script) | `model.safetensors` 1,94 GB + `dwpose/*.onnx` (217 MB + 134 MB) |
| CUDA (torch) | `torch.cuda.is_available()` | `True`, RTX 3060, `is_bf16_supported() == True` |
| Comando del README (30 pasos) | `basic_inference.py ... --category tops` | `EXIT=0`; sampling 30/30 en **43 s** (~1,46 s/paso); **58,9 s** de tiempo total del proceso (carga de modelos incluida) |
| Salida | `outputs/output_00.png` | 576x864 RGB, contenido real (std 58,6; MAD 27,8 vs. persona) |
| Condicionamiento en la prenda | misma persona + prenda roja real vs. azul real | torso con prenda roja (124,57,46) vs. azul (60,51,48): **el modelo responde al color de la prenda** |
| ORT con GPU | `InferenceSession(... providers=['CUDAExecutionProvider'])` | `['CUDAExecutionProvider','CPUExecutionProvider']` |

Salidas de prueba generadas (todas gitignored): `outputs/` (documentada),
`outputs_smoke/`, `outputs_test2/`, `outputs_test_blue/`,
`outputs_test_048508_0/` (prenda roja), `outputs_test_048394_0/` (prenda azul),
`outputs_launcher/`, `outputs_timed/`.

## Pendiente / a tener en cuenta

- **No hay interfaz web en este repositorio** (solo CLI + API Python). La demo
  oficial vive en Hugging Face Spaces (`fashn-ai/fashn-vton-1.5`), no aquí.
- **Calidad visual**: no se pudo inspeccionar la imagen con ojos (el agente no
  puede ver imágenes). Las comprobaciones numéricas indican que el pipeline
  funciona y condiciona con la prenda, pero conviene que el usuario mire
  `outputs/output_00.png` y juzgue el resultado: con una prenda real roja el
  cambio es claro y correcto, mientras que con prendas muy oscuras/azuladas el
  torso salió más apagado de lo esperado (posible comportamiento del modelo con
  tonos oscuros, no un fallo de instalación).
- **Forma de salida**: siempre 576x864 (forma de entrada del modelo); la persona
  se redimensiona respetando aspecto.
- **GPU compartida**: este proyecto no usa `infra.gpu_lock`; si se corre junto
  con el entrenamiento de la Pista A o con `IDM-CUSTOM`, hay que turnarse la GPU
  a mano (el modelo ocupa ~2 GB en bf16 + activaciones).
- `weights/.cache/huggingface` dentro de `weights/` lo crea `hf_hub_download`;
  es inocuo y está gitignored junto con `weights/`.
- Archivos añadidos por este levantamiento (no del repo original):
  `run_fashn_vton.sh` y este `LEVANTAMIENTO_2026-09-15.md`.

## Actualización posterior (misma fecha): arreglos tras probar la UI

Al probar la interfaz en el navegador el usuario no percibió cambios. Se revisó
con evidencia y aparecieron **dos huecos reales** (ya corregidos) más un fallo de
CI que el propio clon arrastraba:

- **El resultado es correcto**: la imagen del camino comercial es idéntica byte a
  byte a la de antes del fork (`sha256 a4d618bf…`), también pedida desde la UI. El
  fork cambia licencias y arquitectura, no la salida del modelo.
- **La UI no mostraba el proveedor usado**: ahora su log añade
  `Segmentación: proveedor=… · licencia=… · máscara persona=… · máscara prenda=…` y,
  cuando corresponde, `AVISO (degradado, resultado no óptimo): …`.
- **La UI acumulaba VRAM al cambiar de proveedor** (9,5 GiB usados tras
  `none`→`sam2`): la caché guarda ahora **un solo** pipeline y libera el anterior.
- **`ruff check` (lint de CI) fallaba con 8 errores** (2 del fork y 6 preexistentes
  del clon): corregidos; `ruff check src scripts tests` → `All checks passed!`.
- **Artefacto nuevo**: `scripts/smoke_webui.py` — prueba de humo de la UI que
  termina en `TODO OK` y comprueba `sha256 a4d618bf…`.
- **Pruebas**: `pytest tests/ -q` → **107 passed** (70 del fork + 37 del clon),
  ejecutadas después de todos los cambios de esta sesión.
