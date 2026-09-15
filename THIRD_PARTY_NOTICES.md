# Third-party notices — fork comercial de FASHN VTON v1.5

Este archivo es el aviso público de componentes de terceros del fork. La versión
machine-readable (con hashes) está en [`licenses/manifest.json`](licenses/manifest.json)
y se verifica con [`licenses/verify_hashes.py`](licenses/verify_hashes.py).

## Qué usa este producto

| Componente | Fuente | Licencia | Uso en el producto |
|---|---|---|---|
| FASHN VTON v1.5 (código y pesos `model.safetensors`) | `fashn-AI/fashn-vton-1.5` | Apache-2.0 | Modelo de try-on (núcleo) |
| DWPose (ONNX: `yolox_l.onnx`, `dw-ll_ucoco_384.onnx`) | `fashn-ai/DWPose` | Apache-2.0 | Detección de persona y pose |
| YOLOX | `Megvii-BaseDetection/YOLOX` | Apache-2.0 | Detector incluido en DWPose |
| PyTorch / torchvision | pytorch.org | BSD-3-Clause | Runtime de inferencia |
| NumPy, SciPy, psutil | PyPI | BSD-3-Clause | Utilidades |
| Pillow | PyPI | MIT-CMU (HPND) | Imágenes |
| OpenCV (`opencv-python`) | opencv.org / PyPI | Apache-2.0 (OpenCV) / MIT (wrapper) | Preprocesado y máscaras |
| tqdm | PyPI | MPL-2.0 / MIT (dual) | Barra de progreso |
| einops | PyPI | MIT | Modelo (`tryon_mmdit`) |
| onnxruntime-gpu | microsoft/onnxruntime | MIT | Ejecución de los ONNX de pose |
| safetensors, huggingface_hub | HuggingFace | Apache-2.0 | Carga de pesos |
| matplotlib | matplotlib.org | PSF-based (BSD-compatible) | Utilidades |
| FastAPI, Starlette, Uvicorn, python-multipart | PyPI | MIT / BSD-3-Clause / BSD-3-Clause / Apache-2.0 | API de servicio (extra `api`) |
| Gradio | `gradio-app/gradio` | Apache-2.0 | Interfaz web de pruebas (extra `webui`) |

## Motor de segmentación (el motivo de este fork)

El pipeline original de FASHN VTON v1.5 dependía de **`fashn-human-parser`**,
cuyos pesos heredan la licencia de **NVIDIA SegFormer**, que restringe el uso a
investigación/evaluación **no comercial**. Este fork **elimina** esa dependencia
y la sustituye por un proveedor de segmentación enchufable
(`src/fashn_vton/segmentation`):

| Proveedor | Licencia | Estado |
|---|---|---|
| `none` (por defecto) | No carga ningún modelo | **Listo** — camino comercial: `segmentation_free=True` + prendas `flat-lay` |
| `pose-heuristic` | Apache-2.0 (DWPose) + geometría propia | **Experimental** — sin parser semántico; validar con benchmark |
| `sam2` | Apache-2.0 (SAM 2, código y checkpoints) | **Implementado** — `pip install sam2==1.1.0` + `facebook/sam2.1-hiera-tiny` (156 MB, hash en el manifiesto) |
| `grounded-sam2` | Apache-2.0 (Grounding DINO + SAM 2) | **Implementado** — `IDEA-Research/grounding-dino-tiny` vía `transformers` (689 MB, hash en el manifiesto) + SAM 2; pensado para prendas puestas por modelos |
| `custom-parser` | Apache-2.0 (Detectron2) + pesos propios | Esqueleto documentado (etapa 3) |

**Ningún proveedor de este producto tiene licencia no comercial.** El código
comprueba además que no se pueda enchufar uno que la tenga
(`SegmentationProvider.info.commercial_ok`, ver `pipeline._setup_segmentation`).

## Datos

- Este fork **no** entrena ni evalúa con VITON-HD ni DressCode (licencias NC y de
  cesión). Si se hace fine-tuning, el dataset debe ser propio o con licencia
  comercial explícita, con trazabilidad por imagen.
- Los ejemplos de `examples/data/` provienen del repositorio Apache-2.0 de FASHN.

## Cómo verificar (antes de cada release)

```bash
python licenses/verify_hashes.py                  # hashes + artefactos prohibidos
python scripts/verify_commercial_readiness.py     # gate completo (fuente, entorno, pesos, proveedores)
```

Cualquier `MISMATCH` o «PROHIBIDOS PRESENTES» bloquea la publicación.

## Aviso

Este documento **no es asesoría legal**. Las conclusiones se basan en las
licencias publicadas por cada proyecto y en hashes verificados localmente; una
revisión legal profesional sigue siendo necesaria antes del lanzamiento
comercial (ver `plan_modelo_comercial/propuestas_fashn_vton_comercial.md` §15).
