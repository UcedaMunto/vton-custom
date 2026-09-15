# `segmentation/` — proveedores de segmentación (sin licencias no comerciales)

Este paquete sustituye la dependencia original `fashn-human-parser` (cuyos pesos
heredan la licencia **no comercial** de NVIDIA SegFormer) por una interfaz
enchufable. **Ningún proveedor de este paquete tiene licencia no comercial** y el
pipeline rechaza cualquiera que la tenga (`info.commercial_ok`).

## Contrato

```python
from fashn_vton.segmentation import build_segmentation_provider

provider = build_segmentation_provider("none")  # default comercial
class_map = provider.predict(image_rgb_uint8, context={"pose": pose_dict, "category": "tops", "role": "person"})
# -> np.ndarray HxW uint8 con ids de labels.py, o None si no aporta máscara
```

- `extract_person_region(image, category, context=None)` y
  `extract_garment(image, category, context=None)` son los nombres usados en el
  plan comercial; por defecto delegan en `predict`.
- `is_available()` indica si el proveedor puede funcionar en este entorno y
  `require_available()` lanza `ProviderNotAvailable` con instrucciones.
- `info: ProviderInfo` describe `name`, `license`, `requires_weights`,
  `commercial_ok`, `experimental`, `weights_dir` y `notes`.

## Catálogo

| Nombre | Licencia | Estado | Cuándo usarlo |
|---|---|---|---|
| `none` | No carga modelos | **Listo** (default) | Camino comercial: `segmentation_free=True` + prenda `flat-lay`. Cero modelos de segmentación |
| `pose-heuristic` | Apache-2.0 (DWPose) + geometría propia | **Experimental** | Flujos que necesitan máscara sin añadir dependencias: regiones torso/brazos/piernas desde los keypoints COCO-18 que el pipeline ya calcula |
| `sam2` | Apache-2.0 | **Implementado (etapa 2)** | Máscaras precisas sin dependencias NC: cajas derivadas de DWPose → SAM 2. Verificado el 2026-09-15 (`sam2==1.1.0` + `facebook/sam2.1-hiera-tiny`) |
| `grounded-sam2` | Apache-2.0 | **Implementado (etapa 2b)** | Prendas fotografiadas **puestas por otra persona**: Grounding DINO (prompts de texto por categoría) → SAM 2. Verificado el 2026-09-15 (`IDEA-Research/grounding-dino-tiny` vía `transformers`) |
| `custom-parser` | Apache-2.0 (Detectron2) + pesos propios | Esqueleto (etapa 3) | Independencia total: parser propio entrenado con datos propios |

### Instalación de los proveedores opcionales

```bash
# SAM 2 (etapa 2)
pip install sam2==1.1.0
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('facebook/sam2.1-hiera-tiny', 'sam2.1_hiera_tiny.pt', local_dir='weights/sam2')"

# Grounding DINO (etapa 2b, se usa vía transformers: no hace falta compilar groundingdino)
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('IDEA-Research/grounding-dino-tiny', local_dir='weights/grounding_dino', \
  allow_patterns=['*.json','*.txt','*.safetensors'])"

# Comprobar hashes y disponibilidad
python licenses/verify_hashes.py
python -c "import json; from fashn_vton.segmentation import available_providers; print(json.dumps(available_providers(), indent=1, ensure_ascii=False))"
```

```bash
# Ver qué proveedores están disponibles y con qué licencia
python -c "import json; from fashn_vton.segmentation import available_providers; print(json.dumps(available_providers(), indent=1, ensure_ascii=False))"

# Comparar proveedores sobre los mismos pares (plan §14)
python scripts/benchmark_providers.py --examples --providers none,pose-heuristic
```

## Cómo actúa el pipeline cuando no hay máscara

Con `NoSegmentationProvider` (o cualquier proveedor que devuelva `None`):

- `segmentation_free=True` → la persona se genera sin enmascarar (comportamiento
  del modelo "maskless" de FASHN, idéntico al original).
- `segmentation_free=False` o `garment_photo_type="model"` → **no falla**: se
  procesa sin máscara, se registra un `WARNING` y queda constancia en
  `PipelineOutput.metadata["segmentation"]["degraded"]`.
- Con `strict_segmentation=True` en el constructor, esos casos lanzan excepción
  (útil para despliegues con QA estricto).

## Añadir un proveedor nuevo

1. Crea `mi_proveedor.py` con una clase que herede de `SegmentationProvider`.
2. Define `info = ProviderInfo(name="mi-proveedor", license=..., requires_weights=..., commercial_ok=True)`.
3. Implementa `predict(image, context=None)` devolviendo un mapa `HxW uint8` con
   los ids de `labels.py` (o `None`).
4. Regístralo en `PROVIDERS` (`__init__.py`).
5. Añade la entrada al inventario (`licenses/manifest.json` y
   `THIRD_PARTY_NOTICES.md`) **con el hash del checkpoint** y una prueba en
   `tests/test_segmentation_providers.py`.

Regla no negociable: si la licencia del checkpoint no permite uso comercial,
`commercial_ok` debe ser `False` y el proveedor **no se registra**.
