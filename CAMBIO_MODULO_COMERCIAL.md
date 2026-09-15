# Cambio necesario para hacer el módulo comercial — FASHN VTON v1.5

**Qué es este documento.** La especificación de cambio, en orden de dependencia,
que convierte el repositorio upstream [`fashn-AI/fashn-vton-1.5`](https://github.com/fashn-AI/fashn-vton-1.5)
(clon en `7c0f10a`, Apache-2.0) en un **módulo utilizable comercialmente**.
Está escrito a nivel de *cambio*: causa → cambio → archivos → antes/después →
cómo se verifica. Sirve para tres cosas:

1. **Auditar** qué se tocó y por qué (por ejemplo, en una revisión legal o de compras).
2. **Reaplicarlo** sobre un clon limpio, en orden, sin saltarse pasos (Apéndice C).
3. **Completarlo**: la Etapa 3 (parser propio) todavía está pendiente (§9).

Documentos que lo complementan (no se duplican aquí):
[`README_COMERCIAL.md`](README_COMERCIAL.md) (uso) ·
[`plan_modelo_comercial/PLAN_IMPLEMENTACION_COMERCIAL.md`](plan_modelo_comercial/PLAN_IMPLEMENTACION_COMERCIAL.md)
(plan y evidencia) · [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) (licencias) ·
[`licenses/manifest.json`](licenses/manifest.json) (inventario con hashes) ·
[`src/fashn_vton/segmentation/README.md`](src/fashn_vton/segmentation/README.md) (contrato de proveedores).

## Resumen: los 7 cambios

| # | Cambio | Archivos clave | Por qué es obligatorio |
|---|---|---|---|
| 1 | **Abstracción del motor de segmentación** (el cambio central) | `src/fashn_vton/segmentation/` (nuevo), `src/fashn_vton/pipeline.py` | El pipeline estaba acoplado a `fashn-human-parser`, cuyos pesos heredan la licencia **no comercial** de NVIDIA SegFormer |
| 2 | **Camino comercial explícito** y con huella verificable | `pipeline.py`, `preprocessing/agnostic.py` | Permite operar sin segmentación y *demostrar* que el resultado no cambió (byte a byte) |
| 3 | **Dependencias, pesos y etiquetas** | `pyproject.toml`, `scripts/download_weights.py`, `src/fashn_vton/segmentation/labels.py` | Sin la dependencia NC, sin sus pesos en caché y sin depender de su tabla de etiquetas |
| 4 | **Cumplimiento automatizado** (gate + CI + imagen) | `src/fashn_vton/compliance.py`, `licenses/`, `scripts/verify_commercial_readiness.py`, `.github/workflows/ci.yml`, `Dockerfile` | Una promesa de licencia que no se comprueba en cada build no sirve para producción |
| 5 | **Interfaces de producto** (CLI, API, UI) con licencia visible | `examples/basic_inference.py`, `src/fashn_vton/api/`, `webui/` | El módulo debe poder consumirse y decir por sí mismo qué licencia usó |
| 6 | **Métricas de calidad y umbrales** | `src/fashn_vton/eval/`, `scripts/benchmark_providers.py` | La calidad del camino sin parser hay que medirla, no suponerla (plan §14) |
| 7 | **Datos**: nada de VITON-HD/DressCode | `THIRD_PARTY_NOTICES.md`, `licenses/manifest.json` | Ambos datasets están prohibidos para uso comercial/empresarial |

Regla de oro que gobierna todo lo anterior: **el módulo comercial no cambia el
modelo ni la calidad; cambia de dónde salen las máscaras y lo demuestra.**

---

## 1. El bloqueo: por qué el repo upstream no es comercial

El pipeline original dependía de `fashn-human-parser` en **cinco** puntos. Sus
pesos se derivan de **NVIDIA SegFormer**, cuya licencia limita el uso a
investigación/evaluación **no comercial**; por tanto todo el producto quedaba
contaminado por esa dependencia, aunque el resto del repo fuese Apache-2.0.

| Punto de acoplamiento | Antes (upstream) |
|---|---|
| Dependencia declarada | `pyproject.toml:40` → `"fashn-human-parser>=0.1.1",` |
| Import del pipeline | `pipeline.py:11` → `from fashn_human_parser import CATEGORY_TO_BODY_COVERAGE, FashnHumanParser` |
| Instanciación incondicional | `pipeline.py:138-143` → `self.logger.info("Loading FashnHumanParser")` + `self.hp_model = FashnHumanParser(device=hp_device)` |
| Tabla de etiquetas del preprocesado | `preprocessing/agnostic.py:8` → `from fashn_human_parser import BODY_COVERAGE_TO_LABELS, IDENTITY_LABELS, LABELS_TO_IDS` |
| Descarga de pesos + herramientas | `scripts/download_weights.py:52-55` (`def download_human_parser()`), `scripts/debug_masks.py`, `examples/basic_inference.py` |

Además, el paquete cacheaba sus pesos en
`~/.cache/huggingface/models--fashn-ai--fashn-human-parser` (**245 MB**): había
que purgarlos, porque una imagen de producción con esos pesos dentro seguiría
teniendo la licencia NC aunque el código ya no los importara.

**Criterio de aceptación del cambio 1:** `importlib.util.find_spec("fashn_human_parser") is None`
y una generación real que funcione sin él.

---

## 2. Cambio 1 (el central): abstracción del motor de segmentación

### 2.1 El contrato que sustituye al parser

Nuevo paquete `src/fashn_vton/segmentation/` con `base.py`, `labels.py`, `none.py`,
`pose_heuristic.py`, `sam2.py`, `grounded_sam2.py`, `custom_parser.py`, `__init__.py`.
El contrato completo (cópialo tal cual si implementas un proveedor nuevo):

```python
@dataclass(frozen=True)
class ProviderInfo:
    name: str                 # nombre en el registro ("sam2", ...)
    license: str              # texto humano, se muestra en API/UI/gate
    requires_weights: bool
    commercial_ok: bool = True   # False => el pipeline lo RECHAZA
    experimental: bool = False   # avisa: calidad no validada
    weights_dir: str | None = None
    notes: str = ""              # instrucciones si no está disponible

class SegmentationProvider(ABC):
    info: ProviderInfo

    @abstractmethod
    def predict(self, image: np.ndarray, context: dict | None = None) -> np.ndarray | None:
        """RGB uint8 HxWx3 -> mapa HxW uint8 con ids de labels.py, o None si no
        aporta máscara. context = {"pose": ..., "category": ..., "role": "person"|"garment"}."""

    def is_available(self) -> bool: ...        # ¿puede correr en este entorno?
    def require_available(self) -> None        # lanza ProviderNotAvailable con `notes`
    def describe(self) -> str                  # "sam2 (licencia: Apache-2.0 ...) [experimental]"
    def extract_person_region(self, image, category, context=None)  # nombres del plan §11
    def extract_garment(self, image, category, context=None)

def validate_class_map(class_map, provider) -> np.ndarray | None:
    """Comprueba 2 dimensiones, no vacío y rango [0, NUM_CLASSES-1]."""
```

Registro y fábrica (`segmentation/__init__.py`):

```python
PROVIDERS: dict[str, type[SegmentationProvider]] = {
    "none": NoSegmentationProvider,            # DEFAULT_PROVIDER, camino comercial
    "pose-heuristic": PoseHeuristicProvider,
    "sam2": Sam2SegmentationProvider,
    "grounded-sam2": GroundedSam2Provider,
    "custom-parser": CustomHumanParserProvider,
}
DEFAULT_PROVIDER = "none"

build_segmentation_provider(name: str | None = None, **kwargs) -> SegmentationProvider
available_providers() -> dict[str, dict]   # name, license, available, commercial_ok, experimental
```

### 2.2 Antes y después en el pipeline

| Aspecto | Antes (upstream) | Después (módulo comercial) |
|---|---|---|
| Import | `from fashn_human_parser import CATEGORY_TO_BODY_COVERAGE, FashnHumanParser` | `from .segmentation import NoSegmentationProvider, SegmentationProvider, validate_class_map` |
| Construcción | `FashnHumanParser(device=hp_device)` siempre | `TryOnPipeline(..., segmentation_provider=<provider>, strict_segmentation=False)`; por defecto `NoSegmentationProvider()` (**no carga nada**) |
| Cambio de motor | editar el código | `build_segmentation_provider("sam2")` (CLI `--segmentation-provider`, API `provider`, UI desplegable) |
| Salida | máscara del parser o fallo | máscara validada, o `None` + **degradación explícita** en `metadata["segmentation"]["degraded"]` (con `strict_segmentation=True` sí lanza) |
| Licencia | implícita | `ProviderInfo.license` + rechazo en `_setup_segmentation()` si `commercial_ok=False` |

Los dos puntos donde vive la garantía legal (`pipeline.py`):

```python
def _setup_segmentation(self):
    provider = self.segmentation_provider
    if not provider.info.commercial_ok:
        raise RuntimeError(
            f"El proveedor de segmentación '{provider.info.name}' tiene licencia "
            f"'{provider.info.license}' y está marcado como NO comercial: "
            "el fork comercial no lo permite (ver THIRD_PARTY_NOTICES.md)."
        )
    self.logger.info(f"Segmentation provider: {provider.describe()}")
    if provider.info.experimental:
        self.logger.warning("Proveedor de segmentación EXPERIMENTAL ('%s') ...", provider.info.name)

def _provider_class_map(self, image, role, context):
    try:
        provider.require_available()
        return validate_class_map(provider.predict(image, {**context, "role": role}), provider)
    except Exception as exc:
        if self.strict_segmentation:
            raise
        self.logger.warning("... Se continúa sin máscara (degradación explícita).")
        return None
```

### 2.3 Catálogo de proveedores resultante

| Proveedor | Licencia | Estado | Cómo obtiene la máscara | Coste medido (RTX 3060, 8 pasos) |
|---|---|---|---|---|
| `none` | No carga ningún modelo | **Listo** (default) | N/A: habilita el camino sin máscara | 13,1 s totales, 0 MiB extra |
| `pose-heuristic` | Apache-2.0 (DWPose) + geometría propia | **Experimental** | Regiones torso/brazos/piernas desde los keypoints COCO-18 que el pipeline ya calcula (+ protección de cara/manos/pies) | ~0 s extra (reutiliza la pose); 12,5 s totales |
| `sam2` | Apache-2.0 (código y checkpoints de `facebookresearch/sam2`) | **Implementado** (etapa 2) | Cajas derivadas de los keypoints → `SAM2ImagePredictor` → mapa FASHN de 5 regiones | **1,0 s** de segmentación |
| `grounded-sam2` | Apache-2.0 (Grounding DINO + SAM 2) | **Implementado** (etapa 2b) | Prompts de texto por categoría → cajas → SAM 2 (pensado para **prendas puestas por otra persona**) | Detección **2,8 s** + segmentación 1,3 s |
| `custom-parser` | Apache-2.0 (Detectron2) + pesos/datos propios | **Esqueleto** (etapa 3) | Parser propio entrenado; `COMMERCIAL_TO_FASHN` ya define la tabla de conversión | Pendiente (§9) |

### 2.4 Regla de admisión de un proveedor nuevo

1. `mi_proveedor.py` con una clase que herede de `SegmentationProvider`.
2. `info = ProviderInfo(name="mi-proveedor", license=..., requires_weights=..., commercial_ok=True)`.
3. `predict(image, context=None)` → mapa `HxW uint8` con ids de `labels.py` (o `None`).
4. Registrarlo en `PROVIDERS` (`segmentation/__init__.py`).
5. Añadirlo a `licenses/manifest.json` **y** `THIRD_PARTY_NOTICES.md` **con el sha256
   del checkpoint** y una prueba en `tests/test_segmentation_providers.py`.

**No negociable:** si la licencia del checkpoint no permite uso comercial,
`commercial_ok=False` y el proveedor **no se registra** (si se intenta construir un
pipeline con él, `_setup_segmentation()` lanza `RuntimeError`).

### 2.5 Qué pasa cuando no hay máscara (no se rompe: degrada)

| Configuración | Comportamiento | Registro |
|---|---|---|
| `segmentation_free=True` + prenda `flat-lay` | Se genera sin enmascarar la persona (comportamiento «maskless» original) | Normal |
| `segmentation_free=False` sin máscara | Genera sin enmascarar | `degraded`: `"persona: se pidió segmentation_free=False pero el proveedor '<x>' no aportó máscara; se generó sin enmascarar la persona"` |
| Prenda `garment_photo_type="model"` sin máscara | Trata la prenda como flat-lay (sin aislarla) | `degraded`: `"prenda: se pidió garment_photo_type='model' sin máscara disponible; se trató la prenda como flat-lay (sin aislarla)"` |
| `strict_segmentation=True` | Lanza excepción en los dos casos anteriores | Excepción (QA estricto) |

El detalle viaja en `PipelineOutput.metadata["segmentation"]`:
`provider`, `license`, `experimental`, `person_mask`, `garment_mask`, `degraded`
(lista de textos). De ahí lo consumen la API, la UI, el benchmark y las métricas.

---

## 3. Cambio 2: camino comercial explícito y con huella verificable

### 3.1 Cuál es el camino comercial (y por qué ese)

```
modelo .......................... TryOnModel de FASHN (mismo checkpoint, sin tocar)
proveedor de segmentación ....... none        (no carga ningún modelo)
segmentation_free ............... True
tipo de foto de prenda .......... flat-lay    (prenda fotografiada como producto)
salida .......................... 576×864 RGB
```

Con esa combinación no se usa **ninguna** máscara semántica, así que el resultado
es el del modelo «maskless» original. Es el camino por defecto en la CLI, la API y
la UI, y `none` es el `DEFAULT_PROVIDER`.

### 3.2 El invariante que demuestra que el fork no cambió el producto

`flat-lay` + `none` + 8 pasos + semilla 42 + guidance 1.5 sobre
`examples/data/model.webp` + `examples/data/garment.webp` (`tops`) deben dar
**exactamente**:

```
sha256 = a4d618bf4d50910374caf3770ee580d9a7df4b7c9a65d4a51831033ea631c53e
```

Ese hash es el del resultado **antes** del fork y sigue siendo el mismo después,
por las tres vías:

| Vía | Comando | Resultado |
|---|---|---|
| CLI | `./run_fashn_vton.sh python examples/basic_inference.py --weights-dir ./weights --person-image examples/data/model.webp --garment-image examples/data/garment.webp --category tops --garment-photo-type flat-lay` | `a4d618bf…` |
| API | `curl -X POST 'localhost:8000/v1/tryon?raw=true' …` | `a4d618bf…` |
| UI | `./run_fashn_vton.sh python scripts/smoke_webui.py` | `OK: idéntico a la referencia pre-fork = True` |

Cualquier release futuro debe volver a dar ese hash. Si cambia, algo del camino
comercial se movió (versión de torch, `disable_masking`, preprocesado…) y hay que
justificarlo.

### 3.3 El único cambio en el preprocesado

`src/fashn_vton/preprocessing/agnostic.py` (5 líneas) cambia el origen de las
tablas y añade un interruptor explícito:

```python
# antes
from fashn_human_parser import BODY_COVERAGE_TO_LABELS, IDENTITY_LABELS, LABELS_TO_IDS
# después
from ..segmentation.labels import BODY_COVERAGE_TO_LABELS, IDENTITY_LABELS, LABELS_TO_IDS
# ... y se mantienen los alias para no tocar los call sites existentes:
FASHN_LABELS_TO_IDS = LABELS_TO_IDS
BODY_COVERAGE_TO_FASHN_LABELS = BODY_COVERAGE_TO_LABELS
IDENTITY_FASHN_LABELS = tuple(IDENTITY_LABELS)
```

`create_clothing_agnostic_image(..., disable_masking=segmentation_free or person_class_map is None, ...)`:
es lo que permite generar **sin** máscara conservando el comportamiento original.
`labels.py` reproduce el diccionario original y una prueba lo ancla contra la
evidencia registrada del paquete antiguo
(`plan_modelo_comercial/evidencia_labels_originales.json`, `test_segmentation_labels.py`).

### 3.4 Lo que deliberadamente NO se cambió (contrato con el upstream)

| Elemento | Estado |
|---|---|
| Pesos del modelo (`weights/model.safetensors`, 1,94 GB) | Intactos; hash en el manifiesto |
| DWPose (`yolox_l.onnx`, `dw-ll_ucoco_384.onnx`) | Intactos |
| Arquitectura (`tryon_mmdit`, `TryOnModel`) y calidad de salida | Sin tocar |
| API pública `TryOnPipeline(...)(person_image=, garment_image=, category=, ...)` | Compatible (solo se **añaden** parámetros con default) |
| Forma de salida 576×864, preprocesado, máscaras de difusión | Sin tocar |
| Ejemplos y pesos: mismos repos de HuggingFace | Sin tocar |
| Licencia del código (Apache-2.0) | Se conserva; versión marcada `1.5.0+commercial.1` |

Es decir: **el fork es aditivo**. Lo que se elimina es la dependencia no
comercial y lo que se añade es la capa que la sustituye, el cumplimiento y las
interfaces.

---

## 4. Cambio 3: dependencias, pesos y etiquetas

### 4.1 `pyproject.toml`

| Cambio | Detalle |
|---|---|
| **Quitar** la dependencia NC | `-    "fashn-human-parser>=0.1.1",` (queda un comentario explicando por qué) |
| Versión | `1.5.0` → `1.5.0+commercial.1` (identificable en `pip show`) |
| Extras nuevos | `api` (FastAPI/uvicorn/python-multipart), `webui` (gradio, psutil), `metrics` (scikit-image, scipy), `providers-sam2` (`sam2>=1.1`), `providers-grounded-sam2`, `dev` (pytest/black/ruff) |
| Metadatos | `description` menciona el fork comercial; licencia del código sigue Apache-2.0 |

Instalación resultante: `pip install -e ".[api,webui,metrics,dev]"`.

### 4.2 `scripts/download_weights.py`

Se elimina `download_human_parser()` (que hacía `from fashn_human_parser import
FashnHumanParser` para forzar la descarga de sus pesos) y se añade la descarga
**opcional** de los checkpoints auditados:

```bash
python scripts/download_weights.py --weights-dir ./weights                 # modelo + DWPose (~2,2 GB)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam2.1-hiera-tiny', 'sam2.1_hiera_tiny.pt', local_dir='weights/sam2')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('IDEA-Research/grounding-dino-tiny', local_dir='weights/grounding_dino', allow_patterns=['*.json','*.txt','*.safetensors'])"
```

### 4.3 Purga de los pesos prohibidos ya cacheados

No basta con dejar de importar el paquete: hay que borrar lo que ya está en disco.

```bash
pip uninstall -y fashn-human-parser
rm -rf ~/.cache/huggingface/hub/models--fashn-ai--fashn-human-parser      # 245 MB
find / -name '*fashn*human*parser*' -o -name '*segformer*' 2>/dev/null    # debe salir vacío
```

El manifiesto deja ese patrón en `forbidden_artifacts` para que el gate lo detecte
si vuelve a aparecer (`**/models--fashn-ai--fashn-human-parser/**`, `**/*segformer*`).

### 4.4 Etiquetas propias (`segmentation/labels.py`)

La tabla de clases FASHN (18 ids: background, face, hair, top, dress, skirt, pants,
belt, scarf, arms, legs, hands, feet, …) se reimplementa localmente, sin importar
nada del paquete NC, y se **ancla** con pruebas a la evidencia registrada del
comportamiento original (`evidencia_labels_originales.json`, capturado de
`fashn-human-parser 0.1.1`). Así se garantiza que los ids siguen siendo los mismos
que espera el modelo.

---

## 5. Cambio 4: cumplimiento automatizado (la parte que hace *vendible* el módulo)

Sin esto, el fork es «un repo donde alguien borró una dependencia». Con esto, la
propiedad se comprueba sola en cada build.

### 5.1 `src/fashn_vton/compliance.py` (nuevo)

| Comprobación | Qué hace |
|---|---|
| `scan_source_tree(root)` | Recorre `.py` y falla si hay `import fashn_human_parser`, uso de `FashnHumanParser(` o imports de `*segformer*` (allowlist: el propio `compliance.py` y su test) |
| `check_pyproject_dependencies()` | La lista `dependencies` no contiene `fashn-human-parser`, `fashn_human_parser` ni `segformer` |
| `check_installed_distributions(manifest)` | Las distribuciones prohibidas **no están instaladas** en el entorno |
| `find_forbidden_files(root, manifest)` | No hay archivos que casen los globs prohibidos (pesos NC, SegFormer, cachés, datasets NC) |
| `verify_manifest_hashes()` | Cada artefacto local casa con el sha256 registrado |

Se agregan en `commercial_report()` y `assert_commercial_ready()`; lo consumen los
tests, el gate de release y (`/healthz` de la API) → `commercial_ready: true`.

### 5.2 Inventario con hashes

- `licenses/manifest.json`: 5 artefactos locales con sha256 (modelo FASHN, los 2 ONNX
  de DWPose, checkpoint de SAM 2, Grounding DINO tiny), las librerías sin hash, y
  **4 familias prohibidas** (paquete/pesos NC, SegFormer, VITON-HD/DressCode, cachés NC).
- `licenses/verify_hashes.py`: verificador ejecutable (`--no-forbidden-scan`,
  `--capture`); hoy **5/5 OK** y 0 prohibidos presentes.
- `THIRD_PARTY_NOTICES.md`: el mismo contenido en formato legal para el consumidor.

### 5.3 Gate de release

```bash
python scripts/verify_commercial_readiness.py          # 0 = listo; 1 = incumplimiento; 2 = artefacto prohibido
python scripts/verify_commercial_readiness.py --json   # salida machine-readable para CI/expediente
```

Imprime 4 bloques: cumplimiento de código/entorno · archivos de licencias
obligatorios · hashes de pesos · licencia y disponibilidad de cada proveedor, y
cierra con `RESULTADO: LISTO PARA COMERCIAL | proveedores no comerciales: ninguno`.

### 5.4 CI y contenedor

- `.github/workflows/ci.yml`: job **compliance-and-tests** (gate bloqueante, hashes,
  `grep` de que no queda rastro del paquete NC, `pytest -m "not integration"` con
  torch CPU) + job **lint** (`ruff check src scripts tests`).
- `Dockerfile` + `.dockerignore`: el gate se ejecuta **dentro** del build
  (`verify_commercial_readiness.py` + `verify_hashes.py --no-forbidden-scan`), los
  pesos **no** se hornean (se montan en `/app/weights`), usuario no root y
  `HEALTHCHECK` contra `/healthz`.
- **Aviso**: la imagen no se ha construido en esta máquina (no hay Docker); hay que
  construirla y publicarla en un host con Docker antes de usarla en producción.

---

## 6. Cambio 5: interfaces de producto (CLI, API y UI)

El módulo tiene que poder **consumirse** y **decir qué licencia usó** en cada
respuesta. Los tres caminos exponen el mismo desplegable de proveedores.

### 6.1 CLI (`examples/basic_inference.py`, +29 líneas)

- Nuevos flags: `--segmentation-provider` (default `none`) y `--garment-photo-type`
  (`flat-lay` | `model`).
- Al terminar imprime el resumen de segmentación: proveedor, licencia, si hubo
  máscara de persona/prenda y la lista de degradaciones.

### 6.2 API HTTP (nueva: `src/fashn_vton/api/main.py`, 308 líneas)

| Endpoint | Método | Qué devuelve |
|---|---|---|
| `/healthz` | GET | `status`, `package_version`, **`commercial_ready`** (resultado del compliance), `default_provider`, `providers`, `device`, `gpu_lock` |
| `/v1/version` | GET | Paquete, modelo con su licencia, rutas del manifiesto y de los avisos |
| `/v1/tryon` | POST (multipart) | Con `?raw=true` y 1 muestra: PNG binario + cabeceras `X-Elapsed-Seconds`, `X-Segmentation-Provider`, `X-Segmentation-Degraded`. En cualquier otro caso: JSON con `images[{base64, sha256, width, height}]`, `metadata` y `elapsed_seconds` |
| `/docs` | GET | OpenAPI interactivo (FastAPI) |

Parámetros de `/v1/tryon`: `person`, `garment`, `category`, `garment_photo_type`,
`num_timesteps`, `guidance_scale`, `seed`, `num_samples`, `provider`.

Reglas de la capa HTTP (todas probadas en `tests/test_api.py`):

| Situación | Respuesta |
|---|---|
| `VTON_API_KEYS` configurado y falta/`X-API-Key` inválida | **401** |
| Sin keys configuradas y sin `VTON_ALLOW_ANONYMOUS=1`, cliente no local | **503** (el servicio no se deja abierto por accidente) |
| Sin keys y cliente *loopback* (`127.0.0.1`, `testclient`) | Permitido (modo desarrollo) |
| `category`, `garment_photo_type`, `num_samples`, `num_timesteps` o `provider` inválidos | **422** con las opciones válidas |
| Proveedor con `commercial_ok=False` | **403** («no tiene licencia comercial: prohibido en el servicio») |
| GPU ocupada por otro proceso (turno de GPU) | **503** con el motivo |
| Error interno | **500** con `TipoDeError: mensaje` (nunca traceback crudo) |

Concurrencia: **una sola generación a la vez** (`asyncio.Lock` + `--workers 1`),
porque el pipeline no es thread-safe y comparte la VRAM. Variables de entorno:

| Variable | Default | Para qué |
|---|---|---|
| `FASHN_WEIGHTS_DIR` / `FASHN_DEVICE` / `FASHN_PROVIDER` | `weights` / auto / `none` | Modelo y proveedor por defecto |
| `VTON_API_KEYS` | (vacío) | Lista separada por comas; activa la autenticación |
| `VTON_ALLOW_ANONYMOUS` | `0` | Permitir sin keys (no recomendado) |
| `VTON_USE_GPU_LOCK`, `VTON_GPU_LOCK_PATH` | `0` / ruta por defecto | Turno único de GPU compartida |
| `VTON_MAX_UPLOAD_MB` / `VTON_MAX_SAMPLES` / `VTON_MAX_STEPS` | 12 / 4 / 50 | Límites de entrada |
| `VTON_STRICT_SEGMENTATION` | `0` | `strict_segmentation` (fallar en vez de degradar) |

### 6.3 Interfaz web de pruebas (`webui/`, opcional)

Envoltorio de `TryOnPipeline` sin modificar el repo: **selector de proveedor** (con
su licencia), `flat-lay` por defecto, progreso por paso real, memoria (RAM/VRAM),
log de la petición con la línea
`Segmentación: proveedor=… · licencia=… · máscara persona=… · máscara prenda=…` y el
aviso de degradación; guarda PNG + `_params.txt` en `outputs/webui/`; caché de **un
solo** pipeline (al cambiar de proveedor libera el anterior para no acumular VRAM).
Prueba de humo: `scripts/smoke_webui.py`.

### 6.4 Lanzadores y turno de GPU

| Script | Para qué |
|---|---|
| `run_fashn_vton.sh` | Activa el env y exporta `LD_LIBRARY_PATH` a las libs CUDA de `nvidia-*` (sin eso DWPose cae a CPU silenciosamente). **Usar siempre este envoltorio.** |
| `run_api.sh` | Lanza el servicio (`API_PORT`, `VTON_WORKERS`, comprueba puerto libre) |
| `run_web.sh` | Lanza la UI (`GRADIO_SERVER_PORT`, `FASHN_PRELOAD`, …) |
| `src/fashn_vton/utils/gpu_lock.py` | Turno único de GPU opcional, con el **mismo archivo de lock** que IDM-CUSTOM, para no chocar con otros proyectos que usan la GPU |

---

## 7. Cambio 6: métricas de calidad y umbrales

Para poder decir «el camino comercial es suficiente» hay que medirlo. Todo se
calcula **sin redes preentrenadas** (solo scikit-image/scipy/opencv, permisivos),
así que las métricas no introducen ninguna licencia nueva.

`src/fashn_vton/eval/quality_gates.py` (274 líneas):

| Métrica | Qué mide |
|---|---|
| `identity_preservation` | SSIM en cara/pelo/manos/pies: ¿se conservó la identidad de la persona? |
| `outside_change` | Cambio medio **fuera** de la región de la prenda (fondo y cuerpo) |
| `color_fidelity` | Distancia de Wasserstein del color entre la prenda pedida y la generada |
| `pattern_fidelity` | SSIM del estampado/textura de la prenda |
| `garment_change` | Cuánto cambió la zona de la prenda (debe cambiar: es el objetivo) |
| `sharpness` | Nitidez (varianza del laplaciano) |
| `failure_rate` | Fracción de pares que fallan (error o degradado) |
| `score` | Combinación ponderada (`color_fidelity` ×3, `identity_preservation` ×2) |
| `passes_gate` | Veredicto contra `QualityGateConfig` |

```python
@dataclass
class QualityGateConfig:
    """Thresholds for passes_quality_gate (starting values, to calibrate)."""
    min_identity_preservation: float = 0.60
    max_outside_change: float = 8.0
    min_color_fidelity: float = 0.55
    min_pattern_fidelity: float = 0.20
    min_sharpness: float = 20.0
    max_failure_rate: float = 0.05
```

**Estos umbrales son valores de partida, no calibrados** (pendiente §9). Medición ya
realizada con `scripts/benchmark_providers.py --examples --metrics` (prenda `model`,
donde la segmentación sí importa):

| Proveedor | Estado | identity_preservation | outside_change | color_fidelity | score |
|---|---|---|---|---|---|
| `none` (sin máscara) | degraded | 0,8232 | 29,98 | 0,5962 | 0,687 |
| `pose-heuristic` | ok | 0,9551 | 22,89 | 0,5869 | 0,734 |
| `sam2` | ok | **0,9650** | **22,66** | **0,5894** | **0,740** |
| `grounded-sam2` | ok | 0,9528 | 25,97 | 0,5878 | 0,734 |

Lectura: sin máscara la identidad se conserva peor (0,82) y se cambia más fuera de
la prenda (30,0); con máscaras reales sube a ~0,95-0,97. En el camino comercial
(`flat-lay` + `segmentation_free`) los cuatro proveedores dan la **misma** salida:
ahí no se usa máscara.

---

## 8. Checklist de aceptación (con el comando y el resultado esperado)

| # | Comprobación | Comando | Resultado esperado | Hoy |
|---|---|---|---|---|
| 1 | El paquete NC no está en el entorno | `python -c "import importlib.util as u; print(u.find_spec('fashn_human_parser'))"` | `None` | ✅ |
| 2 | No hay código que lo use | `grep -rn 'import .*fashn_human_parser' src/ scripts/ tests/ \| wc -l` | `0` | ✅ |
| 3 | No está declarado como dependencia | `grep fashn-human-parser pyproject.toml` | solo el comentario | ✅ |
| 4 | Sus pesos no están en caché | `find / -name '*fashn*human*parser*' 2>/dev/null` | vacío | ✅ |
| 5 | Hashes de los pesos | `python licenses/verify_hashes.py` | `5/5 OK`, 0 prohibidos | ✅ |
| 6 | **Gate de release** | `python scripts/verify_commercial_readiness.py` | `RESULTADO: LISTO PARA COMERCIAL`, exit `0` | ✅ |
| 7 | Pruebas | `python -m pytest tests/ -q` | `107 passed` | ✅ |
| 8 | Lint de CI | `ruff check src scripts tests` | `All checks passed!` | ✅ |
| 9 | **Invariante de salida** | `examples/basic_inference.py … --garment-photo-type flat-lay` | `sha256 a4d618bf…` | ✅ |
| 10 | API | `curl -s localhost:8000/healthz` | `"commercial_ready": true` | ✅ |
| 11 | UI | `curl -s -o /dev/null -w '%{http_code}' localhost:7863` | `200` + selector de proveedor | ✅ |
| 12 | Imagen Docker | `docker build -t fashn-vton-commercial .` | build OK con el gate dentro | ⛔ sin Docker en la máquina |

---

## 9. Lo que **falta** para completar el módulo

El módulo ya es comercializable en el camino `flat-lay` + `none`. Queda por
completar la independencia total (Etapa 3) y cerrar lo que no se puede hacer sin
recursos externos:

### 9.1 Etapa 3 — parser propio (`custom-parser`)

Es el único hueco funcional: hoy, para prendas **puestas por una modelo** la
alternativa comercial es `sam2` (Apache-2.0) o `grounded-sam2`; el parser propio
daría control total de arquitectura y pesos.

| Paso | Qué hacer | Dónde |
|---|---|---|
| 1 | **Dataset propio** con licencia comercial y trazabilidad por imagen (prohibido VITON-HD/DressCode) | `licenses/manifest.json` (`forbidden_artifacts`) lo vigila |
| 2 | Entrenar segmentación semántica con **Detectron2** (Apache-2.0) sobre las etiquetas comerciales de `COMMERCIAL_TO_FASHN` | `src/fashn_vton/segmentation/custom_parser.py` (tabla ya definida) |
| 3 | Rellenar **solo** `predict()` para devolver el mapa `HxW uint8` de etiquetas FASHN y cargar el checkpoint de `weights/custom-parser/*.pt` | `custom_parser.py` (hoy `is_available()` ya detecta el `.pt`) |
| 4 | Registrar el hash del checkpoint | `licenses/manifest.json` + `THIRD_PARTY_NOTICES.md` + `licenses/verify_hashes.py` |
| 5 | Validar calidad y compararlo con `sam2` | `scripts/benchmark_providers.py --providers custom-parser,sam2 --metrics` |

El contrato ya está preparado: no hay que tocar el pipeline, la API ni la UI para
activarlo (aparece solo en el registro y en el desplegable).

### 9.2 Resto de pendientes

| Pendiente | Detalle | Riesgo si no se hace |
|---|---|---|
| **Benchmark completo** (plan §14) | 100 tops + 100 bottoms + 100 one-pieces con imágenes propias: el arnés (`--pairs CSV`) y las métricas están listos, faltan los datos | No hay base estadística para decidir `none` vs `sam2` vs parser propio |
| **Calibrar los umbrales** | `QualityGateConfig` son valores de partida; ajustarlos con ese benchmark | `passes_gate` puede dar falsos verdes/rojos |
| **Construir y publicar la imagen Docker** | No hay Docker en la máquina de desarrollo; el `Dockerfile` se revisó a mano | Sin artefacto reproducible de despliegue |
| **Revisión legal profesional** | El gate técnico **no** sustituye la asesoría legal; revisar checkpoints y datasets concretos del despliegue | Riesgo jurídico residual |
| **Concurrencia/multiusuario** | Hoy: un proceso, una generación a la vez; sin colas ni autenticación multi-tenant | No sirve tal cual para SaaS abierto |

---

## Apéndice A — Inventario de archivos del cambio

**Nuevos (31 rutas, todas del fork):**

| Grupo | Rutas | Para qué |
|---|---|---|
| Motor de segmentación | `src/fashn_vton/segmentation/{__init__,base,labels,none,pose_heuristic,sam2,grounded_sam2,custom_parser}.py`, `README.md` | Sustituye al paquete NC |
| Cumplimiento | `src/fashn_vton/compliance.py`, `licenses/{manifest.json,verify_hashes.py,README.md,Apache-2.0.txt}`, `THIRD_PARTY_NOTICES.md`, `scripts/verify_commercial_readiness.py` | Detecta y bloquea licencias NC |
| Servicio | `src/fashn_vton/api/{main.py,__init__.py}`, `run_api.sh` | API HTTP del producto |
| UI de pruebas | `webui/{app.py,README.md}`, `run_web.sh`, `PLAN_INTERFAZ_WEB.md` | Validación manual |
| Métricas | `src/fashn_vton/eval/{quality_gates.py,__init__.py}`, `scripts/benchmark_providers.py`, `scripts/smoke_webui.py` | Calidad y humo |
| GPU | `src/fashn_vton/utils/gpu_lock.py` | No chocar con la Pista A / IDM-CUSTOM |
| Empaquetado | `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml` | Build y CI con gate |
| Pruebas | `tests/test_{segmentation_labels,segmentation_providers,sam2_provider,grounded_sam2_provider,quality_gates,pipeline_parser_free,compliance,api,gpu_lock}.py` | 70 pruebas nuevas |
| Docs | `README_COMERCIAL.md`, `CAMBIO_MODULO_COMERCIAL.md`, `plan_modelo_comercial/*`, `LEVANTAMIENTO_2026-09-15.md` | Uso, cambio y evidencia |

**Modificados del upstream (10 archivos, +299 / −57 líneas):**

| Archivo | Cambio | Δ |
|---|---|---|
| `src/fashn_vton/pipeline.py` | Proveedor enchufable, rechazo de licencias NC, degradaciones, `metadata` | +154 |
| `scripts/debug_masks.py` | Usa proveedores; sin parser; limpieza de lint | +78 |
| `pyproject.toml` | Fuera la dependencia NC, extras, versión `+commercial.1` | +39 |
| `examples/basic_inference.py` | `--segmentation-provider`, `--garment-photo-type`, resumen | +29 |
| `scripts/download_weights.py` | Fuera la descarga del parser | ~23 |
| `README.md` | Nota de fork comercial (+ enlace a la especificación del cambio) | +16 |
| `src/fashn_vton/__init__.py` | Exporta la capa nueva | +8 |
| `src/fashn_vton/preprocessing/agnostic.py` | Tablas locales + `disable_masking` | +5 |
| `src/fashn_vton/dwpose/{__init__,dwpose}.py` | `__all__` y limpieza (lint de CI) | +3/−1 |

---

## Apéndice B — Licencias y hashes (fuente de verdad)

Artefactos verificados localmente (`sha256` calculado sobre los archivos reales):

| Artefacto | Licencia | Bytes | sha256 (prefijo) |
|---|---|---|---|
| `fashn-ai/fashn-vton-1.5` `model.safetensors` | Apache-2.0 | 1.943.668.048 | `d6cd3828…` |
| `fashn-ai/DWPose` `yolox_l.onnx` | Apache-2.0 | 216.746.733 | `7860ae79…` |
| `fashn-ai/DWPose` `dw-ll_ucoco_384.onnx` | Apache-2.0 | 134.399.116 | `724f4ff2…` |
| `facebook/sam2.1-hiera-tiny` `sam2.1_hiera_tiny.pt` | Apache-2.0 | ~156 MB | en `licenses/manifest.json` |
| `IDEA-Research/grounding-dino-tiny` `model.safetensors` | Apache-2.0 | 689.359.096 | `1a2412ef…` |

Librerías (declaradas, sin hash): PyTorch/torchvision (BSD-3), NumPy/SciPy/psutil
(BSD-3), Pillow (MIT-CMU), OpenCV (Apache-2.0/MIT), tqdm (MPL-2.0/MIT), einops
(MIT), onnxruntime-gpu (MIT), safetensors/huggingface_hub (Apache-2.0),
matplotlib (PSF), FastAPI/Starlette/Uvicorn/python-multipart (MIT/BSD/Apache),
Gradio (Apache-2.0), sam2 (Apache-2.0).

Familias **prohibidas** registradas (y comprobadas por el gate):

| Prohibido | Motivo |
|---|---|
| `fashn-human-parser` (paquete o pesos) | Sus pesos heredan la licencia NC de NVIDIA SegFormer |
| `SegFormer` / `nvidia/segformer-*` | Licencia NC de NVIDIA |
| `VITON-HD` / `DressCode` | CC BY-NC / prohibido a empresas privadas |
| Cachés NC (`models--fashn-ai--fashn-human-parser`, `models--*segformer*`) | Restos que contaminarían una imagen de producción |

---

## Apéndice C — Reaplicar el cambio sobre un clon limpio

Orden probado (cada paso deja el repo en un estado funcional):

```bash
# 0) Clon y entorno
git clone https://github.com/fashn-AI/fashn-vton-1.5 && cd fashn-vton-1.5
conda create -n fashn-vton python=3.11 -y && conda activate fashn-vton
pip install -e .          # todavía con fashn-human-parser (estado upstream)

# 1) Romper el acoplamiento (los 5 puntos de §1)
#    - pyproject.toml: borrar "fashn-human-parser>=0.1.1" y subir la versión a 1.5.0+commercial.1
#    - añadir src/fashn_vton/segmentation/ (base, labels, none, ...)
#    - pipeline.py: importar de .segmentation, _setup_segmentation, _provider_class_map,
#      metadata["segmentation"] y las degradaciones
#    - preprocessing/agnostic.py: importar de ..segmentation.labels y añadir disable_masking
#    - download_weights.py / debug_masks.py / basic_inference.py: quitar el parser, añadir flags
pip uninstall -y fashn-human-parser
rm -rf ~/.cache/huggingface/hub/models--fashn-ai--fashn-human-parser

# 2) Cumplimiento (antes de seguir: es lo que garantiza lo anterior)
#    compliance.py + licenses/{manifest.json,verify_hashes.py} + THIRD_PARTY_NOTICES.md
#    + scripts/verify_commercial_readiness.py
python scripts/verify_commercial_readiness.py       # debe decir LISTO PARA COMERCIAL

# 3) Interfaces (opcional según el producto)
pip install -e ".[api,webui,metrics]"               # FastAPI/uvicorn, Gradio, scikit-image
#    src/fashn_vton/api/, webui/, run_api.sh, run_web.sh, run_fashn_vton.sh

# 4) Pruebas, CI y contenedor
python -m pytest tests/ -q                          # 107 passed
ruff check src scripts tests                        # All checks passed!
#    .github/workflows/ci.yml + Dockerfile + .dockerignore

# 5) Verificación final (las 12 filas del §8), incluida la huella del camino comercial
./run_fashn_vton.sh python examples/basic_inference.py --weights-dir ./weights \
  --person-image examples/data/model.webp --garment-image examples/data/garment.webp \
  --category tops --garment-photo-type flat-lay --num-timesteps 8 --seed 42
# -> sha256 a4d618bf4d50910374caf3770ee580d9a7df4b7c9a65d4a51831033ea631c53e
```

**Trampas conocidas** (ya resueltas en este repo, no las repitas):

1. Sin `LD_LIBRARY_PATH` a las libs CUDA de `nvidia-*`, **DWPose cae a CPU sin
   avisar** → usar `run_fashn_vton.sh`.
2. Una caché de pipelines por proveedor **acumula VRAM** (9,5 GiB con dos): guardar
   uno solo y liberar el anterior (`gc.collect()` + `torch.cuda.empty_cache()`).
3. La galería de Gradio devuelve una copia **WebP re-codificada**; para comparar
   hashes hay que usar el PNG de `outputs/webui/`.
4. `ruff check` del clon limpio **ya falla con 6 errores preexistentes** (imports no
   usados en `dwpose/` y `debug_masks.py`); hay que limpiarlos para que el CI pase.
5. El gate trata los pesos ausentes como **aviso** (`MISSING`), no como violación:
   en el build de Docker no hay `weights/`, y eso es correcto.
