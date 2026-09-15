# Plan de implementación comercial — FASHN VTON v1.5 sin `fashn-human-parser`

Fecha: 2026-09-15. Origen: `propuestas_fashn_vton_comercial.md` (misma carpeta).
Repo: `/home/uceda/Documents/fashn-vton-1.5` (fork comercial del clon
`fashn-AI/fashn-vton-1.5`, commit base `7c0f10a`).
Referencias de patrón: `/home/uceda/Documents/IDM-CUSTOM` (inventario de
licencias + verificador de hashes, guardas anti-artefactos prohibidos, pruebas
unitarias/integración, turno único de GPU, runbooks de operación).

## 1. Objetivo

Convertir el repositorio en un **fork comercialmente utilizable**: mismo modelo
(FASHN VTON v1.5, Apache-2.0) y misma calidad, **sin ninguna dependencia de
licencia no comercial** (`fashn-human-parser`, cuyos pesos derivan de NVIDIA
SegFormer = uso no comercial).

Etapa 1 del plan es el objetivo de esta implementación: **MVP parser-free** con
prendas `flat-lay` + `segmentation_free=True`, más la infraestructura de
cumplimiento y de servicio necesaria para operarlo. Las etapas 2 (SAM 2) y 3
(parser propio) quedan como interfaces listas + hoja de ruta.

## 2. Diagnóstico: dónde está acoplado `fashn-human-parser`

Verificado con `grep` sobre el repo (no supuesto):

| Archivo | Acoplamiento | Se resuelve con |
|---|---|---|
| `pyproject.toml:40` | dependencia `fashn-human-parser>=0.1.1` | quitarla |
| `src/fashn_vton/pipeline.py:11` | `from fashn_human_parser import CATEGORY_TO_BODY_COVERAGE, FashnHumanParser` | `CATEGORY_TO_BODY_COVERAGE` pasa a la tabla propia; el parser se sustituye por un proveedor de segmentación |
| `src/fashn_vton/pipeline.py:84` | `self._setup_hp_model()` (carga el parser siempre, incluso con `segmentation_free=True`) | `_setup_segmentation()` (sin pesos NC) |
| `src/fashn_vton/pipeline.py:266-267` | `person_seg_pred = self.hp_model.predict(...)`, `garment_seg_pred = ...` | solo se llama al proveedor si el flujo realmente necesita máscara |
| `src/fashn_vton/preprocessing/agnostic.py:7` | `from fashn_human_parser import BODY_COVERAGE_TO_LABELS, IDENTITY_LABELS, LABELS_TO_IDS` | tabla propia `segmentation/labels.py` (**no lo menciona el plan original: era un segundo acoplamiento oculto**) |
| `scripts/debug_masks.py:19` | instancia `FashnHumanParser` | pasa a usar el proveedor |
| `scripts/download_weights.py:52-59` | descarga los pesos del parser | se elimina ese paso |
| `README.md:137` | declara `fashn-human-parser` como componente de terceros | sustituir por los avisos comerciales reales |

Punto clave confirmado en el código: con `segmentation_free=True` la imagen
agnóstica es la persona tal cual (`create_clothing_agnostic_image(...,
disable_masking=True)` devuelve la imagen sin tocar), **pero la prenda sí se
procesaba con el parser** cuando `garment_photo_type="model"`. Por eso el MVP
comercial define el camino `flat-lay` (+ `segmentation_free=True`) como el
soportado sin proveedor de máscaras.

## 3. Decisiones de diseño

| Decisión | Elección | Motivo |
|---|---|---|
| Interfaz de segmentación | `SegmentationProvider` (ABC) con `predict(image) -> class_map \| None`, más `extract_person_region` / `extract_garment` | Igual que la propuesta 11 del plan, y encaja con el `seg_pred` que ya consume el preprocesado |
| Proveedor por defecto | `NoSegmentationProvider` | Cero modelos, cero pesos, cero licencias NC; es el camino comercial del MVP |
| Tabla de etiquetas | `segmentation/labels.py` propia, copiada **verificada por igualdad** contra el paquete original antes de retirarlo (18 clases + coberturas + labels de identidad) | Es una tabla de interoperabilidad (el modelo espera esos ids); se congela con valores dorados en pruebas para detectar desviaciones |
| Comportamiento cuando falta máscara | **Degradar con aviso explícito** (no romper): si se pide `segmentation_free=False` o prenda `model` sin proveedor capaz, se procesa sin máscara y se registra en el log y en `PipelineOutput.metadata["segmentation"]` | Un servicio comercial no debe devolver 500 por una combinación de parámetros; pero tampoco debe degradar en silencio |
| Proveedores implementados | `none` (completo), `pose-heuristic` (heurístico con DWPose, **experimental**, licencia limpia), `sam2`, `grounded-sam2`, `custom-parser` (esqueletos documentados que fallan con mensaje accionable hasta que existan los pesos) | Da la interfaz completa y prueba de que la sustitución es enchufable sin tocar el pipeline |
| Licencias | `licenses/manifest.json` + `licenses/verify_hashes.py --capture` + `THIRD_PARTY_NOTICES.md` + texto `Apache-2.0.txt` | Patrón de IDM-CUSTOM: inventario machine-readable con hashes reales verificables |
| Guardas de cumplimiento | `src/fashn_vton/compliance.py` + `scripts/verify_commercial_readiness.py` (gate con exit code) | Detecta que ninguna dependencia/referencia prohibida vuelva a entrar (equivalente al CP-5 de IDM-CUSTOM) |
| Servicio | `src/fashn_vton/api/main.py` (FastAPI ya disponible por Gradio) con API key, límites y turno único de GPU | El plan pide `src/api/main.py`; FastAPI trae validación, OpenAPI y multipart |
| Turno único de GPU | `src/fashn_vton/utils/gpu_lock.py` propio, **mismo formato y ruta por defecto** (`~/.idm_gpu.lock`) que el de IDM-CUSTOM | En esta máquina la GPU es compartida con la Pista A y con IDM-CUSTOM; el formato común permite el turno cruzado sin acoplar el código |
| Idioma | Código y docstrings en **inglés** (convención del upstream, diffs limpios); documentación y mensajes de UI en **español** | Mantener el fork fusionable hacia arriba y los docs legibles para el equipo |
| Referencia de comportamiento | Antes de tocar nada se guardan salidas de referencia (`/tmp/ref_before`, semilla 42, 8 pasos) para comparar después | Prueba de que el fork no cambia el resultado del camino comercial (`flat-lay`) frente al original |

## 4. Fases

| Fase | Entregable | Verificación |
|---|---|---|
| **F1** Tabla de etiquetas | `segmentation/labels.py` + verificación de igualdad contra el paquete original | script de igualdad `OK` antes de retirar la dependencia |
| **F2** Fork parser-free | `segmentation/{base,none,pose_heuristic,sam2,grounded_sam2,custom_parser}.py`, `pipeline.py` sin import NC, `preprocessing/agnostic.py` sin import NC, `pyproject.toml` sin la dependencia, `scripts/debug_masks.py` y `download_weights.py` adaptados | `examples/basic_inference.py` corre **con el paquete desinstalado** y el PNG `flat-lay` coincide con la referencia; prueba con bloqueador de import |
| **F3** Licencias y cumplimiento | `licenses/` (manifest con hashes reales, verificador, textos), `THIRD_PARTY_NOTICES.md`, `compliance.py`, `scripts/verify_commercial_readiness.py` | `verify_hashes.py` con `OK` en los pesos presentes, gate en verde y detección de un prohibido inyectado en un directorio temporal |
| **F4** Pruebas | `tests/test_segmentation_labels.py`, `test_segmentation_providers.py`, `test_pipeline_parser_free.py`, `test_compliance.py` | `pytest` en verde (se instala `pytest` en el env) |
| **F5** API | `src/fashn_vton/api/main.py`, `run_api.sh`, `utils/gpu_lock.py` | `/healthz`, `/v1/version`, 401 sin API key, 422 con categoría inválida, `/v1/tryon` (dependencia simulada en pruebas + 1 corrida real) |
| **F6** UI y herramientas | `webui/app.py` con selector de proveedor, `scripts/benchmark_providers.py` | la UI sigue dando `HTTP 200` y genera con proveedor `none`; el benchmark escribe CSV |
| **F7** Documentación | `README_COMERCIAL.md`, `src/fashn_vton/segmentation/README.md`, actualización de `README.md`/`LEVANTAMIENTO_2026-09-15.md`/`webui/README.md`, y §8 con resultados | revisión de enlaces y lectura final |

## 5. Criterios de aceptación

1. `python -c "import fashn_vton"` y una generación real funcionan **con
   `fashn-human-parser` desinstalado** y sin sus pesos en la caché de HF.
2. El PNG del camino comercial (`flat-lay`, `segmentation_free=True`, semilla
   fija) es **idéntico** al generado antes del fork.
3. `grep -r fashn_human_parser src/ scripts/ pyproject.toml` no devuelve código
   ejecutable que lo use.
4. `licenses/verify_hashes.py` verifica los pesos descargados y el gate
   `verify_commercial_readiness.py` termina con exit 0.
5. `pytest` en verde: etiquetas, proveedores, degradación, ausencia del import
   prohibido y cumplimiento.
6. La API arranca con `run_api.sh`, responde `/healthz` y rechaza peticiones sin
   API key cuando está configurada.
7. La UI sigue funcionando (HTTP 200 + generación real) con el selector de
   proveedor.

## 6. Riesgos y limitaciones

| Riesgo | Mitigación |
|---|---|
| El camino `model` (prenda puesta) pierde calidad al perder el parser | Documentado: el MVP comercial se limita a `flat-lay`; `pose-heuristic` es un puente experimental y SAM 2 / Grounded-SAM 2 son las etapas 2 |
| `pose-heuristic` sin validar | Marcado como experimental en código, UI, API y docs; el benchmark existe para medirlo contra el original |
| La tabla de etiquetas podría no cubrir un caso futuro | Valores dorados en pruebas + verificación de igualdad registrada antes de la retirada |
| Los esqueletos de SAM 2 / Grounded DINO no se pueden probar hoy | `is_available()` + error accionable; las pruebas verifican ese contrato, no la inferencia |
| El paquete NC sigue en el entorno y en la caché de HF | Se retira del entorno y se purga su caché; el gate avisa si reaparece |
| No hay Docker en esta máquina | El punto "imagen sin artefactos NC" del checklist §15 del plan original queda como pendiente documentado en la hoja de ruta |

## 7. Registro de ejecución

**Estado: F1–F7 implementadas y verificadas el 2026-09-15.** El fork está
funcionando (UI en `http://127.0.0.1:7863`, API en `http://127.0.0.1:8000`) y el
gate de cumplimiento comercial pasa en verde.

### Cambios aplicados

| Archivo / paquete | Cambio |
|---|---|
| `pyproject.toml` | fuera `fashn-human-parser`; extras `api`/`webui`/`providers-sam2`/`providers-grounded-sam2`; `[tool.pytest.ini_options]`; versión `1.5.0+commercial.1` |
| `src/fashn_vton/segmentation/` (nuevo) | `base.py` (contrato + validación), `labels.py` (tabla FASHN propia verificada), `none.py`, `pose_heuristic.py`, `sam2.py`, `grounded_sam2.py`, `custom_parser.py`, `README.md` |
| `src/fashn_vton/pipeline.py` | sin import NC; `segmentation_provider` + `strict_segmentation`; degradación explícita; `PipelineOutput.metadata`; guarda de categoría; guarda anti-proveedor-no-comercial |
| `src/fashn_vton/preprocessing/agnostic.py` | importaba `fashn_human_parser` (2º acoplamiento oculto): ahora usa `segmentation.labels` |
| `src/fashn_vton/compliance.py` (nuevo) | guardas: imports prohibidos, dependencias, distribuciones instaladas, artefactos prohibidos, hashes |
| `src/fashn_vton/utils/gpu_lock.py` (nuevo) | turno único de GPU compatible con IDM-CUSTOM/Pista A |
| `src/fashn_vton/api/` (nuevo) | servicio FastAPI (`/healthz`, `/v1/version`, `/v1/tryon`), API key, límites, single-flight + lock de GPU |
| `licenses/` (nuevo) | `manifest.json` (20 artefactos, 4 prohibidos), `verify_hashes.py`, `Apache-2.0.txt`, `README.md` |
| `THIRD_PARTY_NOTICES.md`, `README_COMERCIAL.md` (nuevos) | aviso público de terceros y guía del fork |
| `examples/basic_inference.py` | `--segmentation-provider` y resumen de segmentación/degradaciones |
| `scripts/` | `verify_commercial_readiness.py` (gate), `benchmark_providers.py` (plan §14); `download_weights.py` sin parser; `debug_masks.py` con proveedores |
| `webui/app.py` | selector de proveedor, `flat-lay` por defecto, `gpu_lock` propio |
| `run_api.sh` (nuevo) | lanzador del servicio |
| `tests/` | +5 archivos (70 pruebas): etiquetas, proveedores, parser-free, compliance, API, gpu_lock |
| `README.md` (upstream) | nota de fork comercial y aviso de que `fashn-human-parser` ya no es dependencia |

### Verificación (criterios de §5)

| # | Criterio | Evidencia | Resultado |
|---|---|---|---|
| 1 | Funciona **sin** el paquete NC | `pip uninstall -y fashn-human-parser` + `importlib.util.find_spec('fashn_human_parser') is None` + generación real | ✅ |
| 1b | Sin sus pesos en caché | `~/.cache/huggingface/hub/models--fashn-ai--fashn-human-parser` (245 MB) eliminada; 0 entradas NC | ✅ |
| 2 | El camino comercial es **idéntico** | `flat-lay`, 8 pasos, semilla 42: `a4d618bf4d5091…` antes y después del fork | ✅ byte a byte |
| 2b | También por la **API** y por la **UI** | `POST /v1/tryon?raw=true` → `a4d618bf4d5091…`; `gradio_client` a `webui` → `a4d618bf4d5091…` | ✅ |
| 2c | Camino `model` sin parser | degrada con aviso explícito y `metadata.segmentation.degraded` (y con `pose-heuristic` usa máscara propia, `EXIT=0`) | ✅ documentado |
| 3 | Sin referencias ejecutables al paquete NC | `grep -rn 'import .*fashn_human_parser' src/ scripts/ tests/` → **0**; las 3 apariciones de la cadena en `src/` son la *deny-list* y el regex de `compliance.py`; `pyproject.toml` sin la dependencia; `scan_source_tree()` → `[]` | ✅ |
| 4 | Licencias y gate | `licenses/verify_hashes.py` → 3/3 `OK`, 0 prohibidos; `scripts/verify_commercial_readiness.py` → exit 0 «LISTO PARA COMERCIAL» | ✅ |
| 5 | Pruebas | `pytest tests/ -q` → **107 passed** (70 nuevos del fork + 37 del repo upstream) | ✅ |
| 6 | API | `/healthz` 200 con `commercial_ready: true`; `/v1/version`; 401 sin API key (probado en tests); 422 con categoría/límites inválidos; 403 con proveedor NC | ✅ |
| 7 | UI | `./run_web.sh` → `HTTP 200`; selector de proveedor presente; generación real con `none` (13,5 s, VRAM 1,83/3,50 GiB) | ✅ |
| 7b | UI: el cambio es **visible** | log de la UI con `Segmentación: proveedor=… · licencia=… · máscara persona=… · máscara prenda=…` y aviso de degradación cuando corresponde; `scripts/smoke_webui.py` → `TODO OK` con `sha256 a4d618bf…` (ver «Revisión del 2026-09-15») | ✅ |

### Otros datos medidos

- **Etiquetas**: igualdad exacta entre `segmentation/labels.py` y la evidencia
  registrada del paquete original (`plan_modelo_comercial/evidencia_labels_originales.json`,
  `fashn-human-parser 0.1.1`), anclada en pruebas.
- **Benchmark de humo** (`scripts/benchmark_providers.py --examples`, 8 pasos):
  `none` 13,1 s · `pose-heuristic` 12,5 s · `sam2` registrado como no disponible;
  VRAM pico 5,16 GiB en ambos. CSV y resumen JSON generados.
- **Proveedor `pose-heuristic`**: mapa de clases determinista desde keypoints
  COCO-18 (torso/brazos/piernas + protección de cara/manos/pies), marcado como
  **experimental** en código, UI, API y docs; su calidad no está validada.
- **Turno de GPU**: `utils/gpu_lock` verificado contra la CLI de IDM-CUSTOM
  (prueba de interoperabilidad con el mismo archivo de lock).

### Continuación (misma fecha) — Etapa 2 y 2b implementadas

| Pieza | Detalle | Verificación |
|---|---|---|
| **`sam2` (etapa 2)** | Proveedor real: cajas derivadas de los keypoints COCO-18 → `SAM2ImagePredictor` → mapa de clases FASHN (torso/brazos/piernas + etiqueta de prenda por categoría + protección de cara/pelo/manos/pies) | Carga + segmentación de 5 regiones en **1,0 s**; pipeline completo con `--garment-photo-type model` deja de degradar (`prenda=sí`) y con `--no-segmentation-free` da `persona=sí` |
| **`grounded-sam2` (etapa 2b)** | Grounding DINO (vía `transformers`, sin compilar `groundingdino`) con prompts por categoría → cajas → SAM 2 → máscara de prenda | Detección en **2,8 s** (3 cajas, scores 0,34-0,40) y predicción total en 1,3 s (28% del área como prenda); pipeline con prenda `model` OK |
| **Checkpoints auditados** | `facebook/sam2.1-hiera-tiny` (`7402e0d8…`, 156 MB) e `IDEA-Research/grounding-dino-tiny` (`1a2412ef…`, 689 MB), ambos **Apache-2.0** | `licenses/verify_hashes.py` → **5/5 `OK`** |
| **Métricas de calidad (§14)** | `src/fashn_vton/eval/quality_gates.py`: `identity_preservation`, `outside_change`, `color_fidelity` (Wasserstein), `pattern_fidelity`, `garment_change`, `sharpness`, umbrales (`QualityGateConfig`), `aggregate()` y `weighted_score()`; **sin redes preentrenadas** | 17 pruebas nuevas; integrado en el benchmark (`--metrics`) con veredicto `passes_gate` |
| **Benchmark** | Compara N proveedores sobre los mismos pares y escribe CSV + resumen con métricas; proveedores no disponibles se registran sin romper | 4 proveedores: `none` 13,1 s · `pose-heuristic` 12,5 s · `sam2` 12,6 s · `grounded-sam2` 12,7 s; VRAM 5,16 GiB; RAM 2,3→4,6 GiB |
| **CI** | `.github/workflows/ci.yml`: gate comercial + verificación de hashes + comprobación anti-NC + `pytest -m "not integration"` + `ruff` | (pendiente de ejecutarse en GitHub; localmente se replicó paso a paso) |
| **Contenedor** | `Dockerfile` + `.dockerignore`: ejecuta el gate **dentro del build** y monta `./weights` en runtime | **No verificado**: esta máquina no tiene Docker (documentado) |
| **Pruebas** | Añadidos `test_sam2_provider.py`, `test_grounded_sam2_provider.py`, `test_quality_gates.py`; ajustadas las existentes | `pytest tests/` → **107 passed** (incluye integración real con los dos checkpoints) |

### Evidencia del gate de calidad (camino `model`, 8 pasos, 1 par)

El benchmark con `--metrics` sobre el camino donde el proveedor **sí** influye
(prenda fotografiada puesta por una persona) demuestra que las máscaras importan:

| Proveedor | Estado | `identity_preservation` | `outside_change` | `color_fidelity` | `score` |
|---|---|---:|---:|---:|---:|
| `none` (sin máscara) | **degraded** | 0.8232 | 29.98 | 0.5962 | 0.687 |
| `pose-heuristic` | ok | 0.9551 | 22.89 | 0.5869 | 0.734 |
| `sam2` | ok | **0.9650** | **22.66** | **0.5894** | **0.740** |
| `grounded-sam2` | ok | 0.9528 | 25.97 | 0.5878 | 0.734 |

Lectura: sin máscara (`none` degradado) la persona se conserva peor (0,82) y el
modelo cambia más fuera de la región de la prenda (30,0); con máscaras reales la
conservación sube a ~0,95-0,97 y el cambio fuera de la región baja. SAM 2 obtiene
el mejor resultado global en este par, y por eso es la opción recomendada para
prendas puestas por modelos mientras no exista el parser propio (etapa 3).

En el camino comercial (`flat-lay` + `segmentation_free`) los cuatro proveedores
producen la **misma** salida (byte a byte), como corresponde: ahí no se usa
ninguna máscara.

### Revisión del 2026-09-15: «la UI no parece distinta»

Al probar la interfaz en marcha (http://127.0.0.1:7863) el usuario no percibió
cambios. Comprobación y resultado:

- **Correcto por diseño**: la *imagen* del camino comercial es byte a byte la
  misma que antes del fork (`sha256 a4d618bf…`), también desde la UI
  (`scripts/smoke_webui.py` → `TODO OK`). El fork cambia licencias y arquitectura,
  no la salida del modelo que ya funcionaba.
- **Huecos reales de UX detectados y corregidos (2)**:
  1. La UI no mostraba qué proveedor/licencia se usó ni el aviso de degradación
     (iban solo al log del proceso) → ahora el log de la UI incluye
     `Segmentación: proveedor=… · licencia=… · máscara persona=… · máscara prenda=…`
     y `AVISO (degradado, resultado no óptimo): …`.
  2. La UI acumulaba un pipeline por proveedor en VRAM (9,5 GiB usados tras pasar
     de `none` a `sam2`) → caché de **un solo** pipeline: al cambiar de proveedor
     se libera el anterior (`Cambió el proveedor/device: liberando el pipeline
     anterior (VRAM)…`) y la VRAM reservada quedó en 2,13/5,19 GiB.
- **Diferencia visible** (misma petición, prenda tipo `model`): con `none` →
  `máscara prenda=no` + aviso de degradación; con `sam2` → `máscara prenda=sí`
  sin aviso. Antes del fork el camino `model` exigía el paquete NC; ahora hay una
  alternativa Apache-2.0 auditada.
- **Artefacto nuevo**: `scripts/smoke_webui.py` (prueba de humo de la UI:
  proveedor en el log, PNG de referencia, aviso de degradación según proveedor).
- **Detalle verificado**: la galería de Gradio sirve una copia re-codificada a
  WebP en `/tmp/gradio/...`; para comparar hashes hay que usar el PNG de
  `outputs/webui/`.

### Pendiente actualizado

1. **Etapa 3**: entrenar el parser propio (`custom-parser`) con datos propios (sigue siendo esqueleto documentado).
2. **Benchmark completo** del plan §14 (100 tops + 100 bottoms + 100 one-pieces) con imágenes propias: el arnés y las métricas están listos, faltan los datos (no se pueden usar VITON-HD/DressCode: están en la lista de prohibidos).
3. **Calibrar** los umbrales de `QualityGateConfig` con ese benchmark (hoy son valores de partida documentados).
4. **Construir y publicar la imagen Docker** en un host con Docker.
5. **Revisión legal profesional** antes del lanzamiento (el gate técnico no sustituye la asesoría legal).
