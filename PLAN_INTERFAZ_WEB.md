# Plan — Interfaz web de pruebas para FASHN VTON v1.5

Fecha: 2026-09-15. Proyecto: `/home/uceda/Documents/fashn-vton-1.5`
(commit base `7c0f10a`, ya levantado y verificado ese mismo día, ver
`LEVANTAMIENTO_2026-09-15.md`).

## 1. Objetivo

Dotar al repositorio de una **interfaz web local** para probar el modelo con
imágenes propias, sin escribir Python y sin depender del script `examples/`
(que solo acepta una combinación de argumentos por ejecución). Debe permitir:

1. Subir persona + prenda, elegir categoría y parámetros de difusión, y generar
   el try-on viendo **progreso y tiempos reales**.
2. Probar variantes rápido (mismo par, distintos `seed`/`steps`/`guidance`) sin
   recargar el modelo entre intentos.
3. Guardar cada resultado en disco (`outputs/webui/`) además de mostrarlo.
4. Levantarse con un comando (`./run_web.sh`) y detenerse con `Ctrl-C`/`kill`,
   igual que los otros proyectos de esta máquina.

## 2. Alcance

**Sí**:
- UI Gradio local (127.0.0.1) sobre `fashn_vton.TryOnPipeline` (API pública del
  repo, sin tocar su código).
- Carga perezosa y reutilización del pipeline (evita los ~20 s de carga en cada
  petición), serialización de peticiones y mensajes de error legibles.
- Progreso por paso de difusión (hook de lectura sobre `tqdm`, sin modificar el
  repo), tiempo transcurrido y uso de VRAM/RAM por paso en un log de la UI.
- Guardado automático de salidas + galería de resultados.
- Lanzador `run_web.sh` con el fix de `LD_LIBRARY_PATH` ya identificado en el
  levantamiento (sin él, DWPose cae a CPU).

**No** (fuera de alcance, se deja documentado):
- Autenticación, multiusuario, `share=True` público o despliegue en red.
- Cambiar el modelo, añadir fine-tuning, o soportar vídeo/múltiples prendas.
- Tocar `src/fashn_vton/**` o `examples/**` del repo (todo lo nuevo vive en
  `webui/`, `run_web.sh`, `logs/`, `outputs/webui/`).

## 3. Decisiones de diseño

| Decisión | Elección | Por qué |
|---|---|---|
| Framework | **Gradio 6.27.0** (Apache-2.0) | Es el que usa la demo oficial del proyecto (HF Spaces), ya está disponible en el índice pip, y es el patrón de las UIs de esta máquina (`IDM-CUSTOM/src/serve/app.py`). Se instala en el env `fashn-vton`, no se añade a `pyproject.toml` del repo |
| Layout | `gr.Blocks` (no `gr.Interface`) | Permite parámetros agrupados, galería + log y evita el botón "Flag" roto que ya apareció en `gr.Interface` (ver `16_REVISION_DOCUMENTACION_2026-09-15.md` D-08 de IDM-CUSTOM) |
| Puerto | **7863** por defecto | 7860 = demo IDM-VTON, 7861 = UI IDM-CUSTOM, 7862 = pruebas puntuales. Configurable con `GRADIO_SERVER_PORT` |
| Carga del modelo | Perezosa + caché por `(weights_dir, device)` con `threading.Lock` | La carga cuesta ~20 s y ~2 GB de VRAM; el usuario hace varias pruebas seguidas |
| Concurrencia | `concurrency_limit=1` en el evento de generar | El pipeline no es thread-safe y comparten VRAM; se serializan las peticiones |
| Progreso | Generador de Gradio + hilo de trabajo + hook sobre `fashn_vton.pipeline.tqdm` (subclase que notifica en `update()`, se restaura en `finally`) | Da pasos reales ("paso 12/30") sin modificar el código del repo; si el hook falla, la UI sigue mostrando tiempo transcurrido |
| Device / dtype | `auto` (usa la lógica del propio pipeline: `cuda` + bf16 si está soportado), con opción forzada `cuda`/`cpu` | Permite comparar GPU vs CPU y usar CPU si la GPU está ocupada |
| GPU compartida | Opcional `FASHN_USE_GPU_LOCK=1` → usa `infra.gpu_lock` de `IDM-CUSTOM` vía su CLI de solo stdlib | En esta máquina hay 3 consumidores de GPU (Pista A, IDM-CUSTOM, este). **Por defecto desactivado** para no acoplar proyectos |
| Salidas | Cada imagen se guarda en `outputs/webui/<YYYYmmdd_HHMMSS>_s<seed>_<i>.png` y se muestra en `gr.Gallery`; el log se guarda en `logs/webui_<fecha>.log` | Trazabilidad de pruebas y comparación fuera de la UI |
| Imágenes de entrada | El modelo **no** es maskless a nivel de código: necesita la persona completa y la prenda (foto de modelo o flat-lay). La UI explica ambos modos y trae los ejemplos del repo | Evita el error típico de usar un recorte de prenda con `--garment-photo-type model` |

## 4. Arquitectura

```
run_web.sh  (env fashn-vton + LD_LIBRARY_PATH + puerto libre + logs)
    └── python webui/app.py
            ├── build_app()            -> gr.Blocks (UI)
            ├── _get_pipeline()        -> TryOnPipeline cacheado (1 vez)
            ├── _generate(...)         -> generador Gradio (yield de estado)
            │      ├── hilo: pipeline(person, garment, ...)  [tqdm hook -> pasos]
            │      ├── _gpu_lock (opcional)  acquire/release
            │      └── guardado en outputs/webui/
            └── _memory_snapshot()     -> RAM (psutil) + VRAM (torch.cuda)
```

Archivos nuevos (ninguno modifica el repo original):

| Archivo | Contenido |
|---|---|
| `PLAN_INTERFAZ_WEB.md` | Este plan |
| `webui/app.py` | UI Gradio + lógica de ejecución/progreso/guardado |
| `webui/README.md` | Uso corto de la UI (parámetros, límites, troubleshooting) |
| `run_web.sh` | Lanzador (env, `LD_LIBRARY_PATH`, puerto, logs, PID) |

Directorios que se crean en tiempo de ejecución: `logs/`, `outputs/webui/`
(ambos ya cubiertos por `.gitignore` del repo).

## 5. Parámetros expuestos en la UI

| Control | Valores | Default | Nota |
|---|---|---|---|
| Persona | imagen (jpg/png/webp) | `examples/data/model.webp` | Imagen completa de la persona |
| Prenda | imagen | `examples/data/garment.webp` | Foto de modelo o producto |
| Categoría | `tops` / `bottoms` / `one-pieces` | `tops` | `TryOnPipeline.CATEGORY_TO_LABEL` |
| Tipo de prenda | `model` (puesta) / `flat-lay` (producto) | `model` | `flat-lay` → `disable_masking` de la prenda y pose sintética |
| Muestras | 1–4 | 1 | `num_samples` |
| Pasos | 8–50 (marcas 20/30/50) | 30 | `num_timesteps`; 8 para pruebas rápidas |
| Guidance | 0.5–5.0 | 1.5 | `guidance_scale` |
| Semilla | entero (aleatoria opcional) | 42 | `seed` |
| Sin segmentación | on/off | on | `segmentation_free` (default del README) |
| Device | `auto` / `cuda` / `cpu` | `auto` | `cpu` es mucho más lento (referencia) |

Salidas: `gr.Gallery` (todas las muestras) + `gr.Textbox` con el log numerado
(carga de modelos, parámetros efectivos, paso i/N con tiempo y RAM/VRAM,
tiempo total, ruta de los PNG guardados).

## 6. Comportamiento operativo

1. **Primera petición**: carga perezosa del pipeline (aviso "Cargando modelos…",
   ~20 s + descarga del human parser la primera vez) y se queda residente.
2. **Peticiones siguientes**: reutilizan el pipeline (~1,5 s/paso a 30 pasos;
   ~59 s de proceso completo medidos en el levantamiento).
3. **Serialización**: `concurrency_limit=1`; si llegan dos peticiones, la
   segunda espera en la cola de Gradio (no rompe VRAM).
4. **Errores**: cualquier excepción se muestra en el log con su tipo y mensaje
   (`gr.Error`), sin tumbar el servidor. Casos previstos: pesos faltantes
   (`weights/model.safetensors` o `weights/dwpose/*`), imagen no válida, GPU
   ocupada (si el lock opcional está activo), sin GPU y device forzado.
5. **Variables de entorno**: `GRADIO_SERVER_NAME`, `GRADIO_SERVER_PORT`,
   `GRADIO_SHARE`, `FASHN_WEIGHTS_DIR`, `FASHN_DEVICE`, `FASHN_OUTPUT_DIR`,
   `FASHN_USE_GPU_LOCK`, `FASHN_GPU_LOCK_PATH`, `CONDA_ENV`, `CONDA_SH`,
   `LOG_DIR`.
6. **Detención**: `bash run_web.sh` deja el proceso en segundo plano y escribe
   el PID; se detiene con `kill <PID>` (el lanzador usa `nohup ... & disown`,
   como `IDM-CUSTOM/run_web.sh`).

## 7. Fases de implementación

| Fase | Entregable | Verificación |
|---|---|---|
| **F1** | `gradio==6.27.0` instalado en el env `fashn-vton` (sin tocar `pyproject.toml`) | `python -c "import gradio; print(gradio.__version__)"` |
| **F2** | `webui/app.py`: pipeline cacheado, `_generate` generador, hook de progreso, guardado de salidas, manejo de errores, `build_app()` | Llamada directa a `_generate` con los ejemplos del repo (8 pasos) devuelve PIL 576×864 y escribe el PNG |
| **F3** | `webui/README.md` + `run_web.sh` (puerto, logs, PID, `LD_LIBRARY_PATH`) | `./run_web.sh` levanta y `curl -o /dev/null -w '%{http_code}'` → 200 |
| **F4** | Verificación end-to-end a través del **servidor** (no solo de la función) con `gradio_client` contra `api_name="/tryon"` | Cliente HTTP devuelve una imagen no vacía 576×864 |
| **F5** | Registro de la ejecución y de los tiempos en este mismo documento + actualización de `webui/README.md` con lo observado | Tabla §10 completa |

> **Estado: las cinco fases quedaron implementadas y verificadas el 2026-09-15**
> (evidencia y tiempos reales en §10). La UI sigue levantada en el puerto 7863.

## 8. Criterios de aceptación

1. `./run_web.sh` levanta la UI en `http://127.0.0.1:7863` y responde 200.
2. Una generación de 30 pasos por la UI devuelve 1 imagen 576×864 RGB, la
   guarda en `outputs/webui/` y muestra el log con tiempos y RAM/VRAM.
3. Cambiar `seed`/`num_timesteps`/`category` desde la UI cambia el resultado y
   el log lo refleja.
4. Sin `weights/`, la UI arranca igual y muestra un error legible (no un
   traceback crudo) al generar.
5. `run_web.sh` falla con mensaje claro si el puerto está ocupado; con
   `GRADIO_SERVER_PORT=7864` usa otro puerto.
6. El repo no queda modificado en su código: `git status` solo muestra los
   archivos nuevos listados en §4.

## 9. Riesgos y limitaciones

| Riesgo | Mitigación |
|---|---|
| El hook de progreso (subclase de `tqdm`) podría romperse con otra versión de tqdm | Se envuelve en `try/except`, se restaura en `finally` y la UI sigue funcionando sin pasos (solo tiempo) |
| VRAM de 12 GB compartida con Pista A / IDM-CUSTOM | Pipeline en bf16 (~2 GB), `concurrency_limit=1`, opción `cpu` y `FASHN_USE_GPU_LOCK=1` |
| La primera petición tarda (~20 s de carga) | Aviso en la UI + carga opcional al arrancar con `FASHN_PRELOAD=1` |
| `flat-lay` con prendas muy oscuras da resultados apagados (observado en el levantamiento) | Documentado en `webui/README.md`; se recomienda `model` para fotos de prenda puestas |
| Gradio no está en `pyproject.toml` del repo | Se documenta como dependencia del entorno (no del paquete): `pip install gradio==6.27.0` |

## 10. Registro de ejecución

**Estado: F1–F5 completas y verificadas el 2026-09-15.** El servicio queda
levantado en `http://127.0.0.1:7863` (PID registrado por `run_web.sh`).

| # | Verificación | Comando / evidencia | Resultado |
|---|---|---|---|
| F1 | Gradio en el entorno | `pip install gradio==6.27.0` + `import gradio` | `EXIT=0`; `gradio 6.27.0` (más `psutil 7.2.2` para reportar RAM) |
| F2 | Manejador de la UI (llamada directa) | `python /tmp/test_webui_generate.py` (8 pasos, ejemplos del repo) | `EXIT=0`; 1 imagen `(576, 864)`; `Modelos cargados en 8.3 s (device=cuda, dtype=torch.bfloat16)`; paso real 8/8; PNG guardado |
| F3 | Levantar y responder | `bash run_web.sh` + `curl -o /dev/null -w '%{http_code}'` | `HTTP=200`, escuchando en `127.0.0.1:7863`, log en `logs/webui_<fecha>.log` |
| F4 | End-to-end por HTTP | `gradio_client.Client("http://127.0.0.1:7863/").predict(..., api_name="/tryon")` | endpoint `/tryon` con la firma de 10 entradas; devuelve galería + log; `EXIT=0` |
| F5 | Criterios 1–6 de §8 | ver filas siguientes | todos ✅ |
| C1 | UI arriba y 200 | `curl` a 7863 | ✅ `HTTP=200` |
| C2 | 30 pasos por la UI | `/tryon` con `num_timesteps=30, seed=42` | ✅ 1 imagen 576×864 RGB guardada en `outputs/webui/20260915_125110_s42_00.png`; **46,5 s** totales (1,55 s/paso); log con `RAM=2.54GiB VRAM=2.07/5.08GiB` |
| C3 | Cambiar parámetros cambia el resultado | `/tryon` con `num_timesteps=8, seed=7` | ✅ `sha256` distinto (`46867179…` vs `9ec0d6d8…`); log refleja los parámetros; y **sin recargar modelos** (12,7 s, prueba de la caché) |
| C4 | Error legible sin pesos | `FASHN_WEIGHTS_DIR=/tmp/pesos_que_no_existen` + `webui.app.generate` | ✅ `gr.Error`: «No se pudieron cargar los modelos: Faltan pesos en … Descárgalos con: ./run_fashn_vton.sh python scripts/download_weights.py --weights-dir ./weights» (sin traceback crudo) |
| C5 | Puerto ocupado / alternativo | `bash run_web.sh` (con 7863 en uso) y `GRADIO_SERVER_PORT=7864 bash run_web.sh` | ✅ `ERROR: el puerto 7863 ya está en uso (¿otra UI levantada?).` con `exit_code=1`; y con 7864 levanta y responde `HTTP_7864=200` |
| C6 | Repo sin modificar | `git status --short` | ✅ solo nuevos: `PLAN_INTERFAZ_WEB.md`, `webui/`, `run_web.sh` (más los del levantamiento) |

Tiempos medidos (RTX 3060, bf16, caché de disco caliente):

| Escenario | Modelos | Sampling | Total |
|---|---|---|---|
| 8 pasos (primer uso del proceso) | 8,3 s | 13,2 s | ~22 s |
| 8 pasos (pipeline ya cargado) | — | 12,3 s | 12,7 s |
| 30 pasos (pipeline ya cargado) | — | ~46 s | 46,5 s |

Picos de memoria observados: **RAM 2,5 GiB** del proceso y **VRAM 2,07 GiB
asignada / 5,08 GiB reservada** por generación (cabe de sobra en 12 GB, incluso
compartiendo con IDM-CUSTOM si no se solapan las peticiones).

Diferencias de API encontradas con Gradio 6 (anotadas por si se actualiza la
versión): `gr.Gallery` ya no acepta `show_download_button` (se usa `preview`);
y `theme` debe pasarse a `launch()`, no al constructor de `Blocks()`.
