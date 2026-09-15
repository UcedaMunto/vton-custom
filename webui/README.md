# `webui/` — interfaz web local de pruebas (Gradio)

> **Actualización comercial (2026-09-15)**: la UI ya no depende de
> `fashn-human-parser`. Ahora tiene un **selector de proveedor de segmentación**
> (`none` por defecto = camino comercial sin máscaras; `pose-heuristic`
> experimental; `sam2` / `grounded-sam2` / `custom-parser` documentados como
> etapas 2/3) y usa el turno único de GPU del propio fork
> (`fashn_vton.utils.gpu_lock`, mismo archivo que IDM-CUSTOM). El tipo de foto de
> prenda por defecto pasa a ser `flat-lay`, que es el camino soportado
> comercialmente. Ver [`../README_COMERCIAL.md`](../README_COMERCIAL.md) y
> [`../src/fashn_vton/segmentation/README.md`](../src/fashn_vton/segmentation/README.md).

Envuelve la API pública del repositorio (`fashn_vton.TryOnPipeline`) sin
modificar su código. Diseño y fases en [`../PLAN_INTERFAZ_WEB.md`](../PLAN_INTERFAZ_WEB.md).

## Levantar

```bash
cd /home/uceda/Documents/fashn-vton-1.5
./run_web.sh                            # http://127.0.0.1:7863, segundo plano
GRADIO_SERVER_PORT=7864 ./run_web.sh    # otro puerto
tail -f logs/webui_*.log                # log del servidor
kill <PID>                              # detener (el PID lo imprime el lanzador)
```

El lanzador activa el env `fashn-vton`, exporta `LD_LIBRARY_PATH` con las libs
CUDA de `nvidia-*` (sin eso la pose cae a CPU) y comprueba que el puerto esté
libre. Alternativa directa, sin `run_web.sh`:

```bash
./run_fashn_vton.sh python webui/app.py
# o: FASHN_PRELOAD=1 GRADIO_SERVER_PORT=7864 python webui/app.py   (con el env activado)
```

## Uso

1. Sube la **persona** (foto completa) y la **prenda** (foto de modelo o
   producto). El botón «Cargar ejemplos del repo» usa `examples/data/`.
2. Elige categoría (`tops` / `bottoms` / `one-pieces`), tipo de foto de prenda
   (`model` = puesta, `flat-lay` = producto), pasos, guidance, muestras y semilla.
3. «Generar try-on»: verás la barra de progreso, el **paso i/N real**, el tiempo
   y el uso de VRAM/RAM; al terminar, la galería con el resultado y el log.
4. Los PNG se guardan en `outputs/webui/<fecha>_s<semilla>_<i>.png` junto a un
   `..._params.txt` con el log de la petición.

## Parámetros y coste (medido en esta máquina, RTX 3060)

| Parámetro | Default | Notas |
|---|---|---|
| Pasos (`num_timesteps`) | 30 | 8 pasos ≈ 13 s · 30 pasos ≈ 43 s de sampling (~1,5 s/paso) |
| Guidance (`guidance_scale`) | 1.5 | Default del repo; a más valor, más fidelidad a la prenda y más artefactos |
| Muestras (`num_samples`) | 1 | 1–4; cada muestra extra cuesta otra pasada completa |
| Tipo de foto de prenda | `flat-lay` | `flat-lay` es el camino comercial; `model` requiere un proveedor con máscaras |
| Proveedor de segmentación | `none` | `none` = sin modelos extra (comercial) · `pose-heuristic` = experimental |
| `segmentation_free` | on | On = camino comercial; off solo tiene sentido con un proveedor que aporte máscaras |
| Device | `auto` | `auto` = GPU + bfloat16 si está disponible; `cpu` es ~10-20× más lento |
| Primera petición | — | ~8-20 s de carga de modelos (según caché de disco) y ~2 GB de VRAM al generar |

## Proveedores de segmentación en la UI

El desplegable «Proveedor de segmentación» muestra todos los proveedores
registrados con su licencia; los que no están disponibles aparecen igualmente y
fallan con un mensaje accionable si se seleccionan. Con un proveedor que no
aporta máscaras (`none`) y `flat-lay` + `segmentation_free`, la generación es
idéntica al pipeline original de FASHN (verificado por sha256) y sin ninguna
dependencia de licencia no comercial.

## Variables de entorno

| Variable | Default | Para qué |
|---|---|---|
| `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT` | `127.0.0.1` / `7863` | Escucha y puerto (7860 = demo IDM-VTON, 7861 = IDM-CUSTOM) |
| `GRADIO_SHARE` | `0` | `1` para túnel público de Gradio (no recomendado) |
| `FASHN_WEIGHTS_DIR` | `./weights` | Directorio de pesos |
| `FASHN_DEVICE` | `auto` | Device inicial del desplegable |
| `FASHN_OUTPUT_DIR` | `./outputs/webui` | Dónde guardar los PNG |
| `FASHN_PRELOAD` | `0` | `1` carga los modelos al arrancar en vez de en la primera petición |
| `FASHN_USE_GPU_LOCK` | `0` | `1` pide el turno único de GPU con `infra.gpu_lock` de IDM-CUSTOM (para no chocar con la Pista A) |
| `FASHN_GPU_LOCK_PATH` | (el de `infra.gpu_lock`) | Lock alternativo (pruebas) |
| `FASHN_GPU_LOCK_IDM_SRC` | `/home/uceda/Documents/IDM-CUSTOM/src` | Ruta a `infra.gpu_lock` |

## Comportamiento

- **Carga perezosa y cacheada** del pipeline por `(weights_dir, device, proveedor)`.
  La caché guarda **un solo** pipeline: al cambiar de proveedor o de device se
  libera el anterior (`del` + `torch.cuda.empty_cache()`, con la línea
  `Cambió el proveedor/device: liberando el pipeline anterior (VRAM)…`) para no
  acumular copias del modelo en la VRAM. Cambiar de proveedor cuesta ~6-8 s de
  recarga.
- **Transparencia de licencia**: al terminar cada generación el log de la UI
  muestra qué proveedor y licencia se usaron y si hubo máscaras:

  ```
  8. Segmentación: proveedor=none · licencia=N/A (no carga ningún modelo) · máscara persona=no · máscara prenda=no
  9. AVISO (degradado, resultado no óptimo): prenda: se pidió garment_photo_type='model'
     sin máscara disponible; se trató la prenda como flat-lay (sin aislarla)
  ```

  Con `sam2` (Apache-2.0) la misma petición sale sin aviso y con `máscara prenda=sí`.
- **Peticiones serializadas** (`concurrency_limit=1`): el pipeline no es
  thread-safe y comparte VRAM; la segunda petición espera en la cola.
- **Progreso real** por paso: se envuelve `tqdm` dentro de `fashn_vton.pipeline`
  con una subclase que notifica y se restaura siempre en `finally`. Si ese hook
  fallara, la UI seguiría mostrando tiempo y memoria.
- **Errores**: se muestran en el log de la UI y como error de Gradio (no hay
  traceback crudo en pantalla). Ejemplo con pesos ausentes:

  ```
  2. ERROR al cargar los modelos: RuntimeError: Faltan pesos en /ruta/weights:
       - /ruta/weights/model.safetensors
     Descárgalos con: ./run_fashn_vton.sh python scripts/download_weights.py --weights-dir ./weights
  ```

## Limitaciones conocidas

- Salida siempre **576×864** (forma de entrada del modelo).
- Con prendas muy oscuras o azuladas el resultado puede salir apagado (visto en
  `LEVANTAMIENTO_2026-09-15.md`); el modo `flat-lay` es más sensible a esto.
- No hay autenticación ni soporte multiusuario: es una UI local de pruebas.
- Gradio **no** está en el `pyproject.toml` del repo (no se modificó): es
  dependencia del entorno (`pip install gradio==6.27.0`, ya instalada en
  `fashn-vton`). `psutil` se usa solo para reportar RAM y se degrada si falta.

## Usar la UI como API (sin navegador)

El evento de generación está expuesto como endpoint, así que se puede probar por
HTTP (útil para scripts o pruebas automatizadas):

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7863/")
galeria, log = client.predict(
    handle_file("examples/data/model.webp"),   # persona
    handle_file("examples/data/garment.webp"), # prenda
    "tops",      # categoría
    "flat-lay",  # tipo de foto de prenda
    1,           # muestras
    8,           # pasos
    1.5,         # guidance
    42,          # semilla
    True,        # segmentation_free
    "none",      # proveedor de segmentación
    "auto",      # device
    api_name="/tryon",
)
print(galeria, log)
```

### Script de humo (recomendado)

[`../scripts/smoke_webui.py`](../scripts/smoke_webui.py) hace lo anterior y además
comprueba las promesas del fork: que el log reporte el proveedor, que el **camino
comercial** (`flat-lay` + `segmentation_free` + `none`, 8 pasos, semilla 42)
reproduzca el PNG de referencia pre-fork (`sha256 a4d618bf…`) y que el aviso de
degradación aparezca (o no) según el proveedor:

```bash
./run_fashn_vton.sh python scripts/smoke_webui.py
# compara proveedores y muestra el aviso en el log de la UI:
./run_fashn_vton.sh python scripts/smoke_webui.py --providers none sam2 --garment-photo-type model
```

Compara el **PNG guardado** en `outputs/webui/` (la galería sirve una copia
re-codificada a WebP en `/tmp/gradio/...`, cuyo hash no es comparable).

## Notas de implementación (Gradio 6)

- `gr.Gallery` ya **no** acepta `show_download_button` (se usa `preview=True`).
- `theme` se pasa a `launch()`, **no** al constructor de `Blocks()` (da aviso de
  deprecación en Gradio 6).
- El progreso real por paso se obtiene envolviendo `tqdm` dentro de
  `fashn_vton.pipeline` con una subclase que notifica en `update()` y se
  restaura en `finally`; no requiere modificar el código del repo.
- Los resultados de `/tryon` pasan por la caché de Gradio (rutas temporales en
  `/tmp/gradio/...`); los PNG definitivos están en `outputs/webui/`.
