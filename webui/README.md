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
5. El **resultado final aparece también arriba a la derecha**; el botón «**Ver en
   grande (modal)**» lo abre a pantalla completa (o usa el icono de pantalla
   completa de la propia imagen).

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

## Tela propia: color, textura, tamaño y ángulo

Acordeón «**Tela propia: color, textura y ángulo (opcional)**». Sustituye el color
y el estampado de la prenda con la foto de una tela **antes** de generar el try-on
(es una transformación de *entrada*: el modelo y el camino comercial no se tocan).

1. Sube la **tela** (una foto del estampado; cuanto más plana y repetible, mejor)
   o elige un **color plano** si solo quieres cambiar el color.
2. Pulsa «**Previsualizar prenda con la tela**» para ver el resultado y ajustar.
3. Activa «**Aplicar tela a la prenda**» y genera. Con el acordeón inactivo el
   resultado es exactamente el de antes (mismo `sha256`).

Se **asume que la prenda está sobre fondo blanco o gris** (foto de producto): el
sistema mide la mediana del borde y separa la prenda por distancia de color
(«Máscara de la prenda = auto», con la «Tolerancia de fondo» ajustable). Si el
fondo no es uniforme, la UI avisa en el log y conviene subir la tolerancia o usar
«tela-completa». Como referencia, el modo `flat-lay` es el que mejor encaja.

| Grupo | Control | Qué hace |
|---|---|---|
| Tamaño | **Cómo se mide el motivo**: `centímetros reales` / `píxeles de la imagen` / `relativo (escala)` | Elige la unidad; el patrón **se repite** siempre (mosaico infinito) |
| Tamaño | **Tamaño del motivo (cm)** + **Ancho real de la prenda (cm)** | Calibración real: 50 cm de prenda con motivo de 12 cm ⇒ **~4,2 repeticiones** |
| Tamaño | **Tamaño del motivo (px)** | Para técnicos: ancho del motivo en píxeles de la imagen de la prenda |
| Tamaño | **Escala relativa** | `1.0` = la tela cubre el ancho de la prenda |
| Ángulo | **Ángulo del estampado (°)** | Gira el mosaico (para que las rayas sigan la dirección deseada) |
| Ángulo | **Enderezar la foto de la tela (°)** | Corrige una foto de tela torcida antes de aplicarla |
| Ángulo | **Inclinación X/Y (sesgo)** y **Profundidad Z (perspectiva)** | Simula el plano de la prenda: la tela «cae» con el volumen de la superficie |
| Encaje | **Desplazamiento X/Y** y **Mosaico** (`repetir` / `espejo`) | Mueve el patrón y elige si el mosaico refleja (sin costuras visibles) o repite |
| Resultado | **Fuerza de la tela**, **Conservar sombras**, **Conservar detalles (costuras)**, **Brillo de la tela** | Mezcla con la prenda original, mantiene pliegues y volumen, o reintroduce costuras/botones |
| Resultado | **Solo panel frontal (banda central)** | `1.0` = toda la prenda; menos, una banda central (aproximación geométrica del frente) |
| Fondo | **Fondo del resultado**: `original` / `blanco` / `gris` / `color` (+ selector) | Recompone el fondo de la prenda sintética |
| Ayuda | **Cuadrícula de 10 cm (solo previsualización)** | Dibuja una cuadrícula de referencia **solo en la previsualización**; la imagen que entra al try-on nunca la lleva |

Bajo el acordeón hay una línea viva que informa del tamaño calculado, por ejemplo:

```
Motivo: 138 px de ancho (~12,0 cm sobre 50 cm de prenda) · repeticiones a lo ancho: 4.17
```

Implementación: `src/fashn_vton/preprocessing/fabric.py` (solo numpy/OpenCV/PIL,
**sin nuevos pesos ni licencias**; 17 pruebas en `tests/test_fabric_transfer.py`).
Los parámetros se pueden usar también desde Python o la API:

```python
from fashn_vton.preprocessing.fabric import FabricTransferConfig, retexture_garment

config = FabricTransferConfig(
    enabled=True, repeat_mode="cm", repeat_cm=12, garment_width_cm=50,
    rotation=45, tilt_y=0.15, strength=1.0, shading=1.0, background="blanco",
)
imagen_con_tela, mascara, info = retexture_garment(prenda_rgb, tela_rgb, config)
print(info.describe(), "|", info.describe_repeat())
```

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
galeria, log, preview = client.predict(
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
    # --- tela propia (opcional; 21 parámetros, ver el acordeón) ---
    handle_file("tela.png"),  # imagen de la tela (None = no usar)
    True,        # fabric_enabled
    "#ffffff",   # color plano
    "blanco",    # fondo del resultado: original|blanco|gris|color
    "#ffffff",   # color de fondo
    "auto",      # máscara: auto|tela-completa
    30.0,        # tolerancia de fondo
    1.0,         # solo panel frontal (1.0 = toda la prenda)
    1.0,         # escala relativa
    0.0,         # ángulo del estampado
    0.0,         # inclinación X
    0.0,         # inclinación Y
    0.0,         # desplazamiento X
    0.0,         # desplazamiento Y
    0.0,         # profundidad Z
    0.0,         # enderezar la foto de la tela
    1.0,         # brillo
    1.0,         # fuerza
    1.0,         # conservar sombras
    0.0,         # conservar detalles
    "repetir",   # mosaico: repetir|espejo
    "cm",        # medida del motivo: cm|px|scale
    12.0,        # tamaño del motivo (cm)
    50.0,        # ancho real de la prenda (cm)
    120.0,       # tamaño del motivo (px)
    api_name="/tryon",
)
print(galeria, log, preview)
```

El endpoint devuelve **tres salidas**: la galería, el log y la **previsualización
del resultado final** (la misma que se ve arriba a la derecha en la interfaz).
Leer las salidas por posición (no desempaquetando) evita romperse si se añaden más.

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
- El **modal** no existe como componente en Gradio 6.27: se implementa con un
  `gr.Column` en posición fija y `visible=False` (CSS `.vton-modal` en `MODAL_CSS`)
  que un botón muestra/oculta; así el visor grande no depende de componentes
  internos ni de librerías nuevas.
- Las imágenes usan `buttons=["download", "fullscreen"]`: el icono de pantalla
  completa es el visor rápido y «Ver en grande (modal)» es el overlay propio.
- Cada salida nueva del evento `/tryon` (hoy: galería, log y **previsualización**)
  obliga a actualizar los clientes; leer las salidas por posición, no
  desempaquetando.
- `theme` se pasa a `launch()`, **no** al constructor de `Blocks()` (da aviso de
  deprecación en Gradio 6).
- El progreso real por paso se obtiene envolviendo `tqdm` dentro de
  `fashn_vton.pipeline` con una subclase que notifica en `update()` y se
  restaura en `finally`; no requiere modificar el código del repo.
- Los resultados de `/tryon` pasan por la caché de Gradio (rutas temporales en
  `/tmp/gradio/...`); los PNG definitivos están en `outputs/webui/`.
