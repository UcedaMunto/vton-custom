# Guía de ejecución del entrenamiento — FASHN VTON v1.5 (fork comercial)

> **Estado: 2026-09-17.** Arnés de fine-tuning **implementado, probado y ejecutado**
> de punta a punta con el dataset local `IDM-VTON/dataset` (DressCode, licencia de
> investigación / no comercial) y con los pesos comerciales reales (RTX 3060, 12 GB).
> Este documento es la guía operativa; el *por qué* y las fases están en
> [`PLAN_FINE_TUNING.md`](PLAN_FINE_TUNING.md).

---

## Dónde se ejecuta (infraestructura)

Todo esto corre en **`nodo-gpu-1` = `server`** (IP fija `192.168.0.100`), la **única**
máquina con GPU del clúster de 3 nodos (`nodo-orq` = `anfitrion` `192.168.0.10`;
`nodo-cpu-1` = `asus-tuf` `192.168.0.20`, sin GPU). Implicaciones prácticas:

- El entrenamiento **comparte la RTX 3060** con el servicio del clúster, la UI local y
  `IDM-CUSTOM`: usar el turno único (`FASHN_USE_GPU_LOCK=1` en la UI /
  `VTON_USE_GPU_LOCK=1` en la API y en el worker, archivo `~/.idm_gpu.lock`) y preferir
  horarios en los que el servicio no tenga tráfico.
- Los pesos y los candidatos viven en `weights/` de esta máquina. El worker GPU del clúster
  monta **solo** `model.safetensors` + `dwpose/` (nunca `candidates/`), y al promover un
  candidato hay que reiniciar el worker:
  `kubectl -n vton rollout restart daemonset/vton-gpu-worker`.
- Acceso a los otros nodos desde `server`: `ssh uceda@192.168.0.10` y
  `ssh uceda@192.168.0.20` (llave `~/.ssh/id_ed25519`, verificado 2026-09-19).
- Inventario de red, IPs, cortafuegos y hallazgos:
  [`../../kubernetes/01_RED_E_INVENTARIO.md`](../../kubernetes/01_RED_E_INVENTARIO.md).

---

## 0. Resumen en una página

**Regla de oro:** el entrenamiento **nunca** escribe `weights/model.safetensors`.
Escribe candidatos en `weights/candidates/<tag>/`. El modelo comercial sigue siendo
el punto de partida y el modelo activo hasta que alguien lo promueva a mano, con
verificación previa y (si los datos son no comerciales) ni eso.

| Pieza | Estado | Evidencia |
|---|---|---|
| LoRA sobre el MMDiT (sin `peft`) | ✅ | `src/fashn_vton/train/lora.py` · 72 módulos, 7.700.480 params entrenables (0,79 % de 979 M) |
| Dataset con el preprocesado exacto de la inferencia | ✅ | `src/fashn_vton/train/data.py` |
| Bucle *rectified flow* + checkpointing + resume | ✅ | `src/fashn_vton/train/trainer.py` |
| Guardia de datos/pesos NC (entrada y salida) | ✅ | `src/fashn_vton/train/nc_policy.py` |
| Convertidor de DressCode / prep IDM-VTON a CSV | ✅ | `scripts/prepare_nc_dataset.py` |
| CLI de entrenamiento | ✅ | `scripts/fine_tune.py` |
| Bloqueo de promoción de candidatos NC | ✅ | `scripts/model_registry.py promote` → exit 3 |
| Pruebas | ✅ | `tests/test_train_lora.py`, `test_train_data.py`, `test_train_step.py`, `test_nc_policy.py` (60 pruebas, sin GPU y sin pesos) |

**Ciclo completo en 5 comandos** (con `FASHN_ALLOW_NC_TRAINING=1` exportado):

```bash
cd /home/uceda/Documents/fashn-vton-1.5

# 0) Empaquetar el dataset local en un CSV de pares
python scripts/prepare_nc_dataset.py dresscode \
    --root /home/uceda/Documents/IDM-VTON/dataset/DATA_DIR \
    --out datos/nc/dresscode-train-500 --splits train --max-pairs 500 --verify 20

# 1) Validar plan (no toca GPU ni pesos)
python scripts/fine_tune.py --pairs datos/nc/dresscode-train-500/pairs.csv --check-only

# 2) Prueba de humo (2 pares, 2 pasos): mide tiempo y VRAM reales
./run_fashn_vton.sh python scripts/fine_tune.py --pairs datos/nc/dresscode-train-500/pairs.csv \
    --limit 2 --steps 2 --out weights/candidates/smoke --no-save-optimizer

# 3) Entrenar de verdad (ejemplo: 1.000 pasos ≈ 4,3 h en esta máquina)
./run_fashn_vton.sh python scripts/fine_tune.py --pairs datos/nc/dresscode-train-500/pairs.csv \
    --steps 1000 --rank 16 --out weights/candidates/lora-dresscode-v1

# 4) ¿Empeora? (0 = no empeora · 1 = regresión · 2 = error)
./run_fashn_vton.sh python scripts/baseline.py verify --name baseline-2026-09-15 \
    --weights-dir weights/candidates/lora-dresscode-v1
```

---

## 1. El dataset local: qué es exactamente y qué licencia tiene

Ruta analizada: `/home/uceda/Documents/IDM-VTON/dataset` (53.792 pares en total).

### 1.1 `DATA_DIR/` — **DressCode** en su layout original

```
DATA_DIR/
├── upper_body/   lower_body/   dresses/
│   ├── images/       xxxxxx_0.jpg  (persona con la prenda puesta = objetivo)
│   │                 xxxxxx_1.jpg  (la prenda, foto de producto = entrada)
│   ├── label_maps/   xxxxxx_4.png  (mapa de etiquetas del modelo, ids ATR)
│   ├── keypoints/    xxxxxx_5.json (18 keypoints, calculados a 512x384)
│   ├── skeletons/    xxxxxx_5.jpg  (esqueleto renderizado)
│   ├── dense/        xxxxxx_5.png + xxxxxx_5_uv.npz  (DensePose)
│   └── train_pairs.txt, test_pairs_paired.txt, test_pairs_unpaired.txt
├── train_pairs.txt          (48.392 pares, 3ª columna = categoría 0/1/2)
├── test_pairs_paired.txt    (5.400 pares: 1.800 por categoría)
└── multigarment_test_triplets.txt
```

Comprobado en esta máquina: imágenes de 768×1024, **48.392 pares de train**
(upper 13.563 · lower 7.151 · dresses 27.678) y **5.400 de test**
(1.800 por categoría) — coincide con los totales publicados de DressCode (53.792).

Etiquetas del `label_maps` (esquema ATR/SCHP de 18 clases) y lo que el
convertidor usa como «prenda» de cada categoría:

| Carpeta | Categoría del modelo | Etiqueta de la prenda | Por qué |
|---|---|---|---|
| `upper_body` | `tops` | `4` (upper-clothes) | 11,5 % de la imagen; es la prenda que lleva puesta el modelo |
| `lower_body` | `bottoms` | `5;6` (falda, pantalón) | DressCode mezcla faldas y pantalones en `lower_body` |
| `dresses` | `one-pieces` | `7` (dress) | 18,1 % de la imagen |

### 1.2 `DATA_DIR_PREP/` — DressCode `upper_body` **preparado estilo IDM-VTON**

```
DATA_DIR_PREP/
├── train/  (13.563 pares)  y  test/
│   ├── image/           xxxxxx_0.jpg       (persona)
│   ├── cloth/           xxxxxx_0.jpg       (prenda)
│   ├── image-densepose/ xxxxxx_0.jpg       (DensePose renderizado)
│   ├── agnostic-mask/   xxxxxx_0_mask.png  (máscara binaria de la prenda, 255)
│   └── vitonhd_train_tagged.json           (file_name → categoría: TOPS)
├── train_pairs.txt        (13.563)
├── train_pairs_clean.txt  (11.349, la versión filtrada que usa IDM-VTON)
└── test_pairs.txt         (2.032)
```

`DressCode/` (aparte, 35,5 GB) es la descarga original (`DressCode_part_aa`); el
arnés no la usa, se apoya en `DATA_DIR`/`DATA_DIR_PREP` para no duplicar disco.

### 1.3 Licencia y consecuencias (importante)

| | |
|---|---|
| Licencia real | Dataset de **investigación** (VITON-HD/DressCode): no comercial; DressCode además prohíbe el uso por empresas privadas |
| Clase que declara el CSV | `nc-dresscode` |
| Efecto en el código | Entrenar exige `FASHN_ALLOW_NC_TRAINING=1` (o `--allow-nc`); el candidato queda marcado `commercial_use: false`; `model_registry.py promote` **se niega** (exit 3) |
| Efecto legal | Los pesos resultantes **no** pueden ser el producto comercial. Sirven para investigación/tesis y para medir si la receta mejora |

El gate comercial del repo (`scripts/verify_commercial_readiness.py`) sigue en
verde porque los datos viven **fuera** del repo y el modelo activo no se toca
(verificado hoy: exit 0 → «LISTO PARA COMERCIAL»).

### 1.4 Cómo se construye cada muestra de entrenamiento

Para cada par del CSV, el dataset reproduce **exactamente** el preprocesado de
`TryOnPipeline.__call__` (es lo que evita la desalineación entrenamiento/inferencia
que documentó la Pista A):

| Tensor | Qué es | Forma |
|---|---|---|
| `ca_images` | foto de la persona con la prenda **borrada** (gris 127) vía `create_clothing_agnostic_image` | (3, 864, 576) |
| `person_poses` | render DWPose en escala de grises | (1, 864, 576) |
| `garment_images` | la prenda tal cual (modo `flat-lay`, como el camino comercial) | (3, 864, 576) |
| `garment_poses` | pose vacía (`get_dummy_dw_keypoints`) | (1, 864, 576) |
| `x1` | **objetivo**: la foto real de la persona con esa prenda (`xxxxxx_0.jpg`) | (3, 864, 576) |
| `garment_categories` | 1 = tops · 2 = bottoms · 3 = one-pieces | escalar |

Notas de diseño:

- La imagen agnóstica es imprescindible: si se pasara la foto original como
  `ca_images`, el modelo podría limitarse a **copiar la entrada** y aprendería un
  atajo en vez de la transferencia de prenda. Para pruebas rápidas existe
  `--ca-mode none` (avisa por log de que no sirve para entrenar de verdad).
- Las poses se cachean en `datos/nc/pose_cache/` (DWPose se ejecuta una vez por
  imagen): la segunda época es mucho más rápida.
- `--split test` reserva pares para validación: nunca se entrena con el test.

---

## 2. Estado de partida: el modelo comercial intacto

Comprobación rápida antes de tocar nada:

```bash
cd /home/uceda/Documents/fashn-vton-1.5
python scripts/model_registry.py current          # qué pesos hay activos y con qué tag
python scripts/model_registry.py list             # estados registrados (hash + tamaño)
python scripts/verify_commercial_readiness.py     # exit 0 = LISTO PARA COMERCIAL
python -m pytest tests/ -q                        # suite completa
```

Lo que protege al modelo actual:

| Protección | Dónde | Efecto |
|---|---|---|
| Baseline congelado | `baselines/baseline-2026-09-15/manifest.json` | Huella de pesos, entorno, semillas y métricas |
| Registro de pesos | `model_registry/registry.json` + `weights/registry/<tag>/model.safetensors` | `snapshot` / `promote` / `rollback` con hash |
| El CLI nunca escribe los pesos activos | `scripts/fine_tune.py` (`_resolve_output_dir`) | Falla si `--out` no está dentro de `weights/candidates/` |
| Guardia anti-regresión | `scripts/baseline.py verify` | Exit 1 si el candidato empeora >2 % alguna métrica |
| Bloqueo de promoción NC | `scripts/model_registry.py` + `train/nc_policy.py` | Exit 3 con candidatos no comerciales |

---

## 3. Requisitos y entorno

- Entorno conda **`fashn-vton`** (Python 3.11) ya instalado; ejecutar siempre con
  `./run_fashn_vton.sh` (arregla `LD_LIBRARY_PATH` para `onnxruntime-gpu`, y sin
  él DWPose cae silenciosamente a CPU).
- Pesos base: `weights/model.safetensors` (1,94 GB) + `weights/dwpose/` (351 MB).
  Si faltan: `python scripts/download_weights.py --weights-dir ./weights`.
- GPU: RTX 3060 12 GB. El entrenamiento usa el **turno único de GPU**
  (`fashn_vton.utils.gpu_lock`, mismo `~/.idm_gpu.lock` que IDM-CUSTOM): si la UI,
  IDM-CUSTOM o el watchdog de la Pista A la tienen tomada, el CLI aborta con un
  mensaje claro (usa `--no-gpu-lock` solo si sabes que estás solo).
- Disco: el candidato ocupa ~1,94 GB (modelo mergeado) + 30 MB (adaptador LoRA) +
  caché de poses (~13 MB por 1.000 pares). El CSV no copia imágenes.

---

## 4. Paso 1 — Preparar los pares (CSV)

```bash
cd /home/uceda/Documents/fashn-vton-1.5

# (a) DressCode completo, train, submuestra de 500 pares por split
python scripts/prepare_nc_dataset.py dresscode \
    --root /home/uceda/Documents/IDM-VTON/dataset/DATA_DIR \
    --out datos/nc/dresscode-train-500 --splits train --max-pairs 500 --verify 20

# (b) Todo el train de DressCode (48.392 pares; el CSV pesa ~15 MB)
python scripts/prepare_nc_dataset.py dresscode \
    --root /home/uceda/Documents/IDM-VTON/dataset/DATA_DIR \
    --out datos/nc/dresscode-full --splits train

# (c) Solo una categoría, con el test aparte para validar
python scripts/prepare_nc_dataset.py dresscode \
    --root /home/uceda/Documents/IDM-VTON/dataset/DATA_DIR \
    --out datos/nc/dresscode-upper --splits train,test \
    --categories upper_body=tops:4

# (d) Layout preparado estilo IDM-VTON (upper_body, con su lista filtrada)
python scripts/prepare_nc_dataset.py prep \
    --root /home/uceda/Documents/IDM-VTON/dataset/DATA_DIR_PREP \
    --out datos/nc/dresscode-upper-prep --splits train --pairs-file train_pairs_clean.txt

# (e) Solo resumir (no escribe nada)
python scripts/prepare_nc_dataset.py dresscode \
    --root /home/uceda/Documents/IDM-VTON/dataset/DATA_DIR --info
```

Qué produce cada ejecución:

| Fichero | Contenido |
|---|---|
| `pairs.csv` | una fila por par: `pair_id,person,garment,target,category,garment_photo_type,agnostic,agnostic_kind,agnostic_labels,license_class,source,split` |
| `provenance.json` | dataset, licencia, `commercial_use`, nº de pares, layout y estadísticas |
| `NOTICE.md` | aviso legal del dataset (no comercial, fuera del repo) |

Verificado hoy con el dataset real: `48.392` pares detectados (upper 13.563 ·
lower 7.151 · dresses 27.678), `--verify 8` → «los primeros 8 pares existen», y
`provenance.json` con `"commercial_use": false`.

Opciones útiles: `--max-pairs N` (submuestra reproducible con `--seed`),
`--splits train,test`, `--categories carpeta=categoria:ids`, `--license-class`
(por defecto `nc-dresscode`; si declaras una licencia comercial para estos datos
el script se niega salvo `--force-license`).

---

## 5. Paso 2 — Prueba de humo (obligatoria antes de entrenar en serio)

```bash
export FASHN_ALLOW_NC_TRAINING=1        # opt-in explícito para datos NC

# (a) Validar datos, licencias y plan sin tocar la GPU ni los pesos
python scripts/fine_tune.py --pairs datos/nc/dresscode-train-500/pairs.csv --check-only

# (b) 2 pares y 2 pasos reales (carga de pesos incluida ≈ 1,5 min)
./run_fashn_vton.sh python scripts/fine_tune.py --pairs datos/nc/dresscode-train-500/pairs.csv \
    --limit 2 --steps 2 --out weights/candidates/smoke --dtype bf16 --no-save-optimizer
```

Salida esperada (medida el 2026-09-17 en esta máquina, RTX 3060):

```
trainer - INFO - LoRA r=16 alpha=16.0 en 72 módulos (7,700,480 params entrenables de 979,514,288)
trainer - INFO - Dataset: 2 pares · resolución 576x864 · batch 1 · acumulación 4 · dtype torch.bfloat16
trainer - INFO - paso 1/2 · loss 0.0171 (media 0.0171) · |grad| 0.001 · |delta| 0.1027 · 15.01 s/paso · ETA 0.2 min · VRAM pico 3.03 GiB
trainer - INFO - Checkpoint mergeado: .../weights/candidates/smoke/model.safetensors
trainer - INFO - Fin: 2 pasos en 0.5 min (15.31 s/paso).
```

Qué comprobar tras la prueba de humo:

```bash
ls -la weights/candidates/smoke          # model.safetensors (1,94 GB), lora/, summary.json,
                                         # train_log.jsonl, train_config.json, provenance.json, dwpose -> symlink
python -m json.tool weights/candidates/smoke/summary.json | head -20
python -m json.tool weights/candidates/smoke/provenance.json   # "commercial_use": false
```

Criterios de aceptación de la prueba de humo: loss finita (no `nan`), VRAM pico
< 10 GiB, `model.safetensors` escribible y **cargable por el pipeline** (lo
comprueba el paso 4 con `baseline.py verify`).

---

## 6. Paso 3 — Entrenamiento real

### 6.1 Comando base

```bash
export FASHN_ALLOW_NC_TRAINING=1

./run_fashn_vton.sh python scripts/fine_tune.py \
    --pairs datos/nc/dresscode-train-500/pairs.csv \
    --steps 1000 \
    --rank 16 --alpha 16 \
    --lr 1e-4 --grad-accum 4 \
    --out weights/candidates/lora-dresscode-v1 \
    2>&1 | tee logs/train-lora-dresscode-v1.log
```

### 6.2 Hiperparámetros (valores por defecto = receta del plan §4)

| Parámetro | Valor | Nota |
|---|---|---|
| `--rank` / `--alpha` | 16 / 16 | 7,7 M params (0,79 %). Subir a 32 si el `verify` no mejora |
| `--targets` | `qkv,proj,linear1,linear2` | Atención (qkv/proj) + MLP fusionado de todos los bloques |
| `--lr` / `--weight-decay` | 1e-4 / 1e-2 | Rango LoRA; bajar a 5e-5 si la loss oscila |
| `--steps` / `--grad-accum` | 1000 / 4 | Batch efectivo 4 sin más VRAM |
| `--resolution` | 576x864 | La resolución real del modelo: no desalinear con la inferencia |
| `--dtype` | bf16 | En esta GPU (Ampere) es nativo y ahorra la mitad de VRAM |
| `--cond-drop` | 0,1 | Mantiene útil el CFG de la inferencia |
| `--time-shift` | 1,5 | El mismo `mu` del muestreo (`get_rf_schedule`) |
| `--save-every` | 100 | Checkpoints del adaptador (30 MB) cada 100 pasos |
| `--merge-every` | 0 | Merge solo al final (cada merge escribe 1,94 GB) |
| `--no-grad-checkpointing` | ❌ no usar | Sin él las activaciones no caben (plan §2) |
| `--augment-flip` | opcional | Volteo horizontal (medido aparte, no incluido en la receta base) |
| `--num-workers` | 0 (defecto) | >0 acelera el preprocesado pero multiplica la RAM de DWPose |

### 6.3 Coste real medido (RTX 3060 12 GB)

| Métrica | Valor medido |
|---|---|
| Tiempo por paso de optimizador (batch 1, acumulación 4, 576×864, bf16, checkpointing) | **15,3 s** |
| Tiempo por micro-paso | ~3,8 s |
| VRAM pico | **3,09 GiB** (de 12 GiB; los pesos en bf16 son 1,94 GB) |
| 100 pasos | ~26 min |
| **1.000 pasos** | **~4,3 h** |
| 2.000 pasos | ~8,5 h (una noche) |
| Adaptador LoRA (r=16) | 30,8 MB |
| Modelo mergeado | 1,94 GB |

El cálculo de VRAM del plan (≤ 8 GiB para LoRA) se queda corto por arriba: con
`gradient_checkpointing` bloque a bloque y bf16 el consumo real es **3,1 GiB**, así
que hay margen de sobra para subir `--rank` a 32 o `--batch-size` a 2.

Estimación de tiempo con el `ETA` del propio log: cada línea de progreso trae
`loss`, `media`, `|grad|`, `|delta|`, `s/paso` y `ETA` en minutos.

### 6.4 Reanudar tras una caída o un apagón

```bash
./run_fashn_vton.sh python scripts/fine_tune.py --pairs datos/nc/dresscode-train-500/pairs.csv \
    --steps 2000 --resume --out weights/candidates/lora-dresscode-v1
```

`--resume` lee `<out>/state.json`, recarga el adaptador
(`<out>/lora/adapter.safetensors`) y el optimizador (`optimizer.pt` si se guardó)
y continúa desde el último paso. `--steps` es el **total acumulado**, no los pasos
nuevos.

### 6.5 Fases recomendadas (coherentes con el plan)

| Fase | Qué hacer | Criterio de salida |
|---|---|---|
| F1 (hecha) | Prueba de humo de 2 pasos | loss finita, VRAM < 10 GiB, candidato cargable |
| F2 | 200-500 pares, 1.000 pasos | `baseline.py verify` → exit 0 y mejora en la métrica objetivo |
| F3 | 10-20 k pasos con `--save-every 1000`, verificando cada bloque | Métricas y revisión humana aprueban |
| F4 | Si F3 no basta: dataset propio con *ground truth* o full fine-tune en GPU alquilada | Salto claro en transferencia de prenda |

Disciplina: **no acumular pasos sin verificar**. Cada bloque termina con el paso 4.

---

## 7. Paso 4 — Verificar (y solo entonces decidir)

```bash
# 0 = no empeora (PROMOVER) · 1 = regresión (NO PROMOVER) · 2 = error
./run_fashn_vton.sh python scripts/baseline.py verify --name baseline-2026-09-15 \
    --weights-dir weights/candidates/lora-dresscode-v1
```

Resultado de la prueba real de hoy con el candidato de humo (2 pasos):

```
  pesos  : d6cd38286885…       (baseline)
  actual : 1ac338de6594…       (DISTINTOS (candidato nuevo))
  par                    seed  sha256    ssim_base   mae   identity  outside  color  sharpness
  ejemplo_tops             42  b09d68b8…    0.9971   0.32    0.9795    17.73   0.61     165.12
  ejemplo_tops           1234  6cf8bfee…    0.9972   0.30    0.9499    19.34   0.64     173.90
Candidato: score=0.7588 identity=0.9647 outside=18.53 color=0.62 ssim_media=0.9971
DECISIÓN: PROMOVER (no empeora)      ← exit 0
Informe: baselines/baseline-2026-09-15/verifications/<fecha>.json
```

Esto demuestra dos cosas: (1) el candidato mergeado **es un checkpoint normal**
que el pipeline carga sin saber que hubo LoRA; (2) la guardia anti-regresión ya
tiene línea base para comparar métricas.

Con el catálogo propio conviene congelar antes un baseline propio
(`scripts/baseline.py capture --name baseline-catalogo --pairs pares.csv`), porque
la señal de los ejemplos del repo es muy pobre.

### 7.1 Promoción: bloqueada a propósito para candidatos NC

```bash
python scripts/model_registry.py promote --candidate weights/candidates/lora-dresscode-v1 --tag lora-dresscode-v1
# ERROR: El candidato ... se entrenó con datos NO COMERCIALES (dataset='dresscode', licencia='nc-dresscode')
# exit 3
```

Para usar el candidato en pruebas, **no** hace falta promoverlo: basta decirle al
pipeline o al CLI que use ese directorio:

```bash
./run_fashn_vton.sh python examples/basic_inference.py --weights-dir weights/candidates/lora-dresscode-v1 \
    --person-image examples/data/model.webp --garment-image examples/data/garment.webp --category tops
```

Y para volver al modelo comercial:

```bash
python scripts/model_registry.py rollback --tag baseline-2026-09-15
python scripts/baseline.py verify --name baseline-2026-09-15     # debe dar exit 0 otra vez
```

---

## 8. Qué se probó hoy (evidencia de punta a punta)

| Prueba | Resultado |
|---|---|
| `prepare_nc_dataset.py dresscode --info` | 48.392 pares (upper 13.563 · lower 7.151 · dresses 27.678) |
| `prepare_nc_dataset.py dresscode --max-pairs 8 --verify 8` | 8 pares, todos los ficheros existen, `provenance.json` con `commercial_use: false` |
| `fine_tune.py --check-only` sin opt-in | **exit 3** con el mensaje de licencia NC |
| `fine_tune.py --check-only` con `FASHN_ALLOW_NC_TRAINING=1` | plan completo impreso, exit 0 |
| Entrenamiento real (2 pares, 2 pasos, bf16, pesos reales) | loss 0.0171 → 0.0227, 15,3 s/paso, VRAM 3,09 GiB |
| Corrida de verificación (8 pares, **12 pasos**) | 173 s · 14,4 s/paso · VRAM 3,09 GiB · `delta_norm` 0,099 → 0,576 (crece de forma monótona) · loss finita/no nula (0,0088 → 0,0126) |
| `--resume` (12 → 16 pasos) | «Reanudando desde el paso 12 (adaptador cargado en 72 módulos)» · `delta_norm` continúa 0,576 → 0,688 |
| `--resume` + `--save-every 2` (16 → 18 pasos) | `lora/adapter-step-000018.safetensors` creado y **idéntico** al final (144 claves = 72 módulos × `lora_a`/`lora_b`) |
| Módulos adaptados (comprobado) | `double_blocks.N.{img_attn,txt_attn}.{qkv,proj}`, `single_blocks.N.{linear1,linear2}`, `x_patch_mixer.N.{linear1,linear2}` |
| `baseline.py verify` del candidato de 18 pasos | **exit 0** · «PROMOVER (no empeora)» · mejora en `sharpness` y `outside_change` · `ssim_media` 0,9943 |
| Candidato generado | `model.safetensors` (1,94 GB) + `lora/adapter.safetensors` (30,8 MB) + `summary.json`, `train_log.jsonl`, `state.json`, `adapter_card.json`, `provenance.json`, `dwpose` |
| `baseline.py verify --weights-dir weights/candidates/smoke` | **exit 0** · «PROMOVER (no empeora)» · ssim_media 0.9971 |
| `model_registry.py promote` (candidato NC) | **exit 3** · bloqueado por procedencia |
| `verify_commercial_readiness.py` | **exit 0** · «LISTO PARA COMERCIAL» (el modelo activo sigue intacto) |
| `pytest tests/ -q` | **201 pruebas, 0 fallos** (141 previas + 60 nuevas) · `ruff check src scripts tests` limpio |

---

## 9. Problemas conocidos y qué hacer

| Síntoma | Causa | Solución |
|---|---|---|
| `La licencia 'nc-dresscode' es NO COMERCIAL` | Falta el opt-in | `export FASHN_ALLOW_NC_TRAINING=1` o `--allow-nc` |
| `GPU en uso por '<consumidor>'` | Otro proceso tiene el turno (UI, IDM-CUSTOM, watchdog) | Ciérralo o espera; `--no-gpu-lock` solo si estás seguro |
| `CUDA out of memory` | UI/otro modelo cargado a la vez | Libera la GPU; sube `--grad-accum`; no desactives el checkpointing |
| `invalid literal for int() with base 10: 'cuda'` | `DWPose`/`Wholebody` espera `cuda:0` | Ya resuelto en `build_pose_fn`; si aparece en código propio, usa `cuda:0` |
| `Expected all tensors to be on the same device` | Adaptador creado en CPU | Resuelto en `LoRALinear` (los parámetros nacen en el dispositivo de la base) |
| Aviso «la máscara agnóstica no marcó ningún píxel» | `agnostic_labels`/`agnostic_kind` mal en el CSV | Revisa el CSV: `labelmap` necesita los ids (`4`, `5,6`, `7`) |
| `No encuentro los pesos de DWPose` | Falta `weights/dwpose` | `python scripts/download_weights.py --weights-dir ./weights` |
| El entrenamiento va lento la primera época | DWPose se ejecuta por imagen | Deja la caché (`--pose-cache`, activa por defecto) |
| Loss `nan` | lr demasiado alto o fp16 inestable | Baja `--lr` a 5e-5 o usa `--dtype bf16` |
| `promote` bloqueado con exit 3 | El candidato es NC | Es lo correcto: para probarlo usa `--weights-dir`, no `promote` |
| `verify` da exit 1 (regresión) | El candidato empeora alguna métrica >2 % | **No** promover: ajustar receta, volver al baseline y repetir |

---

## 10. Mapa de ficheros del arnés

| Fichero | Qué hace |
|---|---|
| `src/fashn_vton/train/lora.py` | LoRA propio (inyección, merge/unmerge, adaptador y checkpoint mergeado) |
| `src/fashn_vton/train/data.py` | CSV de pares + dataset con el preprocesado exacto de la inferencia (poses cacheadas) |
| `src/fashn_vton/train/trainer.py` | Bucle *rectified flow*, `gradient_checkpointing`, AdamW, resume, logging y exportación del candidato |
| `src/fashn_vton/train/nc_policy.py` | Puertas NC: entrada (entrenar) y salida (promover) |
| `scripts/prepare_nc_dataset.py` | Convertidor DressCode / prep IDM-VTON → `pairs.csv` + `provenance.json` + `NOTICE.md` |
| `scripts/fine_tune.py` | CLI de entrenamiento (licencias, turno de GPU, plan, corrida) |
| `scripts/model_registry.py` | Registro de pesos; `promote` ahora consulta la procedencia del candidato |
| `tests/test_train_lora.py` | Invariantes del LoRA (identidad al inicio, merge exacto, checkpoint cargable) |
| `tests/test_train_data.py` | Dataset, CSV, enmascarado agnóstico y caché de poses |
| `tests/test_train_step.py` | Entrenamiento real diminuto de punta a punta (CPU, sin pesos) |
| `tests/test_nc_policy.py` | Política NC (opt-in, procedencia, bloqueo de promoción) |
| `plan_entrenamiento/PLAN_FINE_TUNING.md` | Plan completo (fases, riesgos, protocolo de promoción) |
| `plan_entrenamiento/GUIA_EJECUCION_ENTRENAMIENTO.md` | Este documento |

### Salidas de una corrida (`weights/candidates/<tag>/`)

```
<tag>/
├── model.safetensors        checkpoint mergeado (base + LoRA): lo carga el pipeline tal cual
├── lora/adapter.safetensors adaptador puro (30 MB) → ligero para versionar/experimentar
├── lora/adapter-step-NNNNNN.safetensors   copias si se usó --save-every
├── merged/model-step-NNNNNN.safetensors   merges intermedios si se usó --merge-every
├── dwpose -> ../../dwpose   enlace para usar el candidato como --weights-dir
├── train_config.json        configuración + resumen del LoRA inyectado
├── train_log.jsonl          una línea JSON por log (loss, grad, delta, s/paso, ETA, VRAM)
├── state.json               último paso (para --resume)
├── optimizer.pt             estado de AdamW (si no se usó --no-save-optimizer)
├── summary.json             resumen final (pasos, tiempos, VRAM, delta_norm, rutas)
├── adapter_card.json        ficha del adaptador (módulos, params, hash de los pesos base)
└── provenance.json          dataset, licencia y `commercial_use` (lo lee `model_registry promote`)
```

### Glosario mínimo

- **LoRA**: adaptadores de bajo rango que se suman a las proyecciones del MMDiT;
  la base queda congelada y el arranque es idéntico al modelo actual.
- **Rectified flow**: el modelo predice la velocidad `v = limpio − ruido` en
  `x_t = (1−t)·ruido + t·limpio`; la pérdida es un MSE contra ese objetivo.
- **Imagen agnóstica (`ca_images`)**: foto de la persona con la prenda borrada
  (gris 127). Es la entrada que obliga al modelo a usar la prenda nueva.
- **Merge**: incorporar el delta LoRA a los pesos base para producir un
  `model.safetensors` normal (lo único que consume el pipeline).
- **Procedencia**: `provenance.json` del dataset y del candidato; es lo que separa
  «uso comercial» de «investigación» en este fork.

---

## 11. Recordatorio final

- Este arnés **no cambia el pipeline de inferencia**: si no se promueve nada, el
  producto comercial sigue siendo exactamente el mismo (mismo `sha256` de pesos y
  misma salida byte a byte).
- Los datos de DressCode/VITON-HD son de **investigación**: sirven para medir si la
  receta mejora, no para producir el modelo comercial.
- Antes de cualquier decisión de producto: `verify` + revisión humana de las
  imágenes + revisión legal de la procedencia de los datos usados.






