# Plan de fine-tuning — FASHN VTON v1.5 (fork comercial)

**Objetivo:** mejorar el modelo con datos propios **sin arriesgar el estado que ya
funciona**. Cada mejora se mide contra un baseline congelado y solo se promueve si
no empeora; si empeora, se vuelve atrás con un comando.

**Estado de partida (2026-09-15, ya hecho):**

| Pieza | Estado | Evidencia |
|---|---|---|
| Baseline congelado (pesos + salidas + métricas + entorno) | ✅ | `baselines/baseline-2026-09-15/manifest.json` (semillas 42 y 1234; pesos `d6cd3828…`; torch 2.14.0+cu130; score 0.7588) |
| Guardia anti-regresión (capturar / verificar) | ✅ | `scripts/baseline.py` + `src/fashn_vton/eval/regression.py` (11 pruebas) |
| Prueba de que **bloquea** un modelo peor | ✅ | candidato perturbado: `identity 0.0064` vs `0.9648`, `outside_change 115.13` vs `18.44`, `ssim 0.0039` → `NO PROMOVER (regresión)`, exit 1 |
| Código respaldado | ✅ | commit `ed86694` + tag `upstream-base-7c0f10a` en `UcedaMunto/vton-custom` |
| Tag de release del baseline | ⏳ | `v1.5.0-commercial.1-baseline` (paso 1 del §8) |

---

## 1. El modelo en números (qué se entrena exactamente)

| Dato | Valor | Fuente |
|---|---|---|
| Parámetros | **972 M** (bf16 ≈ 1,94 GB) | `tryon_mmdit.py`, checkpoint `model.safetensors` |
| Arquitectura | MMDiT estilo FLUX (adaptación Apache-2.0): 4 *patch mixer* + 8 *double stream* + 16 *single stream* | `tryon_mmdit.py`, modelo card |
| Hidden / cabezas / patch | 1280 / 10 / 12×12 píxeles | modelo card |
| Salida | **576×864**, en espacio de píxeles (**sin VAE**) | `input_shape=(864, 576)` |
| Tokens por imagen | (864/12)×(576/12) = **3456**; la secuencia de los bloques *double* es 3456 (imagen) + 3456 (prenda) = **6912** | cálculo sobre `forward()` |
| Objetivo | **rectified flow**: `x_t = (1−t)·x0 + t·x1`, se predice `v = x1 − x0`; muestreo Euler con `get_rf_schedule(mu=1.5)` | `pipeline._sample`, `utils/sampling.py` |
| Pérdida a implementar | `MSE(model(x_t, t, ca, prenda, poses, categoría), x1 − x0)` | derivado del muestreo |
| Condicionamiento | `ca_images` (persona agnóstica) + `person_poses` (render DWPose) + `garment_images` + `garment_poses` + `garment_categories` | `forward()` |
| CFG en entrenamiento | ya integrado: `mask` aplica *conditional dropout* en `forward()` | `apply_conditional_dropout` |

Consecuencia práctica: **no hay VAE ni encoder de texto** que congelar; el
entrenamiento es un bucle directo sobre píxeles (más simple) pero caro en VRAM
porque cada token es un parche de 12×12 píxeles.

## 2. Viabilidad en esta máquina (RTX 3060, 12 GB)

Cálculo de memoria (batch 1, bf16, grad checkpointing):

| Estrategia | Pesos | Grads | Optimizador | Activaciones | Total | ¿Cabe en 12 GB? |
|---|---|---|---|---|---|---|
| Full fine-tune + AdamW fp32 | 1,9 | 1,9 | 7,8 | ~2-3 | **~14-15 GB** | ❌ |
| Full + Adam 8-bit + checkpointing | 1,9 | 1,9 | 1,9 | ~2-3 | ~8-9 GB | ⚠️ al límite (sin margen; la UI usa 6,9 GB → OOM) |
| **LoRA / DoRA (r=16-32)** | 1,9 (congelados) | — | 0,1-0,4 | ~2-3 | **~5-8 GB** | ✅ **sí, con margen** |

Velocidad estimada (≈40 TFLOPs por muestra: `6 × 972M × 6912` tokens; ≈25 TFLOPs
efectivos en 3060):

| Régimen | Coste por paso | 1.000 pasos | 20.000 pasos |
|---|---|---|---|
| LoRA a 576×864, batch 1 | ~1,5-2,5 s | 25-40 min | **8-14 h** (una noche) |
| Piloto a 576×432 (1728 tokens, fuera de distribución) | ~2× más rápido | 15 min | 4-7 h |

**Aviso medido hoy:** con la UI ocupando 6,9 GB hubo un fallo de asignación
(`memory allocation failed with OOM … trying to allocate 125829120 bytes`) durante
la captura del baseline; se recuperó, pero **para entrenar hay que parar la UI o
turnar la GPU** (`FASHN_USE_GPU_LOCK=1` / `VTON_USE_GPU_LOCK=1`).

Si el LoRA no bastara, el full fine-tune se hace en GPU alquilada (A100 80 GB o
H100, orientativo 1,5-3 USD/h): 50-100 k pasos ≈ 1-3 días de cómputo.

---

## 3. Datos: el cuello de botella real

Regla dura (la vigila el gate): **prohibido VITON-HD y DressCode**, y cualquier
dataset sin licencia comercial explícita (`licenses/manifest.json`,
`forbidden_artifacts`).

### Opción A — Triplets sintéticos filtrados (recomendada para empezar)

Es la fase 2 que usó el propio FASHN («4M synthetic triplets generados desde el
checkpoint de la fase 1»), y aquí es replicable porque ya tenemos las métricas:

```
pares propios (persona + prenda flat-lay del catálogo)
        │
        ├─ generar N candidatos (varias semillas / pasos) con el modelo ACTUAL
        │
        ├─ puntuar con fashn_vton.eval: color_fidelity, identity_preservation,
        │  outside_change, pattern_fidelity, sharpness
        │
        ├─ quedarse con el top-k (rejection sampling)  ← "el modelo como profesor"
        │
        └─ fine-tuning LoRA sobre esos pares  →  verify contra el baseline
```

| Ventaja | Coste | Límite |
|---|---|---|
| Cero licencias nuevas; usa tu catálogo real | GPU: ~13 s por candidato a 8 pasos (1.000 pares × 2 candidatos ≈ 7 h en la 3060) | Es *self-training*: adapta el modelo a **tu dominio** (colores, fondos, tipos de prenda) pero no inventa capacidades nuevas |

Caso de uso documentado y concreto: las prendas **oscuras/azuladas** salen más
apagadas (`LEVANTAMIENTO_2026-09-15.md`) y `flat-lay` es el modo más sensible.
Con rejection sampling por `color_fidelity` + LoRA se puede corregir ese sesgo sin
tocar el resto del comportamiento (y el `verify` dirá si de verdad mejora).

### Opción B — Dataset propio con *ground truth* (la que más mejora)

Trípletas reales: `persona (foto base)` + `prenda (flat-lay)` + `target (la misma
persona con esa prenda puesta)`. Es el estándar del try-on y es lo único que
enseña transferencia real de prenda. Requiere producir los datos (sesión de
fotografía: 1 persona × N prendas, o licencia explícita de un tercero).

Formato propuesto (CSV + carpetas, con procedencia obligatoria):

```csv
name,person,garment,target,category,garment_photo_type,license,source,split
look001,img/p001_front.jpg,img/g001_flat.jpg,img/p001_g001.jpg,tops,flat-lay,propia-propia,estudio-2026-10,train
look002,img/p001_front.jpg,img/g002_flat.jpg,img/p001_g002.jpg,bottoms,flat-lay,propia-propia,estudio-2026-10,val
```

Reglas: `val` congelado (los mismos pares que el baseline), sin duplicados
(perceptual hash), y ninguna imagen de cliente sin consentimiento.

### Opción C — Reconstrucción con máscara (auxiliar, gratis)

Con fotos propias de personas ya vestidas: enmascarar la prenda y pedir al modelo
que la reconstruya (in-painting auto-supervisado). Barato y sin licencias, pero
enseña a copiar la prenda original más que a transferir una nueva: úsalo solo como
pre-entrenamiento o regularizador.

---

## 4. Estrategia de entrenamiento: LoRA hoy, full fine-tune después

### Qué se entrena (LoRA/DoRA)

| Módulo | Capa | Por qué |
|---|---|---|
| `SelfAttention` (bloques double y single) | `qkv`, `proj` | Es donde vive la atención imagen↔prenda: controla la transferencia |
| `DoubleStreamBlock` | `img_mlp`, `txt_mlp` | Mezcla de información multimodal |
| `SingleStreamBlock` | `linear1`, `linear2` | Refinado global |
| `Modulation` / `LastLayer` | `lin`, `adaLN_modulation`, `linear` | Modulación por tiempo/categoría y proyección final a píxeles |

Rank 16-32, `alpha = rank`, dropout 0,05. Todo lo demás congelado; adaptadores en
fp32 (maestro) con autocast bf16.

### Exportación: **merge**, no adaptadores en inferencia

Al terminar, se fusionan los pesos LoRA en el `state_dict` y se guarda un
`model.safetensors` normal. Ventajas: cero cambios en el pipeline/API/UI, cero
coste de inferencia y el mismo contrato de cumplimiento (hash del checkpoint al
manifiesto).

### Hiperparámetros de arranque

| Parámetro | Valor inicial | Nota |
|---|---|---|
| Optimizador | AdamW (o 8-bit si hace falta) | `betas=(0.9, 0.99)`, `weight_decay=0.01` |
| LR | 1e-4 (LoRA) con coseno + warmup 100 | Si diverge: 3e-5 |
| Batch | 1 × acumulación 8 | La VRAM manda |
| `t` | logit-normal desplazada (shift ~1.5, como el muestreo) | Uniforme también vale para el piloto |
| CFG dropout | 10 % (`mask` del `forward`) | Ya implementado en el modelo |
| Precisión | bf16 + `torch.cuda.amp` | 3060 soporta bf16 |
| Checkpoints | cada 500 pasos + `verify` cada 1.000 | Nunca sobrescribir el anterior |
| EMA | opcional | Útil si el LR es alto |

### Protocolo de parada

1. Si en 1.000 pasos la loss no baja de forma sostenida → revisar datos/LR, no
   seguir.
2. Si `verify` empeora una sola vez de forma clara → **no promover** y volver al
   baseline (el checkpoint intermedio se descarta).
3. Tope de pasos por fase (2.000 → 10.000 → 20.000): el coste crece lineal y la
   ganancia no.

---

## 5. Arnés a construir (mapa de ficheros)

| Fichero | Estado | Qué hace |
|---|---|---|
| `src/fashn_vton/eval/regression.py` | ✅ hecho | Comparación de métricas contra el baseline (guardia) |
| `tests/test_regression_gate.py` | ✅ hecho (11 pruebas) | Prueba la guardia: mejora/regresión/tolerancias/faltantes |
| `scripts/baseline.py` | ✅ hecho y probado | `capture` (congelar) y `verify` (decidir) |
| `baselines/<name>/manifest.json` | ✅ hecho | Huella: pesos, entorno, semillas, sha256 y métricas por imagen |
| `scripts/model_registry.py` | ⏳ | `snapshot` / `list` / `promote` / `rollback` de pesos con hash |
| `src/fashn_vton/train/lora.py` | ⏳ | Inyectar/quitar/mergear adaptadores LoRA en el MMDiT |
| `src/fashn_vton/train/data.py` | ⏳ | Dataset de trípletas + CSV de procedencia + preprocesado reutilizado del pipeline |
| `src/fashn_vton/train/trainer.py` | ⏳ | Bucle rectified-flow, grad checkpointing, AdamW (8-bit opcional), resume, EMA |
| `src/fashn_vton/train/callbacks.py` | ⏳ | `verify` automático cada N pasos y parada temprana |
| `scripts/make_synthetic_triplets.py` | ⏳ | Generar candidatos + puntuar + filtrado top-k (Opción A) |
| `scripts/fine_tune.py` | ⏳ | CLI: `--data`, `--rank`, `--steps`, `--lr`, `--out weights/candidates/<tag>` |
| `tests/test_lora_injection.py` | ⏳ | LoRA con B=0 no cambia la salida; merge correcto; solo los adaptadores tienen gradiente |
| `tests/test_training_step.py` | ⏳ | Un paso de entrenamiento real (loss finita, VRAM medida, checkpoint escrito) |

Regla de diseño: **el pipeline de inferencia no se toca**. Todo lo de entrenamiento
vive en `src/fashn_vton/train/` y en `scripts/`, y la salida es un checkpoint normal.

---

## 6. Protocolo de evaluación, promoción y **vuelta atrás**

### 6.1 Reglas de promoción (nada se promueve sin pasar esto)

| # | Condición | Cómo se comprueba |
|---|---|---|
| 1 | El candidato **no empeora** ninguna métrica más del 2 % | `baseline.py verify` → exit 0 |
| 2 | Mejora clara en la métrica objetivo (p. ej. `color_fidelity` +5 %) | salida del `verify` (`mejora en: …`) |
| 3 | Revisión humana de la hoja de contactos | `baselines/<name>/candidate/*.png` vs `images/*.png` |
| 4 | Cumplimiento y tests | `verify_commercial_readiness.py` (exit 0) + `pytest -q` |
| 5 | Hash del nuevo checkpoint registrado | `licenses/manifest.json` / `model_registry.py` |

### 6.2 Comandos (procedimiento completo)

```bash
# 0) Congelar el estado bueno (ya hecho hoy)
python scripts/baseline.py capture --name baseline-2026-09-15
python scripts/model_registry.py snapshot --tag baseline-2026-09-15

# 1) Entrenar → deja el candidato en weights/candidates/<tag>/model.safetensors
python scripts/fine_tune.py --data datos/looks.csv --rank 16 --steps 2000 \
    --out weights/candidates/lora-v1

# 2) ¿Empeora? (0 = no empeora · 1 = regresión · 2 = error)
python scripts/baseline.py verify --name baseline-2026-09-15 \
    --weights-dir weights/candidates/lora-v1

# 3) Promover SOLO si el paso 2 dio 0
python scripts/model_registry.py promote --candidate weights/candidates/lora-v1 --tag lora-v1

# 4) Volver atrás en cualquier momento (un comando, con verificación)
python scripts/model_registry.py rollback --tag baseline-2026-09-15
python scripts/baseline.py verify --name baseline-2026-09-15      # debe dar 0 otra vez
```

### 6.3 Qué NO cambia y qué sí

| | Camino comercial `flat-lay` + `none` |
|---|---|
| Con los pesos originales | Salida **byte a byte** idéntica (`sha256 a4d618bf…`); el invariante vale |
| Con pesos fine-tuneados | El hash **cambia por diseño**; a partir de ahí manda el protocolo de métricas del §6.1 (el `verify` reporta `pesos DISTINTOS` y compara métricas) |
| Cumplimiento de licencias | Igual: los pesos nuevos son propios, así que el gate sigue verde (y hay que registrar su hash) |

Los pesos originales siguen disponibles en HuggingFace (`fashn-ai/fashn-vton-1.5`,
Apache-2.0, oid `d6cd3828…`), así que la vuelta atrás es posible incluso si se
pierde el disco: `python scripts/download_weights.py --weights-dir ./weights`.

---

## 7. Riesgos y límites (con su mitigación)

| Riesgo | Mitigación |
|---|---|
| El fine-tuning **empeora** el modelo | Baseline congelado + `verify` bloqueante (probado hoy con un candidato degradado: exit 1) |
| *Self-training* refuerza artefactos y reduce diversidad | Filtrado estricto por métricas, tope de pasos, val fijo con imágenes que **no** se usaron para generar |
| Sobreajuste al catálogo propio | Mezclar datos sintéticos con reales y con variedad de personas/fondos |
| OOM en 12 GB (visto hoy con la UI abierta) | Parar la UI o `FASHN_USE_GPU_LOCK=1`; batch 1 + acumulación; checkpointing |
| Datos sin licencia | El gate bloquea VITON-HD/DressCode; CSV de procedencia obligatorio por imagen |
| El hash `a4d618bf…` deja de valer | Solo vale para los pesos originales; para candidatos manda el §6.1 (documentado aquí) |
| Se pierde el disco / el repo | Pesos en HF (oid verificado) + `baselines/*/manifest.json` commiteado + tag de git |
| Coste de GPU descontrolado | Fases con tope de pasos y criterio de salida (abajo) |

---

## 8. Fases, criterio de salida y coste

| Fase | Trabajo | Criterio de salida | Coste estimado |
|---|---|---|---|
| **F0** (hecho hoy) | Baseline + guardia + tag/push + prueba de que bloquea | `verify` con los mismos pesos → exit 0; con pesos degradados → exit 1 | ~1 h |
| **F1** (medio día) | Arnés LoRA + `fine_tune.py` + un paso real de entrenamiento sobre 1 par (overfit) | La loss baja; `verify` del candidato overfiteado **no** empeora; VRAM medida < 10 GB | 0,5 día de trabajo |
| **F2** (1 día) | Tríplets sintéticos v1 con **tu catálogo** (200-500 pares) + LoRA 2.000 pasos | Mejora ≥5 % en la métrica objetivo sobre el val, sin regresión | ~7 h de GPU (generación) + 1-1,5 h (entrenamiento) |
| **F3** (1 semana) | LoRA 10-20 k pasos, `verify` cada 1.000, promoción si gana | Métricas y revisión humana aprueban; se promueve y se registra | 8-14 h de GPU |
| **F4** (si F3 no basta) | Dataset propio con *ground truth* (Opción B) o full fine-tune en A100/H100 | Salto claro en transferencia de prenda | Producción de datos + 30-100 USD de alquiler |

Coste de datos orientativo: 1.000 pares de entrenamiento a 576×864 en PNG ≈ 0,5 GB;
los candidatos sintéticos conviene guardarlos en JPEG/WebP (×5 menos) o en `.npy`
uint8.

---

## 9. Qué se puede hacer ya (sin escribir el arnés)

```bash
# 1) Volver a comprobar que el estado actual sigue siendo el bueno
./run_fashn_vton.sh python scripts/baseline.py verify --name baseline-2026-09-15

# 2) Congelar un baseline con TU catálogo (mejor señal que los ejemplos del repo)
./run_fashn_vton.sh python scripts/baseline.py capture --name baseline-catalogo \
    --pairs pares.csv --seeds 42,7,1234 --steps 30

# 3) Medir el sesgo que se quiere corregir (prendas oscuras/azuladas)
./run_fashn_vton.sh python scripts/benchmark_providers.py --pairs pares.csv \
    --providers none --metrics --num-timesteps 30
```

## 10. Referencias internas

- `CAMBIO_MODULO_COMERCIAL.md` — qué se cambió para poder usar esto comercialmente.
- `plan_modelo_comercial/PLAN_IMPLEMENTACION_COMERCIAL.md` — plan y evidencia del fork.
- `src/fashn_vton/eval/quality_gates.py` — métricas (sin redes preentrenadas).
- `src/fashn_vton/eval/regression.py` — guardia anti-regresión.
- `scripts/baseline.py` — capturar/verificar el estado bueno.
- `baselines/baseline-2026-09-15/` — baseline actual (manifest + imágenes + informes).
