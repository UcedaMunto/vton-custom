# FASHN VTON v1.5 — fork comercial

Fork de [`fashn-AI/fashn-vton-1.5`](https://github.com/fashn-AI/fashn-vton-1.5)
(Apache-2.0) preparado para **uso comercial**: mismo modelo y misma calidad, sin
la dependencia `fashn-human-parser` cuyos pesos heredan la licencia **no
comercial** de NVIDIA SegFormer.

- Cambio central: la segmentación es un **proveedor enchufable**
  (`src/fashn_vton/segmentation/`), con `none` como opción por defecto.
- Camino comercial soportado: **`segmentation_free=True` + prenda `flat-lay`**.
- Documentos: [`CAMBIO_MODULO_COMERCIAL.md`](CAMBIO_MODULO_COMERCIAL.md) (especificación del cambio),
  [`plan_modelo_comercial/PLAN_IMPLEMENTACION_COMERCIAL.md`](plan_modelo_comercial/PLAN_IMPLEMENTACION_COMERCIAL.md),
  [`plan_modelo_comercial/propuestas_fashn_vton_comercial.md`](plan_modelo_comercial/propuestas_fashn_vton_comercial.md),
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), [`licenses/`](licenses/).

## Instalación

```bash
cd /home/uceda/Documents/fashn-vton-1.5
conda create -n fashn-vton python=3.11 -y && conda activate fashn-vton
pip install -e ".[api,webui,metrics,dev]"    # api = FastAPI/uvicorn; webui = Gradio; metrics = scikit-image/scipy
python scripts/download_weights.py --weights-dir ./weights   # ~2,2 GB
python licenses/verify_hashes.py                              # 3/3 OK si están los pesos

# Proveedores opcionales de segmentación (etapas 2 y 2b, todos Apache-2.0)
pip install -e ".[providers-sam2]"           # sam2==1.1.0
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam2.1-hiera-tiny', 'sam2.1_hiera_tiny.pt', local_dir='weights/sam2')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('IDEA-Research/grounding-dino-tiny', local_dir='weights/grounding_dino', allow_patterns=['*.json','*.txt','*.safetensors'])"
python licenses/verify_hashes.py                              # 5/5 OK con los dos checkpoints
```

## Uso

```bash
# 1) CLI (proveedor none = camino comercial)
./run_fashn_vton.sh python examples/basic_inference.py \
  --weights-dir ./weights \
  --person-image examples/data/model.webp \
  --garment-image examples/data/garment.webp \
  --category tops --garment-photo-type flat-lay

# 2) API HTTP
./run_api.sh                                  # http://127.0.0.1:8000  (docs en /docs)
curl -s localhost:8000/healthz | python -m json.tool

# 3) Interfaz web de pruebas
./run_web.sh                                  # http://127.0.0.1:7863
```

API Python:

```python
from fashn_vton import TryOnPipeline
from fashn_vton.segmentation import build_segmentation_provider

pipeline = TryOnPipeline(
    weights_dir="./weights",
    segmentation_provider=build_segmentation_provider("none"),   # sin modelos extra
)
result = pipeline(person_image=person, garment_image=garment,
                  category="tops", garment_photo_type="flat-lay")
print(result.metadata["segmentation"])   # proveedor, máscaras y degradaciones
```

## Cumplimiento (gate antes de publicar)

```bash
python scripts/verify_commercial_readiness.py     # 0 = listo, 2 = artefacto prohibido presente
python licenses/verify_hashes.py                  # hashes de pesos + scan de prohibidos
python -m pytest tests/ -q                        # 107 pruebas
```

Qué comprueba el gate: que no haya imports ni dependencias del paquete NC, que
no haya distribuciones prohibidas instaladas, que no aparezcan artefactos
prohibidos en el árbol, que los hashes de los pesos registrados coincidan y que
todos los proveedores registrados tengan licencia comercial.

## Pruebas y métricas (medidas el 2026-09-15, RTX 3060)

| Escenario | Resultado |
|---|---|
| Suite de pruebas | **107 passed** (compliance, proveedores, SAM 2, Grounded-SAM 2, métricas, parser-free, API, gpu_lock, etiquetas; incluye las de integración con pesos reales) |
| Generación comercial (flat-lay, 8 pasos, semilla 42) | **byte-idéntica** al pipeline original (mismo sha256) |
| Generación de 30 pasos | ~46 s (1,55 s/paso), VRAM 2,07 GiB asignada / 5,08 GiB reservada |
| Benchmark de 4 proveedores (8 pasos) | `none` 13,1 s · `pose-heuristic` 12,5 s · `sam2` 12,6 s · `grounded-sam2` 12,7 s; VRAM pico 5,16 GiB; RAM 2,3 → 4,6 GiB |
| SAM 2 (etapa 2) | carga + segmentación de 5 regiones en **1,0 s**; camino `model` ya no degrada |
| Grounded-SAM 2 (etapa 2b) | detecta la prenda en **2,8 s** (3 cajas, score 0,34-0,40) y segmenta en 1,3 s (28% del área) |
| Paquete NC | desinstalado del entorno y purgado de la caché de HuggingFace |

## Métricas de calidad (plan §14)

`src/fashn_vton/eval/quality_gates.py` mide, **sin redes preentrenadas** (solo
scikit-image/scipy/opencv, todo permisivo):

- `identity_preservation` (SSIM en cara/pelo/manos/pies), `outside_change` (cuánto
  cambia fuera de la prenda), `color_fidelity` (Wasserstein del color de la prenda),
  `pattern_fidelity` (SSIM del estampado), `garment_change`, `sharpness`.

```bash
# Comparar proveedores con métricas y veredicto del gate
./run_fashn_vton.sh python scripts/benchmark_providers.py --examples \
    --providers none,pose-heuristic,sam2,grounded-sam2 --metrics

# Con tu catálogo: CSV de pares (person_path,garment_path,category[,garment_photo_type])
./run_fashn_vton.sh python scripts/benchmark_providers.py --pairs pares.csv \
    --providers none,sam2 --metrics --num-timesteps 30
```

Salidas: `outputs/benchmark/benchmark.csv`, `benchmark_summary.json` (incluye
`quality` con medias, `failure_rate`, `score` y `passes_gate`) y los PNG por proveedor.

## CI y contenedor

- `.github/workflows/ci.yml`: gate de cumplimiento + tests (sin GPU) + ruff en cada push/PR.
- `Dockerfile` (+ `.dockerignore`): imagen del servicio que **ejecuta el gate dentro
  del build** (si aparece un artefacto prohibido, la imagen no se construye) y monta
  `./weights` en tiempo de ejecución. *No verificada localmente: esta máquina no
  tiene Docker instalado.*

## Despliegue en el clúster de 3 nodos (2026-09-19/20)

El servicio comercial (frontend + API + workers) corre en un clúster k3s de **3 nodos**
sobre una LAN privada. Este repositorio aporta **la imagen del worker GPU**, el modelo y
el código de los componentes (gateway, workers, frontend); la orquestación, las
direcciones y las credenciales viven **fuera del repositorio**:

- Planos y manifiestos: [`../kubernetes/`](../kubernetes/) (no versionado aquí).
- **Datos de infraestructura** (IPs fijas, hostnames, rango de MetalLB, registry y acceso
  SSH): `~/.vton-cluster.env` en el nodo de operación (modo 600, **nunca** en git).
  Uso: `set -a; . ~/.vton-cluster.env; set +a`.
- Credenciales del clúster: `~/.vton-secrets.env` y los `Secret` de Kubernetes.

| Nodo | Hardware | Rol en el despliegue |
|---|---|---|
| `nodo-orq` | 12 vCPU / 30 GiB, sin GPU | control-plane k3s, ingress + TLS, frontend `vton-web`, API Gateway, Redis, MinIO transitorio, registry privado, workers CPU |
| `nodo-gpu-1` | 6 vCPU / 32 GiB, **RTX 3060 12 GB** | worker GPU (`DaemonSet`) con `TryOnPipeline`; es también la máquina de entrenamiento y de la UI local |
| `nodo-cpu-1` | 8 vCPU / 15 GiB, sin GPU | preprocesado CPU (validación, resize, re-encode, pose) |

Notas de operación:

- **Estado (2026-09-20): desplegado y sirviendo.** Verificado de punta a punta: un job
  enviado al gateway público se completa en ~28 s (8 pasos) y devuelve un PNG 576×864;
  el objeto se borra al entregarlo (retención cero: la segunda descarga responde 404).
- **La GPU es compartida**: el worker del clúster, la UI local (`run_web.sh`), el
  entrenamiento (`plan_entrenamiento/`) y `IDM-CUSTOM` usan la misma RTX 3060 del nodo GPU
  mediante el turno único `~/.idm_gpu.lock` (variables `VTON_USE_GPU_LOCK` /
  `VTON_GPU_LOCK_PATH`).
- **Los pesos no se hornean en la imagen**: el worker los monta en solo lectura desde el
  nodo GPU (únicamente `model.safetensors` + `dwpose/`, nunca `candidates/`). Tras promover
  un candidato: `kubectl -n vton rollout restart daemonset/vton-gpu-worker`.
  *Pendiente al añadir nodos GPU*: distribuir los pesos (initContainer que descargue y
  verifique el `sha256` del manifiesto, o un StorageClass compartido).
- **Cero retención**: las imágenes de los clientes no se conservan en el clúster (plan §11
  de [`../kubernetes/00_PLAN_ARQUITECTURA.md`](../kubernetes/00_PLAN_ARQUITECTURA.md)); este
  repositorio no almacena material de terceros.
- **Acceso SSH**: por clave (`~/.ssh/id_ed25519`) hacia los nodos; usuarios, hosts e IPs
  están en `~/.vton-cluster.env`. **Pendiente de seguridad**: desactivar
  `PasswordAuthentication` (hoy la contraseña es compartida) y aplicar la NetworkPolicy (F8).

## Hoja de ruta (etapas 2 y 3)

1. ~~**SAM 2** (`sam2`)~~ ✅ implementado y verificado (etapa 2).
2. ~~**Grounding DINO + SAM 2** (`grounded-sam2`)~~ ✅ implementado y verificado (etapa 2b).
3. **Parser propio** (`custom-parser`, Detectron2) entrenado con datos propios (etapa 3).
4. **Benchmark completo** del plan §14 (100 tops + 100 bottoms + 100 one-pieces) con
   imágenes propias: el arnés y las métricas están listos, faltan los datos.
5. **Calibrar los umbrales** de `QualityGateConfig` con ese benchmark.
6. **Construir y publicar la imagen Docker** en un host con Docker y revisión legal final.

## Aviso legal

Este repositorio documenta su cumplimiento de licencias (avisos, inventario y
verificador), pero **no constituye asesoría legal**. Antes del lanzamiento
comercial hace falta una revisión profesional y la verificación de los
checkpoints y datasets concretos que se usen.
