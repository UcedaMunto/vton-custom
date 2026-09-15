#!/usr/bin/env bash
# Lanzador del servicio HTTP comercial (FastAPI) de FASHN VTON v1.5.
#
# Igual que run_web.sh: activa el entorno conda del fork y exporta
# LD_LIBRARY_PATH con las librerías CUDA de los paquetes nvidia-* (sin eso,
# onnxruntime-gpu no encuentra libcublasLt.so.13 y DWPose cae a CPU).
#
# Uso:
#   ./run_api.sh                                   # http://127.0.0.1:8000
#   VTON_API_KEYS=clave1,clave2 ./run_api.sh       # exigir API key
#   VTON_USE_GPU_LOCK=1 ./run_api.sh               # turno único de GPU compartido
#   API_PORT=9000 ./run_api.sh                     # otro puerto
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${CONDA_SH:-/home/uceda/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-fashn-vton}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
WORKERS="${VTON_WORKERS:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_DIR}/api_$(date '+%Y%m%d_%H%M%S').log"

if [ ! -f "${CONDA_SH}" ]; then
  echo "ERROR: no existe CONDA_SH=${CONDA_SH} (exporta CONDA_SH=/ruta/a/conda.sh)." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

if ss -ltn 2>/dev/null | grep -q ":${API_PORT} "; then
  echo "ERROR: el puerto ${API_PORT} ya está en uso." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
NVIDIA_LIBS="$(find "${SITE_PACKAGES}/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)"
if [ -n "${NVIDIA_LIBS}" ]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${PROJECT_ROOT}"

echo "Servicio en http://${API_HOST}:${API_PORT} (workers=${WORKERS}, logs: ${LOG_FILE})"
# NOTA: con más de 1 worker cada proceso cargaría su propia copia del modelo en
# la GPU (~2 GB); para esta tarjeta de 12 GB lo recomendado es 1 worker.
PYTHONUNBUFFERED=1 exec uvicorn "fashn_vton.api.main:app" \
  --host "${API_HOST}" --port "${API_PORT}" --workers "${WORKERS}" \
  2>&1 | tee "${LOG_FILE}"
