#!/usr/bin/env bash
# Lanzador de la interfaz web de pruebas de FASHN VTON v1.5 (webui/app.py).
#
# Hace dos cosas que el README del repo no cubre:
#   1) activa el entorno conda dedicado `fashn-vton`;
#   2) exporta LD_LIBRARY_PATH con las librerías CUDA que PyTorch instala como
#      paquetes `nvidia-*` en site-packages. Sin esto, onnxruntime-gpu no
#      encuentra `libcublasLt.so.13` y la detección de pose (DWPose) cae a CPU
#      (ver LEVANTAMIENTO_2026-09-15.md).
#
# Uso:
#   ./run_web.sh                          # http://127.0.0.1:7863 (segundo plano, logs en logs/)
#   GRADIO_SERVER_PORT=7864 ./run_web.sh  # otro puerto
#   FASHN_USE_GPU_LOCK=1 ./run_web.sh     # pide el turno único de GPU de IDM-CUSTOM
#   FASHN_PRELOAD=1 ./run_web.sh          # carga los modelos al arrancar (~20 s)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${CONDA_SH:-/home/uceda/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-fashn-vton}"
GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-127.0.0.1}"
# 7860 = demo de IDM-VTON, 7861 = servicio de IDM-CUSTOM, 7863 = esta UI.
GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7863}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_DIR}/webui_$(date '+%Y%m%d_%H%M%S').log"

if [ ! -f "${CONDA_SH}" ]; then
  echo "ERROR: no existe CONDA_SH=${CONDA_SH} (exporta CONDA_SH=/ruta/a/conda.sh)." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

if ss -ltn 2>/dev/null | grep -q ":${GRADIO_SERVER_PORT} "; then
  echo "ERROR: el puerto ${GRADIO_SERVER_PORT} ya está en uso (¿otra UI levantada?)." >&2
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

echo "Lanzando interfaz web en http://${GRADIO_SERVER_NAME}:${GRADIO_SERVER_PORT} (logs: ${LOG_FILE})"
# PYTHONUNBUFFERED: que el log no se pierda si el proceso muere (OOM-kill).
PYTHONUNBUFFERED=1 GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME}" GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT}" \
  python webui/app.py > "${LOG_FILE}" 2>&1 &
disown
echo "PID: $!"
