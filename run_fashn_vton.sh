#!/usr/bin/env bash
# Lanzador de FASHN VTON v1.5 (entorno conda dedicado `fashn-vton`).
#
# Por qué existe este wrapper (no forma parte del repo original): `onnxruntime-gpu`
# no encuentra las librerías CUDA que PyTorch instala como paquetes `nvidia-*`
# dentro de `site-packages` (`libcublasLt.so.13`, `libcudnn.so.9`, ...). Sin
# `LD_LIBRARY_PATH`, `CUDAExecutionProvider` falla con
#   "Failed to load library libonnxruntime_providers_cuda.so ... libcublasLt.so.13"
# y la detección de pose (DWPose, ONNX) cae silenciosamente a CPU. El modelo de
# try-on (PyTorch) sí usa la GPU en ambos casos.
#
# Uso:
#   ./run_fashn_vton.sh python examples/basic_inference.py --weights-dir ./weights \
#       --person-image examples/data/model.webp \
#       --garment-image examples/data/garment.webp --category tops
#   ./run_fashn_vton.sh python -c "import onnxruntime as ort; print(ort.get_available_providers())"
#   ./run_fashn_vton.sh python     # REPL con el entorno listo
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${CONDA_SH:-/home/uceda/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-fashn-vton}"

if [ ! -f "${CONDA_SH}" ]; then
  echo "ERROR: no existe CONDA_SH=${CONDA_SH} (exporta CONDA_SH=/ruta/a/conda.sh)." >&2
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

if [ "$#" -eq 0 ]; then
  cat <<EOF
Entorno: ${CONDA_ENV}   Proyecto: ${PROJECT_ROOT}
LD_LIBRARY_PATH (libs CUDA de nvidia-*): ${NVIDIA_LIBS:-<vacío>}

Uso: $0 <comando...>
Ejemplos:
  $0 python examples/basic_inference.py --weights-dir ./weights \\
      --person-image examples/data/model.webp \\
      --garment-image examples/data/garment.webp --category tops
  $0 python scripts/download_weights.py --weights-dir ./weights
  $0 python -c "import onnxruntime as ort; print(ort.get_available_providers())"
EOF
  exit 0
fi

exec "$@"
