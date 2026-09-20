# syntax=docker/dockerfile:1
# Imagen de producción del fork comercial de FASHN VTON v1.5.
#
# Construida y verificada el 2026-09-19 (nodo-gpu-1 del cluster, ver
# kubernetes/02_ESTADO_DE_EJECUCION.md): el paso de cumplimiento se ejecuta dentro
# del build, de forma que la imagen falla si aparece un artefacto de licencia no
# comercial. Una sola imagen sirve los 4 roles del despliegue (worker GPU, worker
# CPU, gateway y web); el rol lo decide el comando del manifiesto.
#
# Uso (validación, CPU):
#   docker build -t fashn-vton-commercial .
#   docker run --rm -p 8000:8000 -v $PWD/weights:/app/weights:ro fashn-vton-commercial
#
# Uso (GPU, requiere nvidia-container-toolkit en el host):
#   docker build --build-arg BASE=nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04 \
#                --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 \
#                -t fashn-vton-commercial:gpu .
#   docker run --rm --gpus all -p 8000:8000 -v $PWD/weights:/app/weights:ro fashn-vton-commercial:gpu

ARG BASE=python:3.11-slim
FROM ${BASE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    FASHN_WEIGHTS_DIR=/app/weights \
    VTON_MAX_UPLOAD_MB=12

# Dependencias de sistema mínimas para OpenCV/ONNX y healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# La base CUDA (Ubuntu) trae `python3` pero no `python`, y los manifiestos invocan
# `python -m fashn_vton.queue_worker`: se deja un enlace estable antes de instalar nada.
RUN if ! command -v python >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev && \
        ln -sf /usr/bin/python3 /usr/local/bin/python && \
        ln -sf /usr/bin/pip3 /usr/local/bin/pip && \
        rm -rf /var/lib/apt/lists/*; \
    fi

COPY pyproject.toml README.md README_COMERCIAL.md THIRD_PARTY_NOTICES.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY licenses ./licenses
COPY examples ./examples

# El cache mount de pip evita re-descargar torch/onnxruntime (varios GB) en cada
# reconstruccion: solo se invalida la capa de instalacion, no la descarga.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && pip install torch torchvision --index-url "${TORCH_INDEX}" \
    && pip install -e ".[api,cluster]"

# onnxruntime-gpu (DWPose) busca libcublasLt/libcudnn dentro de site-packages/nvidia/*/lib;
# sin esto cae silenciosamente a CPU (es el mismo problema que resuelve run_fashn_vton.sh
# fuera del contenedor). Se deja en ld.so.conf para que no dependa de una variable de entorno.
RUN SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')" \
    && (find "$SITE_PACKAGES/nvidia" -maxdepth 2 -type d -name lib -print 2>/dev/null || true) > /etc/ld.so.conf.d/nvidia-pip.conf \
    && echo "$SITE_PACKAGES/torch/lib" >> /etc/ld.so.conf.d/nvidia-pip.conf \
    && ldconfig

# Gate de cumplimiento DENTRO del build: si aparece un artefacto prohibido o una
# dependencia no comercial, la imagen no se construye. (Con pesos ausentes el
# gate solo avisa: MISSING no es una violación.)
RUN python scripts/verify_commercial_readiness.py \
    && python licenses/verify_hashes.py --no-forbidden-scan

# Usuario sin privilegios
RUN useradd --create-home --uid 10001 vton && chown -R vton:vton /app
USER vton

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# Los pesos NO se hornean en la imagen (licencia y tamaño): montar ./weights en /app/weights.
CMD ["uvicorn", "fashn_vton.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
