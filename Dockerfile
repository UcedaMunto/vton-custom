# Imagen de producción del fork comercial de FASHN VTON v1.5.
#
# IMPORTANTE: esta imagen **no se ha construido ni verificado** en la máquina de
# desarrollo (no hay Docker/containerd instalado allí; ver
# plan_modelo_comercial/PLAN_IMPLEMENTACION_COMERCIAL.md §7 "Pendiente"). El
# Dockerfile se revisó a mano y el paso de cumplimiento se ejecuta dentro del
# build, de forma que la imagen falla si aparece un artefacto de licencia no
# comercial. Antes de publicarla: construirla en un host con Docker y ejecutar
# el smoke test del README_COMERCIAL.md.
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

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    FASHN_WEIGHTS_DIR=/app/weights \
    VTON_MAX_UPLOAD_MB=12

# Dependencias de sistema mínimas para OpenCV/ONNX y healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md README_COMERCIAL.md THIRD_PARTY_NOTICES.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY licenses ./licenses
COPY examples ./examples

RUN python -m pip install --upgrade pip \
    && pip install torch torchvision --index-url "${TORCH_INDEX}" \
    && pip install -e ".[api]"

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
