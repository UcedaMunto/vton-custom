"""Distributed-deployment components (queue, transient storage, health).

Only the cluster processes (``queue_worker``, ``preprocess_worker``, ``gateway``,
``web``) import this package. The inference pipeline and the original synchronous
API keep working without Redis or MinIO (see ``src/fashn_vton/api/main.py`` and
``kubernetes/00_PLAN_ARQUITECTURA.md``).
"""
from __future__ import annotations

#: Queue receiving the freshly uploaded job (gateway -> CPU workers).
STREAM_PREPROCESS = "vton:jobs:preprocess"
#: Queue receiving a validated/normalised job (CPU workers -> GPU worker).
STREAM_GPU = "vton:jobs:gpu"
#: Consumer groups: one per worker type, so Redis delivers each job to exactly one.
GROUP_CPU = "vton-cpu"
GROUP_GPU = "vton-gpu"
#: Redis hash prefix holding the state of one job.
JOB_KEY_PREFIX = "vton:job:"
#: Default transient bucket (MinIO/S3).
DEFAULT_BUCKET = "vton-jobs"

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_ERROR = "error"
VALID_STATUSES = (STATUS_PENDING, STATUS_PROCESSING, STATUS_DONE, STATUS_ERROR)

#: Object keys inside a job prefix.
KEY_PERSON_RAW = "person_raw"
KEY_GARMENT_RAW = "garment_raw"
KEY_PERSON = "person.jpg"
KEY_GARMENT = "garment.jpg"
KEY_RESULT = "result.png"

__all__ = [
    "STREAM_PREPROCESS",
    "STREAM_GPU",
    "GROUP_CPU",
    "GROUP_GPU",
    "JOB_KEY_PREFIX",
    "DEFAULT_BUCKET",
    "STATUS_PENDING",
    "STATUS_PROCESSING",
    "STATUS_DONE",
    "STATUS_ERROR",
    "VALID_STATUSES",
    "KEY_PERSON_RAW",
    "KEY_GARMENT_RAW",
    "KEY_PERSON",
    "KEY_GARMENT",
    "KEY_RESULT",
]
