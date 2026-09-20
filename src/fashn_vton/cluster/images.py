"""Image validation and normalisation shared by the gateway and CPU worker."""
from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError

#: Longest side kept when re-encoding a client upload (the model resizes later).
MAX_SIDE = 1600


def open_rgb(raw: bytes, role: str, max_mb: int = 12) -> Image.Image:
    """Validate an upload and return an RGB image.

    Raises ``ValueError`` with a user-facing (Spanish) message, mirroring the
    messages already used by ``fashn_vton.api.main``.
    """
    if max_mb and len(raw) > max_mb * 1024 * 1024:
        raise ValueError(f"La imagen de {role} supera el maximo de {max_mb} MB.")
    if not raw:
        raise ValueError(f"La imagen de {role} esta vacia.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"La imagen de {role} no es una imagen valida ({exc}).") from exc
    if image.width < 64 or image.height < 64:
        raise ValueError(f"La imagen de {role} es demasiado pequena ({image.width}x{image.height}).")
    return image.convert("RGB")


def normalize_jpeg(image: Image.Image, max_side: int = MAX_SIDE, quality: int = 95) -> bytes:
    """Re-encode to JPEG, dropping EXIF and capping the longest side.

    Re-encoding is also what guarantees that the original bytes (and any
    metadata they carried) do not survive in the transient bucket.
    """
    working = image.copy()
    if max(working.size) > max_side:
        scale = max_side / float(max(working.size))
        new_size = (max(1, int(working.width * scale)), max(1, int(working.height * scale)))
        working = working.resize(new_size, Image.LANCZOS)
    buffer = io.BytesIO()
    working.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
