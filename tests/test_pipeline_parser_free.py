"""Pruebas del fork comercial: el pipeline NO depende de `fashn-human-parser`.

Estas pruebas son el seguro del cambio central del plan comercial: si alguien
vuelve a introducir la dependencia no comercial, fallan.
"""

from __future__ import annotations

import logging
import subprocess
import sys

import numpy as np
import pytest

from fashn_vton.pipeline import TryOnPipeline
from fashn_vton.segmentation import NoSegmentationProvider, ProviderInfo, SegmentationProvider
from fashn_vton.segmentation.labels import LABELS_TO_IDS

# ---------------------------------------------------------------------------
# 1) El pipeline se puede importar/usar con el paquete prohibido BLOQUEADO
# ---------------------------------------------------------------------------
_BLOCKER_PRELUDE = r"""
import sys

FORBIDDEN = "fashn_human_parser"


class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == FORBIDDEN or fullname.startswith(FORBIDDEN + "."):
            raise ImportError(f"bloqueado por la prueba: {fullname}")
        return None


sys.meta_path.insert(0, Blocker())
"""

PIPELINE_SCRIPT = (
    _BLOCKER_PRELUDE
    + r"""
import fashn_vton.pipeline as pipeline  # noqa: E402
from fashn_vton.segmentation import build_segmentation_provider  # noqa: E402

assert FORBIDDEN not in sys.modules, "el paquete prohibido acabó importado"
provider = build_segmentation_provider("none")
assert provider.predict(None) is None
assert pipeline.TryOnPipeline is not None
print("PARSER_FREE_OK")
"""
)

PREPROCESSING_SCRIPT = (
    _BLOCKER_PRELUDE
    + r"""
import fashn_vton.preprocessing as preprocessing  # noqa: E402
from fashn_vton.preprocessing import create_clothing_agnostic_image  # noqa: E402

assert FORBIDDEN not in sys.modules, "el paquete prohibido acabó importado"
assert preprocessing.FASHN_LABELS_TO_IDS["top"] == 3
assert callable(create_clothing_agnostic_image)
print("PREPROCESSING_PARSER_FREE_OK")
"""
)


def test_pipeline_imports_without_forbidden_package():
    """Importar el pipeline no debe tocar el paquete de licencia no comercial."""
    result = subprocess.run(
        [sys.executable, "-c", PIPELINE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "PARSER_FREE_OK" in result.stdout


def test_source_tree_has_no_forbidden_imports():
    """Comprobación directa sobre el código (sin depender del entorno)."""
    from fashn_vton.compliance import scan_source_tree

    assert scan_source_tree() == []


def test_preprocessing_module_does_not_need_the_forbidden_package():
    """`preprocessing.agnostic` era el segundo acoplamiento oculto: debe importar."""
    result = subprocess.run(
        [sys.executable, "-c", PREPROCESSING_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "PREPROCESSING_PARSER_FREE_OK" in result.stdout


# ---------------------------------------------------------------------------
# 2) Degradación explícita / rechazo de proveedores no comerciales
# ---------------------------------------------------------------------------
class _Harness:
    """Objeto mínimo para ejercitar métodos de `TryOnPipeline` sin cargar modelos."""

    def __init__(self, provider, strict: bool = False) -> None:
        self.segmentation_provider = provider
        self.strict_segmentation = strict
        self.logger = logging.getLogger("test_pipeline_parser_free")


class _NoneProvider(SegmentationProvider):
    info = ProviderInfo(name="fake-none", license="Apache-2.0 (prueba)", requires_weights=False)

    def predict(self, image, context=None):
        return None


class _BrokenProvider(SegmentationProvider):
    info = ProviderInfo(name="fake-broken", license="Apache-2.0 (prueba)", requires_weights=False)

    def predict(self, image, context=None):
        raise RuntimeError("fallo simulado del proveedor")


class _BadMapProvider(SegmentationProvider):
    info = ProviderInfo(name="fake-badmap", license="Apache-2.0 (prueba)", requires_weights=False)

    def predict(self, image, context=None):
        return np.full((4, 4), 99, dtype=np.uint8)  # id fuera de rango


class _NonCommercialProvider(SegmentationProvider):
    info = ProviderInfo(
        name="fake-nc",
        license="No comercial (prueba)",
        requires_weights=True,
        commercial_ok=False,
    )

    def predict(self, image, context=None):
        return np.zeros((4, 4), dtype=np.uint8)


def test_provider_without_mask_degrades_and_reports_none():
    harness = _Harness(_NoneProvider())
    assert TryOnPipeline._provider_class_map(harness, np.zeros((8, 8, 3), np.uint8), "persona", {}) is None


def test_provider_failure_degrades_by_default_and_raises_in_strict_mode():
    image = np.zeros((8, 8, 3), np.uint8)
    assert TryOnPipeline._provider_class_map(_Harness(_BrokenProvider()), image, "prenda", {}) is None
    with pytest.raises(RuntimeError):
        TryOnPipeline._provider_class_map(_Harness(_BrokenProvider(), strict=True), image, "prenda", {})


def test_invalid_class_map_degrades():
    image = np.zeros((8, 8, 3), np.uint8)
    assert TryOnPipeline._provider_class_map(_Harness(_BadMapProvider()), image, "persona", {}) is None


def test_non_commercial_provider_is_rejected_at_setup():
    with pytest.raises(RuntimeError) as excinfo:
        TryOnPipeline._setup_segmentation(_Harness(_NonCommercialProvider()))
    assert "NO comercial" in str(excinfo.value)


def test_default_provider_is_the_commercial_one():
    from fashn_vton.segmentation import DEFAULT_PROVIDER, build_segmentation_provider

    assert DEFAULT_PROVIDER == "none"
    assert isinstance(build_segmentation_provider(), NoSegmentationProvider)


def test_class_map_ids_used_by_pipeline_belong_to_the_table():
    """Los ids de prenda que el pipeline pide existen en la tabla local."""
    for name in ("top", "dress", "scarf", "skirt", "pants", "belt", "arms", "torso", "legs"):
        assert name in LABELS_TO_IDS
