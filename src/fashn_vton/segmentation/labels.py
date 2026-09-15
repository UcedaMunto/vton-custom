"""FASHN VTON v1.5 label table and category mappings.

The model conditions on a *human-parsing class map* whose integer ids are part
of the frozen model interface (it was trained with them). This module provides
that interoperability table locally so the commercial fork does **not** depend
on ``fashn-human-parser`` (whose weights inherit NVIDIA SegFormer's
non-commercial license).

Provenance: the values were recorded from the package that the upstream
pipeline used (``fashn-human-parser`` 0.1.1) *before* removing the dependency,
into ``plan_modelo_comercial/evidencia_labels_originales.json``, and are pinned
by ``tests/test_segmentation_labels.py``. They are plain name -> id facts
required for interoperability, not copied code.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Class id table (0..17) expected by FASHN VTON v1.5.
# ---------------------------------------------------------------------------
LABELS_TO_IDS: dict[str, int] = {
    "background": 0,
    "face": 1,
    "hair": 2,
    "top": 3,
    "dress": 4,
    "skirt": 5,
    "pants": 6,
    "belt": 7,
    "bag": 8,
    "hat": 9,
    "scarf": 10,
    "glasses": 11,
    "arms": 12,
    "hands": 13,
    "legs": 14,
    "feet": 15,
    "torso": 16,
    "jewelry": 17,
}

# Body coverage per category: which labels have to be masked for each garment
# category (used to build the clothing-agnostic person image).
BODY_COVERAGE_TO_LABELS: dict[str, list[str]] = {
    "upper": ["top", "dress", "scarf"],
    "lower": ["skirt", "pants", "belt"],
    "full": ["top", "dress", "scarf", "skirt", "pants", "belt"],
}

# Labels that identify the person and must be preserved (never masked).
IDENTITY_LABELS: list[str] = ["face", "hair", "jewelry", "bag", "glasses", "hat"]

# Public API of the pipeline: category -> body coverage.
CATEGORY_TO_BODY_COVERAGE: dict[str, str] = {
    "tops": "upper",
    "bottoms": "lower",
    "one-pieces": "full",
}

# Reverse view, handy for providers that build class maps label by label.
ID_TO_LABEL: dict[int, str] = {value: key for key, value in LABELS_TO_IDS.items()}

# Clothing-region labels, grouped by coverage (derived, not hardcoded twice).
COVERAGE_LABEL_IDS: dict[str, list[int]] = {
    coverage: [LABELS_TO_IDS[label] for label in labels] for coverage, labels in BODY_COVERAGE_TO_LABELS.items()
}

NUM_CLASSES = len(LABELS_TO_IDS)

__all__ = [
    "BODY_COVERAGE_TO_LABELS",
    "CATEGORY_TO_BODY_COVERAGE",
    "COVERAGE_LABEL_IDS",
    "IDENTITY_LABELS",
    "ID_TO_LABEL",
    "LABELS_TO_IDS",
    "NUM_CLASSES",
]
