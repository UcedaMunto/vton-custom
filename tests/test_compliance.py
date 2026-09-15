"""Pruebas del módulo de cumplimiento comercial."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fashn_vton import compliance

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_repo_sources_are_clean():
    assert compliance.scan_source_tree(REPO_ROOT) == []


def test_real_repo_has_no_forbidden_dependencies():
    assert compliance.check_pyproject_dependencies(REPO_ROOT / "pyproject.toml") == []


def test_real_repo_has_no_forbidden_artifacts():
    assert compliance.find_forbidden_files(REPO_ROOT) == []


def test_forbidden_import_is_detected(tmp_path):
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "bad.py").write_text("from fashn_human_parser import FashnHumanParser\n", encoding="utf-8")
    violations = compliance.scan_source_tree(tmp_path, allowlist=set())
    assert violations and "fashn_human_parser" in violations[0]


def test_fashn_human_parser_instantiation_is_detected(tmp_path):
    (tmp_path / "bad.py").write_text("model = FashnHumanParser(device='cpu')\n", encoding="utf-8")
    violations = compliance.scan_source_tree(tmp_path, allowlist=set())
    assert any("FashnHumanParser" in violation for violation in violations)


def test_forbidden_dependency_is_detected(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\ndependencies = ["torch>=2.0", "fashn-human-parser>=0.1.1"]\n',
        encoding="utf-8",
    )
    violations = compliance.check_pyproject_dependencies(pyproject)
    assert violations and "fashn-human-parser" in violations[0]


def test_forbidden_file_is_detected(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "segformer_b2_clothes.safetensors").write_bytes(b"x")
    manifest = {
        "artifacts": [],
        "forbidden_artifacts": [
            {"name": "SegFormer", "match": ["**/*segformer*"]},
        ],
    }
    violations = compliance.find_forbidden_files(tmp_path, manifest)
    assert violations and "segformer" in violations[0].lower()


def test_manifest_hashes_ok_mismatch_and_missing(tmp_path):
    payload = b"contenido de prueba"
    (tmp_path / "peso.bin").write_bytes(payload)
    import hashlib

    good = hashlib.sha256(payload).hexdigest()
    manifest = {
        "artifacts": [
            {"name": "ok", "local_path": "peso.bin", "sha256": good},
            {"name": "malo", "local_path": "peso.bin", "sha256": "0" * 64},
            {"name": "falta", "local_path": "no-existe.bin", "sha256": good},
            {"name": "libreria", "local_path": None, "sha256": None},
        ]
    }
    rows = {row["name"]: row["status"] for row in compliance.verify_manifest_hashes(tmp_path, manifest)}
    assert rows == {"ok": "OK", "malo": "MISMATCH", "falta": "MISSING", "libreria": "NO_PATH"}


def test_commercial_report_is_green_on_the_real_repo():
    report = compliance.commercial_report(REPO_ROOT)
    assert report.ok, report.violations
    assert report.checked["source_scan"]["violations"] == []
    assert report.checked["forbidden_files"]["violations"] == []


def test_assert_commercial_ready_raises_on_violation(tmp_path):
    (tmp_path / "bad.py").write_text("import segformer\n", encoding="utf-8")
    with pytest.raises(compliance.ComplianceError):
        compliance.assert_commercial_ready(tmp_path, check_environment=False, check_dependencies=True)


def test_manifest_structure_is_valid():
    manifest = compliance.load_manifest()
    assert manifest["artifacts"], "el manifiesto debe declarar artefactos"
    assert manifest["forbidden_artifacts"], "el manifiesto debe declarar prohibidos"
    for artifact in manifest["artifacts"]:
        assert {"name", "source", "license"} <= set(artifact)
    for artifact in manifest["forbidden_artifacts"]:
        assert artifact.get("match"), f"{artifact.get('name')} debe declarar globs de detección"


def test_manifest_forbidden_list_covers_the_nc_parser():
    manifest = compliance.load_manifest()
    names = " ".join(item["name"].lower() for item in manifest["forbidden_artifacts"])
    assert "fashn-human-parser" in names
    assert "segformer" in names


def test_weights_hashes_recorded_in_manifest_are_present_when_weights_exist():
    """Si los pesos están descargados, su hash registrado debe coincidir."""
    rows = compliance.verify_manifest_hashes(REPO_ROOT)
    statuses = {row["status"] for row in rows}
    assert "MISMATCH" not in statuses, rows
    if "OK" in statuses:
        assert {"OK"} <= statuses


def test_segmentation_evidence_file_matches_labels():
    evidence = json.loads(
        (REPO_ROOT / "plan_modelo_comercial" / "evidencia_labels_originales.json").read_text(encoding="utf-8")
    )
    from fashn_vton.segmentation import labels

    assert evidence["LABELS_TO_IDS"] == labels.LABELS_TO_IDS
