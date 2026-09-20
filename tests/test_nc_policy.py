"""Pruebas de la política de datos no comerciales (`fashn_vton.train.nc_policy`).

Lo que se fija aquí es la propiedad que protege el proyecto: **sin opt-in
explícito no se entrena con VITON-HD/DressCode, y sin procedencia comercial no
se promueve un candidato**.
"""

from __future__ import annotations

import json

import pytest

from fashn_vton.train.nc_policy import (
    REPO_ROOT,
    NcDataInsideRepo,
    NcTrainingNotAllowed,
    NonCommercialCandidateError,
    assert_commercially_promotable,
    assert_dataset_outside_repo,
    assert_nc_training_allowed,
    build_provenance,
    is_nc_license,
    provenance_banner,
    read_provenance,
    write_provenance,
)


@pytest.mark.parametrize(
    "license_class,expected",
    [
        ("commercial", False),
        ("Apache-2.0", False),
        ("propia", False),
        ("nc-dresscode", True),
        ("nc-vitonhd", True),
        ("non-commercial", True),
        ("research", True),
        ("", True),
        (None, True),
    ],
)
def test_is_nc_license_is_conservative(license_class, expected):
    assert is_nc_license(license_class) is expected


def test_nc_training_is_blocked_without_opt_in():
    with pytest.raises(NcTrainingNotAllowed) as excinfo:
        assert_nc_training_allowed(license_class="nc-dresscode")
    message = str(excinfo.value)
    assert "NO COMERCIAL" in message
    assert "FASHN_ALLOW_NC_TRAINING" in message


def test_nc_training_allowed_with_cli_flag():
    guard = assert_nc_training_allowed(license_class="nc-dresscode", allow_nc=True)
    assert guard["commercial_use"] is False
    assert guard["non_commercial"] is True
    assert guard["opt_in"] == "flag-cli"


def test_nc_training_allowed_with_env_flag():
    guard = assert_nc_training_allowed(license_class="nc-dresscode", env={"FASHN_ALLOW_NC_TRAINING": "1"})
    assert guard["opt_in"] == "env"


def test_commercial_data_does_not_need_opt_in():
    guard = assert_nc_training_allowed(license_class="commercial")
    assert guard["commercial_use"] is True
    assert guard["non_commercial"] is False
    assert guard["opt_in"] == "no-aplica"


def test_dataset_root_is_resolved_when_outside_the_repo(tmp_path):
    guard = assert_nc_training_allowed(
        license_class="nc-dresscode", dataset_root=tmp_path, pairs_csv=tmp_path / "pairs.csv", allow_nc=True
    )
    assert guard["dataset_root"] == str(tmp_path.resolve())
    assert guard["pairs_csv"] == str((tmp_path / "pairs.csv").resolve())


def test_dataset_inside_the_repo_is_rejected():
    with pytest.raises(NcDataInsideRepo):
        assert_dataset_outside_repo(REPO_ROOT / "datos")
    assert assert_dataset_outside_repo(REPO_ROOT.parent) == REPO_ROOT.parent.resolve()


def test_provenance_roundtrip(tmp_path):
    provenance = build_provenance(
        dataset="dresscode",
        license_class="nc-dresscode",
        commercial_use=False,
        dataset_root=tmp_path,
        pairs_csv=tmp_path / "pairs.csv",
        pairs=10,
        extra={"layout": "dresscode"},
    )
    path = write_provenance(tmp_path / "candidate", provenance)
    assert path.name == "provenance.json"
    assert read_provenance(tmp_path / "candidate")["commercial_use"] is False
    assert read_provenance(path)["pairs"] == 10
    assert read_provenance(tmp_path / "no-existe") is None


def test_commercial_promotion_of_nc_candidate_is_blocked(tmp_path):
    candidate = tmp_path / "candidate"
    write_provenance(
        candidate,
        build_provenance(dataset="dresscode", license_class="nc-dresscode", commercial_use=False),
    )
    with pytest.raises(NonCommercialCandidateError) as excinfo:
        assert_commercially_promotable(candidate)
    assert "NO COMERCIALES" in str(excinfo.value)

    # Con el opt-in explícito pasa, pero queda registrado en la procedencia.
    provenance = assert_commercially_promotable(candidate, allow_nc=True)
    assert provenance["license_class"] == "nc-dresscode"


def test_candidate_without_provenance_is_not_blocked_but_flagged(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    provenance = assert_commercially_promotable(candidate)
    assert provenance["commercial_use"] is None
    assert "desconocida" in provenance_banner(provenance)


def test_commercial_candidate_is_promotable(tmp_path):
    candidate = tmp_path / "candidate"
    write_provenance(
        candidate,
        build_provenance(dataset="catalogo-propio", license_class="propia", commercial_use=True),
    )
    provenance = assert_commercially_promotable(candidate)
    assert provenance["commercial_use"] is True
    assert "uso comercial" in provenance_banner(provenance)


def test_provenance_file_is_valid_json(tmp_path):
    path = write_provenance(
        tmp_path, build_provenance(dataset="x", license_class="comercial", commercial_use=True)
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "x"
    assert "created_at" in payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
