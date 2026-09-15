"""Pruebas del servicio HTTP (sin cargar el modelo real: se inyecta una fábrica)."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fashn_vton.api.main import Settings, create_app


@dataclass
class _FakeResult:
    images: list
    metadata: dict


def _png_bytes(size=(96, 128), color=(120, 30, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_pipeline_factory(provider_name: str, device):
    """Sustituye al TryOnPipeline real: devuelve una imagen sintética."""

    def _pipeline(person_image, garment_image, **kwargs):
        base = Image.new("RGB", (576, 864), (10, 20, 30))
        images = [base.copy() for _ in range(int(kwargs.get("num_samples", 1)))]
        return _FakeResult(
            images=images,
            metadata={
                "segmentation": {
                    "provider": provider_name,
                    "license": "N/A (prueba)",
                    "person_mask": False,
                    "garment_mask": False,
                    "degraded": ["prenda: prueba"] if kwargs.get("garment_photo_type") == "model" else [],
                },
                "request": dict(kwargs),
            },
        )

    return _pipeline


@pytest.fixture()
def client() -> TestClient:
    app = create_app(Settings(weights_dir="weights", device="cpu"))
    app.state.pipeline_factory = _fake_pipeline_factory
    return TestClient(app)


def _post(client: TestClient, **overrides):
    data = {
        "category": "tops",
        "garment_photo_type": "flat-lay",
        "num_timesteps": 8,
        "num_samples": 1,
        "seed": 42,
    }
    data.update(overrides)
    files = {"person": ("person.png", _png_bytes(), "image/png"), "garment": ("garment.png", _png_bytes(), "image/png")}
    return client.post("/v1/tryon", data=data, files=files)


def test_healthz_reports_commercial_status(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["commercial_ready"] is True
    assert body["providers"]
    assert body["compliance_violations"] == []


def test_version_endpoint(client):
    body = client.get("/v1/version").json()
    assert "fork comercial" in body["package"]
    assert "none" in body["providers"]


def test_tryon_returns_json_with_base64_png(client):
    response = _post(client)
    assert response.status_code == 200
    body = response.json()
    assert len(body["images"]) == 1
    image = Image.open(io.BytesIO(base64.b64decode(body["images"][0]["base64"])))
    assert image.size == (576, 864)
    assert body["metadata"]["segmentation"]["provider"] == "none"
    assert body["elapsed_seconds"] >= 0


def test_tryon_raw_returns_png_with_metadata_headers(client):
    response = _post(client, num_timesteps=8)
    raw = client.post(
        "/v1/tryon?raw=true",
        data={
            "category": "tops",
            "garment_photo_type": "model",
            "num_timesteps": 8,
            "num_samples": 1,
            "seed": 1,
        },
        files={"person": ("p.png", _png_bytes(), "image/png"), "garment": ("g.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert raw.status_code == 200
    assert raw.headers["content-type"] == "image/png"
    assert raw.headers["x-segmentation-provider"] == "none"
    assert raw.headers["x-segmentation-degraded"] == "true"


def test_invalid_category_is_rejected(client):
    assert _post(client, category="hats").status_code == 422


def test_invalid_photo_type_is_rejected(client):
    assert _post(client, garment_photo_type="catalog").status_code == 422


def test_limits_are_enforced(client):
    assert _post(client, num_samples=99).status_code == 422
    assert _post(client, num_timesteps=999).status_code == 422


def test_unknown_provider_is_rejected(client):
    assert _post(client, provider="magic").status_code == 422


def test_missing_images_are_rejected(client):
    response = client.post("/v1/tryon", data={"category": "tops"}, files={})
    assert response.status_code == 422


def test_invalid_image_bytes_are_rejected(client):
    files = {
        "person": ("person.png", b"no soy una imagen", "image/png"),
        "garment": ("garment.png", _png_bytes(), "image/png"),
    }
    assert client.post("/v1/tryon", data={"category": "tops"}, files=files).status_code == 422


def test_api_key_is_required_when_configured():
    app = create_app(Settings(api_keys=("secreta",), allow_anonymous=True))
    app.state.pipeline_factory = _fake_pipeline_factory
    client = TestClient(app)
    assert _post(client).status_code == 401

    files = {"person": ("p.png", _png_bytes(), "image/png"), "garment": ("g.png", _png_bytes(), "image/png")}
    ok = client.post(
        "/v1/tryon",
        data={"category": "tops", "num_timesteps": 8},
        files=files,
        headers={"X-API-Key": "secreta"},
    )
    assert ok.status_code == 200


def test_non_commercial_provider_is_forbidden(monkeypatch):
    """Un proveedor marcado como no comercial no puede usarse por la API."""
    from fashn_vton import api as api_module
    from fashn_vton.segmentation.base import ProviderInfo
    from fashn_vton.segmentation.none import NoSegmentationProvider

    class _NC(NoSegmentationProvider):
        info = ProviderInfo(name="none", license="NC (prueba)", requires_weights=False, commercial_ok=False)

    monkeypatch.setitem(api_module.main.PROVIDERS, "none", _NC)
    app = create_app(Settings())
    client = TestClient(app)
    assert _post(client).status_code == 403
