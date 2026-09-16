"""Pruebas del botón «Previsualizar prenda con la tela» (sin GPU ni modelos).

Regresión concreta: el botón debe aplicar la tela **aunque** el interruptor
«Aplicar tela a la prenda» esté desactivado (antes devolvía la prenda original, que
es lo que el usuario percibía como «la previsualización no muestra la tela»).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def webui():
    """Carga `webui/app.py` sin arrancar la interfaz (sus imports pesados son perezosos)."""
    spec = importlib.util.spec_from_file_location("fashn_webui_test", ROOT / "webui" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _garment_pil() -> Image.Image:
    """Prenda sintética sobre fondo gris claro."""
    array = np.full((240, 180, 3), 245, np.uint8)
    array[40:200, 40:140] = (60, 90, 170)
    return Image.fromarray(array)


def _fabric_pil() -> Image.Image:
    """Tela sintética con cuadros."""
    tile = np.zeros((64, 64, 3), np.uint8)
    tile[..., 0] = 170
    tile[..., 1] = 60
    tile[::16, :] = (250, 250, 250)
    tile[:, ::16] = (250, 250, 250)
    return Image.fromarray(tile)


def _preview(webui, garment, fabric, enabled: bool, grid: bool = False, color: str = "#ffffff"):
    """Llama al manejador del botón con los parámetros en el orden real de la UI."""
    return webui.preview_fabric(
        garment,
        fabric,
        enabled,
        color,
        "gris",
        "#ffffff",
        "auto",
        30.0,
        1.0,
        1.0,
        20.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        "repetir",
        "cm",
        12.0,
        50.0,
        120.0,
        grid,
    )


def test_aplica_la_tela_aunque_el_interruptor_este_apagado(webui):
    garment = _garment_pil()
    result, log = _preview(webui, garment, _fabric_pil(), enabled=False)
    assert "tela: aplicada" in log
    assert not np.array_equal(np.asarray(result), np.asarray(garment))
    assert "vista previa" in log  # avisa de que al generar no se usará


def test_sin_tela_ni_color_avisa_y_no_cambia_la_imagen(webui):
    garment = _garment_pil()
    result, log = _preview(webui, garment, None, enabled=True, color="#ffffff")
    assert np.array_equal(np.asarray(result), np.asarray(garment))
    assert "Sube una" in log


def test_color_plano_funciona_sin_tela(webui):
    garment = _garment_pil()
    result, log = _preview(webui, garment, None, enabled=True, color="#cc2222")
    assert "color plano" in log
    assert not np.array_equal(np.asarray(result), np.asarray(garment))


def test_la_cuadricula_solo_se_anuncia_en_la_previsualizacion(webui):
    garment = _garment_pil()
    _plain, log_plain = _preview(webui, garment, _fabric_pil(), enabled=True, grid=False)
    _grid, log_grid = _preview(webui, garment, _fabric_pil(), enabled=True, grid=True)
    assert "cuadrícula" not in log_plain
    assert "cuadrícula de 10 cm solo en la previsualización" in log_grid


def test_el_log_informa_del_tamano_y_las_repeticiones(webui):
    _image, log = _preview(webui, _garment_pil(), _fabric_pil(), enabled=True)
    assert "repeticiones a lo ancho" in log
    assert "12.0 cm" in log


def test_se_aplica_al_generar_solo_con_tela_y_casilla_activa(webui):
    """Regla del try-on: manda la tela configurada + la casilla (activada por defecto)."""
    fabric = _fabric_pil()
    assert webui._should_apply_fabric(fabric, "#ffffff", True) is True  # tela subida
    assert webui._should_apply_fabric(None, "#cc2222", True) is True  # color plano elegido
    assert webui._should_apply_fabric(fabric, "#ffffff", False) is False  # desactivado a propósito
    assert webui._should_apply_fabric(None, "#ffffff", True) is False  # nada configurado
    assert webui._should_apply_fabric(None, "#ffffff", False) is False
