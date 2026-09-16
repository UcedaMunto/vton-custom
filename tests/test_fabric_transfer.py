"""Pruebas de la sustitución de color/textura de prendas (sin GPU ni pesos).

Imágenes sintéticas: una «prenda» azul sobre fondo blanco y una «tela» de rayas.
"""

from __future__ import annotations

import numpy as np

from fashn_vton.preprocessing.fabric import (
    FabricTransferConfig,
    apply_fabric,
    composite_background,
    detect_background_color,
    garment_mask_from_background,
    prepare_fabric_tile,
    restrict_to_front,
    retexture_garment,
    tile_to_canvas,
    to_uint8,
)

GARMENT_COLOR = (40, 80, 200)


def _garment_with_white_background(size=(160, 120), margin=25, color=GARMENT_COLOR) -> np.ndarray:
    """Rectángulo de color sobre fondo blanco (foto de producto sintética)."""
    image = np.full((size[0], size[1], 3), 250, np.uint8)
    image[margin:-margin, margin:-margin] = color
    return image


def _striped_fabric(size=(32, 32)) -> np.ndarray:
    """Tela con dos mitades de color bien distinguibles (patrón)."""
    fabric = np.zeros((size[0], size[1], 3), np.uint8)
    fabric[:, : size[1] // 2] = (220, 30, 30)
    fabric[:, size[1] // 2 :] = (240, 210, 40)
    return fabric


def _mask_of(image: np.ndarray) -> np.ndarray:
    return (np.abs(image.astype(np.int16) - 250).max(axis=2) > 30).astype(bool)


def test_detecta_fondo_blanco():
    image = _garment_with_white_background()
    color, uniformity = detect_background_color(image)
    assert np.all(np.abs(color - 250) < 3)
    assert uniformity < 3.0


def test_mascara_separa_la_prenda_del_fondo():
    image = _garment_with_white_background()
    mask = garment_mask_from_background(image, tolerance=30)
    assert mask.dtype == bool
    assert 0.3 < mask.mean() < 0.5  # el rectángulo ocupa ~40 % del lienzo
    assert not mask[0, 0]  # esquina = fondo
    assert mask[80, 60]  # centro = prenda


def test_desactivado_devuelve_la_imagen_intacta():
    image = _garment_with_white_background()
    result, mask, info = retexture_garment(
        image, _striped_fabric(), FabricTransferConfig(enabled=False, flat_color=(255, 0, 0))
    )
    assert np.array_equal(result, image)
    assert not info.applied
    assert mask.all()


def test_color_plano_sustituye_el_color_de_la_prenda():
    image = _garment_with_white_background()
    config = FabricTransferConfig(enabled=True, flat_color=(220, 30, 30), background="blanco")
    result, mask, info = retexture_garment(image, None, config)
    assert info.applied and "color plano" in info.fabric
    inside = result[mask]
    assert inside[:, 0].mean() > 150  # rojo
    assert inside[:, 2].mean() < 100  # ya no azul


def test_tela_con_patron_se_usa_y_se_repite():
    image = _garment_with_white_background()
    config = FabricTransferConfig(enabled=True, scale=1.0, background="blanco")
    result, mask, _ = retexture_garment(image, _striped_fabric(), config)
    inside = result[mask].astype(np.int16)
    # El brillo medio de la tela se normaliza al de la prenda (por diseño), así que
    # se distingue por tono: mitad roja (G bajo) y mitad amarilla (G alto).
    reddish = ((inside[:, 0] > 90) & (inside[:, 1] < 60)).mean()
    yellowish = ((inside[:, 0] > 90) & (inside[:, 1] > 90)).mean()
    assert reddish > 0.2 and yellowish > 0.2, "el patrón debería verse (dos tonos)"


def test_conserva_las_sombras_de_la_prenda():
    image = np.full((160, 120, 3), 250, np.uint8)
    for index, value in enumerate(np.linspace(60, 200, 160, dtype=np.float32)):
        image[index, 25:95] = (value * 0.3, value * 0.4, value)
    config = FabricTransferConfig(enabled=True, flat_color=(200, 200, 200), shading=1.0, background="blanco")
    result, mask, _ = retexture_garment(image, None, config)
    original = image.astype(np.float32).mean(axis=2)[mask]
    generated = result.astype(np.float32).mean(axis=2)[mask]
    correlation = float(np.corrcoef(original, generated)[0, 1])
    assert correlation > 0.8, f"el sombreado original debe conservarse (correlación={correlation:.3f})"


def test_detalle_reinyecta_altas_frecuencias():
    import cv2

    image = _garment_with_white_background()
    image[70:76, 25:-25] = (20, 20, 20)  # costura horizontal
    mask = _mask_of(image)
    tile = prepare_fabric_tile(_striped_fabric(size=(120, 120)))
    pattern = tile_to_canvas(tile, image.shape[:2], scale=1.0)

    without = apply_fabric(image, pattern, mask, detail=0.0)
    with_detail = apply_fabric(image, pattern, mask, detail=0.6)

    def high_frequency(block: np.ndarray) -> float:
        blurred = cv2.GaussianBlur(block.astype(np.float32), (0, 0), 2.0)
        return float(np.abs(block - blurred)[mask].mean())

    assert high_frequency(with_detail) > high_frequency(without)


def test_fondo_configurable():
    image = _garment_with_white_background()
    mask = garment_mask_from_background(image, tolerance=30)
    white = composite_background(image, mask, mode="blanco")
    assert np.all(white[:5, :5] > 250)  # esquina = fondo puro, sin mezcla de borde
    green = composite_background(image, mask, mode="color", color=(10, 200, 10))
    assert np.allclose(green[:5, :5].mean(axis=(0, 1)), (10, 200, 10), atol=2)
    original = composite_background(image, mask, mode="original")
    assert np.array_equal(to_uint8(original), image)


def test_banda_frontal_reduce_la_mascara():
    image = _garment_with_white_background()
    mask = garment_mask_from_background(image, tolerance=30)
    limited = restrict_to_front(mask, fraction=0.4)
    assert limited.sum() < mask.sum()
    assert limited[:, :25].sum() == 0  # no toca los laterales
    assert limited[80, 60]  # el centro sigue dentro


def test_es_determinista():
    image = _garment_with_white_background()
    config = FabricTransferConfig(enabled=True, rotation=12, tilt_x=0.2, perspective=0.2, scale=0.8)
    first, _, _ = retexture_garment(image, _striped_fabric(), config)
    second, _, _ = retexture_garment(image, _striped_fabric(), config)
    assert np.array_equal(first, second)


def test_angulo_y_escala_cambian_el_resultado():
    image = _garment_with_white_background()
    plain, _, _ = retexture_garment(image, _striped_fabric(), FabricTransferConfig(enabled=True, background="blanco"))
    rotated, _, _ = retexture_garment(image, _striped_fabric(), FabricTransferConfig(enabled=True, rotation=45))
    scaled, _, _ = retexture_garment(image, _striped_fabric(), FabricTransferConfig(enabled=True, scale=0.4))
    tilted, _, _ = retexture_garment(image, _striped_fabric(), FabricTransferConfig(enabled=True, tilt_x=0.5))
    for other in (rotated, scaled, tilted):
        assert not np.array_equal(plain, other)


def test_sin_tela_ni_color_no_toca_nada():
    image = _garment_with_white_background()
    result, _, info = retexture_garment(image, None, FabricTransferConfig(enabled=True))
    assert np.array_equal(result, image)
    assert not info.applied
    assert "tela" in info.reason


def test_mosaico_devuelve_forma_y_rango():
    tile = prepare_fabric_tile(_striped_fabric())
    pattern = tile_to_canvas(tile, (64, 96), scale=1.5, rotation=30, perspective=0.3, mosaic="espejo")
    assert pattern.shape == (64, 96, 3)
    assert pattern.min() >= 0 and pattern.max() <= 255


def test_info_describe_la_operacion():
    image = _garment_with_white_background()
    _, mask, info = retexture_garment(image, _striped_fabric(), FabricTransferConfig(enabled=True))
    assert info.applied and info.coverage > 0.2 and mask.dtype == bool
    assert info.mask_source == "auto"
    assert "rgb" in info.describe() and "imagen" in info.describe()


def _banded_fabric(size=(40, 40), band=8) -> np.ndarray:
    """Tela con una banda clara por motivo (fácil de seguir en el perfil)."""
    fabric = np.zeros((size[0], size[1], 3), np.uint8)
    fabric[..., 0] = 150
    fabric[..., 1] = 20
    fabric[..., 2] = 20
    fabric[:, :band] = (250, 250, 250)
    return fabric


def _periodicity(profile: np.ndarray, lag: int) -> float:
    first, second = profile[:-lag], profile[lag:]
    return float(np.corrcoef(first, second)[0, 1])


def test_repeat_reference_calcula_el_tamano_del_motivo():
    from fashn_vton.preprocessing.fabric import repeat_reference_px

    as_px = FabricTransferConfig(enabled=True, repeat_mode="px", repeat_px=40)
    assert repeat_reference_px(as_px, 120) == 40.0

    as_cm = FabricTransferConfig(enabled=True, repeat_mode="cm", repeat_cm=12, garment_width_cm=50)
    assert abs(repeat_reference_px(as_cm, 300) - 72.0) < 1e-6  # 300 * 12/50

    as_scale = FabricTransferConfig(enabled=True, repeat_mode="scale", scale=2.0)
    assert repeat_reference_px(as_scale, 120) == 60.0

    # Nunca por debajo de 8 px (evita mosaicos degenerados).
    tiny = FabricTransferConfig(enabled=True, repeat_mode="px", repeat_px=1)
    assert repeat_reference_px(tiny, 120) == 8.0


def test_el_patron_se_repite_con_el_periodo_pedido():
    from fashn_vton.preprocessing.fabric import prepare_fabric_tile

    tile = prepare_fabric_tile(_banded_fabric(), flatten=False)
    for repeat in (40, 60):
        pattern = tile_to_canvas(tile, (160, 240), reference_width=float(repeat))
        profile = pattern.mean(axis=(0, 2))
        same_period = _periodicity(profile, repeat)
        half_period = _periodicity(profile, repeat // 2)
        assert same_period > 0.8, f"periodo {repeat} px poco marcado ({same_period:.3f})"
        assert same_period > half_period


def test_info_reporta_el_tamano_y_las_repeticiones():
    image = _garment_with_white_background(size=(160, 240))
    config = FabricTransferConfig(enabled=True, repeat_mode="cm", repeat_cm=12, garment_width_cm=48)
    _, _, info = retexture_garment(image, _striped_fabric(), config)
    # 240 px * (12/48) = 60 px por motivo → 4 repeticiones a lo ancho
    assert info.repeat_px == 60.0
    assert abs((info.repeats or 0) - 4.0) < 0.05
    assert info.repeat_cm == 12.0
    assert "repeticiones a lo ancho" in info.describe_repeat()
