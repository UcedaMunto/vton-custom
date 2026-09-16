"""Sustitución de color/textura de una prenda con una tela propia.

**Qué resuelve:** subir una foto de tela (color + estampado) y aplicarla sobre la
prenda, con control del tamaño, el ángulo y la inclinación del mosaico, para
generar la prenda en esa tela antes de probársela a la persona.

**Cómo lo hace** (todo numpy/OpenCV, sin redes ni pesos de terceros → no añade
licencias nuevas):

1. :func:`detect_background_color` / :func:`garment_mask_from_background`: asume
   fondo claro (blanco o gris) — típico de una foto de producto — y separa la
   prenda por distancia de color, con limpieza morfológica y relleno de huecos.
2. :func:`prepare_fabric_tile`: endereza la foto de la tela (ángulo), le quita su
   propia iluminación (`flatten`) y ajusta el brillo.
3. :func:`tile_to_canvas`: mosaico infinito con **escala, rotación, inclinación
   (x/y), desplazamiento (x/y) y perspectiva (z)** para que la tela siga el ángulo
   que se pida.
4. :func:`apply_fabric`: reemplaza **color y estampado** dentro de la máscara
   conservando sombras y volumen de la prenda (`shading`), con `strength` para
   mezclar y `detail` para conservar costuras/cremalleras.
5. :func:`composite_background`: recompone el fondo (original, blanco, gris o el
   color elegido).
6. :func:`retexture_garment`: entrada única (la que usa la UI).

Limitación honesta: con `front_only < 1` el «panel frontal» es una **banda central
geométrica**; para el frente real de una prenda puesta conviene pasar la máscara
de un proveedor de segmentación (`sam2`, `grounded-sam2`) o recortar la foto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

#: Fondo por defecto (blanco) y gris claro típico de fotos de producto.
WHITE = (255, 255, 255)
LIGHT_GREY = (245, 245, 245)

BACKGROUND_MODES = ("original", "blanco", "gris", "color")
MOSAIC_MODES = ("repetir", "espejo")
MASK_SOURCES = ("auto", "tela-completa", "proporcionada")


@dataclass
class FabricTransferConfig:
    """Parámetros de la sustitución de tela (todos con valores seguros).

    ``enabled=False`` devuelve la imagen intacta: es lo que garantiza que el camino
    comercial (`flat-lay` + `none`) siga siendo byte a byte idéntico.
    """

    enabled: bool = False
    #: Tamaño del motivo. ``repeat_mode`` decide qué control manda:
    #: ``cm`` (medida real), ``px`` (píxeles del lienzo) o ``scale`` (relativo).
    repeat_mode: str = "cm"
    #: Ancho real del motivo en centímetros (cuántos cm mide una repetición).
    repeat_cm: float = 12.0
    #: Ancho real de la prenda en centímetros (para calibrar el motivo).
    garment_width_cm: float = 50.0
    #: Ancho del motivo en píxeles del lienzo (modo ``px``).
    repeat_px: float = 0.0
    #: Tamaño del estampado en modo relativo: 1.0 = la tela cubre el ancho de la prenda.
    scale: float = 1.0
    rotation: float = 0.0
    #: Inclinación del mosaico (-1..1) para simular el plano de la prenda.
    tilt_x: float = 0.0
    tilt_y: float = 0.0
    #: Desplazamiento del mosaico en fracción del tamaño de la imagen.
    offset_x: float = 0.0
    offset_y: float = 0.0
    #: Perspectiva / profundidad (0..0.8): 0 = plano.
    perspective: float = 0.0
    #: Giro de la foto de la tela para enderezarla (grados).
    fabric_angle: float = 0.0
    #: Quitar la iluminación propia de la tela (deja el estampado «plano»).
    flatten: bool = True
    brightness: float = 1.0
    #: 1.0 = solo la tela · 0.0 = solo la prenda original.
    strength: float = 1.0
    #: Cuánta sombra/volumen de la prenda se conserva (1 = toda).
    shading: float = 1.0
    #: Cuánto detalle de la prenda se conserva (costuras, botones).
    detail: float = 0.0
    #: Tolerancia de color para separar la prenda del fondo claro.
    background_tolerance: float = 30.0
    #: Suavizado del borde de la máscara (px).
    mask_feather: int = 2
    #: 1.0 = toda la prenda; <1.0 = banda central (aproximación del panel frontal).
    front_only: float = 1.0
    background: str = "original"
    background_color: tuple[int, int, int] = WHITE
    mosaic: str = "repetir"
    mask_source: str = "auto"
    #: Color plano cuando no se sube tela (solo cambiar el color).
    flat_color: tuple[int, int, int] | None = None


@dataclass
class FabricTransferInfo:
    """Qué hizo la sustitución (para el log de la UI/API)."""

    applied: bool = False
    reason: str = ""
    background: tuple[int, int, int] | None = None
    background_uniformity: float | None = None
    mask_source: str = "auto"
    coverage: float = 0.0
    fabric: str | None = None
    #: Ancho del motivo en píxeles del lienzo y repeticiones a lo ancho.
    repeat_px: float | None = None
    repeats: float | None = None
    repeat_cm: float | None = None
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.applied:
            return f"tela: no aplicada ({self.reason})"
        background = "-" if self.background is None else "rgb" + str(tuple(int(v) for v in self.background))
        return (
            f"tela: aplicada · máscara={self.mask_source} · cobertura={self.coverage * 100:.1f} % "
            f"· fondo detectado={background} · tela={self.fabric}"
        )

    def describe_repeat(self) -> str:
        """Resumen del tamaño del motivo (para el log de la UI)."""
        if self.repeat_px is None or self.repeats is None:
            return "motivo: -"
        size_cm = f" ({self.repeat_cm:.1f} cm)" if self.repeat_cm else ""
        return f"motivo: {self.repeat_px:.0f} px{size_cm} · repeticiones a lo ancho: {self.repeats:.2f}"


def to_float(image: np.ndarray) -> np.ndarray:
    """uint8/float → float32 [0, 255] con 3 canales."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    array = array[..., :3]
    return array.astype(np.float32) if array.dtype != np.float32 else array.copy()


def to_uint8(image: np.ndarray) -> np.ndarray:
    """float → uint8 RGB recortando a [0, 255]."""
    return np.clip(np.asarray(image, dtype=np.float32), 0, 255).astype(np.uint8)


def _luminance(image: np.ndarray) -> np.ndarray:
    """Luminancia perceptual (0-255) de una imagen RGB en float."""
    array = np.asarray(image, dtype=np.float32)
    return 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]


def detect_background_color(image: np.ndarray, border: float = 0.06) -> tuple[np.ndarray, float]:
    """Color del fondo (mediana del borde) y su uniformidad (desviación típica).

    Se asume fondo claro y uniforme (blanco o gris de foto de producto). La mediana
    es robusta frente a la prenda que toca el borde. Una uniformidad alta (> ~12)
    indica fondo con textura: el aviso va en :class:`FabricTransferInfo`.
    """
    array = to_float(image)
    height, width = array.shape[:2]
    margin = max(1, int(round(border * min(height, width))))
    ring = np.concatenate(
        [
            array[:margin].reshape(-1, 3),
            array[-margin:].reshape(-1, 3),
            array[:, :margin].reshape(-1, 3),
            array[:, -margin:].reshape(-1, 3),
        ]
    )
    color = np.median(ring, axis=0)
    uniformity = float(np.mean(np.std(ring, axis=0)))
    return color, uniformity


def garment_mask_from_background(
    image: np.ndarray,
    background: np.ndarray | tuple[float, float, float] | None = None,
    tolerance: float = 30.0,
    morph: int = 5,
    keep_largest: bool = True,
) -> np.ndarray:
    """Máscara booleana de la prenda separándola de un fondo claro y uniforme.

    Args:
        image: foto de la prenda (RGB).
        background: color de fondo; si es None se estima del borde.
        tolerance: distancia máxima (por canal) para considerar «fondo».
        morph: tamaño del elemento estructurante para abrir/cerrar (0 = sin morfología).
        keep_largest: conservar solo la componente conectada mayor (quita motas).

    Returns:
        Máscara ``HxW`` booleana (True = prenda).
    """
    array = to_float(image)
    if background is None:
        background, _ = detect_background_color(array)
    reference = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)

    distance = np.abs(array - reference).max(axis=2)
    mask = (distance > float(tolerance)).astype(np.uint8)

    if morph > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Rellenar huecos interiores (flood fill desde el borde sobre el complemento).
    height, width = mask.shape
    filled = mask.copy()
    canvas = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(filled, canvas, (0, 0), 1)
    holes = (filled == 0).astype(np.uint8)
    mask = np.clip(mask + holes, 0, 1)

    if keep_largest:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == largest).astype(np.uint8)

    return mask.astype(bool)


def feather_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Convierte la máscara en alfa [0, 1] con el borde suavizado."""
    alpha = np.asarray(mask, dtype=np.float32)
    if radius and radius > 0:
        size = 2 * int(radius) + 1
        alpha = cv2.GaussianBlur(alpha, (size, size), 0)
    return np.clip(alpha, 0.0, 1.0)


def restrict_to_front(mask: np.ndarray, fraction: float = 1.0) -> np.ndarray:
    """Limita la máscara a una banda central (aproximación del panel frontal).

    ``fraction=1.0`` no toca nada; ``0.5`` conserva el 50 % central del ancho del
    recuadro de la prenda. Es una aproximación geométrica, no segmentación de caras.
    """
    if fraction >= 1.0 or not mask.any():
        return mask
    fraction = float(np.clip(fraction, 0.05, 1.0))
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    left, right = int(cols.min()), int(cols.max()) + 1
    width = right - left
    new_left = left + int(round(width * (1.0 - fraction) / 2.0))
    new_right = right - int(round(width * (1.0 - fraction) / 2.0))
    limited = np.zeros_like(mask)
    limited[rows.min() : rows.max() + 1, new_left:new_right] = mask[rows.min() : rows.max() + 1, new_left:new_right]
    return limited


def _remove_lighting_ramp(tile: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Quita la rampa de iluminación (un plano por canal) sin borrar el estampado.

    Un desenfoque gaussiano ancho eliminaría también los patrones grandes (dos
    tonos, cuadros anchos); ajustar un plano por mínimos cuadrados quita solo la
    tendencia global de luz con la que se fotografió la tela.
    """
    height, width = tile.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx.ravel().astype(np.float64) / max(width - 1, 1) - 0.5
    y = yy.ravel().astype(np.float64) / max(height - 1, 1) - 0.5
    design = np.stack([np.ones_like(x), x, y], axis=1)
    flat = tile.reshape(-1, 3).astype(np.float64)
    coefficients, *_ = np.linalg.lstsq(design, flat, rcond=None)
    plane = design @ coefficients
    corrected = flat - float(strength) * (plane - plane.mean(axis=0, keepdims=True))
    return np.clip(corrected.reshape(tile.shape), 0, 255).astype(np.float32)


def prepare_fabric_tile(
    fabric: np.ndarray,
    angle: float = 0.0,
    flatten: bool = True,
    brightness: float = 1.0,
) -> np.ndarray:
    """Endereza la tela, le quita su iluminación y ajusta el brillo.

    ``flatten`` elimina la rampa de luz de la foto (ver :func:`_remove_lighting_ramp`)
    para que al aplicarla no se dupliquen sombras sobre la prenda.
    """
    tile = to_float(fabric)
    if abs(angle) > 1e-3:
        height, width = tile.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        tile = cv2.warpAffine(
            tile,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    if flatten:
        tile = _remove_lighting_ramp(tile)

    if abs(brightness - 1.0) > 1e-3:
        tile = tile * float(brightness)

    return np.clip(tile, 0, 255)


def flat_color_tile(color: tuple[int, int, int], size: int = 64) -> np.ndarray:
    """Tela uniforme a partir de un color (para cambiar solo el color)."""
    tile = np.zeros((size, size, 3), np.float32)
    tile[..., :] = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return tile


def _build_mosaic(tile: np.ndarray, canvas: tuple[int, int], extra: int, mirrored: bool) -> np.ndarray:
    """Mosaico que cubre el lienzo (con margen) repitiendo o reflejando la tela."""
    height, width = canvas
    ny = int(np.ceil(height / tile.shape[0])) + extra
    nx = int(np.ceil(width / tile.shape[1])) + extra
    rows = []
    for row in range(ny):
        columns = []
        for column in range(nx):
            piece = tile
            if mirrored:
                piece = piece[::-1] if row % 2 else piece
                piece = piece[:, ::-1] if column % 2 else piece
            columns.append(piece)
        rows.append(np.concatenate(columns, axis=1))
    return np.concatenate(rows, axis=0)


def tile_to_canvas(
    tile: np.ndarray,
    shape: tuple[int, int],
    *,
    scale: float = 1.0,
    rotation: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
    perspective: float = 0.0,
    mosaic: str = "repetir",
    reference_width: float | None = None,
) -> np.ndarray:
    """Coloca la tela como mosaico infinito en el lienzo, con ángulo y profundidad.

    Args:
        tile: tela ya preparada (float 0-255).
        shape: ``(alto, ancho)`` del lienzo (la foto de la prenda).
        scale: 1.0 = la tela cubre el ancho de la prenda; >1 estampado más pequeño.
        rotation: giro del mosaico en grados.
        tilt_x, tilt_y: inclinación (-1..1) del plano de la tela.
        offset: desplazamiento del mosaico en fracción del lienzo.
        perspective: 0 (plano) a 0.8 (profundidad acentuada, «z»).
        mosaic: ``repetir`` (borde a borde) o ``espejo`` (sin costuras visibles).
        reference_width: ancho de referencia (por defecto, el del lienzo).

    Returns:
        Patrón ``HxWx3`` float listo para :func:`apply_fabric`.
    """
    height, width = int(shape[0]), int(shape[1])
    if reference_width is not None:
        reference = float(reference_width)
    else:
        reference = float(width) / max(float(scale), 1e-3)
    factor = reference / tile.shape[1]
    resized = cv2.resize(
        tile,
        (max(8, int(round(tile.shape[1] * factor))), max(8, int(round(tile.shape[0] * factor)))),
        interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR,
    )
    block = _build_mosaic(resized, (height, width), extra=2, mirrored=(mosaic == "espejo"))

    # Centro de la tela (punto de referencia) y destino en el lienzo.
    cx, cy = resized.shape[1] / 2.0, resized.shape[0] / 2.0
    target = (width / 2.0 + float(offset[0]) * width, height / 2.0 + float(offset[1]) * height)

    radians = np.deg2rad(float(rotation))
    cos, sin = float(np.cos(radians)), float(np.sin(radians))
    # Rotación alrededor del centro de la tela + colocación en el destino.
    rotate = np.array(
        [
            [cos, -sin, target[0] - cos * cx + sin * cy],
            [sin, cos, target[1] - sin * cx - cos * cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Cizalla (inclinación del plano).
    shear = np.array([[1.0, float(tilt_x), 0.0], [float(tilt_y), 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    # Profundidad: la coordenada w crece con y → el estampado se acorta al fondo.
    depth = max(0.0, min(float(perspective), 0.8)) * 2.0 / max(height, 1)
    project = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, depth, 1.0]], dtype=np.float64)

    matrix = rotate @ shear @ project
    return cv2.warpPerspective(
        block,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def apply_fabric(
    garment: np.ndarray,
    pattern: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 1.0,
    shading: float = 1.0,
    detail: float = 0.0,
    feather: int = 2,
) -> np.ndarray:
    """Sustituye color/estampado dentro de la máscara conservando las sombras.

    - ``shading``: cuánta luminancia de la prenda se conserva (1 = toda → mantiene
      pliegues y volumen; 0 = se queda con la luz de la tela).
    - ``strength``: mezcla con la prenda original (1 = solo tela).
    - ``detail``: cuánto detalle de la prenda se reinyecta (costuras, botones).

    El brillo medio del mosaico se normaliza al brillo medio de la prenda (para no
    dar saltos de luminosidad); si la tela queda oscura o clara de más, se corrige
    con ``FabricTransferConfig.brightness``.
    """
    base = to_float(garment)
    tile = to_float(pattern)
    if not np.asarray(mask).any():
        return base

    alpha = feather_mask(mask, feather)[..., None]
    luminance = _luminance(base)
    reference = float(luminance[mask].mean()) or 1.0
    shade = np.clip(luminance / reference, 0.2, 2.5)

    pattern_luminance = float(_luminance(tile)[mask].mean()) or 1.0
    normalized = tile * (reference / pattern_luminance)
    shaded = normalized * (np.power(shade, float(shading)))[..., None]

    if detail > 0:
        low = cv2.GaussianBlur(base, (0, 0), 2.0)
        shaded = shaded + (base - low) * float(detail)

    mixed = (1.0 - float(strength)) * base + float(strength) * shaded
    return base * (1.0 - alpha) + mixed * alpha


def composite_background(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str = "original",
    color: tuple[int, int, int] = WHITE,
    feather: int = 1,
) -> np.ndarray:
    """Recompone el fondo: ``original``, ``blanco``, ``gris`` o ``color``."""
    base = to_float(image)
    if mode == "original":
        return base
    if mode == "blanco":
        background = WHITE
    elif mode == "gris":
        background = LIGHT_GREY
    elif mode == "color":
        background = color
    else:
        return base

    fill = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    alpha = feather_mask(mask, feather)[..., None]
    return base * alpha + fill * (1.0 - alpha)


def repeat_reference_px(config: FabricTransferConfig, canvas_width: int) -> float:
    """Ancho del motivo en píxeles del lienzo según el modo elegido.

    - ``cm``: calibra con el ancho real de la prenda (``garment_width_cm``) →
      ``ancho_lienzo * repeat_cm / garment_width_cm``. Ej.: prenda de 50 cm y motivo
      de 12 cm ⇒ el motivo ocupa el 24 % del ancho y el patrón se repite ~4,2 veces.
    - ``px``: valor directo en píxeles del lienzo.
    - ``scale``: relativo (``ancho_lienzo / scale``).

    Returns:
        Píxeles de ancho que debe ocupar una repetición del motivo (≥ 8).
    """
    if config.repeat_mode == "cm" and config.repeat_cm > 0 and config.garment_width_cm > 0:
        reference = canvas_width * (config.repeat_cm / config.garment_width_cm)
    elif config.repeat_mode == "px" and config.repeat_px and config.repeat_px > 0:
        reference = float(config.repeat_px)
    else:
        reference = canvas_width / max(float(config.scale), 1e-3)
    return float(max(reference, 8.0))


def retexture_garment(
    garment: np.ndarray,
    fabric: np.ndarray | None = None,
    config: FabricTransferConfig | None = None,
    garment_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, FabricTransferInfo]:
    """Punto de entrada: aplica la tela a la prenda y devuelve el resultado.

    Args:
        garment: foto de la prenda (RGB, uint8 o float).
        fabric: foto de la tela (opcional; si falta se usa ``config.flat_color``).
        config: parámetros (con ``enabled=False`` no se toca la imagen).
        garment_mask: máscara de la prenda ya calculada (p. ej. de un proveedor de
            segmentación); si es None se estima desde el fondo claro.

    Returns:
        ``(imagen uint8 RGB, máscara usada, info)``.
    """
    config = config or FabricTransferConfig()
    image = to_float(garment)
    shape = image.shape[:2]
    info = FabricTransferInfo()

    if not config.enabled:
        info.reason = "desactivado"
        mask = np.ones(shape, dtype=bool) if garment_mask is None else np.asarray(garment_mask).astype(bool)
        return to_uint8(image), mask, info

    # 1) Máscara de la prenda.
    if garment_mask is not None:
        mask = np.asarray(garment_mask).astype(bool)
        info.mask_source = "proporcionada"
    elif config.mask_source == "tela-completa":
        mask = np.ones(shape, dtype=bool)
        info.mask_source = "tela-completa"
    else:
        background, uniformity = detect_background_color(image)
        info.background = tuple(int(value) for value in background)
        info.background_uniformity = round(uniformity, 2)
        if uniformity > 12:
            info.notes.append("el fondo no parece uniforme; sube la tolerancia o usa «tela completa»")
        mask = garment_mask_from_background(image, background, config.background_tolerance)

    mask = restrict_to_front(mask, config.front_only)
    info.coverage = float(np.asarray(mask).mean())
    if info.coverage < 0.01:
        info.reason = "la máscara de la prenda quedó vacía (revisa la tolerancia de fondo)"
        return to_uint8(image), np.asarray(mask), info

    # 2) Tela: imagen subida o color plano.
    if fabric is not None:
        tile = prepare_fabric_tile(fabric, config.fabric_angle, config.flatten, config.brightness)
        info.fabric = "imagen"
    elif config.flat_color is not None:
        color = tuple(int(value) for value in config.flat_color)
        tile = flat_color_tile(color)
        info.fabric = f"color plano {color}"
    else:
        info.reason = "no se subió tela ni se eligió color"
        return to_uint8(image), np.asarray(mask), info

    # 3) Mosaico, sustitución y fondo.
    reference = repeat_reference_px(config, shape[1])
    info.repeat_px = round(reference, 1)
    info.repeats = round(float(shape[1]) / reference, 2)
    if config.repeat_mode == "cm" and config.garment_width_cm > 0:
        info.repeat_cm = round(reference / float(shape[1]) * float(config.garment_width_cm), 2)
    pattern = tile_to_canvas(
        tile,
        shape,
        scale=config.scale,
        rotation=config.rotation,
        tilt_x=config.tilt_x,
        tilt_y=config.tilt_y,
        offset=(config.offset_x, config.offset_y),
        perspective=config.perspective,
        mosaic=config.mosaic,
        reference_width=reference,
    )
    result = apply_fabric(
        image,
        pattern,
        mask,
        strength=config.strength,
        shading=config.shading,
        detail=config.detail,
        feather=config.mask_feather,
    )
    result = composite_background(
        result,
        mask,
        mode=config.background,
        color=config.background_color,
        feather=config.mask_feather,
    )
    info.applied = True
    return to_uint8(result), np.asarray(mask), info


__all__ = [
    "BACKGROUND_MODES",
    "MASK_SOURCES",
    "MOSAIC_MODES",
    "LIGHT_GREY",
    "WHITE",
    "FabricTransferConfig",
    "FabricTransferInfo",
    "apply_fabric",
    "composite_background",
    "detect_background_color",
    "feather_mask",
    "flat_color_tile",
    "garment_mask_from_background",
    "prepare_fabric_tile",
    "repeat_reference_px",
    "restrict_to_front",
    "retexture_garment",
    "tile_to_canvas",
    "to_float",
    "to_uint8",
]
