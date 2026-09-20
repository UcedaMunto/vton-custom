"""Política de datos con licencia **no comercial** (guardia de entrenamiento).

Contexto: el fork comercial (`README_COMERCIAL.md`, `licenses/manifest.json`)
prohíbe VITON-HD y DressCode en el camino comercial. Este módulo añade el carril
de **investigación**: permite entrenar con esos datasets locales de forma
explícita, aislada y trazable, y hace lo posible por que un candidato entrenado
así **no pueda** acabar siendo el modelo activo por accidente.

Dos funciones, cada una cubriendo un lado del problema:

- :func:`assert_nc_training_allowed` — puerta de entrada. Entrenar con datos NC
  exige un *opt-in* explícito (variable de entorno `FASHN_ALLOW_NC_TRAINING=1` o
  `--allow-nc` en el CLI) y que los datos vivan **fuera** del árbol del repo, de
  modo que nunca se cuelen en una imagen Docker ni en un artefacto publicado.
- :func:`assert_commercially_promotable` — puerta de salida. Promover pesos
  exige que la `provenance.json` del candidato declare uso comercial. Los pesos
  NC solo se pueden promover pasando `--allow-nc`, que imprime un aviso legal.

Nada de esto es asesoría legal: es ingeniería defensiva (que el camino cómodo
sea el correcto).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

#: Variable de entorno que actúa como opt-in explícito.
NC_TRAINING_ENV_FLAG = "FASHN_ALLOW_NC_TRAINING"

#: Raíz del repositorio (para el guardia de "datos fuera del repo").
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Clases de licencia consideradas comerciales (todo lo demás se trata como NC).
COMMERCIAL_LICENSE_CLASSES = frozenset(
    {
        "commercial",
        "comercial",
        "propia",
        "propio",
        "proprietary",
        "apache-2.0",
        "mit",
        "cc0",
        "cc-by",
        "cc-by-4.0",
    }
)

#: Clases de licencia NC conocidas (informativas; la detección es por lista blanca).
KNOWN_NC_LICENSE_CLASSES = frozenset({"nc-dresscode", "nc-vitonhd", "nc-sintetico", "nc"})

PROVENANCE_FILENAME = "provenance.json"


class NcTrainingNotAllowed(RuntimeError):
    """Se intentó entrenar con datos NC sin el *opt-in* explícito."""


class NcDataInsideRepo(RuntimeError):
    """El dataset NC vive dentro del árbol del repositorio (riesgo de publicación)."""


class NonCommercialCandidateError(RuntimeError):
    """Se intentó promover a producción un candidato entrenado con datos NC."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_nc_license(license_class: str | None) -> bool:
    """¿La clase de licencia implica uso no comercial?

    Se es conservador: cualquier valor que no esté en la lista blanca comercial
    se considera NC (`desconocida`, `nc-*`, `non-commercial`, `research`...).
    """
    normalized = (license_class or "").strip().lower()
    if not normalized:
        return True
    return normalized not in COMMERCIAL_LICENSE_CLASSES


def assert_dataset_outside_repo(dataset_root: str | Path) -> Path:
    """Falla si el dataset está dentro del repo (podría empaquetarse/publicarse)."""
    resolved = Path(dataset_root).expanduser().resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved
    raise NcDataInsideRepo(
        f"El dataset '{resolved}' está dentro del repositorio ({REPO_ROOT}). Los datos con "
        "licencia NC deben quedar fuera del árbol del repo para que no entren en artefactos "
        "publicables (imagen Docker, wheels, etc.). Muévelos o usa una ruta externa."
    )


def assert_nc_training_allowed(
    *,
    license_class: str,
    dataset_root: str | Path | None = None,
    pairs_csv: str | Path | None = None,
    allow_nc: bool = False,
    env: dict | None = None,
    flag: str = NC_TRAINING_ENV_FLAG,
) -> dict:
    """Autoriza (o bloquea) un entrenamiento según la licencia de los datos.

    Devuelve un dict de contexto, que se reutiliza para la `provenance.json`.
    """
    environment = os.environ if env is None else env
    env_opt_in = str(environment.get(flag, "")).strip().lower() in {"1", "true", "yes", "si", "sí"}
    nc = is_nc_license(license_class)

    if nc and not (allow_nc or env_opt_in):
        raise NcTrainingNotAllowed(
            f"La licencia '{license_class or 'desconocida'}' es NO COMERCIAL. Este arnés no "
            "entrena con esos datos por defecto porque el modelo resultante queda contaminado "
            "para el camino comercial (ver plan_entrenamiento/GUIA_EJECUCION_ENTRENAMIENTO.md).\n"
            f"Si es una prueba de investigación deliberada: export {flag}=1 (o pasa --allow-nc).\n"
            "El candidato resultante se marcará como no comercial y NO se podrá promover."
        )

    resolved_root = None
    if dataset_root is not None:
        resolved_root = assert_dataset_outside_repo(dataset_root)

    return {
        "license_class": license_class or "desconocida",
        "non_commercial": bool(nc),
        "commercial_use": not nc,
        "opt_in": "flag-cli" if allow_nc else ("env" if env_opt_in else ("no-aplica" if not nc else "no")),
        "dataset_root": str(resolved_root) if resolved_root else None,
        "pairs_csv": str(Path(pairs_csv).expanduser().resolve()) if pairs_csv else None,
        "flag": flag,
        "checked_at": _now(),
    }


def build_provenance(
    *,
    dataset: str,
    license_class: str,
    commercial_use: bool,
    dataset_root: str | Path | None = None,
    pairs_csv: str | Path | None = None,
    license_note: str = "",
    pairs: int | None = None,
    extra: dict | None = None,
) -> dict:
    """Construye el diccionario de procedencia de un candidato (o de un dataset)."""
    payload = {
        "dataset": dataset,
        "license_class": license_class,
        "commercial_use": bool(commercial_use),
        "dataset_root": str(Path(dataset_root).expanduser().resolve()) if dataset_root else None,
        "pairs_csv": str(Path(pairs_csv).expanduser().resolve()) if pairs_csv else None,
        "pairs": pairs,
        "license_note": license_note,
        "created_at": _now(),
    }
    if extra:
        payload.update(extra)
    return payload


def write_provenance(path_or_dir: str | Path, provenance: dict) -> Path:
    """Escribe `provenance.json` (acepta el directorio o la ruta del archivo)."""
    path = Path(path_or_dir)
    if path.is_dir() or not path.suffix:
        path = path / PROVENANCE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_provenance(path_or_dir: str | Path) -> dict | None:
    """Lee `provenance.json` de un directorio (o de la ruta directa), o `None`."""
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / PROVENANCE_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def assert_commercially_promotable(candidate_dir: str | Path, allow_nc: bool = False) -> dict | None:
    """Impide promover pesos derivados de datos NC (salvo `allow_nc` explícito)."""
    provenance = read_provenance(candidate_dir)
    if provenance is None:
        # Sin procedencia no se puede afirmar que sea comercial: se avisa, no se bloquea.
        return {"dataset": "desconocido", "license_class": "desconocida", "commercial_use": None}
    if provenance.get("commercial_use") is False and not allow_nc:
        raise NonCommercialCandidateError(
            f"El candidato '{candidate_dir}' se entrenó con datos NO COMERCIALES "
            f"(dataset='{provenance.get('dataset')}', licencia='{provenance.get('license_class')}').\n"
            "Promoverlo como modelo activo contamina el camino comercial del fork. Si de verdad "
            "quieres hacerlo (uso interno de investigación), repite con --allow-nc y asume el "
            "riesgo legal; para experimentar usa --weights-dir apuntando al candidato."
        )
    return provenance


def provenance_banner(provenance: dict | None) -> str:
    """Texto legible de la procedencia, para logs y para el informe del candidato."""
    if not provenance:
        return "procedencia: desconocida (sin provenance.json)"
    label = "NO COMERCIAL" if provenance.get("commercial_use") is False else "uso comercial"
    return (
        f"procedencia: dataset='{provenance.get('dataset')}' · licencia="
        f"'{provenance.get('license_class')}' · {label}"
    )


__all__ = [
    "COMMERCIAL_LICENSE_CLASSES",
    "KNOWN_NC_LICENSE_CLASSES",
    "NC_TRAINING_ENV_FLAG",
    "NcDataInsideRepo",
    "NcTrainingNotAllowed",
    "NonCommercialCandidateError",
    "PROVENANCE_FILENAME",
    "REPO_ROOT",
    "assert_commercially_promotable",
    "assert_dataset_outside_repo",
    "assert_nc_training_allowed",
    "build_provenance",
    "is_nc_license",
    "provenance_banner",
    "read_provenance",
    "write_provenance",
]

