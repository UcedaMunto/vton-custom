"""Comparación de métricas contra un baseline: guardia anti-regresión.

Pieza central del plan de entrenamiento: **nada se promueve si empeora**. Compara
las métricas de un candidato (fine-tuning, nuevo proveedor, cambio de versión de
torch…) contra las de un baseline congelado y decide si hay regresión.

Sin estado, sin GPU y sin dependencias: solo números, para que se pueda probar y
reutilizar desde el script (``scripts/baseline.py``), la API o la UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Métricas donde más es mejor.
HIGHER_IS_BETTER = (
    "identity_preservation",
    "color_fidelity",
    "pattern_fidelity",
    "garment_change",
    "sharpness",
    "score",
)

#: Métricas donde menos es mejor.
LOWER_IS_BETTER = ("outside_change", "failure_rate")

#: Métricas comparadas por defecto.
TRACKED_METRICS = HIGHER_IS_BETTER + LOWER_IS_BETTER


def _value(metrics: dict | None, name: str) -> float | None:
    """Valor numérico finito de ``metrics[name]`` (None si falta o es nan)."""
    if not metrics:
        return None
    value = metrics.get(name)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class MetricDelta:
    """Diferencia de una métrica entre baseline y candidato."""

    name: str
    baseline: float | None
    candidate: float | None
    relative: float | None = None
    ok: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "relative": self.relative,
            "ok": self.ok,
            "reason": self.reason,
        }


@dataclass
class RegressionReport:
    """Resultado de la comparación (``ok`` == no hay regresión)."""

    rows: list[MetricDelta] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons

    @property
    def improved(self) -> list[str]:
        """Métricas que mejoraron de forma apreciable (relative > 0.5 %)."""
        return [row.name for row in self.rows if row.ok and row.relative is not None and row.relative > 0.005]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reasons": self.reasons,
            "missing": self.missing,
            "improved": self.improved,
            "rows": [row.to_dict() for row in self.rows],
        }


def compare_metrics(
    baseline_metrics: dict | None,
    candidate_metrics: dict | None,
    max_regression: float = 0.02,
    min_absolute: float = 1e-6,
    tracked: tuple[str, ...] = TRACKED_METRICS,
) -> RegressionReport:
    """Compara dos diccionarios de métricas.

    Args:
        baseline_metrics: métricas del baseline congelado.
        candidate_metrics: métricas del candidato (pesos nuevos).
        max_regression: regresión relativa tolerada (0.02 = 2 %).
        min_absolute: tolerancia absoluta mínima (para valores pequeños).
        tracked: métricas a comparar; por defecto :data:`TRACKED_METRICS`.

    Returns:
        :class:`RegressionReport` con una fila por métrica y los motivos de fallo.
        ``ok`` es True solo si ninguna métrica comparada empeora por encima de la
        tolerancia y el candidato no tiene fallos.
    """
    report = RegressionReport()

    if _value(candidate_metrics, "failure_rate") or (candidate_metrics or {}).get("failure"):
        report.reasons.append("el candidato tiene corridas fallidas")

    for name in tracked:
        baseline = _value(baseline_metrics, name)
        candidate = _value(candidate_metrics, name)
        row = MetricDelta(name=name, baseline=baseline, candidate=candidate)

        if baseline is None:
            report.missing.append(name)
            report.rows.append(row)
            continue
        if candidate is None:
            row.ok = False
            row.reason = "sin dato en el candidato"
            report.reasons.append(f"{name}: sin dato en el candidato")
            report.rows.append(row)
            continue

        tolerance = max(abs(baseline) * max_regression, min_absolute)
        row.relative = (candidate - baseline) / abs(baseline) if baseline else None

        if name in LOWER_IS_BETTER:
            if candidate > baseline + tolerance:
                row.ok = False
                row.reason = f"empeora: {candidate:.4f} > {baseline:.4f} (+{tolerance:.4f})"
        else:
            if candidate < baseline - tolerance:
                row.ok = False
                row.reason = f"empeora: {candidate:.4f} < {baseline:.4f} (-{tolerance:.4f})"

        if not row.ok:
            report.reasons.append(f"{name}: {row.reason}")
        report.rows.append(row)

    return report


__all__ = [
    "HIGHER_IS_BETTER",
    "LOWER_IS_BETTER",
    "TRACKED_METRICS",
    "MetricDelta",
    "RegressionReport",
    "compare_metrics",
]
