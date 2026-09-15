"""Commercial compliance guard for the FASHN VTON fork.

This module is the code counterpart of ``licenses/manifest.json`` and
``THIRD_PARTY_NOTICES.md``: it detects, automatically, whether any
non-permissive dependency or artifact has sneaked back into the fork.

Checks
------
- ``scan_source_tree``: no Python file imports/uses ``fashn_human_parser`` (or
  SegFormer) again.
- ``check_pyproject_dependencies``: the declared dependencies do not contain a
  forbidden distribution.
- ``find_forbidden_files``: no file under a root (e.g. ``weights/`` or a
  HuggingFace cache) matches the ``match`` globs of the manifest.
- ``check_installed_distributions``: the forbidden distributions are not
  installed in the current environment.
- ``verify_manifest_hashes``: the recorded sha256 of every local artifact matches.

``commercial_report()`` aggregates everything and ``assert_commercial_ready()``
raises when something is wrong, so the same module can be used by tests, by
``scripts/verify_commercial_readiness.py`` and by the API (``/healthz``).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    tomllib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "licenses" / "manifest.json"

#: Patterns that must never appear as *code* again (the fork removed them).
FORBIDDEN_CODE_PATTERNS: dict[str, re.Pattern[str]] = {
    "import de fashn_human_parser": re.compile(r"^\s*(?:from|import)\s+fashn_human_parser\b", re.MULTILINE),
    "uso de FashnHumanParser": re.compile(r"\bFashnHumanParser\s*\("),
    "import de segformer": re.compile(r"^\s*(?:from|import)\s+\w*segformer\w*\b", re.MULTILINE | re.IGNORECASE),
}

#: Files that legitimately contain the forbidden *names* (definition of the
#: checks themselves and the tests that prove they work).
SOURCE_SCAN_ALLOWLIST = {
    "src/fashn_vton/compliance.py",
    "tests/test_compliance.py",
}

#: Substrings that must never appear in pyproject dependencies.
FORBIDDEN_DEPENDENCY_SUBSTRINGS = ("fashn-human-parser", "fashn_human_parser", "segformer")

#: Source/text extensions scanned by ``scan_source_tree``.
SCANNED_SUFFIXES = (".py",)
SKIPPED_DIRS = {".git", "__pycache__", "weights", "outputs", "logs", ".pytest_cache", ".venv", "node_modules"}

_HASH_CHUNK = 1024 * 1024


class ComplianceError(RuntimeError):
    """The commercial compliance gate failed."""


@dataclass
class ComplianceReport:
    """Result of the compliance checks (``ok`` == no violations)."""

    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "violations": self.violations,
            "warnings": self.warnings,
            "checked": self.checked,
        }


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict:
    """Read the third-party manifest (raises if it is missing/corrupt)."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def forbidden_artifacts(manifest: dict | None = None) -> list[dict]:
    return list((manifest or load_manifest()).get("forbidden_artifacts", []))


def sha256_of_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_source_tree(
    root: str | Path = REPO_ROOT,
    allowlist: set[str] | None = None,
    suffixes: tuple[str, ...] = SCANNED_SUFFIXES,
) -> list[str]:
    """Return violations for forbidden imports/usages found under ``root``."""
    root = Path(root)
    allowed = SOURCE_SCAN_ALLOWLIST if allowlist is None else allowlist
    violations: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part in SKIPPED_DIRS for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        if relative in allowed:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - unreadable file
            continue
        for label, pattern in FORBIDDEN_CODE_PATTERNS.items():
            for match in pattern.finditer(content):
                line = content.count("\n", 0, match.start()) + 1
                violations.append(f"{relative}:{line}: {label}")
    return violations


def check_pyproject_dependencies(path: str | Path = REPO_ROOT / "pyproject.toml") -> list[str]:
    """Return violations for forbidden dependency names in ``pyproject.toml``."""
    path = Path(path)
    if not path.is_file():
        return [f"no existe {path}"]

    if tomllib is None:  # Python 3.10 fallback: scan the dependency strings directly
        lowered = path.read_text(encoding="utf-8", errors="ignore").lower()
        return [
            f"pyproject.toml contiene una dependencia prohibida ('{needle}')"
            for needle in FORBIDDEN_DEPENDENCY_SUBSTRINGS
            if f'"{needle}' in lowered or f"'{needle}" in lowered
        ]

    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project", {})
    declared = list(project.get("dependencies", []))
    for extra in (project.get("optional-dependencies") or {}).values():
        declared.extend(extra)
    violations = []
    for requirement in declared:
        lowered = requirement.lower()
        for needle in FORBIDDEN_DEPENDENCY_SUBSTRINGS:
            if needle in lowered:
                violations.append(f"pyproject.toml declara una dependencia prohibida: {requirement}")
    return violations


def find_forbidden_files(root: str | Path, manifest: dict | None = None) -> list[str]:
    """Return matches of the manifest's ``forbidden_artifacts`` globs under ``root``."""
    root = Path(root)
    artifacts = forbidden_artifacts(manifest)
    found: list[str] = []
    if not root.exists():
        return found
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        for artifact in artifacts:
            for pattern in artifact.get("match", []):
                candidates = {pattern, pattern.replace("**/", ""), f"**/{pattern.lstrip('*/')}"}
                if any(fnmatch.fnmatch(relative, candidate) for candidate in candidates) or any(
                    fnmatch.fnmatch(path.name.lower(), candidate.lstrip("*").lower())
                    for candidate in candidates
                ):
                    found.append(f"{relative} (prohibido: {artifact.get('name')})")
                    break
    return sorted(set(found))


def installed_distributions() -> list[str]:
    """Names of the distributions installed in the current environment."""
    from importlib import metadata

    names = set()
    for dist in metadata.distributions():
        name = (dist.metadata or {}).get("Name")
        if name:
            names.add(name)
    return sorted(names)


def check_installed_distributions(manifest: dict | None = None) -> list[str]:
    """Return violations for forbidden distributions installed in the env."""
    installed = {name.lower() for name in installed_distributions()}
    violations = []
    for needle in FORBIDDEN_DEPENDENCY_SUBSTRINGS:
        for name in installed:
            if needle in name:
                violations.append(f"distribución instalada prohibida: {name}")
    return sorted(set(violations))


def verify_manifest_hashes(root: str | Path = REPO_ROOT, manifest: dict | None = None) -> list[dict]:
    """Verify the recorded sha256 of every artifact that has a ``local_path``."""
    root = Path(root)
    manifest = manifest or load_manifest()
    results = []
    for artifact in manifest.get("artifacts", []):
        relative = artifact.get("local_path")
        if not relative:
            results.append({"name": artifact.get("name"), "status": "NO_PATH", "detail": "librería"})
            continue
        path = root / relative
        if not path.is_file():
            results.append({"name": artifact.get("name"), "status": "MISSING", "detail": str(path)})
            continue
        actual = sha256_of_file(path)
        expected = artifact.get("sha256")
        if not expected:
            results.append({"name": artifact.get("name"), "status": "UNVERIFIED", "detail": actual})
        elif actual.lower() == str(expected).lower():
            results.append({"name": artifact.get("name"), "status": "OK", "detail": actual})
        else:
            results.append(
                {
                    "name": artifact.get("name"),
                    "status": "MISMATCH",
                    "detail": f"esperado={expected} obtenido={actual}",
                }
            )
    return results


def commercial_report(
    root: str | Path = REPO_ROOT,
    manifest_path: str | Path = MANIFEST_PATH,
    check_dependencies: bool = True,
    check_environment: bool = True,
) -> ComplianceReport:
    """Run every compliance check and aggregate the result."""
    root = Path(root)
    manifest = load_manifest(manifest_path)
    report = ComplianceReport()

    source_violations = scan_source_tree(root)
    report.violations.extend(source_violations)
    report.checked["source_scan"] = {"violations": source_violations, "root": str(root)}

    pyproject_violations = check_pyproject_dependencies(root / "pyproject.toml")
    report.violations.extend(pyproject_violations)
    report.checked["pyproject"] = {"violations": pyproject_violations}

    if check_environment:
        env_violations = check_installed_distributions(manifest)
        report.violations.extend(env_violations)
        report.checked["environment"] = {"violations": env_violations}

    if check_dependencies:
        file_violations = find_forbidden_files(root, manifest)
        report.violations.extend(file_violations)
        report.checked["forbidden_files"] = {"violations": file_violations, "root": str(root)}

    hashes = verify_manifest_hashes(root, manifest)
    report.checked["hashes"] = hashes
    for row in hashes:
        if row["status"] == "MISMATCH":
            report.violations.append(f"hash incorrecto en {row['name']}: {row['detail']}")
        elif row["status"] == "MISSING":
            report.warnings.append(f"artefacto no descargado: {row['name']} ({row['detail']})")
        elif row["status"] == "UNVERIFIED":
            report.warnings.append(f"artefacto sin hash registrado: {row['name']}")

    report.checked["forbidden_artifacts"] = [item.get("name") for item in forbidden_artifacts(manifest)]
    return report


def assert_commercial_ready(root: str | Path = REPO_ROOT, **kwargs) -> ComplianceReport:
    """Raise :class:`ComplianceError` when the fork is not commercial-ready."""
    report = commercial_report(root, **kwargs)
    if not report.ok:
        raise ComplianceError("Compliance comercial NO superado:\n  - " + "\n  - ".join(report.violations))
    return report


__all__ = [
    "ComplianceError",
    "ComplianceReport",
    "FORBIDDEN_CODE_PATTERNS",
    "FORBIDDEN_DEPENDENCY_SUBSTRINGS",
    "MANIFEST_PATH",
    "REPO_ROOT",
    "assert_commercial_ready",
    "check_installed_distributions",
    "check_pyproject_dependencies",
    "commercial_report",
    "find_forbidden_files",
    "forbidden_artifacts",
    "installed_distributions",
    "load_manifest",
    "scan_source_tree",
    "sha256_of_file",
    "verify_manifest_hashes",
]
