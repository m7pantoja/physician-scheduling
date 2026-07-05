"""Espacio de trabajo en disco: biblioteca de instancias y soluciones de la app.

Las instancias se guardan en el JSON canónico del paquete (`rostering.storage`), de modo
que cualquier archivo del workspace es directamente legible desde código sin pasar por la
app. Cada instancia lleva una ficha sidecar (`<nombre>.meta.json`) con su procedencia y,
si viene del generador, el blueprint `Settings` empleado (reproducibilidad). Las
soluciones son registros propios de la app: el roster disperso más la configuración y las
métricas medidas al resolver.

Estructura en disco (ignorada por git):

    data/workspace/
        instances/<nombre>.json         # Instance canónica (rostering.storage)
        instances/<nombre>.meta.json    # ficha (InstanceMeta)
        solutions/<nombre>.json         # SolutionRecord
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from rostering import storage
from rostering.domain import Instance
from rostering.roster import Roster

_REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = _REPO_ROOT / "data" / "workspace"
INSTANCES_DIR = WORKSPACE / "instances"
SOLUTIONS_DIR = WORKSPACE / "solutions"


class InstanceMeta(BaseModel):
    """Ficha de una instancia guardada (sidecar del JSON canónico)."""

    name: str
    created: str                     # ISO-8601 local
    origin: str = "generada"         # generada | editada | importada
    note: str = ""
    settings: dict | None = None     # blueprint del generador, si procede


class SolutionRecord(BaseModel):
    """Solución guardada: roster + procedencia + métricas registradas al resolver.

    `params` guarda la configuración completa empleada (objetivo + solver) y `metrics`
    las medidas tomadas en el momento de resolver (z, factibilidad, tiempos y los
    diagnósticos propios de cada método)."""

    name: str
    instance_name: str
    solver: str                      # greedy | sa | milp
    created: str
    params: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    assignments: list[tuple[int, int, str]] = Field(default_factory=list)

    def roster(self) -> Roster:
        """Reconstruye el Roster disperso a partir de las ternas guardadas."""
        return Roster(assignments={tuple(a) for a in self.assignments})


def now_iso() -> str:
    """Instante actual en ISO-8601 local (sello de creación de fichas y registros)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slugify(name: str) -> str:
    """Normaliza un nombre libre a identificador de archivo: minúsculas, sin acentos,
    separadores a guion. Devuelve 'sin-nombre' si no queda nada útil."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text or "sin-nombre"


def _unique(base: str, existing: set[str]) -> str:
    """Desambigua un nombre contra los ya existentes añadiendo sufijo -2, -3, ..."""
    if base not in existing:
        return base
    k = 2
    while f"{base}-{k}" in existing:
        k += 1
    return f"{base}-{k}"


def _ensure_dirs() -> None:
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Instancias
# ---------------------------------------------------------------------------

def instance_names() -> list[str]:
    """Nombres de instancia presentes en el workspace (orden alfabético)."""
    _ensure_dirs()
    return sorted(p.stem for p in INSTANCES_DIR.glob("*.json") if not p.name.endswith(".meta.json"))


def save_instance(instance: Instance, name: str, *, origin: str = "generada",
                  note: str = "", settings: dict | None = None,
                  overwrite: bool = False) -> str:
    """Guarda la instancia y su ficha; devuelve el nombre definitivo (slug, único salvo
    overwrite)."""
    _ensure_dirs()
    slug = slugify(name)
    if not overwrite:
        slug = _unique(slug, set(instance_names()))
    storage.save(instance, INSTANCES_DIR / f"{slug}.json")
    meta = InstanceMeta(name=slug, created=now_iso(), origin=origin, note=note, settings=settings)
    (INSTANCES_DIR / f"{slug}.meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return slug


def load_instance(name: str) -> Instance:
    """Carga y valida la instancia `name` del workspace."""
    return storage.load(INSTANCES_DIR / f"{name}.json")


def instance_meta(name: str) -> InstanceMeta:
    """Ficha de la instancia; si falta el sidecar (archivo copiado a mano), se sintetiza."""
    path = INSTANCES_DIR / f"{name}.meta.json"
    if path.exists():
        return InstanceMeta.model_validate_json(path.read_text(encoding="utf-8"))
    mtime = datetime.fromtimestamp((INSTANCES_DIR / f"{name}.json").stat().st_mtime).astimezone()
    return InstanceMeta(name=name, created=mtime.isoformat(timespec="seconds"), origin="importada")


def list_instances() -> list[InstanceMeta]:
    """Fichas de todas las instancias, más reciente primero."""
    return sorted((instance_meta(n) for n in instance_names()),
                  key=lambda m: m.created, reverse=True)


def delete_instance(name: str, *, cascade: bool = True) -> None:
    """Borra la instancia (y, con `cascade`, todas sus soluciones)."""
    (INSTANCES_DIR / f"{name}.json").unlink(missing_ok=True)
    (INSTANCES_DIR / f"{name}.meta.json").unlink(missing_ok=True)
    if cascade:
        for rec in list_solutions(instance_name=name):
            delete_solution(rec.name)


def instance_json(name: str) -> str:
    """Contenido JSON canónico de la instancia (para exportar/descargar)."""
    return (INSTANCES_DIR / f"{name}.json").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Soluciones
# ---------------------------------------------------------------------------

def solution_names() -> list[str]:
    _ensure_dirs()
    return sorted(p.stem for p in SOLUTIONS_DIR.glob("*.json"))


def save_solution(record: SolutionRecord, *, overwrite: bool = False) -> str:
    """Guarda el registro de solución; devuelve el nombre definitivo."""
    _ensure_dirs()
    slug = slugify(record.name)
    if not overwrite:
        slug = _unique(slug, set(solution_names()))
    record = record.model_copy(update={"name": slug})
    (SOLUTIONS_DIR / f"{slug}.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return slug


def load_solution(name: str) -> SolutionRecord:
    path = SOLUTIONS_DIR / f"{name}.json"
    return SolutionRecord.model_validate_json(path.read_text(encoding="utf-8"))


def list_solutions(instance_name: str | None = None) -> list[SolutionRecord]:
    """Registros de solución (opcionalmente solo los de una instancia), más reciente primero."""
    records = [load_solution(n) for n in solution_names()]
    if instance_name is not None:
        records = [r for r in records if r.instance_name == instance_name]
    return sorted(records, key=lambda r: r.created, reverse=True)


def delete_solution(name: str) -> None:
    (SOLUTIONS_DIR / f"{name}.json").unlink(missing_ok=True)


def solution_json(name: str) -> str:
    """Contenido JSON del registro de solución (para exportar/descargar)."""
    return (SOLUTIONS_DIR / f"{name}.json").read_text(encoding="utf-8")


def import_instance(payload: str | bytes, name: str, *, note: str = "") -> str:
    """Valida un JSON externo como Instance y lo incorpora al workspace."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    instance = Instance.model_validate_json(payload)
    return save_instance(instance, name, origin="importada", note=note)
