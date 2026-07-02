"""Persistencia de instancias: guardar y cargar desde JSON en disco"""

from pathlib import Path

from rostering.domain import Instance


def save(instance: Instance, path: str | Path) -> None:
    """Serializa la instancia a JSON indentado y la escribe en path"""

    text = instance.model_dump_json(indent=2)
    Path(path).write_text(text, encoding="utf-8")

def load(path: str | Path) -> Instance:
    """Lee un JSON de disco y lo valida como Instance"""

    text = Path(path).read_text(encoding="utf-8")
    return Instance.model_validate_json(text)


