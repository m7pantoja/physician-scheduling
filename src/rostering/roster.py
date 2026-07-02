"""Representación de una solución: el roster x_{its} como conjunto disperso de asignaciones."""

from pydantic import BaseModel, Field


class Roster(BaseModel):
    """Una solución del problema: el conjunto de asignaciones (médico, día, turno) con x=1.

    Espejo directo de la matriz binaria x_{its}: (i, t, s) en `assignments` <=> x_{its}=1.
    Es DISPERSO (guarda solo los unos) y MUTABLE (los solvers lo editan asignando y
    desasignando turnos). Deliberadamente tonto: la consulta agregada (horas por médico,
    cobertura por turno) y la evaluación viven en el evaluador, que indexa en una pasada;
    aquí solo están los datos y las operaciones atómicas.

    Un médico puede tener MÁS de un turno el mismo día (p.ej. {M, G1} = mañana encadenada a
    guardia): el no-solape (conjunto O) es una hard constraint, no una propiedad de
    esta estructura, así que `assignments` admite varias tripletas con el mismo (i, t)."""

    assignments: set[tuple[int, int, str]] = Field(default_factory=set)  # (id_médico i, día t, nombre_turno s)

    def assign(self, physician_id: int, day: int, shift_name: str) -> None:
        """Pone x_{its}=1 (idempotente)."""
        self.assignments.add((physician_id, day, shift_name))

    def unassign(self, physician_id: int, day: int, shift_name: str) -> None:
        """Pone x_{its}=0 (idempotente: no falla si no estaba asignado)."""
        self.assignments.discard((physician_id, day, shift_name))

    def is_assigned(self, physician_id: int, day: int, shift_name: str) -> bool:
        """¿x_{its}=1?"""
        return (physician_id, day, shift_name) in self.assignments
