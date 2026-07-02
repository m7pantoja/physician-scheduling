"""Modelo de dominio: entidades del problema de Physician Scheduling"""

from datetime import date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator


class Shift(BaseModel):
    """Tipo de turno (shift) del conjunto S"""

    model_config = ConfigDict(frozen=True)

    name: str
    start: time                     # hora de inicio del turno
    end: time                       # hora de fin del turno (puede ser del día siguiente)
    hours: float = Field(gt=0)      # d_s: horas del turno computables
    is_guardia: bool = False        # pertenece a G (guardia de presencia física)

    def clock_span(self) -> tuple[int, int]:
        """Intervalo reloj del turno en minutos desde las 00:00 del día de inicio. Un turno
        nocturno (fin <= inicio, p.ej. una guardia) termina al día siguiente: el fin se
        desplaza +24h. La guardia de 24h (inicio==fin) ocupa [inicio, inicio+24h)."""

        start = self.start.hour * 60 + self.start.minute
        end = self.end.hour * 60 + self.end.minute
        if end <= start:
            end += 24 * 60
        return (start, end)


class MedicalClass(BaseModel):
    """Clase de médico (C_r)"""

    model_config = ConfigDict(frozen=True)

    name: str
    is_resident: bool                   # si es MIR o no
    residency_year: int | None = None   # año MIR (1-5); None para FEA
    weekly_hours: float = Field(gt=0)         # theta^ord_r: jornada ordinaria contratada (base del suelo); el suelo individual es rho_i*weekly_hours
    default_max_guardias: PositiveInt | None = None   # g^max por defecto de la clase (MIR 7; FEA None -> cota trivial |T|*|G|, vacua, Sección 5.9); lo hereda un médico con max_guardias=None. g^max in Z_{>=1}; 0 guardias se modela con la exencion E, no con un tope 0

    @model_validator(mode="after")
    def _check_residency(self):
        """Coherencia: un residente tiene año 1-5; un FEA no tiene año"""
        if self.is_resident:
            if self.residency_year is None or not (1 <= self.residency_year <= 5):
                raise ValueError("un residente debe tener residency_year entre 1 y 5")
        elif self.residency_year is not None:
            raise ValueError("un FEA no debe tener residency_year")
        return self


class Physician(BaseModel):
    """Instancia de médico"""

    model_config = ConfigDict(frozen=True)

    id: int                                             # identificador del médico (i)
    medical_class_name: str                             # nombre de la clase de médico a la que pertenece
    part_time: float = Field(default=1.0, gt=0, le=1)   # factor de jornada (rho_i)
    is_exempt: bool = False                             # si pertenece a E (exento de guardias)
    max_guardias: PositiveInt | None = None             # override de g^max_i (Z_{>=1}); None = hereda el default_max_guardias de su clase. La exencion (E) la lleva is_exempt, no un tope 0


class Calendar(BaseModel):
    """Horizonte temporal del problema"""

    model_config = ConfigDict(frozen=True)

    start_date: date
    days: int = Field(gt=0)                              # número de días del horizonte
    holidays: list[date] = Field(default_factory=list)   # días festivos (T^fes)

    @model_validator(mode="after")
    def _check_holidays_in_horizon(self):
        """Todo festivo cae dentro del horizonte [start_date, start_date + days - 1]"""

        last_day = self.start_date + timedelta(days=self.days - 1)
        for h in self.holidays:
            if not (self.start_date <= h <= last_day):
                raise ValueError(f"el festivo {h} cae fuera del horizonte [{self.start_date}, {last_day}]")
        return self

    def is_weekend(self, t: int) -> bool:
        """¿El día t del horizonte (t=0 es start_date) cae en sábado o domingo? (T^fds)"""

        day = self.start_date + timedelta(days=t)
        return day.weekday() >= 5

    def is_holiday(self, t: int) -> bool:
        """¿El día t es festivo? (T^fes)"""

        return self.start_date + timedelta(days=t) in self.holidays

    def is_working_day(self, t: int) -> bool:
        """¿El día t es laborable? (ni fin de semana ni festivo)"""

        return not self.is_weekend(t) and not self.is_holiday(t)


class Regulations(BaseModel):
    """Régimen legal del que dependen las hard constraints de descanso y jornada.

    Se modela como objeto parametrizable para intercambiarlo entero (otra CCAA, o el borrador
    de 2026: guardia de 17h y techo de 45h) sin tocar el resto del modelo."""

    model_config = ConfigDict(frozen=True)

    min_rest_hours: float = Field(gt=0)          # delta: descanso mínimo entre jornadas (deriva el conjunto I)
    min_weekly_rest_hours: float = Field(gt=0)   # delta^sem: descanso semanal ininterrumpido
    weekly_hours_ceiling: float = Field(gt=0)    # theta^leg: techo legal de jornada semanal de media (base de h_i^max)


class Instance(BaseModel):
    """Instancia de problema de Physician Scheduling"""

    model_config = ConfigDict(frozen=True)   # inmutable (frozen): la instancia es dato del problema, no se muta

    schema_version: str = "1.2"
    seed: int
    calendar: Calendar
    regulations: Regulations                             # régimen legal (descansos, techo de jornada)
    classes: list[MedicalClass]
    shifts: list[Shift]
    physicians: list[Physician]
    demand: dict[int, dict[str, NonNegativeInt]] = Field(default_factory=dict)          # b_ts: {día t: {nombre_turno s: mínimo de médicos}}; b_ts in Z_{>=0}
    eligibility: dict[int, dict[str, bool]] = Field(default_factory=dict)               # eta_ts: {día t: {nombre_turno s: se ofrece}}; elegibilidad turno-día
    qualification: dict[str,dict[str,bool]] = Field(default_factory=dict)               # gamma_rs {nombre_clase: {nombre_turno: habilitado}}
    preferences: dict[int, dict[int, dict[str, float]]] = Field(default_factory=dict)   # p_its: {id_médico i: {día t: {nombre_turno s: valor}}}; >0 desea, <0 rechaza

    @model_validator(mode="after")
    def _check_references(self):
        """Integridad referencial: todo nombre referenciado existe en su registro y todo día está dentro del horizonte"""

        class_names = {c.name for c in self.classes}
        shift_names = {s.name for s in self.shifts}
        physician_ids = {p.id for p in self.physicians}

        # 0. registros sin duplicados
        if len(class_names) != len(self.classes):
            raise ValueError("Hay nombres de clase duplicados en classes")
        if len(shift_names) != len(self.shifts):
            raise ValueError("Hay nombres de turno duplicados en shifts")
        if len(physician_ids) != len(self.physicians):
            raise ValueError("Hay ids de médico duplicados en physicians")

        # 1. cada médico apunta a una clase existente
        for p in self.physicians:
            if p.medical_class_name not in class_names:
                raise ValueError(f"el médico {p.id} referencia una clase inexistente: {p.medical_class_name!r}")

        # 2. demand: día en rango + turno existente
        for day, per_shift in self.demand.items():
            if not (0 <= day < self.calendar.days):
                raise ValueError(f"demand referencia un día fuera de rango: {day}")
            for shift_name in per_shift:
                if shift_name not in shift_names:
                    raise ValueError(f"demand referencia un turno inexistente: {shift_name!r}")

        # 3. eligibility: día en rango + turno existente
        for day, per_shift in self.eligibility.items():
            if not (0 <= day < self.calendar.days):
                raise ValueError(f"eligibility referencia un día fuera de rango: {day}")
            for shift_name in per_shift:
                if shift_name not in shift_names:
                    raise ValueError(f"eligibility referencia un turno inexistente: {shift_name!r}")

        # 4. qualification: clase existente + turno existente
        for class_name, per_shift in self.qualification.items():
            if class_name not in class_names:
                raise ValueError(f"qualification referencia una clase inexistente: {class_name!r}")
            for shift_name in per_shift:
                if shift_name not in shift_names:
                    raise ValueError(f"qualification referencia un turno inexistente: {shift_name!r}")

        # 5. preferences: médico existente + día en rango + turno existente
        for phys_id, per_day in self.preferences.items():
            if phys_id not in physician_ids:
                raise ValueError(f"preferences referencia un médico inexistente: {phys_id}")
            for day, per_shift in per_day.items():
                if not (0 <= day < self.calendar.days):
                    raise ValueError(f"preferences referencia un día fuera de rango: {day}")
                for shift_name in per_shift:
                    if shift_name not in shift_names:
                        raise ValueError(f"preferences referencia un turno inexistente: {shift_name!r}")

        # 6. coherencia eta-b: no se exige cobertura de un turno que no se ofrece (b_ts=0 donde eta_ts=0)
        for day, per_shift in self.eligibility.items():
            for shift_name, offered in per_shift.items():
                if not offered and self.demand.get(day, {}).get(shift_name, 0) != 0:
                    raise ValueError(
                        f"incoherencia eta/b: el día {day} el turno {shift_name!r} no se ofrece (eta=0) pero tiene demanda > 0"
                    )

        # 7. factibilidad estructural gamma-b: toda celda con demanda exige >=1 médico de una clase que la cualifique
        present_classes = {p.medical_class_name for p in self.physicians}
        for day, per_shift in self.demand.items():
            for shift_name, b in per_shift.items():
                if b <= 0:
                    continue
                if not any(self.qualification.get(c, {}).get(shift_name) for c in present_classes):
                    raise ValueError(
                        f"infactibilidad estructural: el día {day} el turno {shift_name!r} exige cobertura "
                        f"(b={b}) pero ninguna clase presente en la plantilla lo cualifica (gamma)"
                    )

        return self

    def class_of(self, physician: Physician) -> MedicalClass:
        """Devuelve la clase del médico (resuelve la referencia por nombre)"""

        for c in self.classes:
            if c.name == physician.medical_class_name:
                return c
        raise ValueError(f"el médico {physician.id} referencia una clase inexistente: {physician.medical_class_name!r}")

    def effective_max_guardias(self, physician: Physician) -> int | None:
        """g^max efectivo: el del médico si lo fija; si es None, hereda el de su clase. El None
        de clase equivale a la cota trivial |T|*|G| (Sección 5.9), que ningún roster alcanza: por
        vacua, el chequeo se omite"""

        if physician.max_guardias is not None:
            return physician.max_guardias
        return self.class_of(physician).default_max_guardias

    def corridor(self, physician: Physician) -> tuple[float, float]:
        """Banda de jornada [h_i^min, h_i^max] del médico (eq:corredor): suelo = jornada
        ordinaria contractual de su clase; techo = tope legal; ambos prorrateados al horizonte
        (|T|/7) y modulados por rho_i."""

        weeks = self.calendar.days / 7
        rho = physician.part_time
        floor = rho * self.class_of(physician).weekly_hours * weeks
        ceiling = rho * self.regulations.weekly_hours_ceiling * weeks
        return (floor, ceiling)

    def overlapping_pairs(self) -> set[frozenset[str]]:
        """Conjunto O: pares (no ordenados) de turnos que se solapan en el mismo día y por
        tanto no pueden coincidir para un mismo médico (hard, eq:intradia). Derivado de los
        tiempos reloj de los turnos. Tocarse en un extremo (M 8-15 y G1 15-08) NO es solape."""

        spans = {s.name: s.clock_span() for s in self.shifts}
        names = list(spans)
        overlapping = set()
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                (sa, ea), (sb, eb) = spans[names[a]], spans[names[b]]
                if sa < eb and sb < ea:                      # solape estricto
                    overlapping.add(frozenset((names[a], names[b])))
        return overlapping

    def short_rest_pairs(self) -> set[tuple[str, str]]:
        """Conjunto I: pares ORDENADOS (s, s') tales que encadenar s un día y s' al día
        siguiente deja menos de delta horas de descanso (hard, eq:descanso-12h). Derivado de
        los tiempos reloj + Regulations.min_rest_hours."""

        delta = self.regulations.min_rest_hours * 60        # umbral en minutos
        spans = {s.name: s.clock_span() for s in self.shifts}
        short = set()
        for a, (_, ea) in spans.items():                    # s termina en ea (puede ser del día t+1)
            for b, (sb, _) in spans.items():                # s' empieza al día siguiente, en 24h+sb
                if (24 * 60 + sb) - ea < delta:
                    short.add((a, b))
        return short