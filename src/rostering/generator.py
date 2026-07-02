"""Generador de instancias sintéticas calibrado al modelo del Cap. 5.

Enfoque FORWARD: la demanda se calcula a través de un patrón clínico determinista + ruido, no se
deriva de un roster semilla. `Settings` es el blueprint COMPLETO y determinista de la
instancia (catálogo, calendario, elegibilidad y cualificación incluidos); la semilla solo
gobierna la capa estocástica (ruido de demanda, distribución de la plantilla y, en un futuro, preferencias)."""

import random
from datetime import date, time, timedelta

from pydantic import BaseModel, Field, NonNegativeFloat, model_validator

from rostering.domain import Shift, MedicalClass, Physician, Calendar, Regulations, Instance


class ShiftSpec(BaseModel):
    """Un turno del catálogo más su peso relativo de demanda"""

    shift: Shift                  # definición del turno (modelo de dominio)
    weight: float = Field(ge=0)   # peso relativo de demanda por turno (se normaliza)


# FUNCIONES PARA LAS DEFAULT SETTINGS (CASO ANDALUCÍA/ESPAÑA)
def _standard_shifts() -> list[ShiftSpec]:
    """Catálogo base del caso de estudio (Cap. 5): mañana ordinaria + guardia
    laborable (17h) + guardia de 24h (finde/festivo). La noche la cubre la
    guardia."""
    return [
        ShiftSpec(shift=Shift(name="M", start=time(8), end=time(15), hours=7), weight=5),
        ShiftSpec(shift=Shift(name="G1", start=time(15), end=time(8), hours=17, is_guardia=True), weight=2),
        ShiftSpec(shift=Shift(name="G2", start=time(8), end=time(8), hours=24, is_guardia=True), weight=2),
    ]


def _standard_classes() -> list[MedicalClass]:
    """Registro de clases del caso español: adjunto (FEA) + residentes R1..R5.
    Jornada ordinaria de 35h (suelo h^min_r) y tope MIR de 7 guardias/mes."""
    fea = MedicalClass(name="FEA", is_resident=False, weekly_hours=35.0)
    residents = [
        MedicalClass(name=f"R{k}", is_resident=True, residency_year=k,
                     weekly_hours=35.0, default_max_guardias=7)
        for k in range(1, 6)
    ]
    return [fea, *residents]


def _standard_calendar() -> Calendar:
    """Horizonte base: 28 días desde un lunes; sin festivos propios (el finde basta
    para el patrón). Los festivos deterministas son un knob futuro."""
    return Calendar(start_date=date(2026, 1, 5), days=28, holidays=[])


def _standard_regulations() -> Regulations:
    """Régimen legal vigente en España (Ley 55/2003 + RD 1146/2006): 12h de descanso entre
    jornadas, 36h semanal, techo de 48h de media. El borrador de 2026 (17h/45h) sería una variante."""
    return Regulations(min_rest_hours=12.0, min_weekly_rest_hours=36.0, weekly_hours_ceiling=48.0)


def _is_full_day_guardia(shift: Shift) -> bool:
    """La guardia que cubre un día no laborable completo (24h), frente a la de 17h
    encadenada a la mañana. Heurística interna de los builders por defecto: distingue
    G2 de G1 en el catálogo base (no es estructura general, solo el caso default)."""
    return shift.is_guardia and shift.hours >= 24


def _standard_eligibility(calendar: Calendar, shifts: list[ShiftSpec]) -> dict[int, dict[str, bool]]:
    """Construye la eta_ts por defecto (matriz completa día x turno). Heurística del
    caso base: la guardia de día completo solo se ofrece en día no laborable; los
    demás turnos, solo en laborable. La eta canónica es esta matriz (día, turno)."""
    eligibility = {}
    for t in range(calendar.days):
        working = calendar.is_working_day(t)
        row = {}
        for spec in shifts:
            if _is_full_day_guardia(spec.shift):
                row[spec.shift.name] = not working   # guardia de 24h: solo finde o festivo
            else:
                row[spec.shift.name] = working       # mañana y guardia de 17h: solo laborable
        eligibility[t] = row
    return eligibility


def _standard_qualification(classes: list[MedicalClass],
                            shifts: list[ShiftSpec]) -> dict[str, dict[str, bool]]:
    """Construye la gamma_rs por defecto (matriz completa clase x turno). Heurística
    del caso base: un residente no cubre la guardia de día completo (solo adjuntos);
    el resto de turnos, habilitado para toda clase."""
    qualification = {}
    for c in classes:
        row = {}
        for spec in shifts:
            attending_only = _is_full_day_guardia(spec.shift)
            row[spec.shift.name] = not (c.is_resident and attending_only)
        qualification[c.name] = row
    return qualification


# DEFINICIÓN DE CLASE SETTINGS
class Settings(BaseModel):
    """Blueprint completo y determinista de una instancia. `Settings()` = caso base andaluz/español.

    Reúne todo lo que no depende de la aleatoriedad: catálogo, clases, calendario, elegibilidad,
    cualificación, y la forma y nivel de la demanda. Las tablas derivadas (`eligibility`,
    `qualification`) se calculan por defecto desde el resto del blueprint si se dejan vacías,
    pero pueden darse explícitas (la matriz es canónica; la heurística es solo el default).
    Para barrer la dificultad se crean variantes (cambiar `n_physicians`, `coverage_ratio`,
    `calendar`, ...); la réplica se da con la semilla."""

    # estructura determinista (registros + tablas derivadas)
    shifts: list[ShiftSpec] = Field(default_factory=_standard_shifts)
    classes: list[MedicalClass] = Field(default_factory=_standard_classes)
    calendar: Calendar = Field(default_factory=_standard_calendar)
    regulations: Regulations = Field(default_factory=_standard_regulations)   # régimen legal (descansos, techo)
    eligibility: dict[int, dict[str, bool]] = Field(default_factory=dict)    # eta_ts; vacío => se calcula por defecto
    qualification: dict[str, dict[str, bool]] = Field(default_factory=dict)  # gamma_rs; vacío => se calcula por defecto

    # plantilla (parámetros del muestreo estocástico)
    n_physicians: int = Field(default=12, gt=0)        # n: tamaño de la plantilla
    fea_ratio: float = Field(default=0.6, ge=0, le=1)  # fracción de FEA; el resto, residentes
    residency_pyramid: list[NonNegativeFloat] = Field(
        default_factory=lambda: [5, 4, 3, 2, 1],       # pesos por clase residente (se normalizan); uno por R1..Rk
    )

    # forma y nivel de la demanda
    weekday_weights: list[NonNegativeFloat] = Field(
        default_factory=lambda: [5, 5, 5, 5, 5, 2, 2], min_length=7, max_length=7
    )  # peso de demanda por día de la semana (lun..dom); el finde pesa menos
    holiday_weight: float = Field(default=1.0, ge=0)
    coverage_ratio: float = Field(default=1.10, gt=0)   # demanda-horas / horas-contratadas; tensiona la dificultad. Default 1.10 (no 1.0): el redondeo de b_ts sesga el realizado a la baja (~0.97), y el caso base debe quedar por encima del suelo del corredor (h_min) para tener región factible; realizado ~1.08, holgura para que el MILP certifique Z*. Ratios <1 = infra-dotación deliberada (eje experimental Cap.7)
    demand_noise: float = Field(default=0.1, ge=0, lt=1.0)  # amplitud del jitter relativo por celda; <1 mantiene el jitter > 0; 0 = patrón puro

    # validaciones
    @model_validator(mode="after")
    def _fill_derived_tables(self):
        """Rellena eta_ts y gamma_rs por defecto desde el resto del blueprint si se dejaron
        vacías. Ambas son deterministas (no dependen del rng de generate()), por eso viven en Settings."""
        if not self.eligibility:
            self.eligibility = _standard_eligibility(self.calendar, self.shifts)
        if not self.qualification:
            self.qualification = _standard_qualification(self.classes, self.shifts)
        return self

    @model_validator(mode="after")
    def _check_composition(self):
        """Coherencia del blueprint antes de muestrear: debe existir una clase no residente
        (FEA) y `residency_pyramid` debe alinear un peso por clase residente, con alguno
        positivo."""
        resident_classes = [c for c in self.classes if c.is_resident]
        if not any(not c.is_resident for c in self.classes):
            raise ValueError("classes debe incluir al menos una clase no residente (FEA): "
                             "_sample_physicians la usa como adjunto y solo ella cubre G2")
        if len(self.residency_pyramid) != len(resident_classes):
            raise ValueError(
                f"residency_pyramid tiene {len(self.residency_pyramid)} pesos pero classes define "
                f"{len(resident_classes)} clases residentes; deben coincidir (uno por R1..Rk)"
            )
        if resident_classes and sum(self.residency_pyramid) <= 0:
            raise ValueError("residency_pyramid debe tener algún peso positivo (no puede ser todo cero)")
        return self


# FUNCIONES AUXILIARES DE generate()
def _sample_physicians(settings: Settings, rng: random.Random) -> list[Physician]:
    """Muestrea n médicos: clase vía fea_ratio + pirámide MIR (sembrado). Todos a jornada
    completa (rho=1) y sin exenciones: part_time e is_exempt son datos que el modelo admite
    pero que este generador no muestrea."""
    classes = settings.classes
    fea_name = next(c.name for c in classes if not c.is_resident)
    resident_names = [c.name for c in classes if c.is_resident]  # R1..R5, alineados con residency_pyramid

    physicians = []
    for i in range(1, settings.n_physicians + 1):
        if rng.random() < settings.fea_ratio:
            class_name = fea_name
        else:
            class_name = rng.choices(resident_names, weights=settings.residency_pyramid, k=1)[0]
        physicians.append(Physician(id=i, medical_class_name=class_name))  # max_guardias=None -> hereda
    return physicians


def _ensure_structural_coverage(settings: Settings, physicians: list[Physician]) -> list[Physician]:
    """Garantiza que todo turno ofrecido tenga >=1 médico de una clase que lo cualifique
    (caso típico: forzar >=1 FEA para que G2 sea cubrible). Repara la
    muestra de forma determinista y sin consumir rng (reasignando los últimos médicos a las
    clases que falten), así la demanda no se ve perturbada. Las instancias tensas-pero-
    cubribles no se tocan; solo se corrige el caso degenerado 'ninguna clase presente cubre s'."""
    offered = {s for row in settings.eligibility.values() for s, off in row.items() if off}
    qual = settings.qualification
    slot = len(physicians) - 1
    for s in sorted(offered):
        present = {p.medical_class_name for p in physicians}
        qualifying = {c for c in qual if qual[c].get(s)}
        if not qualifying or (present & qualifying):
            continue                       # nadie puede cubrirlo (lo veta el validador) o ya está cubierto
        if slot < 0:
            break                          # plantilla insuficiente para reparar; lo capturará Instance
        target = sorted(qualifying)[0]
        physicians[slot] = physicians[slot].model_copy(update={"medical_class_name": target})
        slot -= 1
    return physicians


def _contracted_hours(settings: Settings, physicians: list[Physician]) -> float:
    """Capacidad contractual de la plantilla sobre el horizonte: suma de jornadas
    ordinarias rho_i * weekly_hours(clase) * |T|/7."""
    hours_by_class = {c.name: c.weekly_hours for c in settings.classes}
    weeks = settings.calendar.days / 7
    return sum(p.part_time * hours_by_class[p.medical_class_name] * weeks for p in physicians)


def _build_demand(settings: Settings, physicians: list[Physician],
                  rng: random.Random) -> dict[int, dict[str, int]]:
    """Demanda b_ts (FORWARD): patrón 'peso_día x peso_turno' escalado a coverage_ratio
    más un jitter por celda. Solo se puebla donde la elegibilidad lo permite (b=0 si eta=0)."""
    calendar = settings.calendar
    eligibility = settings.eligibility
    spec_by_name = {spec.shift.name: spec for spec in settings.shifts}

    # peso bruto y contenido horario de cada (t, s) ofrecido
    raw: dict[tuple[int, str], float] = {}
    weighted_hours = 0.0
    for t in range(calendar.days):
        day = calendar.start_date + timedelta(days=t)
        is_holiday = day in calendar.holidays
        day_w = settings.holiday_weight if is_holiday else settings.weekday_weights[day.weekday()]
        for shift_name, offered in eligibility[t].items():
            if not offered:
                continue
            spec = spec_by_name[shift_name]
            w = day_w * spec.weight
            raw[(t, shift_name)] = w
            weighted_hours += w * spec.shift.hours

    if weighted_hours == 0:
        return {}

    # escala alpha para que la demanda-horas total apunte al objetivo relativo a capacidad
    target_hours = settings.coverage_ratio * _contracted_hours(settings, physicians)
    alpha = target_hours / weighted_hours

    demand: dict[int, dict[str, int]] = {}
    for (t, shift_name), w in raw.items():
        jitter = 1.0 + rng.uniform(-settings.demand_noise, settings.demand_noise)
        demand.setdefault(t, {})[shift_name] = max(1, round(alpha * w * jitter))  # todo turno ofrecido: >=1
    return demand


def generate(settings: Settings, seed: int) -> Instance:
    """Genera una Instance sintética reproducible (misma settings + semilla -> misma
    instancia). La semilla gobierna solo lo estocástico, con un substream determinista e
    independiente por componente: así variar la plantilla (fea_ratio, n) no perturba el
    jitter de la demanda. El resto sale tal cual de `settings`."""
    
    staff_rng = random.Random(f"{seed}:staff")     # muestreo de plantilla
    demand_rng = random.Random(f"{seed}:demand")   # jitter de la demanda

    physicians = _sample_physicians(settings, staff_rng)
    physicians = _ensure_structural_coverage(settings, physicians)
    demand = _build_demand(settings, physicians, demand_rng)
    shifts = [spec.shift for spec in settings.shifts]

    return Instance(
        seed=seed,
        calendar=settings.calendar,
        regulations=settings.regulations,
        classes=settings.classes,
        shifts=shifts,
        physicians=physicians,
        demand=demand,
        eligibility=settings.eligibility,
        qualification=settings.qualification,
        preferences={},  # sin preferencias: p_its es dato que el modelo admite, sin muestreo aquí
    )
