"""Evaluación de la función objetivo del Cap. 5-6.

Tres piezas que comparten todos los solvers: el desglose crudo de los siete términos soft
(`evaluate` -> `Breakdown`, eq:terminos-objetivo), la cantidad de violación de cada hard
constraint (`violations` -> `Violations`), y los normalizadores N_k (`normalizers`). Sobre
ellas, tres combinadores ensamblan las tres formulaciones de Z: clásica (cobertura dura),
de modelo (cobertura blanda, eq:objetivo) y aumentada (todo penalizado, Sección 6.3.4)."""

from pydantic import BaseModel, ConfigDict, Field

from rostering.domain import Instance, Physician
from rostering.roster import Roster


class ObjectiveConfig(BaseModel):
    """Política de scoring del objetivo (separada del dato de la instancia, eje dato-vs-política).

    Los seis pesos soft w_k son la política propiamente dicha. El peso de cobertura w_cob y la
    penalización dura mu (el lambda_h de la Sección 6.3.4) se DERIVAN por defecto (None) para que la dominancia
    de cobertura (Prop. dominancia-cobertura) y la exactitud de la penalización se cumplan por
    construcción; pueden fijarse a mano sobreescribiéndolos."""

    l_max: int = Field(default=6, ge=0)   # L^max: tope de jornadas consecutivas sin penalización (eq:stretch)

    # pesos de las seis familias soft (política, eje dato-vs-política); cobertura va aparte
    w_erg: float = Field(default=1.0, ge=0)    # w_erg:  ergonomía
    w_car: float = Field(default=1.0, ge=0)    # w_car:  equidad de carga
    w_gua: float = Field(default=1.0, ge=0)    # w_gua:  equidad de guardias
    w_cal: float = Field(default=1.0, ge=0)    # w_cal:  equidad de calendario
    w_pref: float = Field(default=1.0, ge=0)   # w_pref: satisfacción de preferencias
    w_over: float = Field(default=1.0, ge=0)   # w_over: sobre-jornada / opt-out (eq:sobre-jornada)

    # derivados si None: cobertura por dominancia, mu por exactitud de la penalización
    w_cob: float | None = Field(default=None, ge=0)   # w_cob: peso de cobertura (None -> N_cob*sumw_soft + 1)
    mu: float | None = Field(default=None, ge=0)       # lambda_h uniforme de Sección 6.3.4 (None -> supera la oscilación de Z)

    @property
    def w_soft_sum(self) -> float:
        """sum_{k!=cob} w_k: suma de los seis pesos soft (lado derecho de la desigualdad de dominancia)."""
        return self.w_erg + self.w_car + self.w_gua + self.w_cal + self.w_pref + self.w_over


class Breakdown(BaseModel):
    """Valores CRUDOS de los siete términos del objetivo (sin pesos ni normalización, eq:terminos-objetivo)."""

    model_config = ConfigDict(frozen=True)

    p_cob: float    # déficit de cobertura            sum u_ts                  (eq:cobertura-soft)
    p_erg: float    # ergonomía                       sum (pi_str + sigma)           (eq:stretch, eq:finde)
    p_car: float    # equidad de carga (L1)           sum_i (H^U_i + H^L_i)     (eq:equidad-carga)
    p_gua: float    # equidad de guardias (min-max)   G^U + G^L               (eq:guardias-eq)
    p_cal: float    # equidad de calendario (min-max) sum_c (J^U_c + J^L_c)     (eq:calendario)
    p_over: float   # sobre-jornada / opt-out         sum_i max(0, H_i - h_i)   (eq:sobre-jornada)
    pref: float     # satisfacción de preferencias    Pi = sum p_its x_its       (eq:preferencias)


class Normalizers(BaseModel):
    """Constantes de normalización N_k > 0 (eq:objetivo): amplitud de cada término sobre el
    conjunto factible, que lleva los siete a escala comparable antes de ponderar. Dependen solo
    de la instancia (y de L^max), no del roster: se calculan una vez por instancia.

    Son COTAS de la amplitud verdadera (que sería una optimización), derivadas del dato. N_cob
    y N_pref son exactas; N_car es ajustada; N_erg, N_gua y N_cal sobreestiman (ignoran el
    acoplamiento con horas y descansos) y N_over también (cota holgada sum_i h_i^max), del lado
    seguro: solo N_cob necesita ser ajustada, por la dominancia, y lo es."""

    model_config = ConfigDict(frozen=True)

    n_cob: float    # sum_{t,s} b_ts                         (déficit máx: nada cubierto)
    n_erg: float    # |P|*[max(0, D-L^max) + W_pares]      (pi_str, sigma en {0,1})
    n_car: float    # sum_i max(h_i-tau_i, tau_i-h_i^min)        (máx de |horas_i-tau_i|, válido fuera del corredor)
    n_gua: float    # max_i min(g^max_i, M_G) + tau_G        (min-max: uno hace k_max, otro 0)
    n_cal: float    # sum_c (M_c + tau_c)                       (ídem por categoría c en {fds, fes})
    n_pref: float   # sum_{i,t,s} |p_its| = Pi_max - Pi_min
    n_over: float   # sum_i h_i                              (cota holgada del exceso total; sobre-estima)


class Violations(BaseModel):
    """Cantidad de violación de cada hard constraint (0 = satisfecha). La cobertura NO está
    aquí: su déficit vive en Breakdown.p_cob, el pivote soft/hard entre las tres formulaciones."""

    model_config = ConfigDict(frozen=True)

    v_intraday: float      # eq:intradia (O): pares solapados asignados a un mismo (i,t)
    v_qualif: float        # eq:cualificacion (gamma): asignaciones a turno no cualificado por la clase
    v_elig: float          # eq:elegibilidad (eta): asignaciones a turno no ofrecido ese día
    v_rest: float          # eq:descanso-12h (I): encadenamientos día-a-día con menos de delta horas de descanso
    v_weekly: float        # eq:semanal: semanas sin par de días libres consecutivos
    v_supervision: float   # eq:supervision-r1: R1 en guardia sin adjunto en el mismo turno
    v_exempt: float        # eq:exencion (E): médico exento asignado a una guardia
    v_corridor: float      # eq:corredor: horas por debajo del suelo h_min (el techo es soft, p_over)
    v_gmax: float          # eq:tope-guardias: guardias por encima de g^max

    @property
    def total(self) -> float:
        """Suma de todas las violaciones; 0 si y solo si el roster es factible."""
        return (self.v_intraday + self.v_qualif + self.v_elig + self.v_rest + self.v_weekly
                + self.v_supervision + self.v_exempt + self.v_corridor + self.v_gmax)

    @property
    def is_feasible(self) -> bool:
        return self.total == 0


def evaluate(instance: Instance, roster: Roster, config: ObjectiveConfig) -> Breakdown:
    """Desglose crudo de los siete términos soft del objetivo sobre `roster`. Una pasada sobre
    las asignaciones construye los índices; cada término se computa a partir de ellos."""
    cal = instance.calendar
    shift_by_name = {s.name: s for s in instance.shifts}
    prefs = instance.preferences

    # --- una pasada sobre las asignaciones: los índices que necesitan los términos ---
    coverage: dict[tuple[int, str], int] = {}            # (t,s) -> nº de médicos asignados
    hours = {p.id: 0.0 for p in instance.physicians}     # horas computables por médico
    guardias = {p.id: 0 for p in instance.physicians}    # nº de guardias por médico   (n^G_i)
    active = {p.id: set() for p in instance.physicians}  # días trabajados por médico  (a_it)
    weekend_cnt = {p.id: 0 for p in instance.physicians}  # turnos en fin de semana    (n^fds_i)
    holiday_cnt = {p.id: 0 for p in instance.physicians}  # turnos en festivo          (n^fes_i)
    pref = 0.0

    for (i, t, sname) in roster.assignments:
        shift = shift_by_name[sname]
        coverage[(t, sname)] = coverage.get((t, sname), 0) + 1
        hours[i] += shift.hours
        if shift.is_guardia:
            guardias[i] += 1
        active[i].add(t)
        if cal.is_weekend(t):
            weekend_cnt[i] += 1
        if cal.is_holiday(t):
            holiday_cnt[i] += 1
        pref += prefs.get(i, {}).get(t, {}).get(sname, 0.0)

    return Breakdown(
        p_cob=_coverage_deficit(instance, coverage),
        p_erg=_ergonomics(instance, active, config.l_max),
        p_car=_workload_equity(instance, hours),
        p_gua=_guardia_equity(instance, guardias),
        p_cal=_calendar_equity(instance, weekend_cnt, holiday_cnt),
        p_over=_overtime(instance, hours),
        pref=pref,
    )


def _overtime(instance: Instance, hours: dict[int, float]) -> float:
    """P_over = sum_i max(0, H_i - h_i): horas por encima del techo de 48h (eq:sobre-jornada). Bajo
    opt-out universal el techo h_i deja de vetar y su exceso se penaliza como sobre-jornada soft."""
    total = 0.0
    for p in instance.physicians:
        h_max = instance.corridor(p)[1]
        total += max(0.0, hours[p.id] - h_max)
    return total


def _coverage_deficit(instance: Instance, coverage: dict[tuple[int, str], int]) -> float:
    """P_cob = sum_{t,s} max(0, b_ts - cobertura): médicos que faltan frente a la demanda."""
    total = 0.0
    for t, per_shift in instance.demand.items():
        for sname, b in per_shift.items():
            deficit = b - coverage.get((t, sname), 0)
            if deficit > 0:
                total += deficit
    return total


def _ergonomics(instance: Instance, active: dict[int, set[int]], l_max: int) -> float:
    """P_erg = sum_{i,t} (pi_str + sigma): exceso de rachas largas + fines de semana partidos."""
    cal = instance.calendar
    days = cal.days
    total = 0.0
    for worked_days in active.values():
        # racha: en cada ventana de L^max+1 días, exceso de jornadas sobre L^max
        for t0 in range(days - l_max):
            worked = sum(1 for u in range(t0, t0 + l_max + 1) if u in worked_days)
            if worked > l_max:
                total += worked - l_max
        # fin de semana partido: pares consecutivos ambos en T^fds, |a_t - a_{t+1}|
        for t in range(days - 1):
            if cal.is_weekend(t) and cal.is_weekend(t + 1):
                total += abs((t in worked_days) - (t + 1 in worked_days))
    return total


def mean_workload(instance: Instance) -> float:
    """H = (sum_{t,s} d_s b_ts) / sum_j rho_j: carga media estimada por médico a jornada completa."""
    shift_by_name = {s.name: s for s in instance.shifts}
    total_hours = sum(shift_by_name[s].hours * b
                      for per in instance.demand.values() for s, b in per.items())
    sum_rho = sum(p.part_time for p in instance.physicians)
    return total_hours / sum_rho if sum_rho else 0.0


def _workload_equity(instance: Instance, hours: dict[int, float]) -> float:
    """P_car = sum_i |horas_i - tau_i| (norma L1), con tau_i = rho_i*H (eq:equidad-carga)."""
    mean = mean_workload(instance)
    return sum(abs(hours[p.id] - p.part_time * mean) for p in instance.physicians)


def _minmax_spread(physicians: list[Physician], counts: dict[int, int], tau: float) -> float:
    """Horquilla min-max: mayor exceso + mayor defecto del recuento frente a la cuota rho_i*tau.
    Es la forma de G^U+G^L y de J^U_c+J^L_c (eq:guardias-eq, eq:calendario)."""
    upper = 0.0   # G^U / J^U: mayor exceso sobre la cuota
    lower = 0.0   # G^L / J^L: mayor defecto bajo la cuota
    for p in physicians:
        dev = counts[p.id] - p.part_time * tau
        upper = max(upper, dev)
        lower = max(lower, -dev)
    return upper + lower


def _guardia_equity(instance: Instance, guardias: dict[int, int]) -> float:
    """P_gua = G^U + G^L (min-max sobre P\\E), con tau_G = (sum_{s en G} b_ts) / sum_{j en P\\E} rho_j."""
    shift_by_name = {s.name: s for s in instance.shifts}
    non_exempt = [p for p in instance.physicians if not p.is_exempt]
    sum_rho = sum(p.part_time for p in non_exempt)
    guardia_demand = sum(b for per in instance.demand.values()
                         for s, b in per.items() if shift_by_name[s].is_guardia)
    tau_g = guardia_demand / sum_rho if sum_rho else 0.0
    return _minmax_spread(non_exempt, guardias, tau_g)


def _calendar_equity(instance: Instance, weekend_cnt: dict[int, int], holiday_cnt: dict[int, int]) -> float:
    """P_cal = sum_c (J^U_c + J^L_c), c en {fds, fes}: min-max por categoría sobre P\\E,
    con tau_c = (sum_{(t,s) en c} b_ts) / sum_{j en P\\E} rho_j (eq:calendario)."""
    cal = instance.calendar
    non_exempt = [p for p in instance.physicians if not p.is_exempt]
    sum_rho = sum(p.part_time for p in non_exempt)
    total = 0.0
    for counts, in_category in ((weekend_cnt, cal.is_weekend), (holiday_cnt, cal.is_holiday)):
        cat_demand = sum(b for t, per in instance.demand.items()
                         for s, b in per.items() if in_category(t))
        tau_c = cat_demand / sum_rho if sum_rho else 0.0
        total += _minmax_spread(non_exempt, counts, tau_c)
    return total


def violations(instance: Instance, roster: Roster) -> Violations:
    """Cara HARD del evaluador: cantidad de violación de cada restricción dura sobre `roster`.
    No depende de la config (la factibilidad es independiente de los pesos del objetivo)."""
    shift_by_name = {s.name: s for s in instance.shifts}
    phys_by_id = {p.id: p for p in instance.physicians}
    class_of = {p.id: instance.class_of(p) for p in instance.physicians}
    adjuntos = {pid for pid, c in class_of.items() if not c.is_resident}

    # --- una pasada: índices + violaciones a nivel de asignación (gamma, eta, exención) ---
    by_phys_day: dict[tuple[int, int], set[str]] = {}      # (i,t) -> turnos asignados (O, I)
    adjunto_on: dict[tuple[int, str], int] = {}            # (t,s) -> nº de adjuntos (supervisión)
    hours = {p.id: 0.0 for p in instance.physicians}
    guardias = {p.id: 0 for p in instance.physicians}
    active = {p.id: set() for p in instance.physicians}
    v_qualif = v_elig = v_exempt = 0.0

    for (i, t, sname) in roster.assignments:
        shift = shift_by_name[sname]
        by_phys_day.setdefault((i, t), set()).add(sname)
        hours[i] += shift.hours
        active[i].add(t)
        if shift.is_guardia:
            guardias[i] += 1
            if phys_by_id[i].is_exempt:
                v_exempt += 1
        if i in adjuntos:
            adjunto_on[(t, sname)] = adjunto_on.get((t, sname), 0) + 1
        if not instance.qualification.get(class_of[i].name, {}).get(sname, False):
            v_qualif += 1
        if not instance.eligibility.get(t, {}).get(sname, False):
            v_elig += 1

    overlap = instance.overlapping_pairs()
    short_rest = instance.short_rest_pairs()

    # intradía O: pares solapados dentro de un mismo (i,t)
    v_intraday = 0.0
    for sset in by_phys_day.values():
        names = sorted(sset)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                if frozenset((names[a], names[b])) in overlap:
                    v_intraday += 1

    # descanso I: turno s el día t y s' el día t+1 con (s,s') en I
    v_rest = 0.0
    for (i, t), sset in by_phys_day.items():
        for s in sset:
            for s2 in by_phys_day.get((i, t + 1), ()):
                if (s, s2) in short_rest:
                    v_rest += 1

    # supervisión R1: residente de primer año en guardia sin ningún adjunto en el mismo turno
    v_supervision = 0.0
    for (i, t, sname) in roster.assignments:
        c = class_of[i]
        if c.is_resident and c.residency_year == 1 and shift_by_name[sname].is_guardia:
            if adjunto_on.get((t, sname), 0) == 0:
                v_supervision += 1

    # corredor: bajo opt-out universal solo el SUELO h_min es duro; el techo h_max deja de vetar y
    # su exceso se penaliza como sobre-jornada soft (Breakdown.p_over, eq:sobre-jornada)
    v_corridor = 0.0
    for p in instance.physicians:
        h_min = instance.corridor(p)[0]
        v_corridor += max(0.0, h_min - hours[p.id])

    # g^max: guardias por encima del tope (None = sin tope, adjuntos)
    v_gmax = 0.0
    for p in instance.physicians:
        cap = instance.effective_max_guardias(p)
        if cap is not None and guardias[p.id] > cap:
            v_gmax += guardias[p.id] - cap

    return Violations(
        v_intraday=v_intraday, v_qualif=v_qualif, v_elig=v_elig, v_rest=v_rest,
        v_weekly=_weekly_rest(instance, active), v_supervision=v_supervision,
        v_exempt=v_exempt, v_corridor=v_corridor, v_gmax=v_gmax,
    )


def _weekly_rest(instance: Instance, active: dict[int, set[int]]) -> float:
    """Nº de (médico, semana) sin un par de días libres consecutivos dentro de la semana
    (eq:semanal). Semanas = bloques de 7 días desde el inicio; se omiten las de < 2 días.
    NOTA: la partición en semanas es una decisión de modelado (bloques desde t=0)."""
    days = instance.calendar.days
    weeks = [range(w, min(w + 7, days)) for w in range(0, days, 7)]
    total = 0.0
    for p in instance.physicians:
        worked = active[p.id]
        for week in weeks:
            if len(week) < 2:
                continue
            has_free_pair = any(t not in worked and (t + 1) not in worked
                                for t in week if (t + 1) in week)
            if not has_free_pair:
                total += 1
    return total


# ---------------------------------------------------------------------------
# Normalizadores N_k y los tres combinadores de Z
# ---------------------------------------------------------------------------

def normalizers(instance: Instance, config: ObjectiveConfig) -> Normalizers:
    """Calcula los siete N_k de la instancia (cotas de amplitud sobre el factible). Una vez por
    instancia: no dependen del roster. Cada uno se guarda con `max(1, *)` para que un término
    ausente (p.ej. sin guardias, o sin preferencias) deje su cociente en 0 y no divida por cero."""
    cal = instance.calendar
    days = cal.days
    shift_by_name = {s.name: s for s in instance.shifts}

    # N_cob = sum b_ts: déficit máximo (ningún turno cubierto)
    n_cob = sum(b for per in instance.demand.values() for b in per.values())

    # N_pref = sum |p_its| = Pi_max - Pi_min: rango entre conceder todo lo deseado y todo lo rechazado
    n_pref = sum(abs(v) for by_t in instance.preferences.values()
                 for by_s in by_t.values() for v in by_s.values())

    # N_car = sum_i max(h_i - tau_i, tau_i - h_i^min): máx de |horas_i - tau_i| con horas_i en el corredor,
    # válido aunque la carga media tau_i caiga fuera del corredor (demanda muy baja o muy alta)
    mean = mean_workload(instance)
    n_car = 0.0
    for p in instance.physicians:
        h_min, h_max = instance.corridor(p)
        tau = p.part_time * mean
        n_car += max(h_max - tau, tau - h_min)

    non_exempt = [p for p in instance.physicians if not p.is_exempt]
    sum_rho = sum(p.part_time for p in non_exempt)

    # N_gua = max_i min(g^max_i, M_G) + tau_G: caso peor del min-max (uno hace k_max guardias, otro 0);
    # el +tau_G cubre el lado del defecto cuando hay part-time (rho<1), donde las cuotas no se cancelan
    m_g = sum(1 for per in instance.demand.values()
              for s, b in per.items() if b > 0 and shift_by_name[s].is_guardia)
    guardia_demand = sum(b for per in instance.demand.values()
                         for s, b in per.items() if shift_by_name[s].is_guardia)
    tau_g = guardia_demand / sum_rho if sum_rho else 0.0
    cap_max = 0.0
    for p in non_exempt:
        cap = instance.effective_max_guardias(p)   # None = sin tope (adjuntos)
        cap_max = max(cap_max, float(m_g) if cap is None else float(min(cap, m_g)))
    n_gua = cap_max + tau_g

    # N_cal = sum_c (M_c + tau_c), c en {fds, fes}: misma cota min-max por categoría de día
    n_cal = 0.0
    for in_category in (cal.is_weekend, cal.is_holiday):
        m_c = sum(1 for t, per in instance.demand.items()
                  for s, b in per.items() if b > 0 and in_category(t))
        cat_demand = sum(b for t, per in instance.demand.items()
                         for s, b in per.items() if in_category(t))
        tau_c = cat_demand / sum_rho if sum_rho else 0.0
        n_cal += m_c + tau_c

    # N_erg = |P|*[max(0, D-L^max) + W_pares]: pi_str <= 1 por ventana, sigma <= 1 por par de finde
    n_windows = max(0, days - config.l_max)
    w_pairs = sum(1 for t in range(days - 1) if cal.is_weekend(t) and cal.is_weekend(t + 1))
    n_erg = len(instance.physicians) * (n_windows + w_pairs)

    # N_over = sum_i h_i: cota holgada del exceso total de sobre-jornada (sobre-estima, lado seguro;
    # el exceso de un médico rara vez supera su propio techo). Se afina en el Cap. 7 si hace falta.
    n_over = sum(instance.corridor(p)[1] for p in instance.physicians)

    return Normalizers(
        n_cob=max(1.0, n_cob), n_erg=max(1.0, float(n_erg)), n_car=max(1.0, n_car),
        n_gua=max(1.0, n_gua), n_cal=max(1.0, n_cal), n_pref=max(1.0, n_pref),
        n_over=max(1.0, n_over),
    )


def resolve_w_cob(config: ObjectiveConfig, norm: Normalizers) -> float:
    """Peso de cobertura. Si no se fija a mano, se deriva para garantizar la dominancia
    w_cob/N_cob > sum w_soft (Prop. dominancia-cobertura): w_cob = N_cob*sumw_soft + 1, con lo que
    w_cob/N_cob = sumw_soft + 1/N_cob > sumw_soft. La cobertura recupera así su carácter Hard*."""
    if config.w_cob is not None:
        return config.w_cob
    return norm.n_cob * config.w_soft_sum + 1.0


def resolve_mu(config: ObjectiveConfig, norm: Normalizers) -> float:
    """Penalización dura uniforme mu (= lambda_h para todoh, Sección 6.3.4). Si no se fija a mano, se toma por encima
    de la oscilación de Z sobre el factible (w_cob + sumw_soft): así una sola violación (v_h >= 1,
    entera) cuesta más que cualquier mejora de Z. Es un punto de partida; fuera del factible
    algún término soft puede pasarse de su N_k, de modo que mu se calibra al alza en el Cap. 7."""
    if config.mu is not None:
        return config.mu
    return resolve_w_cob(config, norm) + config.w_soft_sum + 1.0


def soft_terms(b: Breakdown, norm: Normalizers, config: ObjectiveConfig) -> float:
    """Suma ponderada y normalizada de los SEIS términos soft sin cobertura (común a las tres
    formulaciones). La preferencia entra con signo negativo: se maximiza, no se penaliza."""
    return (config.w_erg * b.p_erg / norm.n_erg
            + config.w_car * b.p_car / norm.n_car
            + config.w_gua * b.p_gua / norm.n_gua
            + config.w_cal * b.p_cal / norm.n_cal
            + config.w_over * b.p_over / norm.n_over
            - config.w_pref * b.pref / norm.n_pref)


def z_clasica(b: Breakdown, norm: Normalizers, config: ObjectiveConfig) -> float:
    """Formulación clásica = MODELO REALISTA bajo opt-out universal: la cobertura es una HARD
    constraint (fuera del objetivo) y el techo de 48h es soft (sobre-jornada p_over, dentro). Z
    reúne los seis términos soft. Pareja de factibilidad: violations.total Y p_cob = 0 (cobertura
    completa exigida)."""
    return soft_terms(b, norm, config)


def z_modelo(b: Breakdown, norm: Normalizers, config: ObjectiveConfig) -> float:
    """Formulación adoptada (eq:objetivo): la cobertura es SOFT y dominante. Z son los seis
    términos. Pareja de factibilidad: violations.total = 0 (la cobertura ya está en el objetivo)."""
    return soft_terms(b, norm, config) + resolve_w_cob(config, norm) * b.p_cob / norm.n_cob


def z_aumentada(b: Breakdown, v: Violations, norm: Normalizers, config: ObjectiveConfig) -> float:
    """Objetivo aumentado de la Sección 6.3.4: Z_modelo + mu*sum_h v_h. Definido en todo el hipercubo
    {0,1}^n (las hard se penalizan, no recortan el dominio); es el que minimiza el SA."""
    return z_modelo(b, norm, config) + resolve_mu(config, norm) * v.total

