"""Heurística constructiva voraz (Sección 6.2, Alg. 6.1).

Levanta un roster en una sola pasada: recorre las parejas (día, turno) por criticidad
decreciente y asigna, entre los candidatos que respetan toda hard constraint, el de menor coste
marginal. Es a la vez el suelo de comparación y la solución inicial del SA; su roster es
factible por construcción, a lo sumo con déficit de cobertura si algún turno se queda sin
candidato admisible."""

from rostering.domain import Instance
from rostering.objective import ObjectiveConfig, evaluate, normalizers, soft_terms
from rostering.roster import Roster


def greedy(instance: Instance, config: ObjectiveConfig | None = None) -> Roster:
    """Construye un roster voraz para `instance`. `config` fija los pesos/normalizadores del
    coste marginal (los mismos del objetivo); por defecto la política base. Devuelve el roster
    (su déficit y su valor se leen ex post con `evaluate`/`violations`)."""
    config = config or ObjectiveConfig()
    norm = normalizers(instance, config)

    shift_by_name = {s.name: s for s in instance.shifts}
    phys_by_id = {p.id: p for p in instance.physicians}
    class_of = {p.id: instance.class_of(p) for p in instance.physicians}
    adjuntos = {pid for pid, c in class_of.items() if not c.is_resident}
    overlap = instance.overlapping_pairs()
    short_rest = instance.short_rest_pairs()
    physician_ids = [p.id for p in instance.physicians]   # orden fijo -> desempate determinista
    days = instance.calendar.days

    # estado incremental que consume el filtro de candidatos (barato, se llama |P| veces por plaza)
    by_phys_day: dict[tuple[int, int], set[str]] = {}     # (i,t) -> turnos de i ese día (O, I)
    guardias = {p.id: 0 for p in instance.physicians}      # guardias acumuladas (g^max)
    active = {p.id: set() for p in instance.physicians}    # días trabajados (descanso semanal)
    adjunto_on: dict[tuple[int, str], int] = {}            # (t,s) -> nº de adjuntos (supervisión R1)

    roster = Roster()

    def admissible(i: int, t: int, s: str) -> bool:
        """¿Asignar (i,t,s) respeta TODA hard constraint dura del bucle dado el roster parcial? El
        corredor queda fuera por AMBOS extremos: el suelo porque añadir nunca lo viola (lo viola no
        asignar bastante, prop:factibilidad-workload), y el TECHO porque bajo opt-out universal deja
        de vetar: su exceso es sobre-jornada soft, penalizada en marginal_cost vía p_over."""
        shift = shift_by_name[s]
        if s in by_phys_day.get((i, t), ()):           # ya asignado: no aporta cobertura
            return False
        if not instance.qualification.get(class_of[i].name, {}).get(s, False):   # gamma
            return False
        if not instance.eligibility.get(t, {}).get(s, False):                    # eta
            return False
        if shift.is_guardia and phys_by_id[i].is_exempt:                         # E
            return False
        for s2 in by_phys_day.get((i, t), ()):                                   # O (solape intradía)
            if frozenset((s, s2)) in overlap:
                return False
        for s_prev in by_phys_day.get((i, t - 1), ()):                           # I (descanso, día anterior)
            if (s_prev, s) in short_rest:
                return False
        for s_next in by_phys_day.get((i, t + 1), ()):                           # I (descanso, día siguiente)
            if (s, s_next) in short_rest:
                return False
        if shift.is_guardia:                                                     # g^max
            cap = instance.effective_max_guardias(phys_by_id[i])
            if cap is not None and guardias[i] + 1 > cap:
                return False
        if t not in active[i] and _breaks_weekly_rest(active[i], t, days):       # descanso semanal
            return False
        c = class_of[i]                                                          # supervisión R1
        if c.is_resident and c.residency_year == 1 and shift.is_guardia:
            if adjunto_on.get((t, s), 0) == 0:    # R1 fuera de candidatos mientras no haya adjunto en (t,s)
                return False
        return True

    def marginal_cost(i: int, t: int, s: str) -> float:
        """c(i,t,s): los términos del objetivo aumentado que dependen del médico (ergonomía + las
        tres equidades + sobre-jornada + preferencias; la cobertura se excluye, vale igual para
        todo candidato).
        Se calcula reevaluando el roster con (i,t,s) añadido (exacto, reusa el evaluador)."""
        roster.assign(i, t, s)
        c = soft_terms(evaluate(instance, roster, config), norm, config)
        roster.unassign(i, t, s)
        return c

    for (t, s) in _criticality_order(instance):
        for _ in range(instance.demand[t][s]):
            candidates = [i for i in physician_ids if admissible(i, t, s)]
            if not candidates:
                break    # sin candidato: el resto de plazas de (t,s) quedan en déficit (se mide ex post)
            i_star = min(candidates, key=lambda i: marginal_cost(i, t, s))
            roster.assign(i_star, t, s)
            shift = shift_by_name[s]
            by_phys_day.setdefault((i_star, t), set()).add(s)
            active[i_star].add(t)
            if shift.is_guardia:
                guardias[i_star] += 1
            if i_star in adjuntos:
                adjunto_on[(t, s)] = adjunto_on.get((t, s), 0) + 1

    return roster


def _criticality_order(instance: Instance) -> list[tuple[int, str]]:
    """Parejas (t,s) con demanda, ordenadas por criticidad b_ts / |{i: cualificado para s}|
    decreciente: las más difíciles de cubrir (oferta basal escasa) se atienden primero. El orden
    se fija una vez. Desempate por demanda decreciente y luego (t,s) para reproducibilidad."""
    qualified = {
        s.name: sum(1 for p in instance.physicians
                    if instance.qualification.get(instance.class_of(p).name, {}).get(s.name, False))
        for s in instance.shifts
    }
    pairs = []
    for t, per in instance.demand.items():
        for s, b in per.items():
            if b <= 0:
                continue
            crit = b / qualified[s] if qualified.get(s) else float("inf")
            pairs.append((crit, b, t, s))
    pairs.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    return [(t, s) for _, _, t, s in pairs]


def _breaks_weekly_rest(worked_days: set[int], t: int, days: int) -> bool:
    """¿Marcar el día t como trabajado deja la semana (bloque de 7 días desde el inicio) sin un
    par de días libres consecutivos? (eq:semanal). Espejo de `objective._weekly_rest`: misma
    partición, y las semanas de < 2 días no imponen restricción."""
    week_start = (t // 7) * 7
    week = range(week_start, min(week_start + 7, days))
    if len(week) < 2:
        return False
    worked = worked_days | {t}
    has_free_pair = any(u not in worked and (u + 1) not in worked
                        for u in week if (u + 1) in week)
    return not has_free_pair
