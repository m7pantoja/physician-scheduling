"""Fachada sobre el paquete `rostering` para la interfaz.

Traduce en ambos sentidos: del motor a la UI (resúmenes, matrices para heatmaps, informes
de evaluación como DataFrames) y de la UI al motor (tablas editadas que se convierten de
vuelta en una `Instance` validada). Toda la semántica del modelo vive en el paquete; aquí
no se re-implementa ningún término del objetivo, y la única regla duplicada es
`violation_marks`, que espeja las restricciones duras celda a celda para poder
localizarlas en el cuadrante."""

from __future__ import annotations

import math
import random
import time
from datetime import date, timedelta

import pandas as pd
import pulp

from rostering.domain import Instance, Physician, Shift
from rostering.generator import Settings, ShiftSpec, _build_demand, generate
from rostering.greedy import greedy
from rostering.milp import solve_exact
from rostering.objective import (
    Breakdown,
    Normalizers,
    ObjectiveConfig,
    Violations,
    evaluate,
    mean_workload,
    normalizers,
    resolve_mu,
    resolve_w_cob,
    violations,
    z_aumentada,
    z_clasica,
    z_modelo,
)
from rostering.roster import Roster
from rostering.sa import SAConfig, SAResult, schedule, simulated_annealing

from core import store

WEEKDAYS = ("lun", "mar", "mié", "jue", "vie", "sáb", "dom")

# etiquetas de los siete términos del objetivo (Breakdown), en orden de presentación
TERM_LABELS = {
    "p_cob": "Cobertura (déficit)",
    "p_erg": "Ergonomía",
    "p_car": "Equidad de carga",
    "p_gua": "Equidad de guardias",
    "p_cal": "Equidad de calendario",
    "p_over": "Sobre-jornada",
    "pref": "Preferencias",
}

# etiquetas de las nueve hard constraints (Violations)
HARD_LABELS = {
    "v_intraday": "Solape intradía (O)",
    "v_qualif": "Cualificación (γ)",
    "v_elig": "Elegibilidad (η)",
    "v_rest": "Descanso entre jornadas (I)",
    "v_weekly": "Descanso semanal",
    "v_supervision": "Supervisión de R1 en guardia",
    "v_exempt": "Exención de guardias (E)",
    "v_corridor": "Suelo del corredor (h^min)",
    "v_gmax": "Tope de guardias (g^max)",
}

SOLVER_LABELS = {"greedy": "Greedy", "sa": "Simulated annealing", "milp": "MILP exacto"}

# etiquetas del régimen de cobertura del MILP (valores internos de run_milp)
COVERAGE_LABELS = {"hard": "dura", "soft": "blanda"}


# ---------------------------------------------------------------------------
# Calendario y etiquetas
# ---------------------------------------------------------------------------

def day_date(instance: Instance, t: int) -> date:
    """Fecha del día t del horizonte (t=0 es start_date)."""
    return instance.calendar.start_date + timedelta(days=t)


def day_label(instance: Instance, t: int) -> str:
    """Etiqueta corta del día t: 'lun 05/01'."""
    d = day_date(instance, t)
    return f"{WEEKDAYS[d.weekday()]} {d.day:02d}/{d.month:02d}"


def day_labels(instance: Instance) -> list[str]:
    return [day_label(instance, t) for t in range(instance.calendar.days)]


def weekend_labels(instance: Instance) -> list[str]:
    return [day_label(instance, t) for t in range(instance.calendar.days)
            if instance.calendar.is_weekend(t)]


def holiday_labels(instance: Instance) -> list[str]:
    return [day_label(instance, t) for t in range(instance.calendar.days)
            if instance.calendar.is_holiday(t)]


def shift_names(instance: Instance) -> list[str]:
    """Nombres de turno en orden de catálogo (el orden de presentación en toda la app)."""
    return [s.name for s in instance.shifts]


# ---------------------------------------------------------------------------
# Resumen de instancia
# ---------------------------------------------------------------------------

def contracted_hours(instance: Instance) -> float:
    """Capacidad contractual de la plantilla sobre el horizonte: sum_i rho_i * theta_r * |T|/7."""
    weekly = {c.name: c.weekly_hours for c in instance.classes}
    weeks = instance.calendar.days / 7
    return sum(p.part_time * weekly[p.medical_class_name] * weeks for p in instance.physicians)


def summarize_instance(instance: Instance) -> dict:
    """Resumen agregado para cabeceras y fichas: plantilla, demanda y tensión demanda/capacidad."""
    shift_by = {s.name: s for s in instance.shifts}
    class_counts: dict[str, int] = {}
    for p in instance.physicians:
        class_counts[p.medical_class_name] = class_counts.get(p.medical_class_name, 0) + 1

    demand_slots = sum(b for per in instance.demand.values() for b in per.values())
    demand_hours = sum(shift_by[s].hours * b
                       for per in instance.demand.values() for s, b in per.items())
    capacity = contracted_hours(instance)

    return {
        "n": len(instance.physicians),
        "days": instance.calendar.days,
        "class_counts": dict(sorted(class_counts.items())),
        "demand_slots": demand_slots,
        "demand_hours": demand_hours,
        "contracted_hours": capacity,
        "realized_ratio": demand_hours / capacity if capacity else math.nan,
        "n_preferences": sum(len(by_s) for by_t in instance.preferences.values()
                             for by_s in by_t.values()),
    }


def demand_matrix(instance: Instance) -> pd.DataFrame:
    """Matriz turno x día con la demanda b_ts. NaN donde el turno no se ofrece (eta=0);
    0 donde se ofrece sin demanda registrada."""
    labels = day_labels(instance)
    data = {}
    for s in shift_names(instance):
        row = []
        for t in range(instance.calendar.days):
            if instance.eligibility.get(t, {}).get(s, False):
                row.append(float(instance.demand.get(t, {}).get(s, 0)))
            else:
                row.append(math.nan)
        data[s] = row
    return pd.DataFrame(data, index=labels).T


# ---------------------------------------------------------------------------
# Vistas de una solución
# ---------------------------------------------------------------------------

def coverage_frame(instance: Instance, roster: Roster) -> pd.DataFrame:
    """Una fila por celda (t, s) con demanda: demanda, asignados y déficit."""
    assigned: dict[tuple[int, str], int] = {}
    for (_, t, s) in roster.assignments:
        assigned[(t, s)] = assigned.get((t, s), 0) + 1
    rows = []
    for t in sorted(instance.demand):
        for s in shift_names(instance):
            b = instance.demand[t].get(s)
            if b is None:
                continue
            a = assigned.get((t, s), 0)
            rows.append({"t": t, "día": day_label(instance, t), "turno": s,
                         "demanda": b, "asignados": a, "déficit": max(0, b - a)})
    return pd.DataFrame(rows)


def deficit_matrix(instance: Instance, roster: Roster) -> pd.DataFrame:
    """Matriz turno x día con el déficit de cobertura max(0, b_ts - asignados).
    NaN donde no hay demanda (no aplica)."""
    assigned: dict[tuple[int, str], int] = {}
    for (_, t, s) in roster.assignments:
        assigned[(t, s)] = assigned.get((t, s), 0) + 1
    labels = day_labels(instance)
    data = {}
    for s in shift_names(instance):
        row = []
        for t in range(instance.calendar.days):
            b = instance.demand.get(t, {}).get(s)
            row.append(math.nan if b is None else float(max(0, b - assigned.get((t, s), 0))))
        data[s] = row
    return pd.DataFrame(data, index=labels).T


def physician_label(instance: Instance, p: Physician) -> str:
    return f"{p.id:02d} · {p.medical_class_name}"


def roster_matrix(instance: Instance, roster: Roster) -> pd.DataFrame:
    """Cuadrante médico x día: cada celda lleva los turnos asignados ('M', 'G1', 'M+G1', '')."""
    order = {s: k for k, s in enumerate(shift_names(instance))}
    cells: dict[tuple[int, int], list[str]] = {}
    for (i, t, s) in roster.assignments:
        cells.setdefault((i, t), []).append(s)
    labels = day_labels(instance)
    rows = {}
    for p in sorted(instance.physicians, key=lambda p: p.id):
        row = []
        for t in range(instance.calendar.days):
            names = sorted(cells.get((p.id, t), []), key=lambda s: order.get(s, 99))
            row.append("+".join(names))
        rows[physician_label(instance, p)] = row
    return pd.DataFrame(rows, index=labels).T


def per_physician_frame(instance: Instance, roster: Roster) -> pd.DataFrame:
    """Una fila por médico: horas frente al corredor [h_min, h_max] y a la carga media
    tau_i, guardias frente al tope g^max y a la cuota, y recuentos de calendario."""
    shift_by = {s.name: s for s in instance.shifts}
    cal = instance.calendar
    mean = mean_workload(instance)

    hours = {p.id: 0.0 for p in instance.physicians}
    guardias = {p.id: 0 for p in instance.physicians}
    active = {p.id: set() for p in instance.physicians}
    fds = {p.id: 0 for p in instance.physicians}
    fes = {p.id: 0 for p in instance.physicians}
    for (i, t, s) in roster.assignments:
        hours[i] += shift_by[s].hours
        if shift_by[s].is_guardia:
            guardias[i] += 1
        active[i].add(t)
        if cal.is_weekend(t):
            fds[i] += 1
        if cal.is_holiday(t):
            fes[i] += 1

    # cuota de guardias por médico: rho_i * tau_G, con tau_G sobre los no exentos (P\E)
    non_exempt_rho = sum(p.part_time for p in instance.physicians if not p.is_exempt)
    guardia_demand = sum(b for per in instance.demand.values()
                         for s, b in per.items() if shift_by[s].is_guardia)
    tau_g = guardia_demand / non_exempt_rho if non_exempt_rho else 0.0

    rows = []
    for p in sorted(instance.physicians, key=lambda p: p.id):
        h_min, h_max = instance.corridor(p)
        cap = instance.effective_max_guardias(p)
        rows.append({
            "id": p.id,
            "médico": physician_label(instance, p),
            "clase": p.medical_class_name,
            "rho": p.part_time,
            "exento": p.is_exempt,
            "horas": hours[p.id],
            "h_min": h_min,
            "h_max": h_max,
            "tau": p.part_time * mean,
            "guardias": guardias[p.id],
            "g_max": float(cap) if cap is not None else math.nan,
            "cuota_guardias": 0.0 if p.is_exempt else p.part_time * tau_g,
            "fds": fds[p.id],
            "fes": fes[p.id],
            "días_activos": len(active[p.id]),
        })
    return pd.DataFrame(rows)


def violation_marks(instance: Instance, roster: Roster) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Localiza cada violación dura: espejo celda a celda de `objective.violations`
    (mismas reglas y umbrales; aquí se devuelve DÓNDE en vez de CUÁNTO).

    Devuelve dos tablas: violaciones localizables en una celda (médico, día) y
    violaciones a nivel de médico (suelo del corredor, tope de guardias, descanso
    semanal), que no señalan una celda concreta."""
    shift_by = {s.name: s for s in instance.shifts}
    phys_by_id = {p.id: p for p in instance.physicians}
    class_of = {p.id: instance.class_of(p) for p in instance.physicians}
    adjuntos = {pid for pid, c in class_of.items() if not c.is_resident}

    by_phys_day: dict[tuple[int, int], set[str]] = {}
    adjunto_on: dict[tuple[int, str], int] = {}
    hours = {p.id: 0.0 for p in instance.physicians}
    guardias = {p.id: 0 for p in instance.physicians}
    active = {p.id: set() for p in instance.physicians}
    cells: list[dict] = []

    for (i, t, sname) in roster.assignments:
        shift = shift_by[sname]
        by_phys_day.setdefault((i, t), set()).add(sname)
        hours[i] += shift.hours
        active[i].add(t)
        if shift.is_guardia:
            guardias[i] += 1
            if phys_by_id[i].is_exempt:
                cells.append({"i": i, "t": t, "restricción": HARD_LABELS["v_exempt"],
                              "detalle": f"guardia {sname} asignada a un médico exento"})
        if i in adjuntos:
            adjunto_on[(t, sname)] = adjunto_on.get((t, sname), 0) + 1
        if not instance.qualification.get(class_of[i].name, {}).get(sname, False):
            cells.append({"i": i, "t": t, "restricción": HARD_LABELS["v_qualif"],
                          "detalle": f"la clase {class_of[i].name} no cualifica el turno {sname}"})
        if not instance.eligibility.get(t, {}).get(sname, False):
            cells.append({"i": i, "t": t, "restricción": HARD_LABELS["v_elig"],
                          "detalle": f"el turno {sname} no se ofrece ese día"})

    overlap = instance.overlapping_pairs()
    short_rest = instance.short_rest_pairs()

    for (i, t), sset in by_phys_day.items():
        names = sorted(sset)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                if frozenset((names[a], names[b])) in overlap:
                    cells.append({"i": i, "t": t, "restricción": HARD_LABELS["v_intraday"],
                                  "detalle": f"{names[a]} y {names[b]} se solapan el mismo día"})
        for s in sset:
            for s2 in by_phys_day.get((i, t + 1), ()):
                if (s, s2) in short_rest:
                    detail = f"{s} (día {t}) encadenado a {s2} (día {t + 1}) con descanso < δ"
                    cells.append({"i": i, "t": t, "restricción": HARD_LABELS["v_rest"],
                                  "detalle": detail})
                    cells.append({"i": i, "t": t + 1, "restricción": HARD_LABELS["v_rest"],
                                  "detalle": detail})

    for (i, t, sname) in roster.assignments:
        c = class_of[i]
        if c.is_resident and c.residency_year == 1 and shift_by[sname].is_guardia:
            if adjunto_on.get((t, sname), 0) == 0:
                cells.append({"i": i, "t": t, "restricción": HARD_LABELS["v_supervision"],
                              "detalle": f"R1 en guardia {sname} sin ningún adjunto en el turno"})

    phys: list[dict] = []
    weeks = [range(w, min(w + 7, instance.calendar.days))
             for w in range(0, instance.calendar.days, 7)]
    for p in instance.physicians:
        worked = active[p.id]
        for week in weeks:
            if len(week) < 2:
                continue
            has_free_pair = any(t not in worked and (t + 1) not in worked
                                for t in week if (t + 1) in week)
            if not has_free_pair:
                phys.append({"i": p.id, "restricción": HARD_LABELS["v_weekly"],
                             "detalle": f"semana de los días {week[0]}–{week[-1]} sin par de "
                                        "días libres consecutivos"})
        h_min = instance.corridor(p)[0]
        if hours[p.id] < h_min:
            phys.append({"i": p.id, "restricción": HARD_LABELS["v_corridor"],
                         "detalle": f"{hours[p.id]:.1f} h asignadas < suelo h^min = {h_min:.1f} h"})
        cap = instance.effective_max_guardias(p)
        if cap is not None and guardias[p.id] > cap:
            phys.append({"i": p.id, "restricción": HARD_LABELS["v_gmax"],
                         "detalle": f"{guardias[p.id]} guardias > tope g^max = {cap}"})

    def _with_labels(rows: list[dict], with_day: bool) -> pd.DataFrame:
        out = []
        for r in rows:
            row = {"médico": physician_label(instance, phys_by_id[r["i"]]),
                   "restricción": r["restricción"], "detalle": r["detalle"], "i": r["i"]}
            if with_day:
                row["día"] = day_label(instance, r["t"])
                row["t"] = r["t"]
            out.append(row)
        columns = (["médico", "día", "restricción", "detalle", "i", "t"] if with_day
                   else ["médico", "restricción", "detalle", "i"])
        return pd.DataFrame(out, columns=columns)

    return _with_labels(cells, with_day=True), _with_labels(phys, with_day=False)


def marked_cell_labels(cell_marks: pd.DataFrame) -> set[tuple[str, str]]:
    """Pares (etiqueta de médico, etiqueta de día) con violación, para señalarlos en el
    cuadrante (charts.roster_styler, parámetro `marked`)."""
    if cell_marks.empty:
        return set()
    return set(zip(cell_marks["médico"], cell_marks["día"]))


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluation_report(instance: Instance, roster: Roster, config: ObjectiveConfig) -> dict:
    """Informe completo de una solución: las tres Z, desglose de los siete términos
    (crudo, normalizador, peso y contribución con signo a Z_modelo) y las nueve hard.

    La contribución de preferencias entra con signo negativo (se maximiza); la de
    cobertura usa el w_cob resuelto (dominante por construcción si no se fijó a mano)."""
    b = evaluate(instance, roster, config)
    v = violations(instance, roster)
    norm = normalizers(instance, config)
    w_cob = resolve_w_cob(config, norm)
    mu = resolve_mu(config, norm)

    spec = [
        ("p_cob", b.p_cob, norm.n_cob, w_cob, w_cob * b.p_cob / norm.n_cob),
        ("p_erg", b.p_erg, norm.n_erg, config.w_erg, config.w_erg * b.p_erg / norm.n_erg),
        ("p_car", b.p_car, norm.n_car, config.w_car, config.w_car * b.p_car / norm.n_car),
        ("p_gua", b.p_gua, norm.n_gua, config.w_gua, config.w_gua * b.p_gua / norm.n_gua),
        ("p_cal", b.p_cal, norm.n_cal, config.w_cal, config.w_cal * b.p_cal / norm.n_cal),
        ("p_over", b.p_over, norm.n_over, config.w_over, config.w_over * b.p_over / norm.n_over),
        ("pref", b.pref, norm.n_pref, config.w_pref, -config.w_pref * b.pref / norm.n_pref),
    ]
    terms = pd.DataFrame(
        [{"término": TERM_LABELS[k], "clave": k, "crudo": raw, "normalizador": n,
          "peso": w, "contribución": c} for (k, raw, n, w, c) in spec]
    )
    hard = pd.DataFrame(
        [{"restricción": HARD_LABELS[k], "clave": k, "violación": getattr(v, k),
          "cumple": getattr(v, k) == 0} for k in HARD_LABELS]
    )
    return {
        "breakdown": b,
        "violations": v,
        "normalizers": norm,
        "z_clasica": z_clasica(b, norm, config),
        "z_modelo": z_modelo(b, norm, config),
        "z_aumentada": z_aumentada(b, v, norm, config),
        "w_cob": w_cob,
        "mu": mu,
        "feasible": v.is_feasible,
        "terms": terms,
        "hard": hard,
    }


# ---------------------------------------------------------------------------
# Resolución (envoltorios con medición de tiempo)
# ---------------------------------------------------------------------------

def run_greedy(instance: Instance, config: ObjectiveConfig) -> dict:
    """Heurística constructiva (Alg. 6.1). Devuelve {'roster', 'elapsed'}."""
    t0 = time.perf_counter()
    roster = greedy(instance, config)
    return {"roster": roster, "elapsed": time.perf_counter() - t0}


def auto_sa_config(instance: Instance, *, seed: int = 0, rounds: int = 400) -> SAConfig:
    """Esquema de enfriamiento robusto calculado para la instancia (sa.schedule)."""
    return schedule(instance, seed=seed, rounds=rounds)


def run_sa(instance: Instance, config: ObjectiveConfig, *,
           x0: Roster | None = None, sa_config: SAConfig | None = None,
           seed: int = 0) -> dict:
    """Simulated annealing (Alg. 6.2) desde `x0` (greedy si no se da). Devuelve
    {'result': SAResult, 'roster', 'sa_config', 'elapsed', 't_greedy'}."""
    t_greedy = 0.0
    if x0 is None:
        t0 = time.perf_counter()
        x0 = greedy(instance, config)
        t_greedy = time.perf_counter() - t0
    cfg = sa_config or auto_sa_config(instance, seed=seed)
    t0 = time.perf_counter()
    result = simulated_annealing(instance, x0, cfg, config)
    return {"result": result, "roster": result.roster, "sa_config": cfg,
            "elapsed": time.perf_counter() - t0, "t_greedy": t_greedy}


def milp_engines() -> dict[str, bool]:
    """Motores MILP disponibles en el entorno: {'highs': True, 'gurobi': bool}."""
    engines = {"highs": True}
    try:
        available = pulp.GUROBI(msg=False).available()
    except Exception:
        available = False
    engines["gurobi"] = bool(available)
    return engines


def run_milp(instance: Instance, config: ObjectiveConfig, *,
             time_limit: int = 60, coverage_hard: bool = True,
             engine: str = "highs") -> dict:
    """Resolución exacta (MILP). Si la cobertura dura resulta infactible, se repite en
    régimen blando (déficit penalizado), como en el harness del Cap. 7. Si Gurobi
    rechaza el modelo por licencia (la limitada por tamaño que trae pip cuando no hay
    licencia completa), se repite con HiGHS: `engine_used` informa del motor efectivo.
    Devuelve {'roster', 'info', 'coverage_mode', 'engine_used', 'elapsed'}."""
    def make_solver(eng: str):
        if eng == "gurobi":
            return pulp.GUROBI(msg=False, timeLimit=time_limit)
        return pulp.HiGHS(msg=False, timeLimit=time_limit)

    def attempt(eng: str) -> tuple[Roster, dict, str]:
        roster, info = solve_exact(instance, config, solver=make_solver(eng),
                                   break_symmetry=False, coverage_hard=coverage_hard)
        mode = "hard" if coverage_hard else "soft"
        if coverage_hard and info["status"] == "Infeasible":
            roster, info = solve_exact(instance, config, solver=make_solver(eng),
                                       break_symmetry=False, coverage_hard=False)
            mode = "soft"
        return roster, info, mode

    t0 = time.perf_counter()
    engine_used = engine
    try:
        roster, info, mode = attempt(engine)
    except Exception as exc:
        if engine != "gurobi" or "license" not in str(exc).lower():
            raise
        engine_used = "highs"
        roster, info, mode = attempt("highs")
    return {"roster": roster, "info": info, "coverage_mode": mode,
            "engine_used": engine_used, "elapsed": time.perf_counter() - t0}


def make_solution_record(*, name: str, instance_name: str, solver: str,
                         instance: Instance, roster: Roster, config: ObjectiveConfig,
                         solver_params: dict | None = None,
                         extra_metrics: dict | None = None) -> store.SolutionRecord:
    """Construye el registro de solución con las métricas estándar (las tres Z,
    factibilidad, déficit) más las propias del método (`extra_metrics`)."""
    report = evaluation_report(instance, roster, config)
    metrics = {
        "z_modelo": report["z_modelo"],
        "z_clasica": report["z_clasica"],
        "z_aumentada": report["z_aumentada"],
        "feasible": report["feasible"],
        "p_cob": report["breakdown"].p_cob,
        "violaciones": report["violations"].total,
    }
    metrics.update(extra_metrics or {})
    return store.SolutionRecord(
        name=name,
        instance_name=instance_name,
        solver=solver,
        created=store.now_iso(),
        params={"objetivo": config.model_dump(), "solver": solver_params or {}},
        metrics=metrics,
        assignments=sorted(roster.assignments),
    )


# ---------------------------------------------------------------------------
# Edición de instancias (tablas UI <-> Instance validada)
# ---------------------------------------------------------------------------

def rebuild_instance(instance: Instance, **updates) -> Instance:
    """Reconstruye la instancia (inmutable) con campos sustituidos; los validadores de
    `Instance` corren de nuevo, así que una edición incoherente falla aquí con mensaje."""
    data = instance.model_dump()
    data.update(updates)
    return Instance.model_validate(data)


def physicians_frame(instance: Instance) -> pd.DataFrame:
    """Plantilla como tabla editable: una fila por médico."""
    rows = [{"id": p.id, "clase": p.medical_class_name, "part_time": p.part_time,
             "exento": p.is_exempt,
             "max_guardias": p.max_guardias if p.max_guardias is not None else pd.NA}
            for p in sorted(instance.physicians, key=lambda p: p.id)]
    df = pd.DataFrame(rows)
    return df.astype({"max_guardias": "Int64"})


def physicians_from_frame(df: pd.DataFrame) -> list[Physician]:
    """Tabla editada -> lista de Physician. Filas nuevas sin id reciben el primero libre;
    max_guardias vacío queda en None (hereda el default de la clase)."""
    used = {int(r["id"]) for _, r in df.iterrows() if not pd.isna(r["id"])}
    next_id = (max(used) + 1) if used else 1
    physicians = []
    for _, r in df.iterrows():
        if pd.isna(r["id"]):
            pid, next_id = next_id, next_id + 1
        else:
            pid = int(r["id"])
        cap = r["max_guardias"]
        physicians.append(Physician(
            id=pid,
            medical_class_name=str(r["clase"]),
            part_time=float(r["part_time"]) if not pd.isna(r["part_time"]) else 1.0,
            is_exempt=bool(r["exento"]) if not pd.isna(r["exento"]) else False,
            max_guardias=None if pd.isna(cap) else int(cap),
        ))
    return physicians


def _day_table(instance: Instance, values: dict[int, dict[str, float]],
               default: float, dtype: str) -> pd.DataFrame:
    """Tabla día x turno con columna inicial 'día' (solo lectura en las páginas)."""
    names = shift_names(instance)
    rows = []
    for t in range(instance.calendar.days):
        row = {"día": day_label(instance, t)}
        for s in names:
            row[s] = values.get(t, {}).get(s, default)
        rows.append(row)
    df = pd.DataFrame(rows, index=range(instance.calendar.days))
    return df.astype({s: dtype for s in names})


def demand_frame(instance: Instance) -> pd.DataFrame:
    """Demanda b_ts como tabla editable (0 = sin demanda)."""
    return _day_table(instance, instance.demand, default=0, dtype="int64")


def demand_from_frame(df: pd.DataFrame, instance: Instance) -> dict[int, dict[str, int]]:
    """Tabla editada -> b_ts disperso (se descartan ceros y vacíos)."""
    demand: dict[int, dict[str, int]] = {}
    for t in df.index:
        for s in shift_names(instance):
            v = df.loc[t, s]
            if not pd.isna(v) and int(v) > 0:
                demand.setdefault(int(t), {})[s] = int(v)
    return demand


def eligibility_frame(instance: Instance) -> pd.DataFrame:
    """Elegibilidad eta_ts como tabla editable de booleanos."""
    return _day_table(instance, instance.eligibility, default=False, dtype="bool")


def eligibility_from_frame(df: pd.DataFrame, instance: Instance) -> dict[int, dict[str, bool]]:
    """Tabla editada -> matriz eta completa (día x turno)."""
    return {int(t): {s: bool(df.loc[t, s]) for s in shift_names(instance)} for t in df.index}


def qualification_frame(instance: Instance) -> pd.DataFrame:
    """Cualificación gamma_rs como tabla editable clase x turno."""
    names = shift_names(instance)
    rows = {c.name: [bool(instance.qualification.get(c.name, {}).get(s, False)) for s in names]
            for c in instance.classes}
    return pd.DataFrame.from_dict(rows, orient="index", columns=names)


def qualification_from_frame(df: pd.DataFrame) -> dict[str, dict[str, bool]]:
    return {str(c): {str(s): bool(df.loc[c, s]) for s in df.columns} for c in df.index}


def preferences_frame(instance: Instance, physician_id: int) -> pd.DataFrame:
    """Preferencias p_its de un médico como tabla editable día x turno (0 = indiferente,
    >0 desea, <0 rechaza)."""
    prefs = instance.preferences.get(physician_id, {})
    return _day_table(instance, prefs, default=0.0, dtype="float64")


def preferences_with_update(instance: Instance, physician_id: int,
                            df: pd.DataFrame) -> dict[int, dict[int, dict[str, float]]]:
    """Sustituye las preferencias de un médico por la tabla editada (los ceros se
    descartan) y devuelve el diccionario p completo listo para rebuild_instance."""
    per_day: dict[int, dict[str, float]] = {}
    for t in df.index:
        for s in shift_names(instance):
            v = df.loc[t, s]
            if not pd.isna(v) and float(v) != 0.0:
                per_day.setdefault(int(t), {})[s] = float(v)
    prefs = {i: by_t for i, by_t in instance.preferences.items() if i != physician_id}
    if per_day:
        prefs[physician_id] = per_day
    return prefs


# ---------------------------------------------------------------------------
# Generación
# ---------------------------------------------------------------------------

def generate_instance(settings: Settings, seed: int) -> Instance:
    """Genera la instancia sintética reproducible (misma settings + semilla -> misma
    instancia)."""
    return generate(settings, seed)


def shift_specs_frame(specs: list[ShiftSpec]) -> pd.DataFrame:
    """Catálogo de turnos como tabla editable (una fila por turno; el peso gobierna el
    reparto de la demanda por patrón)."""
    rows = [{"nombre": s.shift.name, "inicio": s.shift.start, "fin": s.shift.end,
             "horas": s.shift.hours, "guardia": s.shift.is_guardia, "peso": s.weight}
            for s in specs]
    return pd.DataFrame(rows)


def shift_specs_from_frame(df: pd.DataFrame) -> list[ShiftSpec]:
    """Tabla editada -> catálogo de ShiftSpec validados (filas sin nombre se descartan)."""
    specs = []
    for _, r in df.iterrows():
        if pd.isna(r["nombre"]) or not str(r["nombre"]).strip():
            continue
        specs.append(ShiftSpec(
            shift=Shift(name=str(r["nombre"]).strip(), start=r["inicio"], end=r["fin"],
                        hours=float(r["horas"]),
                        is_guardia=bool(r["guardia"]) if not pd.isna(r["guardia"]) else False),
            weight=float(r["peso"]) if not pd.isna(r["peso"]) else 1.0,
        ))
    return specs


def build_manual_instance(*, settings: Settings, class_counts: dict[str, int], seed: int,
                          generate_demand: bool = True) -> Instance:
    """Instancia construida de forma DETERMINISTA, sin muestreo de plantilla: los médicos
    salen de los recuentos por clase (ids consecutivos, jornada completa; part-time,
    exenciones y g^max individuales se ajustan después en la página Instancia).

    La demanda, si se pide, usa el mismo patrón forward del generador con su substream de
    semilla (`generator._build_demand`; se invoca la función del paquete para no duplicar
    la semántica). Con `generate_demand=False` la demanda queda vacía, para introducirla a
    mano en el editor. Las tablas eta/gamma salen de los defaults del blueprint (o de las
    que traiga `settings` si se dieron explícitas)."""
    physicians = []
    next_id = 1
    for c in settings.classes:
        for _ in range(int(class_counts.get(c.name, 0))):
            physicians.append(Physician(id=next_id, medical_class_name=c.name))
            next_id += 1
    demand: dict[int, dict[str, int]] = {}
    if generate_demand:
        demand = _build_demand(settings, physicians, random.Random(f"{seed}:demand"))
    return Instance(
        seed=seed,
        calendar=settings.calendar,
        regulations=settings.regulations,
        classes=settings.classes,
        shifts=[spec.shift for spec in settings.shifts],
        physicians=physicians,
        demand=demand,
        eligibility=settings.eligibility,
        qualification=settings.qualification,
        preferences={},
    )
