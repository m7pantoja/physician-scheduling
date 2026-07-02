"""Resolución exacta del modelo del Cap. 5 como programa lineal entero mixto (MILP).

Da el óptimo Z* de la formulación de modelo (`z_modelo`, cobertura blanda), la referencia contra
la que se mide el SA. Se construye con PuLP (capa de modelado portable a otros motores) por
bloques verificables (variables vivas, restricciones hard, auxiliares y objetivo) y se valida
cruzando el Z* del solver con `objective.z_modelo` sobre la propia solución que devuelve."""

import pulp

from rostering.domain import Instance
from rostering.objective import (
    ObjectiveConfig, mean_workload, evaluate, normalizers, resolve_w_cob, violations,
    z_clasica, z_modelo,
)
from rostering.roster import Roster


def _live_triples(instance: Instance) -> list[tuple[int, int, str]]:
    """Ternas (i,t,s) cuya variable x_{its} se crea: las que ninguna hard estructural
    (cualificación gamma, elegibilidad eta, exención E) fija ya a 0. Crear solo las vivas
    reduce el modelo a su tamaño efectivo sin perder ninguna decisión real."""
    class_name = {p.id: instance.class_of(p).name for p in instance.physicians}
    is_guardia = {s.name: s.is_guardia for s in instance.shifts}
    triples = []
    for p in instance.physicians:
        for t in range(instance.calendar.days):
            for s in instance.shifts:
                if not instance.eligibility.get(t, {}).get(s.name, False):                      # eta
                    continue
                if not instance.qualification.get(class_name[p.id], {}).get(s.name, False):     # gamma
                    continue
                if is_guardia[s.name] and p.is_exempt:                                           # E
                    continue
                triples.append((p.id, t, s.name))
    return triples


def _optimality_info(model):
    """Extrae (proven_optimal, gap_relativo, cota_dual) del solver subyacente (Gurobi o
    HiGHS): el status de PuLP no distingue óptimo certificado de mejor solución por tiempo.
    Devuelve (None, None, None) si no puede leerse."""
    sm = getattr(model, "solverModel", None)
    if sm is None:
        return None, None, None
    # Gurobi: solverModel es un gurobipy.Model (Status, MIPGap, ObjBound)
    try:
        import gurobipy as gp
        if isinstance(sm, gp.Model):
            proven = (sm.Status == gp.GRB.OPTIMAL)
            gap = sm.MIPGap if sm.SolCount > 0 else None
            bound = sm.ObjBound if sm.SolCount > 0 else None
            return proven, gap, bound
    except Exception:
        pass
    # HiGHS: solverModel expone getModelStatus()/getInfo()
    try:
        import highspy
        st = sm.getModelStatus()
        h = sm.getInfo()
        proven = (st == highspy.HighsModelStatus.kOptimal)
        return proven, getattr(h, "mip_gap", None), getattr(h, "mip_dual_bound", None)
    except Exception:
        return None, None, None


def solve_exact(instance: Instance, config: ObjectiveConfig | None = None, solver=None,
                break_symmetry: bool = True, coverage_hard: bool = True):
    """Construye y resuelve el MILP exacto de `instance`; devuelve (Roster, info).

    `break_symmetry` ordena por carga a los médicos intercambiables para podar permutaciones
    equivalentes (clave para que el solver certifique). `coverage_hard` (por defecto): la
    cobertura es dura y el cross-check usa `z_clasica`; con `False` la cobertura es blanda
    (déficit u penalizado, régimen de estrés) y se cruza con `z_modelo`."""
    config = config or ObjectiveConfig()
    model = pulp.LpProblem("exact_rostering", pulp.LpMinimize)

    # --- precómputos (espejo de domain.py / objective.py) ---
    shift_by = {s.name: s for s in instance.shifts}
    class_of = {p.id: instance.class_of(p) for p in instance.physicians}
    adjuntos = {p.id for p in instance.physicians if not class_of[p.id].is_resident}   # A: no residentes
    overlap = instance.overlapping_pairs()        # O: pares NO ordenados con solape estricto
    short_rest = instance.short_rest_pairs()       # I: pares ORDENADOS con menos de delta horas de descanso (incl. self-pairs)

    # --- PASO 1: variables de decisión x_{its} (binarias, solo las vivas) ---
    triples = _live_triples(instance)
    x = {(i, t, s): pulp.LpVariable(f"x_{i}_{t}_{s}", cat="Binary") for (i, t, s) in triples}

    # índices auxiliares + expresiones lineales por médico (reusadas aquí y en el paso 3)
    shifts_of: dict[tuple[int, int], list[str]] = {}   # (i,t) -> turnos vivos de i ese día  (O, I)
    phys_of: dict[tuple[int, str], list[int]] = {}     # (t,s) -> médicos con (i,t,s) vivo   (cobertura, supervisión)
    horas: dict[int, list] = {p.id: [] for p in instance.physicians}     # términos d_s*x_{its}  (corredor, equidad)
    guardias: dict[int, list] = {p.id: [] for p in instance.physicians}  # x_{its} con s en G       (g^max, equidad)
    for (i, t, s) in triples:
        shifts_of.setdefault((i, t), []).append(s)
        phys_of.setdefault((t, s), []).append(i)
        horas[i].append(shift_by[s].hours * x[(i, t, s)])
        if shift_by[s].is_guardia:
            guardias[i].append(x[(i, t, s)])

    # --- PASO 2: restricciones HARD que dependen solo de x (las que usan a van en el paso 3) ---

    # (cobertura) coverage_hard: sum_i x_{its} >= b_{ts} dura; si no, déficit u_{ts} >= 0
    # penalizado (eq:cobertura-soft). Fila solo donde b_{ts} > 0; la sobrecobertura no penaliza.
    u: dict[tuple[int, str], pulp.LpVariable] = {}
    for t, per in instance.demand.items():
        for sname, b in per.items():
            if b <= 0:
                continue
            cov = pulp.lpSum(x[(i, t, sname)] for i in phys_of.get((t, sname), []))
            if coverage_hard:
                model += (cov >= b, f"cob_{t}_{sname}")
            else:
                u[(t, sname)] = pulp.LpVariable(f"u_{t}_{sname}", lowBound=0)
                model += (cov + u[(t, sname)] >= b, f"cob_{t}_{sname}")

    # (O, intradía) x_{its} + x_{its'} <= 1 para {s,s'} en O en un mismo (i,t) (eq:intradia)
    for (i, t), sset in shifts_of.items():
        for a in range(len(sset)):
            for b in range(a + 1, len(sset)):
                if frozenset((sset[a], sset[b])) in overlap:
                    model += (x[(i, t, sset[a])] + x[(i, t, sset[b])] <= 1, f"O_{i}_{t}_{sset[a]}_{sset[b]}")

    # (I, descanso entre jornadas) x_{its} + x_{i,t+1,s'} <= 1 para (s,s') en I (eq:descanso-12h)
    for (i, t), sset in shifts_of.items():
        for s in sset:
            for s2 in shifts_of.get((i, t + 1), []):
                if (s, s2) in short_rest:
                    model += (x[(i, t, s)] + x[(i, t + 1, s2)] <= 1, f"I_{i}_{t}_{s}_{s2}")

    # (supervisión R1) x_{R1,t,s} <= sum_{j en A} x_{j,t,s} en guardia (eq:supervision-r1)
    for (i, t, s) in triples:
        c = class_of[i]
        if c.is_resident and c.residency_year == 1 and shift_by[s].is_guardia:
            model += (x[(i, t, s)] <= pulp.lpSum(x[(j, t, s)] for j in phys_of.get((t, s), [])
                                                 if j in adjuntos), f"sup_{i}_{t}_{s}")

    # (corredor de horas) suelo h_i^min duro; el techo no veta bajo opt-out: el exceso
    # o_i >= horas_i - h_i^max se penaliza como sobre-jornada en el objetivo (eq:sobre-jornada).
    o: dict[int, pulp.LpVariable] = {}
    for p in instance.physicians:
        h_min, h_max = instance.corridor(p)
        model += (pulp.lpSum(horas[p.id]) >= h_min, f"corrLo_{p.id}")
        o[p.id] = pulp.LpVariable(f"o_{p.id}", lowBound=0)
        model += (o[p.id] >= pulp.lpSum(horas[p.id]) - h_max, f"over_{p.id}")

    # (g^max) sum_{t, s en G} x_{its} <= g_i^max (eq:tope-guardias). cap=None (sin tope) => sin fila.
    for p in instance.physicians:
        cap = instance.effective_max_guardias(p)
        if cap is not None:
            model += (pulp.lpSum(guardias[p.id]) <= cap, f"gmax_{p.id}")

    # --- PASO 3: variables auxiliares (continuas, salvo a) + objetivo z_modelo ---
    days = instance.calendar.days
    cal = instance.calendar
    physicians = instance.physicians
    L = config.l_max

    # día-activo a_{it} en {0,1}: 1 sii i trabaja algún turno el día t (eq:stretch, eq:finde).
    a = {(p.id, t): pulp.LpVariable(f"a_{p.id}_{t}", cat="Binary")
         for p in physicians for t in range(days)}
    for (i, t, s) in triples:                                   # cota inferior desagregada: a >= x_{its}
        model += (a[(i, t)] >= x[(i, t, s)], f"actLB_{i}_{t}_{s}")
    for p in physicians:                                        # cota superior agregada (exactness-critical)
        for t in range(days):
            model += (a[(p.id, t)] <= pulp.lpSum(x[(p.id, t, s)] for s in shifts_of.get((p.id, t), [])),
                      f"actUB_{p.id}_{t}")

    # equidad de CARGA (L1, sobre TODO P): horas_i - tau_i = H^U_i - H^L_i, tau_i = rho_i*H (eq:equidad-carga)
    mean = mean_workload(instance)
    hu = {p.id: pulp.LpVariable(f"HU_{p.id}", lowBound=0) for p in physicians}
    hl = {p.id: pulp.LpVariable(f"HL_{p.id}", lowBound=0) for p in physicians}
    for p in physicians:
        model += (pulp.lpSum(horas[p.id]) - p.part_time * mean == hu[p.id] - hl[p.id], f"carga_{p.id}")

    # cuotas min-max (denominador SOLO no exentos P\E), idénticas a objective.py
    non_exempt = [p for p in physicians if not p.is_exempt]
    sum_rho = sum(p.part_time for p in non_exempt)
    gd = sum(b for per in instance.demand.values() for s, b in per.items() if shift_by[s].is_guardia)
    fds = sum(b for t, per in instance.demand.items() for s, b in per.items() if cal.is_weekend(t))
    fes = sum(b for t, per in instance.demand.items() for s, b in per.items() if cal.is_holiday(t))
    tau_g = gd / sum_rho if sum_rho else 0.0
    tau_c = {"fds": fds / sum_rho if sum_rho else 0.0, "fes": fes / sum_rho if sum_rho else 0.0}

    # recuentos por categoría POR ASIGNACIÓN (no por día): n^c_i = sum_{t en T^c, s} x_{its}
    n_fds = {p.id: [] for p in physicians}
    n_fes = {p.id: [] for p in physicians}
    for (i, t, s) in triples:
        if cal.is_weekend(t):
            n_fds[i].append(x[(i, t, s)])
        if cal.is_holiday(t):
            n_fes[i].append(x[(i, t, s)])

    # equidad de GUARDIAS (min-max, solo P\E): G^U >= n^G_i - rho_i*tau_G ; G^L >= rho_i*tau_G - n^G_i (eq:guardias-eq)
    gu = pulp.LpVariable("GU", lowBound=0)
    gl = pulp.LpVariable("GL", lowBound=0)
    for p in non_exempt:
        ng = pulp.lpSum(guardias[p.id])
        model += (gu >= ng - p.part_time * tau_g, f"guaU_{p.id}")
        model += (gl >= p.part_time * tau_g - ng, f"guaL_{p.id}")

    # equidad de CALENDARIO (min-max por categoría, solo P\E) (eq:calendario)
    ju = {c: pulp.LpVariable(f"JU_{c}", lowBound=0) for c in ("fds", "fes")}
    jl = {c: pulp.LpVariable(f"JL_{c}", lowBound=0) for c in ("fds", "fes")}
    for c, counts in (("fds", n_fds), ("fes", n_fes)):
        for p in non_exempt:
            nc = pulp.lpSum(counts[p.id])
            model += (ju[c] >= nc - p.part_time * tau_c[c], f"calU_{c}_{p.id}")
            model += (jl[c] >= p.part_time * tau_c[c] - nc, f"calL_{c}_{p.id}")

    # ergonomía (rachas): pi_{i,t0} >= sum_{u=t0}^{t0+L} a_{iu} - L, por ventana de L+1 días (eq:stretch)
    pi = {}
    for p in physicians:
        for t0 in range(days - L):
            pi[(p.id, t0)] = pulp.LpVariable(f"pi_{p.id}_{t0}", lowBound=0)
            model += (pi[(p.id, t0)] >= pulp.lpSum(a[(p.id, u)] for u in range(t0, t0 + L + 1)) - L,
                      f"racha_{p.id}_{t0}")

    # ergonomía (finde partido): sigma_{it} >= +/-(a_{it} - a_{i,t+1}), pares consecutivos ambos finde (eq:finde)
    weekend_pairs = [t for t in range(days - 1) if cal.is_weekend(t) and cal.is_weekend(t + 1)]
    sigma = {}
    for p in physicians:
        for t in weekend_pairs:
            sigma[(p.id, t)] = pulp.LpVariable(f"sig_{p.id}_{t}", lowBound=0)
            model += (sigma[(p.id, t)] >= a[(p.id, t)] - a[(p.id, t + 1)], f"sigA_{p.id}_{t}")
            model += (sigma[(p.id, t)] >= a[(p.id, t + 1)] - a[(p.id, t)], f"sigB_{p.id}_{t}")

    # DESCANSO SEMANAL (hard puro): >=1 par de días consecutivos libres por (médico, semana) (eq:semanal).
    # f_{it} <= 1-a_{it}, f_{it} <= 1-a_{i,t+1}, sum_pares f >= 1. Continua [0,1] (las 2 cotas bastan).
    for p in physicians:
        for w0 in range(0, days, 7):
            week = range(w0, min(w0 + 7, days))
            if len(week) < 2:
                continue
            pairs = [t for t in week if (t + 1) in week]
            fw = {}
            for t in pairs:
                fw[t] = pulp.LpVariable(f"f_{p.id}_{t}", lowBound=0, upBound=1)
                model += (fw[t] <= 1 - a[(p.id, t)], f"freeA_{p.id}_{t}")
                model += (fw[t] <= 1 - a[(p.id, t + 1)], f"freeB_{p.id}_{t}")
            model += (pulp.lpSum(fw.values()) >= 1, f"sem_{p.id}_{w0}")

    # rompe-simetría (opcional): médicos intercambiables ordenados por carga decreciente. Toda
    # solución tiene una permutación equivalente que lo cumple => no se pierde el óptimo, solo
    # sus copias. Se omiten los grupos con preferencias (dejan de ser intercambiables).
    if break_symmetry:
        groups: dict = {}
        for p in physicians:
            sig = (p.medical_class_name, p.part_time, p.is_exempt,
                   instance.effective_max_guardias(p), bool(instance.preferences.get(p.id)))
            groups.setdefault(sig, []).append(p.id)
        for sig, ids in groups.items():
            if sig[4]:                       # tienen preferencias -> no intercambiables, no ordenar
                continue
            ids.sort()
            for i_hi, i_lo in zip(ids, ids[1:]):
                model += (pulp.lpSum(horas[i_hi]) >= pulp.lpSum(horas[i_lo]), f"sym_{i_hi}_{i_lo}")

    # --- objetivo: idéntico a objective.py (mismos w_k, N_k, signos); con coverage_hard el
    # término de cobertura se anula (u vacío) ---
    norm = normalizers(instance, config)
    w_cob = resolve_w_cob(config, norm)
    prefs = instance.preferences
    pref_term = pulp.lpSum(prefs.get(i, {}).get(t, {}).get(s, 0.0) * x[(i, t, s)] for (i, t, s) in triples)
    model += (
        w_cob / norm.n_cob * pulp.lpSum(u.values())
        + config.w_erg / norm.n_erg * (pulp.lpSum(pi.values()) + pulp.lpSum(sigma.values()))
        + config.w_car / norm.n_car * pulp.lpSum(hu[p.id] + hl[p.id] for p in physicians)
        + config.w_gua / norm.n_gua * (gu + gl)
        + config.w_cal / norm.n_cal * (ju["fds"] + jl["fds"] + ju["fes"] + jl["fes"])
        + config.w_over / norm.n_over * pulp.lpSum(o.values())
        - config.w_pref / norm.n_pref * pref_term
    )

    # --- PASO 4: resolver + extraer Roster + cross-check Z* contra objective.py ---
    model.solve(solver or pulp.HiGHS(msg=False))
    roster = Roster(assignments={k for k, var in x.items() if (var.value() or 0.0) > 0.5})

    z_star = pulp.value(model.objective)
    breakdown = evaluate(instance, roster, config)                       # mismo objetivo, vía objective.py
    z_ref = (z_clasica if coverage_hard else z_modelo)(breakdown, norm, config)
    viol = violations(instance, roster)
    proven_optimal, gap, dual_bound = _optimality_info(model)            # del solver, no de PuLP
    # cross-check estricto solo si el óptimo está certificado; el incumbent arrastra el ruido
    # de la tolerancia de integralidad del solver (~1e-5).
    match_tol = 1e-6 if proven_optimal else 1e-4
    info = {
        "status": pulp.LpStatus[model.status],
        "proven_optimal": proven_optimal,       # ¿el solver CERTIFICÓ el óptimo? (gap=0)
        "gap": gap,                             # (incumbente - cota dual)/|incumbente|; >0 = no certificado
        "dual_bound": dual_bound,               # mejor cota inferior conocida (acota el óptimo real)
        "z_star": z_star,                       # valor de la mejor solución que reporta el solver
        "z_ref": z_ref,                         # z_modelo recomputado sobre el roster extraído
        "match": z_star is not None and abs(z_star - z_ref) < match_tol,   # cross-check de oro
        "feasible": viol.is_feasible,
        "n_vars": len(model.variables()),
        "n_cons": len(model.constraints),
    }
    return roster, info
