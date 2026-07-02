"""Evaluador incremental del objetivo aumentado (Sección 6.3.4) para el simulated annealing.

Un movimiento de un bit solo altera los términos que dependen de su terna (i,t,s); manteniendo
esas magnitudes en un estado, el incremento Delta se recalcula en tiempo proporcional a los
términos afectados y no al tamaño del problema. El estado se inicializa con el evaluador
completo; cada `apply` lo actualiza y devuelve el Delta, y es su propia inversa (aplicado dos
veces deja todo igual), lo que el SA usa para deshacer un movimiento rechazado."""

from rostering.domain import Instance
from rostering.objective import (
    ObjectiveConfig, mean_workload, evaluate, normalizers, resolve_mu, resolve_w_cob, violations,
)
from rostering.roster import Roster


class DeltaEvaluator:
    """Estado mutable del objetivo aumentado bajo movimientos de un bit. `score` es Z del estado
    actual; `apply(i,t,s)` togglea la asignación, actualiza el estado y devuelve el Delta del score."""

    def __init__(self, instance: Instance, roster: Roster, config: ObjectiveConfig | None = None):
        self.config = config or ObjectiveConfig()
        norm = normalizers(instance, self.config)
        self.w_cob = resolve_w_cob(self.config, norm)
        self.mu = resolve_mu(self.config, norm)
        c = self.config
        self.we, self.wc, self.wg, self.wl, self.wp = c.w_erg, c.w_car, c.w_gua, c.w_cal, c.w_pref
        self.wo = c.w_over
        self.ne, self.nc, self.ng, self.nl, self.npref, self.ncob = (
            norm.n_erg, norm.n_car, norm.n_gua, norm.n_cal, norm.n_pref, norm.n_cob)
        self.no = norm.n_over

        # --- lookups inmutables derivados de la instancia ---
        self.days = instance.calendar.days
        self.l_max = c.l_max
        self.shift = {s.name: s for s in instance.shifts}
        self.demand = instance.demand
        self.qual = instance.qualification
        self.elig = instance.eligibility
        self.overlap = instance.overlapping_pairs()
        self.short_rest = instance.short_rest_pairs()
        cal = instance.calendar
        self.is_wknd = {t: cal.is_weekend(t) for t in range(self.days)}
        self.is_hol = {t: cal.is_holiday(t) for t in range(self.days)}

        self.phys = [p.id for p in instance.physicians]
        class_of = {p.id: instance.class_of(p) for p in instance.physicians}
        self.class_name = {p.id: class_of[p.id].name for p in instance.physicians}
        self.part_time = {p.id: p.part_time for p in instance.physicians}
        self.exempt = {p.id: p.is_exempt for p in instance.physicians}
        self.is_r1 = {p.id: class_of[p.id].is_resident and class_of[p.id].residency_year == 1
                      for p in instance.physicians}
        self.is_adj = {p.id: not class_of[p.id].is_resident for p in instance.physicians}
        self.corridor = {p.id: instance.corridor(p) for p in instance.physicians}
        self.cap = {p.id: instance.effective_max_guardias(p) for p in instance.physicians}
        self.non_exempt = [p.id for p in instance.physicians if not p.is_exempt]
        self.pref_val = {(pid, t, s): v
                         for pid, by_t in instance.preferences.items()
                         for t, by_s in by_t.items() for s, v in by_s.items()}

        # cuotas de equidad (fijas: dependen solo de la demanda y la plantilla)
        mean = mean_workload(instance)
        self.tau = {p.id: p.part_time * mean for p in instance.physicians}
        sum_rho = sum(self.part_time[i] for i in self.non_exempt)
        gd = sum(b for per in self.demand.values() for s, b in per.items() if self.shift[s].is_guardia)
        fds = sum(b for t, per in self.demand.items() for s, b in per.items() if self.is_wknd[t])
        fes = sum(b for t, per in self.demand.items() for s, b in per.items() if self.is_hol[t])
        self.tau_g = gd / sum_rho if sum_rho else 0.0
        self.tau_fds = fds / sum_rho if sum_rho else 0.0
        self.tau_fes = fes / sum_rho if sum_rho else 0.0

        # --- estado agregado, construido recorriendo el roster ---
        self.assigned = set(roster.assignments)
        self.cov: dict[tuple[int, str], float] = {}
        self.hours = {i: 0.0 for i in self.phys}
        self.guardias = {i: 0.0 for i in self.phys}
        self.wknd = {i: 0.0 for i in self.phys}
        self.hol = {i: 0.0 for i in self.phys}
        self.day_shifts: dict[tuple[int, int], set[str]] = {}
        self.active = {i: set() for i in self.phys}
        self.adj_on: dict[tuple[int, str], float] = {}
        self.r1_on: dict[tuple[int, str], float] = {}
        for (i, t, s) in self.assigned:
            sh = self.shift[s]
            self.cov[(t, s)] = self.cov.get((t, s), 0) + 1
            self.hours[i] += sh.hours
            self.day_shifts.setdefault((i, t), set()).add(s)
            self.active[i].add(t)
            if self.is_wknd[t]:
                self.wknd[i] += 1
            if self.is_hol[t]:
                self.hol[i] += 1
            if sh.is_guardia:
                self.guardias[i] += 1
                if self.is_adj[i]:
                    self.adj_on[(t, s)] = self.adj_on.get((t, s), 0) + 1
                if self.is_r1[i]:
                    self.r1_on[(t, s)] = self.r1_on.get((t, s), 0) + 1

        # --- valores iniciales de los términos: del evaluador completo (verificado) ---
        b = evaluate(instance, roster, self.config)
        v = violations(instance, roster)
        self.p_cob, self.p_erg, self.p_car, self.pref = b.p_cob, b.p_erg, b.p_car, b.pref
        self.p_over = b.p_over
        self.p_gua = self._spread(self.guardias, self.tau_g)
        self.cal_fds = self._spread(self.wknd, self.tau_fds)
        self.cal_fes = self._spread(self.hol, self.tau_fes)
        self.v_intraday, self.v_qualif, self.v_elig = v.v_intraday, v.v_qualif, v.v_elig
        self.v_rest, self.v_weekly, self.v_supervision = v.v_rest, v.v_weekly, v.v_supervision
        self.v_exempt, self.v_corridor, self.v_gmax = v.v_exempt, v.v_corridor, v.v_gmax
        self.score = self._score()

    # ------------------------------------------------------------------ helpers

    def _spread(self, counts: dict[int, float], tau: float) -> float:
        """Horquilla min-max sobre P\\E: mayor exceso + mayor defecto frente a la cuota rho_i*tau."""
        upper = lower = 0.0
        for i in self.non_exempt:
            dev = counts[i] - self.part_time[i] * tau
            upper = max(upper, dev)
            lower = max(lower, -dev)
        return upper + lower

    def _score(self) -> float:
        """Z desde los términos actuales (misma fórmula que `z_aumentada`)."""
        soft = (self.we * self.p_erg / self.ne + self.wc * self.p_car / self.nc
                + self.wg * self.p_gua / self.ng + self.wl * (self.cal_fds + self.cal_fes) / self.nl
                + self.wo * self.p_over / self.no
                - self.wp * self.pref / self.npref)
        v_total = (self.v_intraday + self.v_qualif + self.v_elig + self.v_rest + self.v_weekly
                   + self.v_supervision + self.v_exempt + self.v_corridor + self.v_gmax)
        return soft + self.w_cob * self.p_cob / self.ncob + self.mu * v_total

    def _stretch_windows(self, t: int) -> range:
        """Ventanas de racha [t0, t0+L^max] que contienen el día t."""
        lo = max(0, t - self.l_max)
        hi = min(t, self.days - 1 - self.l_max)
        return range(lo, hi + 1) if hi >= lo else range(0)

    def _erg_around(self, i: int, t: int) -> float:
        """Penalización ergonómica de i que depende del día t: rachas que cubren t + findes
        partidos que tocan t. Usa `self.active[i]`."""
        act = self.active[i]
        total = 0.0
        for t0 in self._stretch_windows(t):
            worked = sum(1 for u in range(t0, t0 + self.l_max + 1) if u in act)
            if worked > self.l_max:
                total += worked - self.l_max
        for a in (t - 1, t):                                 # pares de finde que contienen t
            b = a + 1
            if 0 <= a and b < self.days and self.is_wknd[a] and self.is_wknd[b]:
                total += abs((a in act) - (b in act))
        return total

    def _weekly_at(self, i: int, t: int) -> float:
        """1 si la semana (bloque de 7 días) que contiene t no tiene par de días libres
        consecutivos para i; 0 si lo tiene. Espejo de `objective._weekly_rest`."""
        ws = (t // 7) * 7
        week = range(ws, min(ws + 7, self.days))
        if len(week) < 2:
            return 0.0
        act = self.active[i]
        has_free = any(u not in act and (u + 1) not in act for u in week if (u + 1) in week)
        return 0.0 if has_free else 1.0

    # ------------------------------------------------------------------ flip

    def apply(self, i: int, t: int, s: str) -> float:
        """Togglea x_{i,t,s}, actualiza el estado y devuelve Delta = Z_nuevo - Z_anterior."""
        sign = -1.0 if (i, t, s) in self.assigned else 1.0
        sh = self.shift[s]
        old_score = self.score

        # cobertura
        cov = self.cov.get((t, s), 0)
        b = self.demand.get(t, {}).get(s, 0)
        self.p_cob += max(0.0, b - (cov + sign)) - max(0.0, b - cov)
        self.cov[(t, s)] = cov + sign

        # preferencias
        self.pref += sign * self.pref_val.get((i, t, s), 0.0)

        # carga (p_car), corredor-suelo (v_corridor) y sobre-jornada (p_over): cambian con las horas de i
        h_old = self.hours[i]
        h_new = h_old + sign * sh.hours
        tau = self.tau[i]
        self.p_car += abs(h_new - tau) - abs(h_old - tau)
        hmin, hmax = self.corridor[i]
        self.v_corridor += max(0.0, hmin - h_new) - max(0.0, hmin - h_old)   # solo el suelo (techo->p_over)
        self.p_over += max(0.0, h_new - hmax) - max(0.0, h_old - hmax)        # sobre-jornada (eq:sobre-jornada)
        self.hours[i] = h_new

        # cualificación (gamma) y elegibilidad (eta): locales
        if not self.qual.get(self.class_name[i], {}).get(s, False):
            self.v_qualif += sign
        if not self.elig.get(t, {}).get(s, False):
            self.v_elig += sign

        # solape intradía (O) y descanso (I): leen day_shifts ANTES de togglear s
        dst = self.day_shifts.get((i, t), set())
        others = [s2 for s2 in dst if s2 != s]
        self.v_intraday += sign * sum(1 for s2 in others if frozenset((s, s2)) in self.overlap)
        nxt = self.day_shifts.get((i, t + 1), ())
        prv = self.day_shifts.get((i, t - 1), ())
        self.v_rest += sign * (sum(1 for s2 in nxt if (s, s2) in self.short_rest)
                               + sum(1 for s2 in prv if (s2, s) in self.short_rest))

        # bloque de guardia: v_gmax, v_exempt, p_gua, supervisión
        if sh.is_guardia:
            g_old = self.guardias[i]
            cap = self.cap[i]
            if cap is not None:
                self.v_gmax += max(0.0, (g_old + sign) - cap) - max(0.0, g_old - cap)
            if self.exempt[i]:
                self.v_exempt += sign
            self.guardias[i] = g_old + sign
            if not self.exempt[i]:
                self.p_gua = self._spread(self.guardias, self.tau_g)
            if self.is_r1[i]:                                # R1 en guardia sin adjunto -> viola
                self.v_supervision += sign * (1.0 if self.adj_on.get((t, s), 0) == 0 else 0.0)
                self.r1_on[(t, s)] = self.r1_on.get((t, s), 0) + sign
            elif self.is_adj[i]:                             # cambia adj_on -> vuelca a todos los R1 de (t,s)
                a_old = self.adj_on.get((t, s), 0)
                a_new = a_old + sign
                n_r1 = self.r1_on.get((t, s), 0)
                self.v_supervision += (n_r1 if a_new == 0 else 0) - (n_r1 if a_old == 0 else 0)
                self.adj_on[(t, s)] = a_new

        # calendario (p_cal): recuentos de finde/festivo; el spread solo cuenta P\E
        if self.is_wknd[t]:
            self.wknd[i] += sign
            if not self.exempt[i]:
                self.cal_fds = self._spread(self.wknd, self.tau_fds)
        if self.is_hol[t]:
            self.hol[i] += sign
            if not self.exempt[i]:
                self.cal_fes = self._spread(self.hol, self.tau_fes)

        # ergonomía (p_erg) y descanso semanal (v_weekly): solo si t cambia de activo a i
        was_active = len(dst) > 0
        will_active = True if sign > 0 else len(dst) > 1
        if was_active != will_active:
            old_e, old_w = self._erg_around(i, t), self._weekly_at(i, t)
            if will_active:
                self.active[i].add(t)
            else:
                self.active[i].discard(t)
            self.p_erg += self._erg_around(i, t) - old_e
            self.v_weekly += self._weekly_at(i, t) - old_w

        # mutar day_shifts y assigned
        if sign > 0:
            self.day_shifts.setdefault((i, t), set()).add(s)
            self.assigned.add((i, t, s))
        else:
            dst.discard(s)
            if not dst:
                self.day_shifts.pop((i, t), None)
            self.assigned.discard((i, t, s))

        self.score = self._score()
        return self.score - old_score

    def roster(self) -> Roster:
        """Roster del estado actual (copia del conjunto de asignaciones)."""
        return Roster(assignments=set(self.assigned))
