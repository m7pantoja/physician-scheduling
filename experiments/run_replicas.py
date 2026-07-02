"""Réplicas del SA sobre una rejilla del Cap. 7: 30 ejecuciones por celda, con la semilla del SA
desacoplada de la semilla de instancia; el greedy, determinista, se corre una vez. Produce un
CSV de calidad por réplica y otro de distribución por médico (fairness). Paralelo por celdas y
reanudable; escribe en data/rerun/ para no pisar los CSV congelados.

  uv run python experiments/run_replicas.py base         # rejilla n en {8,10,12}
  uv run python experiments/run_replicas.py ext          # rejilla n en {20,30,40}
  uv run python experiments/run_replicas.py ext --dry    # 1 celda x 2 réplicas, sin escribir
"""
import csv
import sys
import time
from collections import defaultdict
from datetime import date
from multiprocessing import Pool
from pathlib import Path

from rostering.domain import Calendar
from rostering.generator import Settings, generate
from rostering.greedy import greedy
from rostering.objective import ObjectiveConfig, evaluate, normalizers, violations, z_modelo
from rostering.sa import schedule, simulated_annealing

DATA = Path(__file__).resolve().parent.parent / "data"

GRIDS = {"base": ([8, 10, 12], "_base"), "ext": ([20, 30, 40], "_extended")}
GRID_DAYS = [14, 21, 28]
GRID_CR = [1.10, 1.30, 1.50]
SEEDS_INST = [0, 1, 2, 3, 4]
R_DEFAULT = 30
N_PROC = 8

COMPS = ("p_cob", "p_erg", "p_car", "p_gua", "p_cal", "p_over", "pref")
REP_FIELDS = (["n", "days", "seed_inst", "coverage_ratio", "seed_sa",
               "z_greedy", "feas_greedy", "z_sa", "feas_sa", "t_sa", "sa_iters"]
              + [f"greedy_{c}" for c in COMPS] + [f"sa_{c}" for c in COMPS])
FAIR_FIELDS = ["n", "days", "seed_inst", "coverage_ratio", "seed_sa",
               "phys_id", "part_time", "is_exempt",
               "hours", "n_shifts", "n_guardias", "n_weekend", "n_holiday"]
CFG = ObjectiveConfig()


def per_physician(instance, roster):
    """Espejo de objective.evaluate: agrega por médico las cantidades de fairness."""
    sbn = {s.name: s for s in instance.shifts}
    cal = instance.calendar
    hours = {p.id: 0.0 for p in instance.physicians}
    nsh = {p.id: 0 for p in instance.physicians}
    gua = {p.id: 0 for p in instance.physicians}
    wk = {p.id: 0 for p in instance.physicians}
    hol = {p.id: 0 for p in instance.physicians}
    for (i, t, sn) in roster.assignments:
        sh = sbn[sn]
        hours[i] += sh.hours
        nsh[i] += 1
        if sh.is_guardia:
            gua[i] += 1
        if cal.is_weekend(t):
            wk[i] += 1
        if cal.is_holiday(t):
            hol[i] += 1
    return hours, nsh, gua, wk, hol


def comps_of(instance, roster):
    b = evaluate(instance, roster, CFG)
    return {c: getattr(b, c) for c in COMPS}


def run_cell(task):
    """Trabajo de UNA celda: genera instancia + greedy (1 vez) y corre las réplicas pendientes.
    Devuelve (celda, rep_rows, fair_rows). Ejecutado en un proceso worker."""
    n, d, cr, si, pend = task
    inst = generate(Settings(n_physicians=n,
                             calendar=Calendar(start_date=date(2026, 1, 5), days=d, holidays=[]),
                             coverage_ratio=cr), seed=si)
    norm = normalizers(inst, CFG)
    g = greedy(inst, CFG)
    zg = z_modelo(evaluate(inst, g, CFG), norm, CFG)
    fg = violations(inst, g).is_feasible
    cg = comps_of(inst, g)
    rep_rows, fair_rows = [], []
    for s in pend:
        t0 = time.perf_counter()
        sa = simulated_annealing(inst, g, schedule(inst, seed=s), CFG)
        t_sa = time.perf_counter() - t0
        zs = z_modelo(evaluate(inst, sa.roster, CFG), norm, CFG)
        fs = violations(inst, sa.roster).is_feasible
        cs = comps_of(inst, sa.roster)
        row = {"n": n, "days": d, "seed_inst": si, "coverage_ratio": cr, "seed_sa": s,
               "z_greedy": zg, "feas_greedy": fg, "z_sa": zs, "feas_sa": fs,
               "t_sa": t_sa, "sa_iters": sa.iterations}
        row.update({f"greedy_{c}": cg[c] for c in COMPS})
        row.update({f"sa_{c}": cs[c] for c in COMPS})
        rep_rows.append(row)
        hours, nsh, gua, wk, hol = per_physician(inst, sa.roster)
        fair_rows += [{"n": n, "days": d, "seed_inst": si, "coverage_ratio": cr, "seed_sa": s,
                       "phys_id": p.id, "part_time": p.part_time, "is_exempt": p.is_exempt,
                       "hours": hours[p.id], "n_shifts": nsh[p.id], "n_guardias": gua[p.id],
                       "n_weekend": wk[p.id], "n_holiday": hol[p.id]}
                      for p in inst.physicians]
    return (n, d, cr, si), rep_rows, fair_rows


def done_keys(rep_path):
    """Mapa (n,days,cr,seed_inst) -> set de seed_sa ya presentes en `rep_path`; el llamador
    salta las celdas cuyo conjunto está completo (len == R)."""
    if not rep_path.exists():
        return defaultdict(set)
    seen = defaultdict(set)
    with rep_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                seen[(int(r["n"]), int(r["days"]), round(float(r["coverage_ratio"]), 2),
                      int(r["seed_inst"]))].add(int(r["seed_sa"]))
            except (KeyError, ValueError):
                continue
    return seen


def _cost(cell):
    n, d, _, _ = cell
    return n * d          # proxy de coste: celdas grandes primero


def main():
    args = [a for a in sys.argv[1:] if a != "--dry"]
    if len(args) != 1 or args[0] not in GRIDS:
        sys.exit("uso: run_replicas.py base|ext [--dry]")
    grid_n, suffix = GRIDS[args[0]]
    rep_out = DATA / "rerun" / f"sa_replicas{suffix}.csv"
    fair_out = DATA / "rerun" / f"fairness{suffix}.csv"

    if "--dry" in sys.argv:
        # control de fidelidad: la réplica seed_sa == seed_inst == 0 debe reproducir la congelada
        n0 = grid_n[0]
        (_, rep_rows, fair_rows) = run_cell((n0, 14, 1.10, 0, [0, 1]))
        frozen = {int(r["seed_sa"]): r for r in csv.DictReader(open(DATA / f"sa_replicas{suffix}.csv"))
                  if (int(r["n"]), int(r["days"]), int(r["seed_inst"]),
                      round(float(r["coverage_ratio"]), 2)) == (n0, 14, 0, 1.10)}
        fz = float(frozen[0]["z_sa"])
        print(f"[dry] celda {n0}x14 cr1.1 si0 -> {len(rep_rows)} rep_rows, {len(fair_rows)} fair_rows")
        print(f"[dry] z_sa réplica seed_sa=0 : {rep_rows[0]['z_sa']:.6f}")
        print(f"[dry] z_sa congelada        : {fz:.6f}")
        print(f"[dry] fidelidad {'OK' if abs(rep_rows[0]['z_sa'] - fz) < 1e-6 else 'DISCREPA'}")
        return

    R = R_DEFAULT
    done = done_keys(rep_out)
    tasks = []
    for n in grid_n:
        for d in GRID_DAYS:
            for cr in GRID_CR:
                for si in SEEDS_INST:
                    pend = [s for s in range(R) if s not in done[(n, d, round(cr, 2), si)]]
                    if pend:
                        tasks.append((n, d, cr, si, pend))
    tasks.sort(key=lambda t: -_cost(t[:4]))
    total_cells = len(grid_n) * len(GRID_DAYS) * len(GRID_CR) * len(SEEDS_INST)
    print(f"celdas pendientes: {len(tasks)}/{total_cells} | procesos: {N_PROC} | R={R}", flush=True)
    if not tasks:
        print("DONE (nada pendiente)", flush=True)
        return

    rep_out.parent.mkdir(parents=True, exist_ok=True)
    rep_new, fair_new = not rep_out.exists(), not fair_out.exists()
    with rep_out.open("a", newline="", encoding="utf-8") as rf, \
         fair_out.open("a", newline="", encoding="utf-8") as ff:
        rw = csv.DictWriter(rf, fieldnames=REP_FIELDS)
        fw = csv.DictWriter(ff, fieldnames=FAIR_FIELDS)
        if rep_new:
            rw.writeheader(); rf.flush()
        if fair_new:
            fw.writeheader(); ff.flush()
        t_start = time.perf_counter()
        with Pool(N_PROC) as pool:
            for k, (cell, rep_rows, fair_rows) in enumerate(pool.imap_unordered(run_cell, tasks), 1):
                rw.writerows(rep_rows); rf.flush()
                fw.writerows(fair_rows); ff.flush()
                el = time.perf_counter() - t_start
                print(f"[{k:>3}/{len(tasks)}] celda {cell[0]}x{cell[1]} cr{cell[2]} si{cell[3]}: "
                      f"{len(rep_rows)} reps | elapsed {el/60:.1f} min", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
