"""Harness de experimentos del Cap. 7: greedy vs simulated annealing vs exacto (MILP).

Los tres métodos se miden con la misma vara (`z_modelo`). El exacto fija la referencia: su
óptimo Z* si certifica, o su cota dual si agota el tiempo (medir contra la cota sobre-estima la
distancia real, del lado conservador). Además del gap relativo, que un objetivo dominado por la
cobertura infla, se registra el desglose por componente y la diferencia absoluta; `z_modelo`
solo se compara entre rosters factibles (columnas `feas_*`)."""

import csv
import time
from datetime import date
from pathlib import Path

import pulp

from rostering.domain import Calendar
from rostering.generator import Settings, generate
from rostering.greedy import greedy
from rostering.milp import solve_exact
from rostering.objective import (
    ObjectiveConfig, evaluate, normalizers, violations, z_modelo,
)
from rostering.roster import Roster
from rostering.sa import SAConfig, schedule, simulated_annealing

# los siete términos del objetivo (Breakdown), en orden de presentación
_COMPS = ("p_cob", "p_erg", "p_car", "p_gua", "p_cal", "p_over", "pref")


def _quality(instance, roster: Roster, config, norm) -> tuple[float, bool, dict]:
    """(z_modelo, ¿factible en las hard?, desglose por componente): la métrica común a los tres métodos."""
    b = evaluate(instance, roster, config)
    z = z_modelo(b, norm, config)
    feas = violations(instance, roster).is_feasible
    comps = {c: getattr(b, c) for c in _COMPS}
    return z, feas, comps


def _with_prefix(comps: dict, prefix: str) -> dict:
    """Renombra el desglose con prefijo de método: {'p_cob': ..} -> {'greedy_p_cob': ..}."""
    return {f"{prefix}_{k}": v for k, v in comps.items()}


# columnas del CSV, en orden fijo (el capítulo lee de aquí)
FIELDS = (
    ["n", "days", "seed", "coverage_ratio", "regime", "coverage_mode"]
    + ["z_greedy", "feas_greedy", "t_greedy"]
    + ["z_sa", "feas_sa", "t_sa", "sa_stop", "sa_iters"]
    + ["z_exact", "proven", "gap_solver", "dual_bound", "match", "n_vars", "n_cons", "t_exact"]
    + ["ref", "dz_greedy", "gap_greedy", "dz_sa", "gap_sa", "exact_error"]
    + [f"greedy_{c}" for c in _COMPS]
    + [f"sa_{c}" for c in _COMPS]
    + [f"exact_{c}" for c in _COMPS]
)


def run_instance(n: int, days: int, seed: int, *, coverage_ratio: float = 1.10,
                 time_limit: int = 300, obj_config: ObjectiveConfig | None = None,
                 sa_config: SAConfig | None = None,
                 holidays: list[date] | None = None) -> dict:
    """Resuelve una instancia n x days (semilla `seed`, holgura `coverage_ratio`) con los tres
    métodos y devuelve un registro con calidad, desglose, tiempos y régimen. `time_limit` topa el
    solver exacto."""
    config = obj_config or ObjectiveConfig()
    settings = Settings(
        n_physicians=n,
        calendar=Calendar(start_date=date(2026, 1, 5), days=days, holidays=holidays or []),
        coverage_ratio=coverage_ratio,
    )
    instance = generate(settings, seed=seed)
    norm = normalizers(instance, config)
    rec: dict = {"n": n, "days": days, "seed": seed, "coverage_ratio": coverage_ratio}

    # --- greedy (constructivo) ---
    t0 = time.perf_counter()
    g = greedy(instance, config)
    rec["t_greedy"] = time.perf_counter() - t0
    rec["z_greedy"], rec["feas_greedy"], comps_g = _quality(instance, g, config, norm)
    rec.update(_with_prefix(comps_g, "greedy"))

    # --- simulated annealing (arranca del greedy) ---
    sa_cfg = sa_config or schedule(instance, seed=seed)
    t0 = time.perf_counter()
    sa = simulated_annealing(instance, g, sa_cfg, config)
    rec["t_sa"] = time.perf_counter() - t0
    rec["z_sa"], rec["feas_sa"], comps_s = _quality(instance, sa.roster, config, norm)
    rec["sa_stop"], rec["sa_iters"] = sa.stop_reason, sa.iterations
    rec.update(_with_prefix(comps_s, "sa"))

    # --- exacto (Gurobi); puede no certificar (cota dual) o no caber (license_exceeded) ---
    t0 = time.perf_counter()
    ref = None
    try:
        # break_symmetry=False: con Gurobi (detección interna de simetría) el orden lex MANUAL es
        # errático y a veces BLOQUEA la certificación (p.ej. 10x21 cr1.3 certifica en 1.3s sin él,
        # no en 30s con él). Se deja la simetría al motor.
        exact_roster, info = solve_exact(
            instance, config, solver=pulp.GUROBI(msg=False, timeLimit=time_limit),
            break_symmetry=False, coverage_hard=True)
        if info["status"] == "Infeasible":     # celda no cubrible con cobertura dura -> régimen soft
            exact_roster, info = solve_exact(
                instance, config, solver=pulp.GUROBI(msg=False, timeLimit=time_limit),
                break_symmetry=False, coverage_hard=False)
            rec["coverage_mode"] = "soft"
        else:
            rec["coverage_mode"] = "hard"
        rec["t_exact"] = time.perf_counter() - t0
        # z_exact = z_modelo recomputado del roster (limpio); si certifica, == Z*
        rec["z_exact"], _, comps_e = _quality(instance, exact_roster, config, norm)
        rec.update(_with_prefix(comps_e, "exact"))
        rec.update(proven=info["proven_optimal"], gap_solver=info["gap"],
                   dual_bound=info["dual_bound"], match=info["match"],
                   n_vars=info["n_vars"], n_cons=info["n_cons"])
        rec["regime"] = "certified" if info["proven_optimal"] else "bounded"
        ref = rec["z_exact"] if info["proven_optimal"] else info["dual_bound"]
    except Exception as e:                                # GurobiError de tamaño u otro fallo
        rec["t_exact"] = time.perf_counter() - t0
        rec["regime"] = "exact_failed"
        rec["exact_error"] = f"{type(e).__name__}: {str(e)[:80]}"

    # --- gaps contra la referencia: absoluto (titular) y relativo (secundario) ---
    if ref is not None:
        rec["ref"] = ref
        rec["dz_greedy"] = rec["z_greedy"] - ref
        rec["dz_sa"] = rec["z_sa"] - ref
        rec["gap_greedy"] = rec["dz_greedy"] / abs(ref)
        rec["gap_sa"] = rec["dz_sa"] / abs(ref)
    return rec


def run_grid(specs: list[tuple[int, int]], seeds: list[int], **kw) -> list[dict]:
    """Barre la rejilla `specs` (lista de (n, days)) x `seeds`. Imprime una línea por instancia
    y devuelve la lista de registros. Acepta los mismos kwargs que `run_instance`."""
    records = []
    for (n, days) in specs:
        for seed in seeds:
            rec = run_instance(n, days, seed, **kw)
            records.append(rec)
            dz, gap = rec.get("dz_sa"), rec.get("gap_sa")
            tag = (f"Delta z={dz:+.4f} ({gap:+.1%})" if isinstance(dz, float) else "-")
            print(f"  {n:>2}x{days:<2} seed={seed} | {rec['regime']:<13} | "
                  f"z_sa={rec['z_sa']:.4f} vs ref={rec.get('ref', float('nan')):.4f}  {tag} | "
                  f"t: greedy={rec['t_greedy']:.2f}s sa={rec['t_sa']:.2f}s exact={rec['t_exact']:.1f}s")
    return records


def run_full_grid(ns: list[int], days_list: list[int], coverage_ratios: list[float],
                  seeds: list[int], *, time_limit: int, out_path: str | Path,
                  obj_config: ObjectiveConfig | None = None) -> list[dict]:
    """Barre la rejilla COMPLETA `ns x days_list x coverage_ratios x seeds` y guarda
    INCREMENTALMENTE en `out_path` (una fila por instancia, con flush): una corrida larga que se
    interrumpa conserva lo ya hecho. Imprime una línea de progreso por instancia. Cada instancia
    se envuelve en try/except: un fallo inesperado no aborta la rejilla (queda como `run_failed`)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(ns) * len(days_list) * len(coverage_ratios) * len(seeds)
    records: list[dict] = []
    k = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        f.flush()
        for n in ns:
            for days in days_list:
                for cr in coverage_ratios:
                    for seed in seeds:
                        k += 1
                        try:
                            rec = run_instance(n, days, seed, coverage_ratio=cr,
                                               time_limit=time_limit, obj_config=obj_config)
                        except Exception as e:                      # red de seguridad: no abortar la rejilla
                            rec = {"n": n, "days": days, "seed": seed, "coverage_ratio": cr,
                                   "regime": "run_failed",
                                   "exact_error": f"{type(e).__name__}: {str(e)[:80]}"}
                        records.append(rec)
                        writer.writerow({c: rec.get(c, "") for c in FIELDS})
                        f.flush()
                        dz = rec.get("dz_sa")
                        tag = f"Dz_sa={dz:+.4f}" if isinstance(dz, float) else "-"
                        zsa = rec.get("z_sa", float("nan"))
                        print(f"[{k:>3}/{total}] {n:>2}x{days:<2} cr{cr} s{seed} | "
                              f"{rec.get('regime', '?'):<12} cov={rec.get('coverage_mode', '-'):<4} "
                              f"z_sa={zsa:.4f} {tag} | t_exact={rec.get('t_exact', 0):.0f}s", flush=True)
    return records


def save_csv(records: list[dict], path: str | Path) -> None:
    """Vuelca los registros a CSV (columnas FIELDS, en orden). Crea el directorio si no existe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in FIELDS})
