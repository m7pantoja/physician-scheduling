"""Corrida de una rejilla del Cap. 7: greedy + SA + exacto (300 s/celda) por instancia.

Mismo diseño factorial en las dos rejillas (3 tamaños x 3 horizontes x 3 holguras x 5 semillas =
135 celdas): `base` con n en {8, 10, 12} y `ext` en {20, 30, 40}. Reanudable: continúa por las
celdas que falten en el CSV de salida, y escribe en data/rerun/ para no pisar los CSV congelados.

  uv run python experiments/run_grid.py base
  uv run python experiments/run_grid.py ext
"""
import csv
import sys
from pathlib import Path

from rostering.experiment import run_instance, FIELDS

GRIDS = {"base": ("grid_base.csv", [8, 10, 12]), "ext": ("grid_extended.csv", [20, 30, 40])}
DAYS_LIST = [14, 21, 28]
CRS = [1.10, 1.30, 1.50]
SEEDS = [0, 1, 2, 3, 4]
TIME_LIMIT = 300


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in GRIDS:
        sys.exit("uso: run_grid.py base|ext")
    csv_name, ns = GRIDS[sys.argv[1]]
    out = Path(__file__).resolve().parent.parent / "data" / "rerun" / csv_name
    out.parent.mkdir(parents=True, exist_ok=True)

    # reanudación: recoger las celdas ya resueltas en el CSV de salida
    done = set()
    if out.exists():
        for r in csv.DictReader(out.open(encoding="utf-8")):
            try:
                done.add((float(r["coverage_ratio"]), int(r["days"]), int(r["n"]), int(r["seed"])))
            except (ValueError, KeyError):
                pass

    cells = [(cr, d, n, s) for cr in CRS for d in DAYS_LIST for n in ns for s in SEEDS]
    pending = [c for c in cells if c not in done]
    print(f"Total {len(cells)} | ya hechas {len(done)} | pendientes {len(pending)}", flush=True)

    write_header = (not out.exists()) or out.stat().st_size == 0
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
            f.flush()
        for j, (cr, days, n, seed) in enumerate(pending, 1):
            try:
                rec = run_instance(n, days, seed, coverage_ratio=cr, time_limit=TIME_LIMIT)
            except Exception as e:                                   # red de seguridad
                rec = {"n": n, "days": days, "seed": seed, "coverage_ratio": cr,
                       "regime": "run_failed", "exact_error": f"{type(e).__name__}: {str(e)[:80]}"}
            w.writerow({c: rec.get(c, "") for c in FIELDS})
            f.flush()
            print(f"[{len(done) + j:>3}/{len(cells)}] n={n:>2} d{days} cr{cr} s{seed} -> "
                  f"{rec.get('regime', '?'):<12} cov={rec.get('coverage_mode', '-'):<4} "
                  f"z_sa={rec.get('z_sa')} t_ex={rec.get('t_exact', 0):.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
