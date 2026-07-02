"""Análisis transversal del Cap. 7: pool de 270 celdas, corte por régimen del solver.

Lee los CSV congelados de data/ (grid, réplicas y fairness de las rejillas base y extendida),
imprime las cifras de la Sección 7.4 y regenera en out/ los .dat de las figuras de la memoria.

  uv run python experiments/analyze.py
"""
import csv, math
import statistics as st
from collections import defaultdict
from datetime import date
from pathlib import Path

from scipy import stats

from rostering.domain import Calendar
from rostering.generator import Settings, generate
from rostering.objective import ObjectiveConfig, normalizers, resolve_w_cob

_ROOT = Path(__file__).resolve().parent.parent
BASE = str(_ROOT / "data")
FIG = str(_ROOT / "out")
Path(FIG).mkdir(exist_ok=True)
COMPS = ["p_cob", "p_erg", "p_car", "p_gua", "p_cal", "p_over", "pref"]
CFG = ObjectiveConfig()

def num(x):
    try: return float(x)
    except: return None

def key(r, sc):
    return (int(r["n"]), int(r["days"]), int(r[sc]), round(float(r["coverage_ratio"]), 2))

# ---------- carga y pool ----------
grid = {key(r, "seed"): r for r in csv.DictReader(open(f"{BASE}/grid_base.csv"))}
grid.update({key(r, "seed"): r for r in csv.DictReader(open(f"{BASE}/grid_extended.csv"))})

reps = list(csv.DictReader(open(f"{BASE}/sa_replicas_base.csv")))
reps += list(csv.DictReader(open(f"{BASE}/sa_replicas_extended.csv")))
by_cell = defaultdict(list)
for r in reps:
    by_cell[key(r, "seed_inst")].append(r)

fair = list(csv.DictReader(open(f"{BASE}/fairness_base.csv")))
fair += list(csv.DictReader(open(f"{BASE}/fairness_extended.csv")))

print(f"celdas grid: {len(grid)} ; celdas con réplicas: {len(by_cell)} ; réplicas: {len(reps)}")
assert len(grid) == 270 and len(by_cell) == 270, "pool incompleto"
nrep = {c: len(v) for c, v in by_cell.items()}
print(f"réplicas por celda: min={min(nrep.values())} max={max(nrep.values())}")

# normalizadores + clases (para contribuciones y Theil)
norm_cache, class_cache = {}, {}
for (n, d, si, cr) in by_cell:
    inst = generate(Settings(n_physicians=n,
                             calendar=Calendar(start_date=date(2026, 1, 5), days=d, holidays=[]),
                             coverage_ratio=cr), seed=si)
    nm = normalizers(inst, CFG)
    norm_cache[(n, d, si, cr)] = (nm, resolve_w_cob(CFG, nm))
    class_cache[(n, d, si, cr)] = {p.id: inst.class_of(p).is_resident for p in inst.physicians}

def contrib(cv, nm, w):
    W = {"p_cob": w, "p_erg": 1., "p_car": 1., "p_gua": 1., "p_cal": 1., "p_over": 1., "pref": 1.}
    N = {"p_cob": nm.n_cob, "p_erg": nm.n_erg, "p_car": nm.n_car, "p_gua": nm.n_gua,
         "p_cal": nm.n_cal, "p_over": nm.n_over, "pref": nm.n_pref}
    return {c: (-1. if c == "pref" else 1.) * W[c] * cv[c] / N[c] for c in COMPS}

def gini(xs):
    xs = sorted(xs); n = len(xs); s = sum(xs)
    if n == 0 or s == 0: return 0.
    return (2 * sum((i + 1) * x for i, x in enumerate(xs))) / (n * s) - (n + 1) / n

def theil(xs):
    n = len(xs); mu = sum(xs) / n if n else 0.
    return sum((x / mu) * math.log(x / mu) for x in xs if x > 0) / n if mu else 0.

def theil_dec(xs, gr):
    n = len(xs); mu = sum(xs) / n
    if mu == 0: return 0., 0., 0.
    g = defaultdict(list)
    for x, l in zip(xs, gr): g[l].append(x)
    tb = tw = 0.
    for l, v in g.items():
        ng = len(v); mug = sum(v) / ng; sg = ng / n
        if mug > 0:
            tb += sg * (mug / mu) * math.log(mug / mu)
            tw += sg * (mug / mu) * theil(v)
    return theil(xs), tb, tw

def cvar(xs):
    n = len(xs); mu = sum(xs) / n if n else 0.
    return st.pstdev(xs) / mu if mu else 0.

# ---------- resumen por celda ----------
cell = {}
for c, rows in by_cell.items():
    zs = [float(r["z_sa"]) for r in rows]
    cell[c] = {
        "z_med": st.median(zs), "z_sd": st.pstdev(zs), "z_min": min(zs), "z_max": max(zs),
        "comp": {cc: st.median(float(r[f"sa_{cc}"]) for r in rows) for cc in COMPS},
        "feas_sa": sum(1 for r in rows if r["feas_sa"] == "True"),
        "z_greedy": float(rows[0]["z_greedy"]), "feas_greedy": rows[0]["feas_greedy"],
        "t_sa_med": st.median(float(r["t_sa"]) for r in rows),
        "regime": grid[c]["regime"], "z_exact": num(grid[c]["z_exact"]),
        "dual": num(grid[c]["dual_bound"]), "ncons": int(float(grid[c]["n_cons"])),
        "t_exact": num(grid[c]["t_exact"]), "gap_solver": num(grid[c]["gap_solver"]),
    }

CERT = {c: v for c, v in cell.items() if v["regime"] == "certified"}
ACOT = {c: v for c, v in cell.items() if v["regime"] == "bounded"}
print(f"\nrégimen: certificadas={len(CERT)} acotadas={len(ACOT)} sin_resolver={270-len(CERT)-len(ACOT)}")

# ============ A. FRONTERA (tabla 7.1 extendida a 270) ============
print("\n" + "=" * 70 + "\nA. TABLA FRONTERA (270 celdas, cert/acot y t por fila n x días)\n" + "=" * 70)
rowagg = defaultdict(lambda: {"cert": 0, "acot": 0, "t": []})
for c, v in cell.items():
    k = (c[0], c[1])
    rowagg[k]["cert" if v["regime"] == "certified" else "acot"] += 1
    rowagg[k]["t"].append(v["t_exact"])
for k in sorted(rowagg):
    a = rowagg[k]
    print(f"  n={k[0]:>2} d={k[1]:>2}: cert={a['cert']:>2} acot={a['acot']:>2} t={st.mean(a['t']):>5.0f}s")
print("cert por n:", {n: sum(1 for c in CERT if c[0] == n) for n in (8, 10, 12, 20, 30, 40)})
# tiempo medio de las certificadas (para 'certifica en <X min')
tc = [v["t_exact"] for v in CERT.values()]
print(f"t_exact en certificadas: med={st.median(tc):.0f}s max={max(tc):.0f}s media={st.mean(tc):.0f}s")

# ============ B. RÉGIMEN CERTIFICADO (61) ============
print("\n" + "=" * 70 + f"\nB. CERTIFICADO ({len(CERT)}): Delta z = z_SA(med30) - z*\n" + "=" * 70)
dz_all, rel_all, hits, big = [], [], 0, []
comp_delta = defaultdict(list)
for c, v in CERT.items():
    dz = v["z_med"] - v["z_exact"]
    dz_all.append(dz); rel_all.append(dz / v["z_exact"] if v["z_exact"] else 0.)
    if abs(dz) < 1e-6: hits += 1
    if v["z_exact"] >= 0.5: big.append(dz / v["z_exact"])
    nm, w = norm_cache[c]
    csa = contrib(v["comp"], nm, w)
    cex = contrib({cc: float(grid[c][f"exact_{cc}"]) for cc in COMPS}, nm, w)
    for cc in COMPS: comp_delta[cc].append(csa[cc] - cex[cc])
q = st.quantiles(dz_all, n=4)
print(f"Delta z: med={st.median(dz_all):.4f} Q3={q[2]:.4f} max={max(dz_all):.4f} ; SA alcanza óptimo en {hits}/{len(dz_all)}")
print(f"gap rel: med={100 * st.median(rel_all):.1f}% ; celdas z*<0.5: {sum(1 for v in CERT.values() if v['z_exact'] < 0.5)}/{len(CERT)}"
      f" ; gap rel z*>=0.5 ({len(big)}): med={100 * st.median(big):.1f}%")
print("exceso medio de contribución por término (>0 SA peor):")
for cc in COMPS: print(f"   {cc:6s}: {st.mean(comp_delta[cc]):+.5f}")
for n in (8, 10, 12, 20, 30, 40):
    dd = [v["z_med"] - v["z_exact"] for c, v in CERT.items() if c[0] == n]
    hh = sum(1 for c, v in CERT.items() if c[0] == n and abs(v["z_med"] - v["z_exact"]) < 1e-6)
    if dd: print(f"  n={n:>2}: {len(dd):>2} celdas  Delta z med={st.median(dd):+.4f}  hits={hh}")

# ============ C. RÉGIMEN ACOTADO (209) ============
print("\n" + "=" * 70 + f"\nC. ACOTADO ({len(ACOT)}): SA vs incumbent + gap conservador z_SA - cota dual\n" + "=" * 70)
win = tie = lose = 0
dinc, gcon, any_beats = [], [], 0
comp_delta_a = defaultdict(list)
for c, v in ACOT.items():
    d = v["z_med"] - v["z_exact"]; dinc.append(d)
    if d < -1e-9: win += 1
    elif d > 1e-9: lose += 1
    else: tie += 1
    if any(float(r["z_sa"]) < v["z_exact"] - 1e-9 for r in by_cell[c]): any_beats += 1
    if v["dual"] is not None: gcon.append(v["z_med"] - v["dual"])
    nm, w = norm_cache[c]
    csa = contrib(v["comp"], nm, w)
    cex = contrib({cc: float(grid[c][f"exact_{cc}"]) for cc in COMPS}, nm, w)
    for cc in COMPS: comp_delta_a[cc].append(csa[cc] - cex[cc])
print(f"SA gana={win} empata={tie} pierde={lose} (de {len(ACOT)}) ; alguna réplica bate incumbent: {any_beats}")
q = st.quantiles(dinc, n=4)
print(f"Delta(SA-inc): med={st.median(dinc):+.4f} Q3={q[2]:+.4f} max={max(dinc):+.3f}")
print(f"gap conservador z_SA - cota dual: med={st.median(gcon):.4f} Q3={st.quantiles(gcon, n=4)[2]:.4f} max={max(gcon):.3f}")
print("exceso medio de contribución vs incumbent por término:")
for cc in COMPS: print(f"   {cc:6s}: {st.mean(comp_delta_a[cc]):+.5f}")
for n in (8, 10, 12, 20, 30, 40):
    dd = [v["z_med"] - v["z_exact"] for c, v in ACOT.items() if c[0] == n]
    gg = [v["z_med"] - v["dual"] for c, v in ACOT.items() if c[0] == n and v["dual"] is not None]
    ww = sum(1 for c, v in ACOT.items() if c[0] == n and v["z_med"] - v["z_exact"] < -1e-9)
    tt = sum(1 for c, v in ACOT.items() if c[0] == n and abs(v["z_med"] - v["z_exact"]) <= 1e-9)
    if dd: print(f"  n={n:>2}: {len(dd):>3} celdas  Delta(SA-inc) med={st.median(dd):+.4f}  gapConsv med={st.median(gg):.4f}  SA gana={ww} empata={tt}")
# holgura de la cota del solver (cuánto sobreestima el gap conservador)
gs = [v["gap_solver"] for v in ACOT.values() if v["gap_solver"] is not None]
print(f"gap_solver (incumbent vs cota dual, holgura interna del BnB): med={100 * st.median(gs):.1f}% max={100 * max(gs):.1f}%")

# ============ D. FIGURA CLAVE: margen Delta z vs n_cons por régimen ============
# dzplot: dz con suelo 0.0012 para que las celdas de margen nulo sean visibles en eje log
FLOOR = 0.0012
with open(f"{FIG}/cap7-margen-cert.dat", "w") as fc, open(f"{FIG}/cap7-margen-acot.dat", "w") as fa:
    fc.write("ncons dz dzplot\n"); fa.write("ncons dz dzplot\n")
    for c, v in sorted(cell.items()):
        dz = v["z_med"] - v["z_exact"]
        (fc if v["regime"] == "certified" else fa).write(
            f"{v['ncons']} {dz:.5f} {max(dz, FLOOR):.5f}\n")
with open(f"{FIG}/cap7-margen-med.dat", "w") as fm:
    fm.write("ncons dz\n")
    for n in (8, 10, 12, 20, 30, 40):
        cs = [c for c in cell if c[0] == n]
        nc = st.median(cell[c]["ncons"] for c in cs)
        dzm = st.median(cell[c]["z_med"] - cell[c]["z_exact"] for c in cs)
        fm.write(f"{nc:.0f} {dzm:.4f}\n")
alldz = [v["z_med"] - v["z_exact"] for v in cell.values()]
print(f"\nD. margen (270): min={min(alldz):+.4f} max={max(alldz):+.4f} -> ejes figura")
print(f"   ncons: min={min(v['ncons'] for v in cell.values())} max={max(v['ncons'] for v in cell.values())}")
# mediana del margen por n (para prosa: continuidad)
for n in (8, 10, 12, 20, 30, 40):
    dd = [v["z_med"] - v["z_exact"] for c, v in cell.items() if c[0] == n]
    print(f"   n={n:>2}: margen med={st.median(dd):+.4f}")

# ============ E. SA vs GREEDY (270) ============
print("\n" + "=" * 70 + "\nE. SA vs GREEDY (pool 270)\n" + "=" * 70)
g_feas = sum(1 for v in cell.values() if v["feas_greedy"] == "True")
tot_reps = sum(len(v) for v in by_cell.values())
feas_sa_reps = sum(v["feas_sa"] for v in cell.values())
print(f"greedy factible: {g_feas}/270 celdas ; SA factible: {feas_sa_reps}/{tot_reps} réplicas")
by_grid_gf = {"base": sum(1 for c, v in cell.items() if c[0] <= 12 and v["feas_greedy"] == "True"),
              "ext": sum(1 for c, v in cell.items() if c[0] >= 20 and v["feas_greedy"] == "True")}
print(f"  greedy factible por rejilla: {by_grid_gf}")
sa_a = [v["z_med"] for v in cell.values()]; g_a = [v["z_greedy"] for v in cell.values()]
n_better = sum(1 for a, b in zip(sa_a, g_a) if a < b - 1e-9)
W, p = stats.wilcoxon(sa_a, g_a, alternative="less")
res = stats.wilcoxon(sa_a, g_a, alternative="less", method="approx")
rbc = res.zstatistic / math.sqrt(sum(1 for a, b in zip(sa_a, g_a) if abs(a - b) > 1e-9))
print(f"SA mejor en {n_better}/270 ; Wilcoxon W={W:.0f} p_scipy={p:.2e} ; Z/sqrt(N) (Rosenthal)={rbc:.3f}")
# estadística tal y como la reporta el capítulo: con W=0 y sin empates el p es combinatorio
# exacto y la biserial de rangos pareada (T+ - T-)/(T+ + T-) alcanza su extremo
d_nz = [a - b for a, b in zip(sa_a, g_a) if abs(a - b) > 1e-9]
if all(x < 0 for x in d_nz):
    print(f"  las {len(d_nz)} diferencias comparten signo -> p exacto unilateral 2^-{len(d_nz)}"
          f" = {0.5 ** len(d_nz):.2e} ; biserial de rangos r = -1 (dominancia completa)")
# IC 95% libre de distribución (estadísticos de orden) para la mediana de greedy-SA
d_sorted = sorted(g - a for a, g in zip(sa_a, g_a))
n_d = len(d_sorted)
lo_k = int(stats.binom.ppf(0.025, n_d, 0.5))
hi_k = int(stats.binom.isf(0.025, n_d, 0.5)) + 1
print(f"mediana(greedy-SA) = {st.median(d_sorted):.1f} ; IC95 por orden "
      f"[{d_sorted[lo_k - 1]:.1f}, {d_sorted[hi_k - 1]:.1f}] (rangos {lo_k},{hi_k})")
# Friedman con los tres métodos (greedy, SA, exacto/incumbent) + W de Kendall
zx_a = [v["z_exact"] for v in cell.values()]
chi2_f, p_f = stats.friedmanchisquare(g_a, sa_a, zx_a)
w_kendall = chi2_f / (len(g_a) * 2)
strict = sum(1 for g, s, x in zip(g_a, sa_a, zx_a) if x < s - 1e-12 and s < g - 1e-12)
ties_ref = sum(1 for s, x in zip(sa_a, zx_a) if abs(s - x) <= 1e-12)
print(f"Friedman: chi2(2)={chi2_f:.1f} p={p_f:.2e} ; W de Kendall={w_kendall:.4f}")
print(f"  orden estricto exacto<SA<greedy en {strict}/270 ; empates SA=referencia: {ties_ref}")
# desglose de la mejora (por término, media sobre 270): greedy - SA en contribución
imp = defaultdict(list)
for c, v in cell.items():
    nm, w = norm_cache[c]
    csa = contrib(v["comp"], nm, w)
    rows = by_cell[c]
    cgr = contrib({cc: float(rows[0][f"greedy_{cc}"]) for cc in COMPS}, nm, w)
    for cc in COMPS: imp[cc].append(cgr[cc] - csa[cc])
print("mejora media de contribución greedy-SA por término (>0 SA mejor):")
for cc in COMPS: print(f"   {cc:6s}: {st.mean(imp[cc]):+.3f}")
# tiempos
for n in (8, 10, 12, 20, 30, 40):
    tt = [v["t_sa_med"] for c, v in cell.items() if c[0] == n]
    print(f"  t_SA n={n:>2}: med={st.median(tt):.1f}s max={max(tt):.1f}s")
t_base = [v["t_sa_med"] for c, v in cell.items() if c[0] <= 12]
t_ext = [v["t_sa_med"] for c, v in cell.items() if c[0] >= 20]
t_all = [v["t_sa_med"] for v in cell.values()]
print(f"  t_SA base: med={st.median(t_base):.1f}s ; ext: med={st.median(t_ext):.1f}s ; pool: med={st.median(t_all):.1f}s max={max(t_all):.1f}s")

# .dat dominación (270)
with open(f"{FIG}/cap7-domination-gfeas.dat", "w") as okf, open(f"{FIG}/cap7-domination-ginf.dat", "w") as nof:
    okf.write("zg zsa\n"); nof.write("zg zsa\n")
    for c, v in sorted(cell.items()):
        line = f"{v['z_greedy']:.5f} {v['z_med']:.5f}\n"
        (okf if v["feas_greedy"] == "True" else nof).write(line)
print("rango dominación:", f"zg en [{min(g_a):.3f},{max(g_a):.1f}], zsa en [{min(sa_a):.3f},{max(sa_a):.1f}]")

# ============ F. VÁLVULA (270) ============
print("\n" + "=" * 70 + "\nF. VÁLVULA (pool 270)\n" + "=" * 70)
lev = defaultdict(list)
for c, v in sorted(cell.items()):
    lev[c[3]].append((c, v))
n_over = sum(1 for v in cell.values() if v["comp"]["p_over"] > 1e-9)
n_def = sum(1 for v in cell.values() if v["comp"]["p_cob"] > 1e-9)
print(f"celdas con sobre-jornada: {n_over}/270 ; con déficit: {n_def}/270")
for cr in sorted(lev):
    ov = [v["comp"]["p_over"] for _, v in lev[cr]]
    dcells = sum(1 for _, v in lev[cr] if v["comp"]["p_cob"] > 1e-9)
    print(f"  cr={cr}: sobre-jornada med={st.median(ov):.1f}h max={max(ov):.0f}h ; celdas con déficit={dcells}/{len(lev[cr])}")
ok = open(f"{FIG}/cap7-valvula-ok.dat", "w"); ok.write("x pover\n")
df = open(f"{FIG}/cap7-valvula-def.dat", "w"); df.write("x pover\n")
for cr, items in lev.items():
    m = len(items)
    for i, (c, v) in enumerate(items):
        xj = cr + (i / (m - 1) - 0.5) * 0.16
        (df if v["comp"]["p_cob"] > 1e-9 else ok).write(f"{xj:.4f} {v['comp']['p_over']:.2f}\n")
ok.close(); df.close()
print(f"máx sobre-jornada global: {max(v['comp']['p_over'] for v in cell.values()):.0f}h -> ymax figura")

# ============ G. EQUIDAD (270) ============
print("\n" + "=" * 70 + "\nG. EQUIDAD (pool 270): Gini carga, Theil guardias, por tamaño\n" + "=" * 70)
fair_cell = defaultdict(lambda: defaultdict(list))
for fr in fair:
    c = key(fr, "seed_inst"); ss = int(fr["seed_sa"])
    fair_cell[c][ss].append((float(fr["hours"]), int(fr["n_guardias"]), class_cache[c][int(fr["phys_id"])]))
gini_cell, cv_cell, theil_cell, tb_cell, tw_cell = {}, {}, {}, {}, {}
for c, byss in fair_cell.items():
    a, b, d, e, f = [], [], [], [], []
    for ss, rows in byss.items():
        h = [r[0] for r in rows]; gu = [r[1] for r in rows]; res_ = [r[2] for r in rows]
        a.append(gini(h)); f.append(cvar(h))
        t, tbv, twv = theil_dec(gu, res_)
        b.append(t); d.append(tbv); e.append(twv)
    gini_cell[c] = st.median(a); cv_cell[c] = st.median(f)
    theil_cell[c] = st.median(b); tb_cell[c] = st.median(d); tw_cell[c] = st.median(e)
print(f"celdas con fairness: {len(fair_cell)}/270")
gl = list(gini_cell.values()); tg = list(theil_cell.values())
print(f"Gini carga (270): med={st.median(gl):.3f} max={max(gl):.3f} ; CV med={st.median(list(cv_cell.values())):.3f}")
print(f"Theil guardias (270): med={st.median(tg):.4f}")
share = [tb_cell[c] / theil_cell[c] for c in theil_cell if theil_cell[c] > 1e-9]
print(f"cuota entre-clases/total (T>0): med={100 * st.median(share):.1f}%")
for n in (8, 10, 12, 20, 30, 40):
    gg = [gini_cell[c] for c in gini_cell if c[0] == n]
    tt = [theil_cell[c] for c in theil_cell if c[0] == n]
    print(f"  n={n:>2}: Gini carga med={st.median(gg):.4f} max={max(gg):.4f} ; Theil med={st.median(tt):.4f}")
g_base = [gini_cell[c] for c in gini_cell if c[0] <= 12]
g_ext = [gini_cell[c] for c in gini_cell if c[0] >= 20]
print(f"Gini base (n<=12): med={st.median(g_base):.4f} ; escala (n>=20): med={st.median(g_ext):.4f}")

# refuerzo anti-artefacto (citado en Sección 7.4.6): Gini corregido n/(n-1) y CV por rejilla
gcorr_cell = {}
for c, byss in fair_cell.items():
    gs = [gini([r[0] for r in rows]) * len(rows) / (len(rows) - 1) for rows in byss.values()]
    gcorr_cell[c] = st.median(gs)
gc_b = [v for c, v in gcorr_cell.items() if c[0] <= 12]
gc_e = [v for c, v in gcorr_cell.items() if c[0] >= 20]
cv_b = [cv_cell[c] for c in cv_cell if c[0] <= 12]
cv_e = [cv_cell[c] for c in cv_cell if c[0] >= 20]
print(f"Gini corregido n/(n-1): base med={st.median(gc_b):.4f} ; escala med={st.median(gc_e):.4f}")
print(f"CV carga: base med={st.median(cv_b):.3f} ; escala med={st.median(cv_e):.3f}")

# .dat theil (270, por holgura)
by_cr_t = defaultdict(lambda: {"btw": [], "wth": []})
for c in theil_cell:
    by_cr_t[c[3]]["btw"].append(tb_cell[c]); by_cr_t[c[3]]["wth"].append(tw_cell[c])
with open(f"{FIG}/cap7-theil.dat", "w") as f:
    f.write("holgura entre dentro\n")
    for cr in sorted(by_cr_t):
        d = by_cr_t[cr]
        f.write(f"{cr:.2f} {st.median(d['btw']):.5f} {st.median(d['wth']):.5f}\n")

# ============ H. ESTABILIDAD (270) ============
print("\n" + "=" * 70 + "\nH. ESTABILIDAD SA (270)\n" + "=" * 70)
sds = [v["z_sd"] for v in cell.values()]; rng = [v["z_max"] - v["z_min"] for v in cell.values()]
print(f"sd por celda: med={st.median(sds):.4f} max={max(sds):.4f} ; rango max={max(rng):.2f}")
worst = sorted(cell.items(), key=lambda t: t[1]["z_max"] - t[1]["z_min"], reverse=True)[:5]
for c, v in worst:
    print(f"  peor rango: n={c[0]} d={c[1]} seed={c[2]} cr={c[3]} rango={v['z_max'] - v['z_min']:.2f}")

# ============ I. BREAKDOWN (61 cert) para cap7-breakdown.tex ============
# NOTA: cada término se agrega por la MEDIANA de las 30 réplicas (conducta típica),
# no por la media. La mediana no es aditiva: las barras NO suman la media de Delta z
# (con medias sí sumaría exactamente, pero la dominarían las trayectorias atípicas
# raras de las celdas tensas: p_cob pasaría de 0.000 a +0.44). Declarado en nota
# al pie de Sección 7.4.2.
print("\n" + "=" * 70 + "\nI. BREAKDOWN 61 certificadas (coordenadas para el .tex):")
names = {"p_cob": "cobertura", "p_erg": "ergonomía", "p_over": "sobre-jornada",
         "p_gua": "guardias", "p_cal": "calendario", "p_car": "carga"}
vals = {cc: st.mean(comp_delta[cc]) for cc in COMPS if cc != "pref"}
for cc, v in sorted(vals.items(), key=lambda t: t[1]):
    print(f"  ({v:.3f},{names[cc]})")
print("\n.dat escritos en", FIG)
