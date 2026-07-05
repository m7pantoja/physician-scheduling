"""Experimento: rejilla de instancias sintéticas resuelta con greedy y SA (MILP opcional)."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from datetime import date

import pandas as pd
import streamlit as st

from rostering.domain import Calendar
from rostering.generator import Settings
from rostering.objective import ObjectiveConfig

from core import charts, services, ui

ui.page_header(
    "Experimento",
    "Rejilla n × días × ratio × semillas resuelta con <em>greedy</em> y <em>simulated "
    "annealing</em> (MILP opcional), al estilo de los experimentos del Cap. 7.",
    kicker="RESOLUCIÓN",
)

N_OPTIONS = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 60]
DAYS_OPTIONS = [7, 14, 21, 28, 35, 42, 56, 70, 84, 98, 112]
RATIO_OPTIONS = [0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30, 1.50, 1.75, 2.00]
MAX_CELLS = 60

st.caption(
    "Los resultados de esta página no se guardan como soluciones del espacio de trabajo: "
    "son datos de experimento que se descargan como CSV."
)

# ---------------------------------------------------------------------------
# Configuración de la rejilla
# ---------------------------------------------------------------------------

st.subheader("Rejilla")
c1, c2, c3 = st.columns(3)
n_values = c1.multiselect("n (médicos)", N_OPTIONS, default=[8, 12], key="exp_n")
day_values = c2.multiselect("días", DAYS_OPTIONS, default=[28], key="exp_days")
ratio_values = c3.multiselect("coverage_ratio", RATIO_OPTIONS, default=[1.10], key="exp_ratio")

c4, c5, c6 = st.columns(3)
k = c4.number_input("nº de semillas (k)", 1, 10, 3, 1, key="exp_k")
base_seed = c5.number_input("semilla base", 0, 10_000, 0, 1, key="exp_seed_base")
rounds = c6.number_input("rounds del SA", 50, 2000, 400, 10, key="exp_rounds")

include_milp = st.checkbox("Incluir MILP exacto", value=False, key="exp_milp_on")
if include_milp:
    engines = services.milp_engines()
    gurobi_available = engines.get("gurobi", False)
    c7, c8 = st.columns(2)
    time_limit = c7.number_input("Límite de tiempo (s)", 5, 3600, 60, 5, key="exp_milp_time")
    engine_choice = c8.radio("Motor del solver", ["HiGHS (libre)", "Gurobi"],
                            key="exp_milp_engine", disabled=not gurobi_available)
    if not gurobi_available:
        st.caption("Gurobi no está disponible en este entorno: se resuelve con HiGHS.")
    engine = "gurobi" if engine_choice == "Gurobi" else "highs"
else:
    time_limit = 60
    engine = "highs"

total_cells = len(n_values) * len(day_values) * len(ratio_values) * int(k)
st.caption(f"**{total_cells}** celdas en la rejilla (tope de {MAX_CELLS} para mantenerla interactiva).")
if total_cells > MAX_CELLS or total_cells == 0:
    st.error(
        f"La rejilla debe tener entre 1 y {MAX_CELLS} celdas; ajusta n, días, ratios o k "
        f"({total_cells} celdas con la selección actual)."
    )
grid_ok = 0 < total_cells <= MAX_CELLS

st.caption(
    "Toda la rejilla se evalúa con la política base `ObjectiveConfig()`; el SA arranca del "
    "greedy con el esquema automático (`services.auto_sa_config`) sembrado con la semilla "
    "de cada celda, como en los experimentos del Cap. 7."
)
st.markdown(
    "Métodos comparados: " + ui.solver_badge_md("greedy") + " vs " + ui.solver_badge_md("sa")
    + (" vs " + ui.solver_badge_md("milp") if include_milp else "")
)

launch = st.button("Lanzar rejilla", type="primary", disabled=not grid_ok, key="exp_launch")

# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

if launch:
    config = ObjectiveConfig()
    cells = [
        (n, d, cr, s)
        for n in n_values for d in day_values for cr in ratio_values
        for s in range(int(base_seed), int(base_seed) + int(k))
    ]
    progress = st.progress(0.0)
    status = st.empty()
    rows: list[dict] = []
    for idx, (n, d, cr, s) in enumerate(cells):
        status.text(f"celda {idx + 1}/{len(cells)} · n={n}, días={d}, ratio={cr}, semilla={s}")
        row: dict = {"n": n, "días": d, "cr": cr, "semilla": s}
        try:
            settings = Settings(n_physicians=n,
                                calendar=Calendar(start_date=date(2026, 1, 5), days=d),
                                coverage_ratio=cr)
            instance = services.generate_instance(settings, seed=s)

            g = services.run_greedy(instance, config)
            sa = services.run_sa(
                instance, config, x0=g["roster"],
                sa_config=services.auto_sa_config(instance, seed=s, rounds=int(rounds)),
            )
            report_g = services.evaluation_report(instance, g["roster"], config)
            report_sa = services.evaluation_report(instance, sa["roster"], config)
            row.update({
                "z_greedy": report_g["z_modelo"],
                "z_sa": report_sa["z_modelo"],
                "mejora": report_g["z_modelo"] - report_sa["z_modelo"],
                "factible_sa": bool(report_sa["feasible"]),
                "t_greedy": g["elapsed"],
                "t_sa": sa["elapsed"],
            })

            if include_milp:
                m = services.run_milp(instance, config, time_limit=int(time_limit),
                                      coverage_hard=True, engine=engine)
                info = m["info"]
                report_m = services.evaluation_report(instance, m["roster"], config)
                row.update({
                    "z_exacto": report_m["z_modelo"],
                    "certificado": bool(info["proven_optimal"]),
                    "gap": info["gap"],
                    "t_exacto": m["elapsed"],
                    "régimen": services.COVERAGE_LABELS.get(m["coverage_mode"], m["coverage_mode"]),
                })
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
        progress.progress((idx + 1) / len(cells))

    status.empty()
    st.session_state["exp_results"] = pd.DataFrame(rows)
    st.session_state["exp_params"] = {
        "n": n_values, "días": day_values, "coverage_ratio": ratio_values,
        "semillas": list(range(int(base_seed), int(base_seed) + int(k))),
        "rounds": int(rounds), "milp": include_milp,
        "time_limit": int(time_limit) if include_milp else None,
        "engine": engine if include_milp else None,
    }

# ---------------------------------------------------------------------------
# Resultados (siempre desde session_state, sobrevive a reruns)
# ---------------------------------------------------------------------------

if "exp_results" in st.session_state:
    df = st.session_state["exp_results"]
    params = st.session_state.get("exp_params", {})

    st.divider()
    st.subheader("Resultados")
    st.caption(
        f"n={params.get('n')} · días={params.get('días')} · "
        f"coverage_ratio={params.get('coverage_ratio')} · semillas={params.get('semillas')} · "
        f"rounds={params.get('rounds')}"
    )

    n_total = len(df)
    n_ok = int(df["error"].isna().sum()) if "error" in df.columns else n_total
    metrics = [("Celdas resueltas", f"{n_ok}/{n_total}")]
    if "mejora" in df.columns and df["mejora"].notna().any():
        metrics.append(("Mejora media (Δ)", ui.fmt(df["mejora"].mean())))
    if "factible_sa" in df.columns and df["factible_sa"].notna().any():
        metrics.append(("SA factible", ui.fmt_pct(df["factible_sa"].mean())))
    if "certificado" in df.columns:
        n_cert = int(df["certificado"].fillna(False).sum())
        metrics.append(("Certificadas (MILP)", f"{n_cert}/{n_total}"))
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value, border=True)

    column_config = {
        "z_greedy": st.column_config.NumberColumn(
            format="%.4f", help="Z_modelo obtenido por la heurística constructiva (greedy)."),
        "z_sa": st.column_config.NumberColumn(
            format="%.4f", help="Z_modelo obtenido por simulated annealing desde el greedy."),
        "mejora": st.column_config.NumberColumn(
            format="%.4f",
            help="Mejora del SA frente al greedy: z_greedy − z_sa (mayor es mejor)."),
        "z_exacto": st.column_config.NumberColumn(
            format="%.4f", help="Z_modelo de la referencia exacta (MILP)."),
        "factible_sa": st.column_config.CheckboxColumn("factible (SA)"),
        "certificado": st.column_config.CheckboxColumn("certificado (MILP)"),
        "t_greedy": st.column_config.NumberColumn(format="%.2f"),
        "t_sa": st.column_config.NumberColumn(format="%.2f"),
        "t_exacto": st.column_config.NumberColumn(format="%.2f"),
        "gap": st.column_config.NumberColumn(
            format="percent",
            help="Gap relativo del solver frente a la cota dual (0 si está certificado)."),
    }
    st.dataframe(
        df, hide_index=True, width="stretch", height=ui.dataframe_height(n_total),
        column_config={k: v for k, v in column_config.items() if k in df.columns},
    )

    if "mejora" in df.columns:
        groups: dict[str, list[float]] = {}
        for (n, d, cr), sub in df.groupby(["n", "días", "cr"]):
            values = sub["mejora"].dropna().tolist()
            if values:
                groups[f"{n}x{d} cr{cr:.2f}"] = values
        if groups:
            fig = charts.box_by_group(groups, y_title="mejora del SA (z_greedy - z_sa)")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("Cada caja agrupa las réplicas (semillas) de una celda n×días×ratio; "
                      "los puntos son las semillas individuales y el eje es la mejora "
                      "Δ = z_greedy − z_sa (mayor es mejor).")

    c1, c2 = st.columns(2)
    c1.download_button(
        "Descargar CSV", data=df.to_csv(index=False), file_name="experimento.csv",
        mime="text/csv", key="exp_download",
    )
    if c2.button("Vaciar resultados", key="exp_clear"):
        del st.session_state["exp_results"]
        st.session_state.pop("exp_params", None)
        st.rerun()

    if bool(params.get("milp")):
        st.caption(
            "Las celdas certificadas del MILP (columna `certificado`) dan la referencia "
            "exacta de Z_modelo; en el resto, la mejora Δ solo compara heurísticas entre sí."
        )
    else:
        st.caption(
            "Sin MILP en la rejilla, la mejora Δ solo compara heurísticas entre sí: no hay "
            "referencia exacta certificada."
        )
