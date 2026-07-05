"""Comparador: enfrenta varias soluciones de una misma instancia bajo una única política."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from rostering.objective import ObjectiveConfig

from core import charts, services, ui
from core.services import SOLVER_LABELS

ui.page_header(
    "Comparador",
    "Confronta soluciones de la misma instancia bajo una política de evaluación única: "
    "se re-evalúan todas con los mismos pesos w_k y L^max, como en el Cap. 7 al enfrentar "
    "<em>greedy</em>, <em>simulated annealing</em> y MILP exacto.",
    kicker="ANÁLISIS",
)

sel = ui.pick_instance("cmp_inst")
if sel is None:
    st.stop()
instance_name, instance = sel

records = ui.pick_solutions("cmp_sols", instance_name=instance_name)
if len(records) < 2:
    st.info("Selecciona al menos dos soluciones de esta instancia.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Política única de evaluación
# ---------------------------------------------------------------------------

st.subheader("Política de evaluación")
mode = st.radio(
    "Política del objetivo",
    ["base (por defecto)", "la de una solución", "personalizada"],
    key="cmp_cfg_mode",
)

if mode == "la de una solución":
    ref_name = st.selectbox(
        "Tomar la política de", [r.name for r in records], key="cmp_cfg_ref",
        format_func=lambda n: n,
    )
    ref_rec = next(r for r in records if r.name == ref_name)
    try:
        config = ObjectiveConfig.model_validate(ref_rec.params.get("objetivo", {}))
    except ValidationError:
        config = ObjectiveConfig()
elif mode == "personalizada":
    config = ui.objective_config_widget("cmp_obj")
else:
    config = ObjectiveConfig()

st.caption(
    "Comparar exige evaluar todas las soluciones con la misma política: los pesos w_k y "
    "L^max elegidos aquí sustituyen a los que se usaron al resolver cada una."
)

st.divider()

# ---------------------------------------------------------------------------
# Re-evaluación de todas las soluciones bajo la política elegida
# ---------------------------------------------------------------------------

reports = {r.name: services.evaluation_report(instance, r.roster(), config) for r in records}


def _diagnostico(rec: object) -> str:
    """Diagnóstico propio del método si consta en las métricas registradas: stop_reason
    del SA, o certificado/gap del MILP; '—' si no hay ninguno de los dos."""
    m = rec.metrics
    stop = m.get("stop_reason")
    if stop:
        return str(stop)
    gap = m.get("gap")
    if gap is not None:
        return "certificado (gap=0)" if m.get("proven") else f"gap={ui.fmt_pct(gap)}"
    if m.get("proven"):
        return "certificado (gap=0)"
    return "—"


st.subheader("Comparación")
rows = []
for r in records:
    report = reports[r.name]
    rows.append({
        "solución": r.name,
        "método": SOLVER_LABELS.get(r.solver, r.solver),
        "Z_modelo": report["z_modelo"],
        "Δ Z_modelo": None,  # se completa tras conocer la mejor solución factible
        "Z_clásica": report["z_clasica"],
        "factible": report["feasible"],
        "déficit p_cob": report["breakdown"].p_cob,
        "violaciones duras": report["violations"].total,
        "tiempo (s)": r.metrics.get("elapsed"),
        "diagnóstico": _diagnostico(r),
    })

# Δ Z_modelo: diferencia frente a la mejor (menor Z_modelo) solución factible re-evaluada;
# si ninguna es factible, frente a la mejor a secas. La mejor queda en 0.0000.
feasible_rows = [row for row in rows if row["factible"]]
basis_rows = feasible_rows or rows
best_z = min(row["Z_modelo"] for row in basis_rows)
for row in rows:
    row["Δ Z_modelo"] = row["Z_modelo"] - best_z

df = pd.DataFrame(rows)
st.dataframe(
    df, width="stretch", height=ui.dataframe_height(len(df)), hide_index=True,
    column_config={
        "Z_modelo": st.column_config.NumberColumn(
            format="%.4f",
            help="Valor de la función objetivo del modelo (Cap. 6): menor es mejor."),
        "Δ Z_modelo": st.column_config.NumberColumn(
            format="%.4f",
            help="Diferencia frente a la mejor solución factible re-evaluada (0 en su fila)."),
        "Z_clásica": st.column_config.NumberColumn(
            format="%.4f", help="Z con la cobertura como restricción dura, fuera del objetivo."),
        "déficit p_cob": st.column_config.NumberColumn(
            format="%.0f", help="Plazas de demanda sin cubrir bajo esta solución."),
        "violaciones duras": st.column_config.NumberColumn(
            format="%.0f", help="Número total de violaciones duras detectadas (0 = factible)."),
        "tiempo (s)": st.column_config.NumberColumn(format="%.2f"),
    },
)

if feasible_rows:
    best = min(feasible_rows, key=lambda row: row["Z_modelo"])
    st.caption(f"Mejor solución factible: **{best['solución']}** (menor Z_modelo); "
               "Δ Z_modelo en la tabla se mide frente a ella (0.0000 en su fila).")
else:
    st.caption("Ninguna de las soluciones seleccionadas es factible con esta política; "
               "Δ Z_modelo se mide frente a la de menor Z_modelo entre las mostradas.")

st.divider()

# ---------------------------------------------------------------------------
# Contribuciones por término
# ---------------------------------------------------------------------------

st.subheader("Contribuciones por término")
long_df = pd.concat(
    [reports[r.name]["terms"][["término", "contribución"]].assign(solución=r.name)
     for r in records],
    ignore_index=True,
)[["solución", "término", "contribución"]]
st.plotly_chart(charts.compare_terms_chart(long_df), width="stretch",
                config={"displayModeBar": False})
st.caption(
    "Cada barra es la contribución con signo de un término a Z_modelo; las preferencias "
    "satisfechas restan (aparecen en negativo)."
)

st.divider()

# ---------------------------------------------------------------------------
# Distribución de horas por médico
# ---------------------------------------------------------------------------

st.subheader("Distribución de horas por médico")
groups = {
    r.name: services.per_physician_frame(instance, r.roster())["horas"].tolist()
    for r in records
}
st.plotly_chart(charts.box_by_group(groups, y_title="horas por médico"), width="stretch",
                config={"displayModeBar": False})
st.caption("Cada punto es un médico y la caja resume la distribución de horas de esa "
          "solución; menor dispersión equivale a una carga mejor equilibrada (p_car).")

st.divider()

# ---------------------------------------------------------------------------
# Diferencias entre dos soluciones
# ---------------------------------------------------------------------------

if len(records) == 2:
    rec_a, rec_b = records[0], records[1]
    a = set(rec_a.roster().assignments)
    b = set(rec_b.roster().assignments)
    comunes = a & b
    solo_a = a - b
    solo_b = b - a

    st.subheader("Diferencias entre las dos soluciones")
    st.markdown(
        f"**{rec_a.name}** ({ui.solver_badge_md(rec_a.solver)}) frente a "
        f"**{rec_b.name}** ({ui.solver_badge_md(rec_b.solver)})"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Asignaciones comunes", len(comunes), border=True)
    c2.metric(f"Solo en {rec_a.name}", len(solo_a), border=True)
    c3.metric(f"Solo en {rec_b.name}", len(solo_b), border=True)
    st.caption(
        "Cada asignación es una terna (médico, día, turno); aquí se cuentan las que ambas "
        "soluciones comparten y las exclusivas de cada una. Dos cuadrantes con Z_modelo "
        "casi idéntica pueden compartir pocas ternas: el problema tiene muchos óptimos "
        "casi equivalentes (intercambiar dos médicos igualmente cualificados en un turno "
        "apenas mueve Z)."
    )

    phys_by_id = {p.id: p for p in instance.physicians}

    def _diff_frame(triples: set[tuple[int, int, str]]) -> pd.DataFrame:
        rows = [{
            "médico": services.physician_label(instance, phys_by_id[i]),
            "día": services.day_label(instance, t),
            "turno": s,
        } for (i, t, s) in sorted(triples)]
        return pd.DataFrame(rows, columns=["médico", "día", "turno"])

    with st.expander("Ver las ternas que difieren"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"Solo en {rec_a.name}")
            df_a = _diff_frame(solo_a)
            st.dataframe(df_a, width="stretch", height=ui.dataframe_height(len(df_a)),
                        hide_index=True)
        with col_b:
            st.caption(f"Solo en {rec_b.name}")
            df_b = _diff_frame(solo_b)
            st.dataframe(df_b, width="stretch", height=ui.dataframe_height(len(df_b)),
                        hide_index=True)

    st.divider()

st.caption(
    "Los registros guardan las métricas medidas al resolver; aquí todas las soluciones se "
    "re-evalúan con la política elegida arriba, así que los valores pueden diferir de los "
    "registrados en su momento."
)
