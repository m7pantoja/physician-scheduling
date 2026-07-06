"""Página Cuadrante: visualiza una solución guardada como cuadrante médico por día."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from core import charts, services, store, ui

ui.page_header(
    "Cuadrante",
    "Solución guardada como cuadrante médico × día, con localización de violaciones "
    "duras y cargas individuales frente a sus corredores.",
    kicker="ANÁLISIS",
)

rec = ui.pick_solution("cua_sel")
if rec is None:
    st.stop()

try:
    instance = store.load_instance(rec.instance_name)
except Exception:
    st.error(f"La instancia {rec.instance_name} ya no está en el espacio de trabajo.")
    st.stop()

roster = rec.roster()
if not services.roster_matches_instance(instance, roster):
    st.error(f"La solución **{rec.name}** no es compatible con la instancia "
             f"**{rec.instance_name}** actual: referencia médicos, días o turnos que ya "
             "no existen (la instancia se editó y sobrescribió después de resolver).")
    st.stop()

st.caption(
    f"{ui.solver_badge_md(rec.solver)} · "
    f"guardada el {rec.created[:16].replace('T', ' ')} · "
    f"instancia **{rec.instance_name}**"
)

col_z, col_feas, col_def, col_t = st.columns(4)
col_z.metric("Z_modelo", ui.fmt(rec.metrics.get("z_modelo")), border=True)
ui.feasibility_metric(col_feas, bool(rec.metrics.get("feasible")), rec.metrics.get("violaciones"))
col_def.metric("Déficit de cobertura (p_cob)", ui.fmt(rec.metrics.get("p_cob"), 0), border=True)
col_t.metric("Tiempo (s)", ui.fmt(rec.metrics.get("elapsed"), 2), border=True)

st.divider()

# ---------------------------------------------------------------------------
# Cuadrante
# ---------------------------------------------------------------------------
st.subheader("Cuadrante médico × día")

cells_df, phys_df = services.violation_marks(instance, roster)

if cells_df.empty:
    # sin celdas que señalar, el toggle prometería anillos que no pueden existir
    show_marks = False
    if not bool(rec.metrics.get("feasible")):
        st.caption("Las violaciones duras de esta solución son a nivel de médico y no "
                   "señalan celdas concretas del cuadrante: el detalle está más abajo.")
else:
    # la clave incluye la solución y su sello: un widget con key fija arrastraría el
    # estado de la solución anterior (o de una homónima ya borrada)
    show_marks = st.toggle(
        "Señalar violaciones duras en el cuadrante",
        value=True,
        key=f"cua_marks_{rec.name}_{rec.created}",
    )

codes = services.roster_matrix(instance, roster)
marked = services.marked_cell_labels(cells_df) if show_marks else set()
styler = charts.roster_styler(
    codes,
    shift_names=services.shift_names(instance),
    guardia_names={s.name for s in instance.shifts if s.is_guardia},
    weekend_cols=services.weekend_labels(instance),
    holiday_cols=services.holiday_labels(instance),
    marked=marked,
)
st.dataframe(styler, width="stretch", height=ui.dataframe_height(len(codes), max_rows=40))
caption = (
    "Las celdas combinadas como «M+G1» indican un turno de mañana encadenado a una "
    "guardia; las guardias se muestran en negrita. El fondo gris marca fin de semana "
    "o festivo sin asignación."
)
if show_marks:
    caption += " El anillo rojo señala una celda implicada en una violación dura."
st.caption(caption)
st.download_button(
    "Descargar cuadrante (CSV)",
    data=codes.to_csv(),
    file_name=f"{rec.name}-cuadrante.csv",
    mime="text/csv",
    key="cua_dl_cuadrante",
)

if len(cells_df) or len(phys_df):
    with st.expander("Detalle de las violaciones duras",
                     expanded=not bool(rec.metrics.get("feasible"))):
        st.caption(
            "Las violaciones a nivel de médico (suelo del corredor h^min, tope de "
            "guardias g^max, descanso semanal) no señalan una celda concreta del "
            "cuadrante."
        )
        st.markdown("**Celdas implicadas**")
        if len(cells_df):
            st.dataframe(
                cells_df, width="stretch", height=ui.dataframe_height(len(cells_df)),
                hide_index=True, column_order=["médico", "día", "restricción", "detalle"],
            )
        else:
            st.caption("Ninguna violación implica una celda concreta.")
        st.markdown("**Nivel de médico**")
        if len(phys_df):
            st.dataframe(
                phys_df, width="stretch", height=ui.dataframe_height(len(phys_df)),
                hide_index=True, column_order=["médico", "restricción", "detalle"],
            )
        else:
            st.caption("Ninguna violación a nivel de médico.")
else:
    st.caption("Ninguna violación dura localizada.")

st.divider()

# ---------------------------------------------------------------------------
# Cobertura
# ---------------------------------------------------------------------------
st.subheader("Cobertura")

st.plotly_chart(
    charts.deficit_heatmap(services.deficit_matrix(instance, roster)),
    width="stretch", config={"displayModeBar": False},
)
st.caption("Cada celda anota el déficit del turno en ese día (demanda sin cubrir); el "
          "fondo gris marca celdas sin demanda registrada (no aplica).")

cov = services.coverage_frame(instance, roster)
cov_column_config = {
    "t": None,
    "demanda": st.column_config.NumberColumn("demanda", format="%.0f"),
    "asignados": st.column_config.NumberColumn("asignados", format="%.0f"),
    "déficit": st.column_config.NumberColumn(
        "déficit", format="%.0f", help="Plazas de demanda sin cubrir en esa celda (turno, día)."),
}
cov_deficit = cov[cov["déficit"] > 0] if "déficit" in cov.columns else cov
if len(cov_deficit) == 0:
    st.caption("Cobertura completa: ninguna celda con déficit.")
else:
    st.dataframe(cov_deficit, width="stretch",
                 height=ui.dataframe_height(len(cov_deficit)), hide_index=True,
                 column_config=cov_column_config)

with st.expander("Tabla de cobertura completa"):
    st.dataframe(cov, width="stretch", height=ui.dataframe_height(len(cov)), hide_index=True,
                 column_config=cov_column_config)
    st.download_button(
        "Descargar cobertura (CSV)",
        data=cov.to_csv(index=False),
        file_name=f"{rec.name}-cobertura.csv",
        mime="text/csv",
        key="cua_dl_cobertura",
    )

st.divider()

# ---------------------------------------------------------------------------
# Cargas individuales
# ---------------------------------------------------------------------------
st.subheader("Cargas individuales")

pp = services.per_physician_frame(instance, roster)
col_h, col_g = st.columns(2)
with col_h:
    st.plotly_chart(charts.hours_corridor_chart(pp), width="stretch",
                     config={"displayModeBar": False})
    st.caption("Barra = horas asignadas por médico; las marcas verticales delimitan el "
              "corredor [h^min, h^max] y el rombo marca τ_i (carga media ponderada).")
with col_g:
    st.plotly_chart(charts.guardias_chart(pp), width="stretch",
                     config={"displayModeBar": False})
    st.caption("Barra = guardias asignadas; el rombo marca la cuota ρ_i·τ_G y la marca "
              "vertical (si existe) el tope g^max.")

with st.expander("Tabla por médico"):
    st.dataframe(
        pp, width="stretch", height=ui.dataframe_height(len(pp)), hide_index=True,
        column_config={
            "id": None,
            "rho": st.column_config.NumberColumn(
                "ρ (jornada)", format="%.2f",
                help="Fracción de jornada completa; escala horas, corredor y cuota de guardias."),
            "exento": st.column_config.CheckboxColumn("exento", help="Exento de guardias (E)."),
            "horas": st.column_config.NumberColumn("horas", format="%.1f"),
            "h_min": st.column_config.NumberColumn(
                "h^min", format="%.1f", help="Suelo del corredor de horas admisible."),
            "h_max": st.column_config.NumberColumn(
                "h^max", format="%.1f", help="Techo del corredor de horas admisible."),
            "tau": st.column_config.NumberColumn(
                "τ (cuota de horas)", format="%.1f",
                help="Carga media ponderada por ρ_i: referencia de equidad de carga (p_car)."),
            "guardias": st.column_config.NumberColumn("guardias", format="%.0f"),
            "g_max": st.column_config.NumberColumn(
                "g^max", format="%.0f", help="Tope de guardias del médico; vacío si no aplica."),
            "cuota_guardias": st.column_config.NumberColumn(
                "cuota de guardias", format="%.2f",
                help="Referencia de equidad de guardias (p_gua): ρ_i · τ_G sobre los no exentos."),
            "fds": st.column_config.NumberColumn("turnos en fin de semana", format="%.0f"),
            "fes": st.column_config.NumberColumn("turnos en festivo", format="%.0f"),
            "días_activos": st.column_config.NumberColumn("días activos", format="%.0f"),
        },
    )

st.divider()
st.caption(
    "El cuadrante muestra la solución tal cual se guardó; para re-puntuarla con otra "
    "política del objetivo, usar la página Evaluación."
)
