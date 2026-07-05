"""Evaluación: las tres formulaciones de Z, desglose del objetivo y restricciones duras."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from rostering.objective import ObjectiveConfig

from core import services, store, ui
from core.charts import contributions_bar, guardias_chart, hours_corridor_chart

ui.page_header(
    "Evaluación",
    "Informe completo de una solución con la función objetivo de los Caps. 5-6: las tres "
    "formulaciones de Z, el desglose de términos, un veredicto de factibilidad y las nueve "
    "restricciones duras.",
    kicker="ANÁLISIS",
)

rec = ui.pick_solution("eva_sel")
if rec is None:
    st.stop()

try:
    instance = store.load_instance(rec.instance_name)
except Exception as exc:
    st.error(f"No se pudo cargar la instancia **{rec.instance_name}** asociada a esta "
             "solución (puede haberse borrado del espacio de trabajo).")
    with st.expander("Detalle del error"):
        st.code(repr(exc))
    st.stop()

roster = rec.roster()

st.caption(f"Solución **{rec.name}** de la instancia **{rec.instance_name}** · "
           f"{ui.solver_badge_md(rec.solver)}")

st.divider()

# ---------------------------------------------------------------------------
# Política de evaluación
# ---------------------------------------------------------------------------

st.subheader("Política del objetivo")
try:
    initial = ObjectiveConfig.model_validate(rec.params.get("objetivo", {}))
except Exception:
    initial = ObjectiveConfig()

st.caption("Política con la que se resolvió; puedes modificarla para re-puntuar (la "
           "evaluación es independiente del método).")
# la clave incluye la solución: al cambiar de solución, el formulario se siembra de nuevo
# con la política registrada de la nueva (un widget con key fija ignoraría el nuevo initial)
config = ui.objective_config_widget(f"eva_obj_{rec.name}", initial=initial)

report = services.evaluation_report(instance, roster, config)
hard = report["hard"]
n_fail = int((~hard["cumple"]).sum())

st.divider()

# ---------------------------------------------------------------------------
# Titulares
# ---------------------------------------------------------------------------

st.subheader("Titulares")
cols = st.columns(4)
cols[0].metric("Z_clásica", ui.fmt(report["z_clasica"]), border=True)
cols[1].metric("Z_modelo", ui.fmt(report["z_modelo"]), border=True)
cols[2].metric("Z_aumentada", ui.fmt(report["z_aumentada"]), border=True)
ui.feasibility_metric(cols[3], report["feasible"], report["violations"].total)

cols2 = st.columns(4)
cols2[0].metric("w_cob (resuelto)", ui.fmt(report["w_cob"], 2), border=True)
cols2[1].metric("μ (resuelto)", ui.fmt(report["mu"], 2), border=True)

p_cob = report["breakdown"].p_cob
if report["feasible"] and p_cob == 0:
    st.success("Cuadrante factible y con cobertura completa.")
elif report["feasible"]:
    st.warning(f"Cuadrante factible con déficit de cobertura: {p_cob:.0f} plazas sin cubrir.")
else:
    st.error(f"Cuadrante infactible: {n_fail} de las nueve familias duras no se satisfacen. "
             "Consulta la página Cuadrante para localizar las violaciones sobre el "
             "calendario.")

with st.expander("Las tres formulaciones de Z"):
    st.caption("Z_clásica: la cobertura es una restricción dura, fuera de Z.")
    st.caption("Z_modelo: la cobertura es blanda y dominante dentro de Z; es la que comparan "
               "los experimentos del Cap. 7.")
    st.caption("Z_aumentada: además penaliza cada violación dura con μ; es la que minimiza el "
               "simulated annealing.")

st.divider()

# ---------------------------------------------------------------------------
# Restricciones duras
# ---------------------------------------------------------------------------

st.subheader("Restricciones duras")
st.dataframe(
    hard, width="stretch", height=ui.dataframe_height(len(hard)), hide_index=True,
    column_order=("restricción", "violación", "cumple"),
    column_config={
        "violación": st.column_config.NumberColumn(
            "violación", format="%.0f",
            help="Recuento de violaciones detectadas para esta familia dura; 0 = se cumple."),
        "cumple": st.column_config.CheckboxColumn("cumple"),
    },
)
if n_fail == 0:
    st.success("Las nueve familias duras se satisfacen.")
else:
    st.error(f"{n_fail} de las nueve familias duras no se satisfacen.")

st.divider()

# ---------------------------------------------------------------------------
# Términos del objetivo
# ---------------------------------------------------------------------------

st.subheader("Términos del objetivo")
terms = report["terms"]
st.dataframe(
    terms, width="stretch", height=ui.dataframe_height(len(terms)), hide_index=True,
    column_order=("término", "crudo", "normalizador", "peso", "contribución"),
    column_config={
        "crudo": st.column_config.NumberColumn(
            format="%.4f", help="Valor crudo del término, sin normalizar."),
        "normalizador": st.column_config.NumberColumn(
            format="%.4f", help="Escala que adimensionaliza el término antes de ponderarlo."),
        "peso": st.column_config.NumberColumn(format="%.4f", help="Peso w_k del término."),
        "contribución": st.column_config.NumberColumn(
            format="%.4f",
            help="Peso por valor normalizado: aporte con signo del término a Z_modelo."),
    },
)
st.plotly_chart(contributions_bar(terms), width="stretch", config={"displayModeBar": False})
st.caption("La contribución es peso por valor normalizado; las preferencias entran en "
           "negativo porque se maximizan. La cobertura entra en Z_modelo con w_cob "
           "dominante.")

st.divider()

# ---------------------------------------------------------------------------
# Equidad y cargas
# ---------------------------------------------------------------------------

st.subheader("Equidad y cargas")
pp = services.per_physician_frame(instance, roster)
col_h, col_g = st.columns(2)
with col_h:
    st.plotly_chart(hours_corridor_chart(pp), width="stretch",
                     config={"displayModeBar": False})
    st.caption("Barra = horas asignadas por médico; las marcas verticales delimitan el "
              "corredor [h^min, h^max] y el rombo marca τ_i (carga media ponderada).")
with col_g:
    st.plotly_chart(guardias_chart(pp), width="stretch", config={"displayModeBar": False})
    st.caption("Barra = guardias asignadas; el rombo marca la cuota ρ_i·τ_G y la marca "
              "vertical (si existe) el tope g^max.")

calendario = pp[["médico", "fds", "fes", "días_activos"]]
st.dataframe(
    calendario, width="stretch", height=ui.dataframe_height(len(calendario)), hide_index=True,
    column_config={
        "fds": st.column_config.NumberColumn("turnos en fin de semana", format="%.0f"),
        "fes": st.column_config.NumberColumn("turnos en festivo", format="%.0f"),
        "días_activos": st.column_config.NumberColumn("días activos", format="%.0f"),
    },
)

st.divider()

# ---------------------------------------------------------------------------
# Comparación con lo registrado
# ---------------------------------------------------------------------------

if config.model_dump() != initial.model_dump():
    z_registrado = rec.metrics.get("z_modelo")
    st.info(f"Z_modelo registrado al resolver: {ui.fmt(z_registrado)}. "
            f"Z_modelo re-puntuado con la política actual: {ui.fmt(report['z_modelo'])}.")
