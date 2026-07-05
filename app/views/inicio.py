"""Portada: estado del espacio de trabajo, flujo de la herramienta y gestión de archivos."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st

from rostering.generator import Settings

from core import store, ui
from core.services import SOLVER_LABELS, generate_instance, milp_engines, summarize_instance

ui.page_header(
    "Physician scheduling",
    "Generación, resolución y evaluación de instancias sobre el paquete "
    "<code>rostering</code>, el motor de los Caps. 5–7 del TFG, intacto bajo esta interfaz.",
    kicker="INICIO",
)

instances = store.list_instances()
solutions = store.list_solutions()

# --- estado del espacio de trabajo ---
if "ini_engines" not in st.session_state:
    st.session_state["ini_engines"] = milp_engines()
engines = st.session_state["ini_engines"]

c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("Instancias", len(instances), border=True)
c2.metric("Cuadrantes", len(solutions), border=True)
c3.metric("Motores MILP", "HiGHS y Gurobi" if engines.get("gurobi") else "HiGHS", border=True)

# --- flujo de trabajo ---
st.subheader("Flujo de trabajo")
cards = [
    ("PASO 1 · DATOS", "Generar o editar una instancia",
     "Instancias sintéticas parametrizadas sobre el caso base andaluz, o edición validada "
     "de plantilla, demanda, reglas y preferencias.",
     "views/generador.py", "Abrir el Generador"),
    ("PASO 2 · RESOLUCIÓN", "Resolver con los tres métodos",
     "<em>Greedy</em>, <em>simulated annealing</em> y MILP exacto con la política de pesos "
     "que se quiera; o en rejilla con la página Experimento.",
     "views/resolver.py", "Abrir Resolver"),
    ("PASO 3 · ANÁLISIS", "Cuadrante, evaluación y comparación",
     "La solución como cuadrante médico × día, el desglose completo del objetivo y la "
     "comparación entre soluciones con la misma vara.",
     "views/cuadrante.py", "Abrir el Cuadrante"),
]
cols = st.columns(3)
for col, (step, title, desc, target, label) in zip(cols, cards):
    with col:
        st.markdown(
            f'<div class="app-card"><div class="app-step">{step}</div>'
            f'<h3>{title}</h3><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )
        st.page_link(target, label=label, icon=":material/arrow_forward:")

st.divider()

# --- gestión del espacio de trabajo ---
if not instances:
    st.info("El espacio de trabajo está vacío. Crea el caso base de ejemplo o genera una "
            "instancia a medida en el Generador.")
    if st.button("Crear caso base de ejemplo (12 médicos × 28 días, semilla 0)",
                 type="primary"):
        instance = generate_instance(Settings(), seed=0)
        name = store.save_instance(instance, "caso-base", origin="generada",
                                   note="Caso base del generador (Settings por defecto)",
                                   settings=Settings().model_dump(mode="json"))
        st.success(f"Instancia guardada como **{name}**.")
        st.rerun()
    st.stop()

tab_inst, tab_sol = st.tabs([f"Instancias ({len(instances)})", f"Cuadrantes ({len(solutions)})"])

with tab_inst:
    rows = []
    for meta in instances:
        summary = summarize_instance(store.load_instance(meta.name))
        rows.append({
            "nombre": meta.name,
            "origen": meta.origin,
            "creada": meta.created[:16].replace("T", " "),
            "médicos": summary["n"],
            "días": summary["days"],
            "plazas demandadas": summary["demand_slots"],
            "ratio demanda/capacidad": round(summary["realized_ratio"], 3),
            "nota": meta.note,
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df, hide_index=True, width="stretch", height=ui.dataframe_height(len(df)),
        column_config={
            "plazas demandadas": st.column_config.NumberColumn("plazas demandadas", format="%.0f"),
            "ratio demanda/capacidad": st.column_config.NumberColumn(
                "ratio demanda/capacidad", format="%.3f",
                help="Horas de demanda entre horas contratadas disponibles en el horizonte."),
        },
    )

    c1, c2 = st.columns(2)
    with c1:
        target = st.selectbox("Exportar instancia", [m.name for m in instances],
                              key="ini_exp_inst")
        st.download_button("Descargar JSON", data=store.instance_json(target),
                           file_name=f"{target}.json", mime="application/json")
    with c2:
        victim = st.selectbox("Borrar instancia", [m.name for m in instances],
                              key="ini_del_inst")
        sure = st.checkbox("Borrar también sus soluciones y confirmar", key="ini_del_inst_ok")
        if st.button("Borrar", key="ini_del_inst_btn", disabled=not sure):
            store.delete_instance(victim)
            st.rerun()

    st.subheader("Importar instancia")
    uploaded = st.file_uploader("JSON en el formato canónico del paquete", type="json")
    if uploaded is not None:
        default_name = Path(uploaded.name).stem
        name = st.text_input("Nombre en el espacio de trabajo", value=default_name,
                             key="ini_imp_name")
        if st.button("Importar", key="ini_imp_btn"):
            try:
                final = store.import_instance(uploaded.getvalue(), name)
                st.success(f"Instancia importada como **{final}**.")
                st.rerun()
            except Exception as exc:
                ui.show_validation_error(exc)

with tab_sol:
    if solutions:
        rows = [{
            "nombre": r.name,
            "instancia": r.instance_name,
            "método": SOLVER_LABELS.get(r.solver, r.solver),
            "Z_modelo": r.metrics.get("z_modelo"),
            "factible": bool(r.metrics.get("feasible")),
            "creada": r.created[:16].replace("T", " "),
        } for r in solutions]
        df = pd.DataFrame(rows)
        st.dataframe(
            df, hide_index=True, width="stretch", height=ui.dataframe_height(len(df)),
            column_config={
                "Z_modelo": st.column_config.NumberColumn(
                    format="%.4f",
                    help="Valor de la función objetivo del modelo (Cap. 5): menor es mejor."),
            },
        )

        c1, c2 = st.columns(2)
        with c1:
            target = st.selectbox("Exportar cuadrante", [r.name for r in solutions],
                                  key="ini_exp_sol")
            st.download_button("Descargar JSON", data=store.solution_json(target),
                               file_name=f"{target}.json", mime="application/json")
        with c2:
            victim = st.selectbox("Borrar cuadrante", [r.name for r in solutions],
                                  key="ini_del_sol")
            if st.button("Borrar", key="ini_del_sol_btn"):
                store.delete_solution(victim)
                st.rerun()
    else:
        st.caption("Aún no hay cuadrantes guardados.")
