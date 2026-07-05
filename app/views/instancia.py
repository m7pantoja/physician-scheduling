"""Página Instancia: inspección y edición segura de una instancia del espacio de trabajo."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st

from core import charts, services, store, ui

ui.page_header(
    "Instancia",
    "Inspección y edición validada de una instancia del espacio de trabajo: plantilla, "
    "demanda, reglas y preferencias, reconstruida y revalidada en cada cambio aplicado.",
    kicker="DATOS",
)

sel = ui.pick_instance("inst_sel")
if sel is None:
    st.stop()
name, disk_instance = sel

work_key = f"inst_work_{name}"
if work_key not in st.session_state:
    st.session_state[work_key] = disk_instance
work = st.session_state[work_key]

meta = store.instance_meta(name)
st.caption(f"{name}  ·  {meta.origin}  ·  creada {meta.created[:16].replace('T', ' ')}")

if work != disk_instance:
    warn_col, btn_col = st.columns([5, 1])
    with warn_col:
        st.warning("Hay cambios sin guardar en la copia de trabajo.")
    with btn_col:
        st.write("")
        if st.button("Descartar cambios", key="inst_discard"):
            st.session_state[work_key] = disk_instance
            # los editores conservan su propio estado por clave: al descartar hay que vaciarlos
            stale = [k for k in st.session_state
                     if k.startswith((f"inst_ed_{name}_", f"inst_reg_{name}_"))]
            for k in stale:
                del st.session_state[k]
            st.rerun()

tab_resumen, tab_plantilla, tab_demanda, tab_reglas, tab_pref, tab_guardar = st.tabs(
    ["Resumen", "Plantilla", "Demanda", "Reglas", "Preferencias", "Guardar"]
)

# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
with tab_resumen:
    summary = services.summarize_instance(work)
    cols = st.columns(5)
    cols[0].metric("Médicos", summary["n"], border=True)
    cols[1].metric("Días", summary["days"], border=True)
    cols[2].metric("Plazas demandadas", summary["demand_slots"], border=True)
    cols[3].metric("Ratio demanda/capacidad", ui.fmt(summary["realized_ratio"], 3), border=True)
    cols[4].metric("Preferencias no nulas", summary["n_preferences"], border=True)

    st.plotly_chart(charts.demand_heatmap(services.demand_matrix(work)),
                    width="stretch", config={"displayModeBar": False})
    st.caption("Cada celda es la demanda b_ts del turno en ese día; el fondo gris indica "
              "que el turno no se ofrece esa fecha (η_ts=0).")

    st.subheader("Catálogo de turnos")
    shifts_df = pd.DataFrame([{
        "nombre": s.name,
        "inicio": s.start.strftime("%H:%M"),
        "fin": s.end.strftime("%H:%M"),
        "horas": s.hours,
        "guardia": s.is_guardia,
    } for s in work.shifts])
    st.dataframe(
        shifts_df, width="stretch", height=ui.dataframe_height(len(shifts_df)), hide_index=True,
        column_config={"horas": st.column_config.NumberColumn("horas", format="%.1f")},
    )

    st.subheader("Clases")
    classes_df = pd.DataFrame([{
        "nombre": c.name,
        "residente": c.is_resident,
        "año": str(c.residency_year) if c.residency_year is not None else "—",
        "jornada semanal": c.weekly_hours,
        "g^max por defecto": str(c.default_max_guardias) if c.default_max_guardias is not None else "—",
    } for c in work.classes])
    st.dataframe(
        classes_df, width="stretch", height=ui.dataframe_height(len(classes_df)), hide_index=True,
        column_config={
            "jornada semanal": st.column_config.NumberColumn(
                "jornada semanal", format="%.1f",
                help="Horas semanales de jornada completa para la clase."),
            "g^max por defecto": st.column_config.TextColumn(
                "g^max por defecto",
                help="Tope de guardias heredado por los médicos de la clase sin tope individual."),
        },
    )

    st.caption(
        f"Régimen legal: descanso mínimo entre jornadas {work.regulations.min_rest_hours:g} h  ·  "
        f"descanso semanal ininterrumpido {work.regulations.min_weekly_rest_hours:g} h  ·  "
        f"techo semanal de media {work.regulations.weekly_hours_ceiling:g} h"
    )

    overlap = work.overlapping_pairs()
    short_rest = work.short_rest_pairs()
    overlap_items = sorted(" + ".join(sorted(pair)) for pair in overlap)
    short_rest_items = [f"{a} → {b}" for a, b in sorted(short_rest)]

    def _badge_line(items: list[str], color: str, max_badges: int = 20) -> str:
        """Directivas de badge en markdown para hasta `max_badges` elementos; el resto se
        resume en un recuento."""
        if not items:
            return "ninguno"
        shown = " ".join(f":{color}-badge[{it}]" for it in items[:max_badges])
        if len(items) > max_badges:
            shown += f"  ... y {len(items) - max_badges} más"
        return shown

    st.markdown(
        "Conjunto O (solapes intradía prohibidos): " + _badge_line(overlap_items, "gray")
    )
    st.markdown(
        "Conjunto I (encadenamientos día → día+1 con descanso insuficiente): "
        + _badge_line(short_rest_items, "orange")
    )

# ---------------------------------------------------------------------------
# Plantilla
# ---------------------------------------------------------------------------
with tab_plantilla:
    st.caption("Una fila por médico. Las filas nuevas sin id reciben el primero libre; "
              "max_guardias vacío hereda el tope de la clase.")
    edited_phys = st.data_editor(
        services.physicians_frame(work),
        num_rows="dynamic",
        key=f"inst_ed_{name}_phys",
        column_config={
            "id": st.column_config.NumberColumn("id", step=1, min_value=1),
            "clase": st.column_config.SelectboxColumn(
                "clase", options=[c.name for c in work.classes]),
            "part_time": st.column_config.NumberColumn(
                "ρ (jornada)", min_value=0.05, max_value=1.0, step=0.05, format="%.2f",
                help="Fracción de jornada completa; escala el corredor de horas y la cuota de guardias."),
            "max_guardias": st.column_config.NumberColumn(
                "max_guardias", min_value=1, step=1,
                help="Tope individual de guardias; vacío hereda el g^max por defecto de la clase."),
        },
        hide_index=True,
    )
    if st.button("Aplicar plantilla", key="inst_apply_phys"):
        try:
            physicians = services.physicians_from_frame(edited_phys)
            new_work = services.rebuild_instance(
                work, physicians=[p.model_dump() for p in physicians])
        except Exception as exc:
            ui.show_validation_error(exc)
        else:
            st.session_state[work_key] = new_work
            work = new_work
            st.success("Plantilla aplicada.")

# ---------------------------------------------------------------------------
# Demanda
# ---------------------------------------------------------------------------
with tab_demanda:
    st.caption("Demanda b_ts por día y turno; 0 = sin demanda. Los validadores impiden "
              "demanda sobre turnos que no se ofrecen ese día.")
    shift_cols = services.shift_names(work)
    demand_config = {"día": st.column_config.TextColumn("día", disabled=True)}
    demand_config.update({s: st.column_config.NumberColumn(s, min_value=0, step=1)
                          for s in shift_cols})
    edited_demand = st.data_editor(
        services.demand_frame(work),
        key=f"inst_ed_{name}_dem",
        column_config=demand_config,
        hide_index=True,
    )
    if st.button("Aplicar demanda", key="inst_apply_dem"):
        try:
            new_demand = services.demand_from_frame(edited_demand, work)
            new_work = services.rebuild_instance(work, demand=new_demand)
        except Exception as exc:
            ui.show_validation_error(exc)
        else:
            st.session_state[work_key] = new_work
            work = new_work
            st.success("Demanda aplicada.")

# ---------------------------------------------------------------------------
# Reglas
# ---------------------------------------------------------------------------
with tab_reglas:
    st.subheader("Cualificación γ_rs")
    edited_qual = st.data_editor(
        services.qualification_frame(work),
        key=f"inst_ed_{name}_qual",
    )
    if st.button("Aplicar cualificación", key="inst_apply_qual"):
        try:
            new_qual = services.qualification_from_frame(edited_qual)
            new_work = services.rebuild_instance(work, qualification=new_qual)
        except Exception as exc:
            ui.show_validation_error(exc)
        else:
            st.session_state[work_key] = new_work
            work = new_work
            st.success("Cualificación aplicada.")

    st.divider()
    st.subheader("Elegibilidad η_ts")
    st.caption("Quitar la oferta de un turno con demanda registrada hará fallar la "
              "validación de coherencia η–b.")
    eli_config = {"día": st.column_config.TextColumn("día", disabled=True)}
    edited_eli = st.data_editor(
        services.eligibility_frame(work),
        key=f"inst_ed_{name}_eli",
        column_config=eli_config,
        hide_index=True,
    )
    if st.button("Aplicar elegibilidad", key="inst_apply_eli"):
        try:
            new_eli = services.eligibility_from_frame(edited_eli, work)
            new_work = services.rebuild_instance(work, eligibility=new_eli)
        except Exception as exc:
            ui.show_validation_error(exc)
        else:
            st.session_state[work_key] = new_work
            work = new_work
            st.success("Elegibilidad aplicada.")

    st.divider()
    st.subheader("Régimen legal")
    r1, r2, r3 = st.columns(3)
    min_rest = r1.number_input("Descanso mínimo entre jornadas (h)", 0.1, 48.0,
                               float(work.regulations.min_rest_hours), 0.5,
                               key=f"inst_reg_{name}_rest")
    min_weekly_rest = r2.number_input("Descanso semanal ininterrumpido (h)", 0.1, 72.0,
                                      float(work.regulations.min_weekly_rest_hours), 0.5,
                                      key=f"inst_reg_{name}_weekly")
    ceiling = r3.number_input("Techo semanal de media (h)", 0.1, 100.0,
                              float(work.regulations.weekly_hours_ceiling), 0.5,
                              key=f"inst_reg_{name}_ceiling")
    if st.button("Aplicar régimen legal", key="inst_apply_reg"):
        try:
            new_work = services.rebuild_instance(work, regulations={
                "min_rest_hours": min_rest,
                "min_weekly_rest_hours": min_weekly_rest,
                "weekly_hours_ceiling": ceiling,
            })
        except Exception as exc:
            ui.show_validation_error(exc)
        else:
            st.session_state[work_key] = new_work
            work = new_work
            st.success("Régimen legal aplicado.")

# ---------------------------------------------------------------------------
# Preferencias
# ---------------------------------------------------------------------------
with tab_pref:
    by_id = {p.id: services.physician_label(work, p)
            for p in sorted(work.physicians, key=lambda p: p.id)}
    pid = st.selectbox("Médico", list(by_id), key="inst_pref_who",
                       format_func=lambda i: by_id[i])
    st.caption("Preferencias p_its: >0 desea el turno, <0 lo rechaza, 0 indiferente. "
              "Se maximiza la suma ponderada por w_pref.")
    pref_config = {"día": st.column_config.TextColumn("día", disabled=True)}
    pref_config.update({s: st.column_config.NumberColumn(s, step=0.5)
                        for s in services.shift_names(work)})
    edited_pref = st.data_editor(
        services.preferences_frame(work, pid),
        key=f"inst_ed_{name}_pref_{pid}",
        column_config=pref_config,
        hide_index=True,
    )
    if st.button("Aplicar preferencias", key="inst_apply_pref"):
        try:
            new_prefs = services.preferences_with_update(work, pid, edited_pref)
            new_work = services.rebuild_instance(work, preferences=new_prefs)
        except Exception as exc:
            ui.show_validation_error(exc)
        else:
            st.session_state[work_key] = new_work
            work = new_work
            st.success("Preferencias aplicadas.")

# ---------------------------------------------------------------------------
# Guardar
# ---------------------------------------------------------------------------
with tab_guardar:
    st.caption("Sobrescribir no toca las soluciones ya guardadas de la versión anterior: "
              "siguen siendo registros históricos válidos, pero evaluados contra la "
              "instancia antigua.")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Sobrescribir {name}**")
        if st.button(f"Sobrescribir {name}", key="inst_save_overwrite", type="primary"):
            store.save_instance(work, name, origin="editada", overwrite=True)
            st.session_state[work_key] = work
            st.success(f"Instancia **{name}** actualizada.")
            st.rerun()
    with c2:
        st.write("**Guardar como**")
        new_name = st.text_input("Nombre nuevo", key="inst_save_as_name")
        if st.button("Guardar como", key="inst_save_as_btn", disabled=not new_name.strip()):
            final = store.save_instance(work, new_name, origin="editada")
            st.session_state["inst_sel"] = final
            st.session_state[f"inst_work_{final}"] = work
            st.success(f"Instancia guardada como **{final}**.")
            st.rerun()
