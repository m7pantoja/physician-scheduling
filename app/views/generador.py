"""Página Generador: instancias nuevas guardadas en el workspace, sintéticas o manuales."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from datetime import date, timedelta

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from rostering.domain import Calendar, Regulations
from rostering.generator import Settings

from core import charts, services, store, ui
from core.services import WEEKDAYS

ui.page_header(
    "Generador de instancias",
    "Dos modos de construcción: generación sintética (plantilla muestreada y demanda "
    "por patrón, reproducible por semilla) o construcción manual determinista a partir "
    "de recuentos de clase.",
    kicker="DATOS",
)

_DEFAULT_SETTINGS = Settings()   # caso base andaluz: fuente de los defaults de ambas pestañas


def _preview_and_save(instance, settings_dump: dict, *, origin: str, save_key: str,
                      default_name: str) -> None:
    """Vista previa (métricas + heatmap si hay demanda) y guardado en el workspace, común
    a los resultados de las dos pestañas."""
    st.divider()
    st.subheader("Vista previa")
    summary = services.summarize_instance(instance)
    has_demand = bool(instance.demand)

    m = st.columns(6)
    m[0].metric("Médicos", summary["n"], border=True)
    m[1].metric("Días", summary["days"], border=True)
    m[2].metric("Plazas demandadas", summary["demand_slots"] if has_demand else "—", border=True)
    m[3].metric("Horas demandadas",
               ui.fmt(summary["demand_hours"], 2) if has_demand else "—", border=True)
    m[4].metric("Capacidad contratada (h)", ui.fmt(summary["contracted_hours"], 2), border=True)
    m[5].metric("Ratio demanda/capacidad",
               ui.fmt(summary["realized_ratio"], 3) if has_demand else "—", border=True)

    st.markdown("**Composición de la plantilla por clase médica**")
    comp_df = pd.DataFrame([summary["class_counts"]])
    st.dataframe(comp_df, width="stretch", height=ui.dataframe_height(1), hide_index=True)

    if has_demand:
        fig = charts.demand_heatmap(services.demand_matrix(instance))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Cada celda es la demanda b_ts del turno (fila) en el día (columna); "
                   "gris = turno no ofrecido ese día (η=0).")
    else:
        st.info("Instancia sin demanda: introdúcela en la página **Instancia**.")

    st.subheader("Guardar en el espacio de trabajo")
    name = st.text_input("Nombre", value=default_name, key=f"{save_key}_name")
    note = st.text_input("Nota (opcional)", value="", key=f"{save_key}_note")
    if st.button("Guardar en el espacio de trabajo", key=f"{save_key}_go"):
        final = store.save_instance(instance, name, origin=origin, note=note, settings=settings_dump)
        st.success(f"Instancia guardada como **{final}**.")
        st.caption("Continúa en la página **Instancia** para editarla, o en **Resolver** "
                   "para obtener su primera solución.")


tab_synth, tab_manual = st.tabs(["Sintética", "Manual"])

# ---------------------------------------------------------------------------
# Sintética
# ---------------------------------------------------------------------------
with tab_synth:
    st.subheader("Plantilla")
    c1, c2 = st.columns(2)
    n_physicians = c1.number_input("Número de médicos (n)", 1, 60, 12, 1, key="gen_n")
    fea_ratio = c2.slider("Fracción de adjuntos (fea_ratio)", 0.0, 1.0, 0.6, 0.05, key="gen_fea")

    st.caption("Pirámide MIR: pesos relativos por año de residencia (se normalizan).")
    pyramid_defaults = [5.0, 4.0, 3.0, 2.0, 1.0]
    pyramid_cols = st.columns(5)
    pyramid = [
        pyramid_cols[k].number_input(f"R{k + 1}", 0.0, value=pyramid_defaults[k], step=0.5,
                                     key=f"gen_r{k + 1}")
        for k in range(5)
    ]

    st.subheader("Calendario")
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Fecha de inicio", value=date(2026, 1, 5), key="gen_start")
    days = c2.number_input("Días del horizonte", 7, 112, 28, 7, key="gen_days")

    horizon_dates = [start_date + timedelta(days=k) for k in range(int(days))]
    holiday_options = [dd.isoformat() for dd in horizon_dates]
    holiday_key = f"gen_holidays_{start_date.isoformat()}_{int(days)}"
    holiday_choices = st.multiselect("Festivos", holiday_options, default=[], key=holiday_key)
    holidays = [date.fromisoformat(s) for s in holiday_choices]

    st.subheader("Demanda")
    c1, c2, c3 = st.columns(3)
    coverage_ratio = c1.slider("coverage_ratio (demanda-horas / capacidad)", 0.50, 2.00,
                               1.10, 0.05, key="gen_cov")
    c1.caption("Menor que 1: infra-dotación deliberada.")
    demand_noise = c2.slider("demand_noise (jitter relativo por celda)", 0.0, 0.5,
                             0.10, 0.05, key="gen_noise")
    holiday_weight = c3.number_input("holiday_weight", 0.0, 10.0, 1.0, 0.5, key="gen_holw")

    with st.expander("Forma semanal de la demanda"):
        weekday_defaults = [5.0, 5.0, 5.0, 5.0, 5.0, 2.0, 2.0]
        weekday_cols = st.columns(7)
        weekday_weights = [
            weekday_cols[k].number_input(WEEKDAYS[k], 0.0, value=weekday_defaults[k], step=0.5,
                                         key=f"gen_wd{k}")
            for k in range(7)
        ]

    seed = st.number_input("Semilla", min_value=0, value=0, step=1, key="gen_seed")

    st.caption(
        "El generador no muestrea part_time, exenciones ni preferencias: el estudio "
        "experimental del TFG (Cap. 7) fija una plantilla homogénea a jornada completa "
        "para aislar los factores del barrido, y su muestreo queda delegado como línea "
        "futura (§8.2). El modelo sí los admite: esos datos se editan después en la "
        "página **Instancia**."
    )

    if st.button("Generar", key="gen_go", type="primary"):
        try:
            settings = Settings(
                n_physicians=int(n_physicians),
                fea_ratio=float(fea_ratio),
                residency_pyramid=[float(x) for x in pyramid],
                calendar=Calendar(start_date=start_date, days=int(days), holidays=holidays),
                coverage_ratio=float(coverage_ratio),
                demand_noise=float(demand_noise),
                holiday_weight=float(holiday_weight),
                weekday_weights=[float(x) for x in weekday_weights],
            )
            instance = services.generate_instance(settings, int(seed))
        except ValidationError as exc:
            ui.show_validation_error(exc)
        else:
            st.session_state["gen_instance"] = instance
            st.session_state["gen_settings_dump"] = settings.model_dump(mode="json")
            st.session_state["gen_seed_used"] = int(seed)

    gen_instance = st.session_state.get("gen_instance")
    if gen_instance is not None:
        gen_summary = services.summarize_instance(gen_instance)
        seed_used = st.session_state.get("gen_seed_used", 0)
        default_name = f"sintetica-n{gen_summary['n']}-d{gen_summary['days']}-s{seed_used}"
        save_key = f"gen_save_{gen_summary['n']}_{gen_summary['days']}_{seed_used}"
        _preview_and_save(gen_instance, st.session_state["gen_settings_dump"], origin="generada",
                          save_key=save_key, default_name=default_name)

# ---------------------------------------------------------------------------
# Manual
# ---------------------------------------------------------------------------
with tab_manual:
    st.caption(
        "La plantilla se define por recuentos de clase, sin muestreo; part-time, "
        "exenciones, g^max y preferencias se ajustan después en la página **Instancia**."
    )

    st.subheader("Plantilla por recuentos")
    class_defaults = {c.name: (7 if not c.is_resident else 1) for c in _DEFAULT_SETTINGS.classes}
    cls_cols = st.columns(len(_DEFAULT_SETTINGS.classes))
    class_counts = {
        c.name: cls_cols[k].number_input(c.name, 0, 40, class_defaults[c.name], 1,
                                         key=f"gen_m_cls_{c.name}")
        for k, c in enumerate(_DEFAULT_SETTINGS.classes)
    }
    st.caption(f"Total: {sum(class_counts.values())} médicos.")

    st.subheader("Calendario")
    c1, c2 = st.columns(2)
    start_date_m = c1.date_input("Fecha de inicio", value=date(2026, 1, 5), key="gen_m_start")
    days_m = c2.number_input("Días del horizonte", 7, 112, 28, 7, key="gen_m_days")

    horizon_dates_m = [start_date_m + timedelta(days=k) for k in range(int(days_m))]
    holiday_options_m = [dd.isoformat() for dd in horizon_dates_m]
    holiday_key_m = f"gen_m_holidays_{start_date_m.isoformat()}_{int(days_m)}"
    holiday_choices_m = st.multiselect("Festivos", holiday_options_m, default=[], key=holiday_key_m)
    holidays_m = [date.fromisoformat(s) for s in holiday_choices_m]

    st.subheader("Catálogo de turnos")
    edited_shifts = st.data_editor(
        services.shift_specs_frame(_DEFAULT_SETTINGS.shifts),
        num_rows="dynamic",
        key="gen_m_shifts",
        width="stretch",
        hide_index=True,
        column_config={
            "nombre": st.column_config.TextColumn("nombre"),
            "inicio": st.column_config.TimeColumn("inicio", format="HH:mm"),
            "fin": st.column_config.TimeColumn("fin", format="HH:mm"),
            "horas": st.column_config.NumberColumn("horas", min_value=0.5, step=0.5),
            "guardia": st.column_config.CheckboxColumn("guardia"),
            "peso": st.column_config.NumberColumn(
                "peso", min_value=0.0, step=0.5,
                help="peso relativo del turno en la demanda por patrón",
            ),
        },
    )
    st.caption(
        "Una guardia de 24 h (inicio = fin) se ofrece por defecto solo en días no "
        "laborables; el resto de turnos, solo en laborables (heurística de "
        "elegibilidad del blueprint)."
    )

    st.subheader("Demanda")
    demand_mode = st.radio(
        "Demanda",
        ["Generar por patrón (misma fórmula del generador)",
         "Dejarla vacía (introducirla a mano en la página Instancia)"],
        key="gen_m_demand_mode",
    )
    generate_demand = demand_mode.startswith("Generar")

    if generate_demand:
        c1, c2, c3 = st.columns(3)
        coverage_ratio_m = c1.slider("coverage_ratio (demanda-horas / capacidad)", 0.50, 2.00,
                                     1.10, 0.05, key="gen_m_cov")
        c1.caption("Menor que 1: infra-dotación deliberada.")
        demand_noise_m = c2.slider("demand_noise (jitter relativo por celda)", 0.0, 0.5,
                                   0.10, 0.05, key="gen_m_noise")
        holiday_weight_m = c3.number_input("holiday_weight", 0.0, 10.0, 1.0, 0.5, key="gen_m_holw")

        with st.expander("Forma semanal de la demanda"):
            weekday_defaults_m = [5.0, 5.0, 5.0, 5.0, 5.0, 2.0, 2.0]
            wd_cols = st.columns(7)
            weekday_weights_m = [
                wd_cols[k].number_input(WEEKDAYS[k], 0.0, value=weekday_defaults_m[k], step=0.5,
                                        key=f"gen_m_wd{k}")
                for k in range(7)
            ]
    else:
        coverage_ratio_m = _DEFAULT_SETTINGS.coverage_ratio
        demand_noise_m = _DEFAULT_SETTINGS.demand_noise
        holiday_weight_m = _DEFAULT_SETTINGS.holiday_weight
        weekday_weights_m = list(_DEFAULT_SETTINGS.weekday_weights)

    seed_m = st.number_input("Semilla", min_value=0, value=0, step=1, key="gen_m_seed")

    with st.expander("Régimen legal"):
        reg = _DEFAULT_SETTINGS.regulations
        c1, c2, c3 = st.columns(3)
        min_rest = c1.number_input("Descanso mínimo entre jornadas (h)", 0.0,
                                   value=float(reg.min_rest_hours), step=1.0, key="gen_m_reg_rest")
        min_weekly_rest = c2.number_input("Descanso semanal ininterrumpido (h)", 0.0,
                                          value=float(reg.min_weekly_rest_hours), step=1.0,
                                          key="gen_m_reg_weekly")
        ceiling = c3.number_input("Techo semanal de jornada de media (h)", 0.0,
                                  value=float(reg.weekly_hours_ceiling), step=1.0,
                                  key="gen_m_reg_ceiling")

    if st.button("Construir instancia", key="gen_m_go", type="primary"):
        try:
            specs = services.shift_specs_from_frame(edited_shifts)
            settings_m = Settings(
                shifts=specs,
                calendar=Calendar(start_date=start_date_m, days=int(days_m), holidays=holidays_m),
                regulations=Regulations(
                    min_rest_hours=float(min_rest),
                    min_weekly_rest_hours=float(min_weekly_rest),
                    weekly_hours_ceiling=float(ceiling),
                ),
                coverage_ratio=float(coverage_ratio_m),
                demand_noise=float(demand_noise_m),
                holiday_weight=float(holiday_weight_m),
                weekday_weights=[float(x) for x in weekday_weights_m],
            )
            instance = services.build_manual_instance(
                settings=settings_m,
                class_counts={name: int(v) for name, v in class_counts.items()},
                seed=int(seed_m),
                generate_demand=generate_demand,
            )
        except ValidationError as exc:
            ui.show_validation_error(exc)
        else:
            st.session_state["gen_m_instance"] = instance
            st.session_state["gen_m_settings_dump"] = settings_m.model_dump(mode="json")

    gen_m_instance = st.session_state.get("gen_m_instance")
    if gen_m_instance is not None:
        gen_m_summary = services.summarize_instance(gen_m_instance)
        default_name_m = f"manual-n{gen_m_summary['n']}-d{gen_m_summary['days']}"
        save_key_m = f"gen_m_save_{gen_m_summary['n']}_{gen_m_summary['days']}"
        _preview_and_save(gen_m_instance, st.session_state["gen_m_settings_dump"], origin="manual",
                          save_key=save_key_m, default_name=default_name_m)
