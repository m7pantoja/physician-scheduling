"""Resolver una instancia con los tres métodos del TFG y guardar cada resultado en el workspace."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st

from rostering.sa import SAConfig

from core import charts, services, store, ui

ui.page_header(
    "Resolver",
    "Resuelve la instancia seleccionada con <em>greedy</em>, <em>simulated annealing</em> "
    "y MILP exacto bajo la misma política de objetivo, y guarda cada solución en el "
    "espacio de trabajo.",
    kicker="RESOLUCIÓN",
)

sel = ui.pick_instance("sol_sel")
if sel is None:
    st.stop()
inst_name, instance = sel

summary = services.summarize_instance(instance)
st.caption(
    f"{summary['n']} médicos · {summary['days']} días · "
    f"{summary['demand_slots']} plazas demandadas · "
    f"ratio demanda/capacidad {ui.fmt(summary['realized_ratio'], 3)}"
)

st.subheader("Política del objetivo")
config = ui.objective_config_widget("sol_obj")
st.caption(
    "Los pesos afectan al coste marginal del greedy, al objetivo aumentado del SA y al "
    "objetivo del MILP; la evaluación posterior puede re-puntuar con otra política."
)


def _metrics_row(rec: store.SolutionRecord) -> None:
    """Fila común tras resolver: Z_modelo, factibilidad, déficit de cobertura y tiempo."""
    cols = st.columns(4)
    cols[0].metric("Z_modelo", ui.fmt(rec.metrics.get("z_modelo")), border=True)
    ui.feasibility_metric(cols[1], bool(rec.metrics.get("feasible")), rec.metrics.get("violaciones"))
    cols[2].metric("Déficit de cobertura (p_cob)", ui.fmt(rec.metrics.get("p_cob"), 0), border=True)
    cols[3].metric("Tiempo (s)", ui.fmt(rec.metrics.get("elapsed"), 2), border=True)


tab_greedy, tab_sa, tab_milp, tab_3m = st.tabs(
    ["Greedy", "Simulated annealing", "MILP exacto", "Los tres métodos"])

# ---------------------------------------------------------------------------
# Greedy
# ---------------------------------------------------------------------------

with tab_greedy:
    st.caption(
        "Heurística constructiva por criticidad (Alg. 6.1): asigna en orden decreciente de "
        "escasez de cobertura; es el suelo de comparación y la solución inicial habitual "
        "del SA."
    )
    sol_g_name = st.text_input("Nombre de la solución", value=f"{inst_name}-greedy",
                               key=f"sol_g_name_{inst_name}")
    if st.button("Resolver", key="sol_g_btn", type="primary"):
        with st.spinner("Resolviendo con la heurística constructiva..."):
            result = services.run_greedy(instance, config)
        rec = services.make_solution_record(
            name=sol_g_name, instance_name=inst_name, solver="greedy",
            instance=instance, roster=result["roster"], config=config,
            solver_params={}, extra_metrics={"elapsed": result["elapsed"]},
        )
        saved = store.save_solution(rec)
        _metrics_row(rec)
        st.success(f"{ui.solver_badge_md('greedy')} guardada como **{saved}** — revísala en "
                  "**Cuadrante** o **Evaluación**.")

# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------

with tab_sa:
    st.caption(
        "*Simulated annealing* sobre el objetivo aumentado (Alg. 6.2): vecindario flip+swap "
        "con aceptación de Metrópolis, la temperatura anela los términos soft y congela toda "
        "violación dura (Sección 6.3.4)."
    )

    init_choice = st.radio("Solución inicial", ["greedy (se recalcula)", "una solución guardada"],
                           key="sol_sa_init")
    x0 = None
    init_solution_name = None
    if init_choice == "una solución guardada":
        rec0 = ui.pick_solution("sol_sa_init_sol", instance_name=inst_name,
                                label="Solución de partida")
        if rec0 is not None:
            x0 = rec0.roster()
            init_solution_name = rec0.name

    scheme_mode = st.radio("Modo del esquema de enfriamiento",
                           ["automático (schedule)", "manual"], key="sol_sa_scheme")

    if scheme_mode == "automático (schedule)":
        c1, c2 = st.columns(2)
        rounds = c1.number_input("rounds (pasadas sobre las ternas vivas)", 50, 2000, 400, 10,
                                 key="sol_sa_rounds")
        base_seed = c2.number_input("semilla", 0, 10_000, 0, 1, key="sol_sa_seed_auto")

        def _build_cfg(seed_value: int) -> SAConfig:
            return services.auto_sa_config(instance, seed=seed_value, rounds=int(rounds))
    else:
        with st.expander("Parámetros manuales del esquema", expanded=True):
            c1, c2, c3 = st.columns(3)
            t_0 = c1.number_input("T_0 (temperatura inicial)", 0.01, 10000.0, 100.0, 1.0,
                                  key="sol_sa_t0")
            alpha = c2.number_input("alpha (enfriamiento geométrico)", 0.80, 0.999, 0.95, 0.001,
                                    key="sol_sa_alpha", format="%.3f")
            chain_length = c3.number_input("L (pasos por temperatura)", 1, 5000, 200, 10,
                                           key="sol_sa_chain")
            c4, c5, c6 = st.columns(3)
            t_min = c4.number_input("T_min (parada)", 0.0001, 10.0, 0.01, 0.001,
                                    key="sol_sa_tmin", format="%.4f")
            max_iterations = c5.number_input("max_iterations (parada)", 1000, 2_000_000, 200_000,
                                             1000, key="sol_sa_maxit")
            stagnation_limit = c6.number_input("stagnation_limit (parada)", 100, 500_000, 20_000,
                                               100, key="sol_sa_stag")
            c7, c8 = st.columns(2)
            p_swap = c7.number_input("p_swap (prob. de SWAP frente a FLIP)", 0.0, 1.0, 0.5, 0.05,
                                     key="sol_sa_pswap")
            base_seed = c8.number_input("semilla", 0, 10_000, 0, 1, key="sol_sa_seed_manual")

        def _build_cfg(seed_value: int) -> SAConfig:
            return SAConfig(t_0=t_0, alpha=alpha, chain_length=int(chain_length), t_min=t_min,
                            max_iterations=int(max_iterations),
                            stagnation_limit=int(stagnation_limit), p_swap=p_swap,
                            seed=seed_value)

    k = st.number_input("Réplicas (k)", 1, 20, 1, 1, key="sol_sa_k")
    sol_sa_name = st.text_input("Nombre de la solución", value=f"{inst_name}-sa",
                                key=f"sol_sa_name_{inst_name}")
    budget_preview = _build_cfg(int(base_seed)).max_iterations
    st.caption(
        f"Presupuesto de la búsqueda: {budget_preview:,} iteraciones por réplica".replace(",", " ")
        + " (crece con el tamaño de la instancia y con rounds). Orientación: unas 200 000 "
        "iteraciones son ~5 s en un portátil y varias veces más en un despliegue en la nube."
    )

    if st.button("Resolver", key="sol_sa_btn", type="primary"):
        with st.spinner("Recociendo..."):
            if int(k) == 1:
                sa_cfg = _build_cfg(int(base_seed))
                result = services.run_sa(instance, config, x0=x0, sa_config=sa_cfg)
                solver_params = {**sa_cfg.model_dump(), "modo_inicial": init_choice,
                                 "solucion_inicial": init_solution_name}
                rec = services.make_solution_record(
                    name=sol_sa_name, instance_name=inst_name, solver="sa",
                    instance=instance, roster=result["roster"], config=config,
                    solver_params=solver_params,
                    extra_metrics={
                        "stop_reason": result["result"].stop_reason,
                        "iterations": result["result"].iterations,
                        "accepted": result["result"].accepted,
                        "elapsed": result["elapsed"],
                        "t_greedy": result["t_greedy"],
                        "seed": int(base_seed),
                    },
                )
                saved = store.save_solution(rec)
            else:
                rows = []
                replicas = []
                progress = st.progress(0.0)
                for idx, s in enumerate(range(int(base_seed), int(base_seed) + int(k))):
                    sa_cfg = _build_cfg(s)
                    result = services.run_sa(instance, config, x0=x0, sa_config=sa_cfg)
                    report = services.evaluation_report(instance, result["roster"], config)
                    rows.append({
                        "semilla": s, "Z_modelo": report["z_modelo"],
                        "parada": result["result"].stop_reason,
                        "iteraciones": result["result"].iterations,
                        "t (s)": result["elapsed"],
                    })
                    replicas.append((s, sa_cfg, result, report["z_modelo"],
                                     report["feasible"]))
                    progress.progress((idx + 1) / int(k))

                df = pd.DataFrame(rows)
                st.dataframe(
                    df, width="stretch", height=ui.dataframe_height(len(df)), hide_index=True,
                    column_config={
                        "Z_modelo": st.column_config.NumberColumn(
                            format="%.4f",
                            help="Valor de la función objetivo del modelo (Cap. 6): menor es mejor."),
                        "iteraciones": st.column_config.NumberColumn(format="%.0f"),
                        "t (s)": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
                fig = charts.box_by_group({"SA": df["Z_modelo"].tolist()}, y_title="Z_modelo")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption("Cada punto es una réplica (semilla distinta) y la caja resume "
                          "su distribución de Z_modelo.")
                st.caption(
                    f"media {ui.fmt(df['Z_modelo'].mean())} · "
                    f"desviación {ui.fmt(df['Z_modelo'].std())} sobre {int(k)} réplicas."
                )

                # la ganadora: primero factibilidad, luego Z_modelo (una infactible con buen
                # perfil soft no debe ganar a una factible)
                winner_seed, winner_cfg, winner_result, _, _ = min(
                    replicas, key=lambda r: (not r[4], r[3]))
                sol_sa_name = f"{sol_sa_name}-seed{winner_seed}"
                solver_params = {**winner_cfg.model_dump(), "modo_inicial": init_choice,
                                 "solucion_inicial": init_solution_name}
                rec = services.make_solution_record(
                    name=sol_sa_name, instance_name=inst_name, solver="sa",
                    instance=instance, roster=winner_result["roster"], config=config,
                    solver_params=solver_params,
                    extra_metrics={
                        "stop_reason": winner_result["result"].stop_reason,
                        "iterations": winner_result["result"].iterations,
                        "accepted": winner_result["result"].accepted,
                        "elapsed": winner_result["elapsed"],
                        "t_greedy": winner_result["t_greedy"],
                        "seed": winner_seed,
                    },
                )
                saved = store.save_solution(rec)

        _metrics_row(rec)
        st.success(f"{ui.solver_badge_md('sa')} guardada como **{saved}** — revísala en "
                  "**Cuadrante** o **Evaluación**.")

# ---------------------------------------------------------------------------
# MILP exacto
# ---------------------------------------------------------------------------

with tab_milp:
    st.caption(
        "Referencia exacta de la formulación de modelo del Cap. 5 (PuLP): certifica el "
        "óptimo cuando el límite de tiempo lo permite."
    )
    engines = services.milp_engines()
    gurobi_available = engines.get("gurobi", False)
    engine_choice = st.radio("Motor del solver", ["HiGHS (libre)", "Gurobi"],
                             key="sol_m_engine", disabled=not gurobi_available)
    if not gurobi_available:
        st.caption("Gurobi no está disponible en este entorno: se resuelve con HiGHS.")

    time_limit = st.number_input("Límite de tiempo (s)", 5, 3600, 60, 5, key="sol_m_time")
    coverage_hard = st.checkbox("Cobertura dura", value=True, key="sol_m_cov")
    st.caption(
        "Si la cobertura dura resulta infactible, se repite automáticamente en régimen "
        "blando (déficit penalizado), como en el Cap. 7."
    )
    sol_m_name = st.text_input("Nombre de la solución", value=f"{inst_name}-milp",
                               key=f"sol_m_name_{inst_name}")
    st.warning(
        "El solver puede agotar el límite de tiempo sin certificar el óptimo; en ese caso "
        "devuelve la mejor solución encontrada junto con su gap."
    )

    if st.button("Resolver", key="sol_m_btn", type="primary"):
        engine = "gurobi" if engine_choice == "Gurobi" else "highs"
        with st.spinner("Resolviendo el MILP exacto..."):
            result = services.run_milp(instance, config, time_limit=int(time_limit),
                                       coverage_hard=coverage_hard, engine=engine)
        info = result["info"]
        engine = result["engine_used"]
        if engine != "gurobi" and engine_choice == "Gurobi":
            st.warning("Gurobi rechazó el modelo por el límite de tamaño de su licencia; "
                       "se resolvió con HiGHS.")
        solver_params = {"engine": engine, "time_limit": int(time_limit),
                         "coverage_hard": coverage_hard}
        rec = services.make_solution_record(
            name=sol_m_name, instance_name=inst_name, solver="milp",
            instance=instance, roster=result["roster"], config=config,
            solver_params=solver_params,
            extra_metrics={
                "proven": info["proven_optimal"],
                "gap": info["gap"],
                "dual_bound": info["dual_bound"],
                "coverage_mode": result["coverage_mode"],
                "n_vars": info["n_vars"],
                "n_cons": info["n_cons"],
                "engine": engine,
                "elapsed": result["elapsed"],
                "status": info["status"],
            },
        )
        saved = store.save_solution(rec)
        _metrics_row(rec)
        cols = st.columns(6)
        cols[0].metric("Certificado", "sí" if info["proven_optimal"] else "no", border=True)
        cols[1].metric("Gap del solver", ui.fmt_pct(info["gap"]), border=True)
        cols[2].metric("Cota dual", ui.fmt(info["dual_bound"]), border=True)
        cols[3].metric("Cobertura", services.COVERAGE_LABELS.get(
            result["coverage_mode"], result["coverage_mode"]), border=True)
        cols[4].metric("n_vars", info["n_vars"], border=True)
        cols[5].metric("n_cons", info["n_cons"], border=True)
        st.success(f"{ui.solver_badge_md('milp')} guardada como **{saved}** — revísala en "
                  "**Cuadrante** o **Evaluación**.")

# ---------------------------------------------------------------------------
# Los tres métodos
# ---------------------------------------------------------------------------

with tab_3m:
    st.caption(
        "Reproduce una celda de los experimentos del Cap. 7 sobre esta instancia: greedy "
        "como suelo de comparación, SA desde el greedy con esquema automático y, si se "
        "activa, el MILP exacto como referencia certificada."
    )

    sol_3m_name = st.text_input("Nombre base", value=f"{inst_name}-3m",
                                key=f"sol_3m_name_{inst_name}")
    c1, c2 = st.columns(2)
    rounds_3m = c1.number_input("rounds del SA (esquema automático)", 50, 2000, 400, 10,
                                key="sol_3m_rounds")
    seed_3m = c2.number_input("semilla", 0, 10_000, 0, 1, key="sol_3m_seed")

    run_milp_3m = st.checkbox("Incluir MILP exacto", value=True, key="sol_3m_milp")
    c3, c4 = st.columns(2)
    time_limit_3m = c3.number_input("Límite de tiempo del MILP (s)", 5, 3600, 60, 5,
                                    key="sol_3m_time", disabled=not run_milp_3m)
    engines_3m = services.milp_engines()
    gurobi_available_3m = engines_3m.get("gurobi", False)
    engine_choice_3m = c4.radio("Motor del solver", ["HiGHS (libre)", "Gurobi"],
                                key="sol_3m_engine",
                                disabled=not run_milp_3m or not gurobi_available_3m)
    if run_milp_3m and not gurobi_available_3m:
        st.caption("Gurobi no está disponible en este entorno: se resuelve con HiGHS.")

    if st.button("Ejecutar los tres métodos", key="sol_3m_btn", type="primary"):
        records: list[store.SolutionRecord] = []
        with st.status("Resolviendo con los tres métodos...", expanded=True) as status:
            status.write("Greedy: heurística constructiva por criticidad...")
            g_result = services.run_greedy(instance, config)
            g_rec = services.make_solution_record(
                name=f"{sol_3m_name}-greedy", instance_name=inst_name, solver="greedy",
                instance=instance, roster=g_result["roster"], config=config,
                solver_params={}, extra_metrics={"elapsed": g_result["elapsed"]},
            )
            g_saved = store.save_solution(g_rec)
            records.append(g_rec)
            status.write(f"{ui.solver_badge_md('greedy')} guardado como **{g_saved}**.")

            status.write("Simulated annealing: recociendo desde el greedy...")
            sa_cfg_3m = services.auto_sa_config(instance, seed=int(seed_3m),
                                                rounds=int(rounds_3m))
            sa_result = services.run_sa(instance, config, x0=g_result["roster"],
                                        sa_config=sa_cfg_3m)
            sa_solver_params = {
                **sa_cfg_3m.model_dump(),
                "modo_inicial": "greedy (Los tres métodos)",
                "solucion_inicial": g_rec.name,
            }
            sa_rec = services.make_solution_record(
                name=f"{sol_3m_name}-sa", instance_name=inst_name, solver="sa",
                instance=instance, roster=sa_result["roster"], config=config,
                solver_params=sa_solver_params,
                extra_metrics={
                    "stop_reason": sa_result["result"].stop_reason,
                    "iterations": sa_result["result"].iterations,
                    "accepted": sa_result["result"].accepted,
                    "elapsed": sa_result["elapsed"],
                    "t_greedy": sa_result["t_greedy"],
                    "seed": int(seed_3m),
                },
            )
            sa_saved = store.save_solution(sa_rec)
            records.append(sa_rec)
            status.write(f"{ui.solver_badge_md('sa')} guardado como **{sa_saved}**.")

            if run_milp_3m:
                status.write("MILP exacto: resolviendo...")
                engine_3m = "gurobi" if engine_choice_3m == "Gurobi" else "highs"
                milp_result = services.run_milp(instance, config,
                                                time_limit=int(time_limit_3m),
                                                coverage_hard=True, engine=engine_3m)
                info_3m = milp_result["info"]
                if milp_result["engine_used"] != engine_3m:
                    status.write("Gurobi rechazó el modelo por el límite de tamaño de su "
                                 "licencia; se resolvió con HiGHS.")
                engine_3m = milp_result["engine_used"]
                milp_solver_params = {"engine": engine_3m, "time_limit": int(time_limit_3m),
                                      "coverage_hard": True}
                milp_rec = services.make_solution_record(
                    name=f"{sol_3m_name}-milp", instance_name=inst_name, solver="milp",
                    instance=instance, roster=milp_result["roster"], config=config,
                    solver_params=milp_solver_params,
                    extra_metrics={
                        "proven": info_3m["proven_optimal"],
                        "gap": info_3m["gap"],
                        "dual_bound": info_3m["dual_bound"],
                        "coverage_mode": milp_result["coverage_mode"],
                        "n_vars": info_3m["n_vars"],
                        "n_cons": info_3m["n_cons"],
                        "engine": engine_3m,
                        "elapsed": milp_result["elapsed"],
                        "status": info_3m["status"],
                    },
                )
                milp_saved = store.save_solution(milp_rec)
                records.append(milp_rec)
                status.write(f"{ui.solver_badge_md('milp')} guardado como **{milp_saved}**.")

            status.update(label="Los tres métodos resueltos.", state="complete")

        summary_rows = []
        for rec in records:
            if rec.solver == "sa":
                diag = f"parada {rec.metrics.get('stop_reason', '—')}"
            elif rec.solver == "milp":
                diag = ("certificado" if rec.metrics.get("proven")
                        else f"gap {ui.fmt_pct(rec.metrics.get('gap'))}")
            else:
                diag = "suelo de comparación"
            summary_rows.append({
                "método": services.SOLVER_LABELS.get(rec.solver, rec.solver),
                "Z_modelo": rec.metrics.get("z_modelo"),
                "factible": bool(rec.metrics.get("feasible")),
                "déficit p_cob": rec.metrics.get("p_cob"),
                "tiempo (s)": rec.metrics.get("elapsed"),
                "diagnóstico": diag,
            })
        df_3m = pd.DataFrame(summary_rows)
        st.dataframe(
            df_3m, hide_index=True, width="stretch", height=ui.dataframe_height(len(df_3m)),
            column_config={
                "Z_modelo": st.column_config.NumberColumn(
                    format="%.4f",
                    help="Valor de la función objetivo del modelo (Cap. 6): menor es mejor."),
                "factible": st.column_config.CheckboxColumn("factible"),
                "déficit p_cob": st.column_config.NumberColumn(
                    format="%.0f", help="Plazas de demanda sin cubrir bajo esta solución."),
                "tiempo (s)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption("Compara las tres en la página Comparador.")
