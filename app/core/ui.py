"""Widgets y utilidades compartidas por todas las páginas.

Concentra los selectores de instancia/solución y el formulario de la política del
objetivo para que todas las páginas se comporten igual: mismos textos, mismas claves de
estado y misma manera de mostrar errores de validación."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from rostering.domain import Instance
from rostering.objective import ObjectiveConfig

from core import store
from core.services import SOLVER_LABELS

_STYLE = """
<style>
/* densidad y tipografía global */
.block-container { padding-top: 2.4rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.01em; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
[data-testid="stMetricLabel"] p { font-size: 0.82rem; color: #52514e; }

/* cabecera de página */
.app-header { border-bottom: 1px solid #e1e0d9; padding-bottom: 0.9rem; margin-bottom: 0.6rem; }
.app-header .app-kicker { color: #a8003b; font-size: 0.92rem; font-weight: 650;
                          letter-spacing: 0.09em; margin: 0 0 0.15rem 0; }
.app-header h1 { margin: 0 0 0.3rem 0; font-size: 1.72rem; font-weight: 650; }
.app-header p.app-sub { margin: 0; color: #52514e; font-size: 0.95rem; max-width: 62rem; }

/* cabeceras de sección de la navegación lateral */
[data-testid="stNavSectionHeader"] { font-size: 0.85rem; }

/* tarjetas de flujo de la portada */
.app-card { border: 1px solid #e1e0d9; border-radius: 0.6rem; padding: 1rem 1.1rem;
            background: #ffffff; height: 100%; }
.app-card .app-step { color: #a8003b; font-size: 0.78rem; font-weight: 650;
                      letter-spacing: 0.09em; margin-bottom: 0.2rem; }
.app-card h3 { margin: 0 0 0.3rem 0; font-size: 1.02rem; font-weight: 620; }
.app-card p { margin: 0; color: #52514e; font-size: 0.86rem; line-height: 1.45; }
</style>
"""


def inject_style() -> None:
    """CSS global de la app; lo aplica el enrutador una vez por ejecución."""
    st.markdown(_STYLE, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", kicker: str = "") -> None:
    """Cabecera homogénea de página: kicker opcional, título y subtítulo."""
    kicker_html = f'<div class="app-kicker">{kicker}</div>' if kicker else ""
    sub_html = f'<p class="app-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f'<div class="app-header">{kicker_html}<h1>{title}</h1>{sub_html}</div>',
                unsafe_allow_html=True)


def dataframe_height(n_rows: int, *, max_rows: int = 20) -> int:
    """Altura en píxeles para un st.dataframe sin scroll hasta `max_rows` filas."""
    return 38 + 35 * min(n_rows, max_rows)


def fmt(x: float | None, dec: int = 4) -> str:
    """Número corto para métricas; '—' si no hay valor."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.{dec}f}"


def fmt_pct(x: float | None, dec: int = 2) -> str:
    """Fracción como porcentaje ('0.0523' -> '5.23%'); '—' si no hay valor."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.{dec}%}"


# ---------------------------------------------------------------------------
# Selectores sobre el workspace
# ---------------------------------------------------------------------------

def pick_instance(key: str, *, label: str = "Instancia") -> tuple[str, Instance] | None:
    """Selector de instancia del workspace; None (con aviso) si no hay ninguna."""
    metas = store.list_instances()
    if not metas:
        st.info("No hay instancias en el espacio de trabajo. Crea una en la página "
                "**Generador** o importa un JSON desde **Inicio**.")
        return None
    by_name = {m.name: m for m in metas}
    name = st.selectbox(
        label, list(by_name),
        key=key,
        format_func=lambda n: f"{n}  ·  {by_name[n].origin}  ·  {by_name[n].created[:10]}",
    )
    return name, store.load_instance(name)


def pick_solution(key: str, *, instance_name: str | None = None,
                  label: str = "Solución") -> store.SolutionRecord | None:
    """Selector de solución (opcionalmente restringido a una instancia)."""
    records = store.list_solutions(instance_name=instance_name)
    if not records:
        st.info("No hay soluciones guardadas"
                + (f" para la instancia **{instance_name}**" if instance_name else "")
                + ". Resuelve una instancia en la página **Resolver**.")
        return None
    by_name = {r.name: r for r in records}
    name = st.selectbox(label, list(by_name), key=key,
                        format_func=lambda n: solution_caption(by_name[n]))
    return by_name[name]


def pick_solutions(key: str, *, instance_name: str | None = None,
                   label: str = "Soluciones") -> list[store.SolutionRecord]:
    """Selector múltiple de soluciones (para el comparador)."""
    records = store.list_solutions(instance_name=instance_name)
    if not records:
        st.info("No hay soluciones guardadas"
                + (f" para la instancia **{instance_name}**" if instance_name else "")
                + ". Resuelve la instancia al menos dos veces en **Resolver**.")
        return []
    by_name = {r.name: r for r in records}
    names = st.multiselect(label, list(by_name), key=key,
                           format_func=lambda n: solution_caption(by_name[n]))
    return [by_name[n] for n in names]


def solution_caption(rec: store.SolutionRecord) -> str:
    """Línea-resumen de una solución para selectores y tablas."""
    z = rec.metrics.get("z_modelo")
    z_txt = f"Z={z:.4f}" if isinstance(z, (int, float)) else "Z=—"
    feas = "factible" if rec.metrics.get("feasible") else "infactible"
    return f"{rec.name}  ·  {SOLVER_LABELS.get(rec.solver, rec.solver)}  ·  {z_txt}  ·  {feas}"


# ---------------------------------------------------------------------------
# Política del objetivo
# ---------------------------------------------------------------------------

def objective_config_widget(key: str, *, initial: ObjectiveConfig | None = None) -> ObjectiveConfig:
    """Formulario de la política de scoring (pesos w_k y L^max). Los derivados w_cob y mu
    quedan automáticos salvo que se fijen a mano en el bloque avanzado."""
    base = initial or ObjectiveConfig()
    cols = st.columns(4)
    w_erg = cols[0].number_input("w_erg (ergonomía)", 0.0, 100.0, base.w_erg, 0.1, key=f"{key}_erg")
    w_car = cols[1].number_input("w_car (eq. carga)", 0.0, 100.0, base.w_car, 0.1, key=f"{key}_car")
    w_gua = cols[2].number_input("w_gua (eq. guardias)", 0.0, 100.0, base.w_gua, 0.1, key=f"{key}_gua")
    w_cal = cols[3].number_input("w_cal (eq. calendario)", 0.0, 100.0, base.w_cal, 0.1, key=f"{key}_cal")
    cols = st.columns(4)
    w_pref = cols[0].number_input("w_pref (preferencias)", 0.0, 100.0, base.w_pref, 0.1, key=f"{key}_pref")
    w_over = cols[1].number_input("w_over (sobre-jornada)", 0.0, 100.0, base.w_over, 0.1, key=f"{key}_over")
    l_max = cols[2].number_input("L^max (racha sin penalizar)", 0, 30, base.l_max, 1, key=f"{key}_lmax")

    with st.expander("Avanzado: w_cob y μ (derivados por defecto)"):
        st.caption("Si no se fijan, w_cob se deriva para garantizar la dominancia de la "
                   "cobertura y μ para que una violación dura cueste más que cualquier "
                   "mejora de Z (Sección 6.3.4).")
        c1, c2 = st.columns(2)
        manual_wcob = c1.checkbox("Fijar w_cob a mano", value=base.w_cob is not None, key=f"{key}_mwc")
        w_cob = c1.number_input("w_cob", 0.0, 1e6, float(base.w_cob or 100.0), 1.0,
                                key=f"{key}_wc", disabled=not manual_wcob)
        manual_mu = c2.checkbox("Fijar μ a mano", value=base.mu is not None, key=f"{key}_mmu")
        mu = c2.number_input("μ", 0.0, 1e7, float(base.mu or 1000.0), 1.0,
                             key=f"{key}_mu", disabled=not manual_mu)

    return ObjectiveConfig(
        l_max=int(l_max), w_erg=w_erg, w_car=w_car, w_gua=w_gua, w_cal=w_cal,
        w_pref=w_pref, w_over=w_over,
        w_cob=w_cob if manual_wcob else None,
        mu=mu if manual_mu else None,
    )


# ---------------------------------------------------------------------------
# Mensajes
# ---------------------------------------------------------------------------

def show_validation_error(exc: Exception) -> None:
    """Error de validación de Instance con el detalle técnico plegado."""
    st.error("La edición no supera la validación de la instancia: revisa la coherencia "
             "(referencias, rangos de día, demanda sobre turnos no ofrecidos, cobertura "
             "estructural por cualificación).")
    detail = str(exc) if isinstance(exc, ValidationError) else repr(exc)
    with st.expander("Detalle del error"):
        st.code(detail)


def feasibility_metric(container, feasible: bool, total_violation: float | None = None) -> None:
    """Métrica homogénea de factibilidad."""
    if feasible:
        container.metric("Factibilidad", "factible", border=True)
    else:
        extra = f"{total_violation:g}" if total_violation is not None else "—"
        container.metric("Factibilidad", "infactible", delta=f"violación {extra}",
                         delta_color="inverse", border=True)


SOLVER_BADGE = {"greedy": ":gray-badge[Greedy]", "sa": ":primary-badge[Simulated annealing]",
                "milp": ":violet-badge[MILP exacto]"}


def solver_badge_md(solver: str) -> str:
    """Directiva de badge por método, para incrustar en markdown/captions."""
    return SOLVER_BADGE.get(solver, f":gray-badge[{solver}]")
