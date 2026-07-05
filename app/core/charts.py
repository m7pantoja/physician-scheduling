"""Gráficos Plotly de la app, sobre una paleta única validada (CVD-safe).

Reglas que siguen todas las figuras: color categórico en orden fijo (nunca ciclado),
magnitud con rampa secuencial de un solo tono, un único eje Y por figura, marcas finas,
rejilla recesiva y capa de hover siempre presente. Las etiquetas de texto van en tinta,
nunca en el color de la serie."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# paleta categórica (orden fijo: el orden ES el mecanismo de seguridad CVD)
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

# rampa secuencial azul (magnitud, claro -> oscuro) y rampa roja (déficit)
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_RED = ["#fdf0ef", "#f6c9c8", "#ee9f9e", "#e57573", "#d03b3b", "#a02525"]

# tintas y superficie (modo claro; el tema de la app es claro)
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
NEUTRAL_BG = "#f0efec"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# tintas suaves por turno para el cuadrante (alineadas con la paleta categórica)
TINTS = ["#cde2fb", "#c9efe0", "#ffeab8", "#cde9cd", "#ded9f7", "#f8d3d3", "#fbdde9", "#fde0d4"]


def _scale(colors: list[str]) -> list[list]:
    """Lista de hex -> colorscale de Plotly con paradas equiespaciadas."""
    n = len(colors) - 1
    return [[k / n, c] for k, c in enumerate(colors)]


def _layout(fig: go.Figure, *, height: int, x_title: str = "", y_title: str = "",
            legend: bool = False) -> go.Figure:
    """Estilo común: superficie, tipografía de sistema, rejilla recesiva, sin zoom."""
    fig.update_layout(
        template="none",
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(size=12, color=INK2)),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(family=FONT, color=INK)),
        dragmode=False,
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     automargin=True, tickfont=dict(color=INK2, size=11),
                     title_font=dict(color=INK2, size=12))
    fig.update_yaxes(title=y_title, gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     automargin=True, tickfont=dict(color=INK2, size=11),
                     title_font=dict(color=INK2, size=12))
    return fig


def _heatmap(df: pd.DataFrame, *, colors: list[str], hover_value: str,
             colorbar_title: str) -> go.Figure:
    """Heatmap turno x día con el valor anotado en cada celda. Las celdas NaN (no aplica)
    quedan en el fondo neutro. La rampa se recorta a su mitad clara para que el número en
    tinta sea legible también en la celda más oscura; la magnitud fina la da la anotación."""
    fig = go.Figure(go.Heatmap(
        z=df.values,
        x=list(df.columns),
        y=list(df.index),
        colorscale=_scale(colors),
        zmin=0,
        xgap=2, ygap=2,   # separador de 2px entre celdas
        texttemplate="%{z:.0f}",
        textfont=dict(color=INK, size=11, family=FONT),
        hovertemplate="%{x} · %{y}<br>" + hover_value + ": %{z:.0f}<extra></extra>",
        colorbar=dict(title=dict(text=colorbar_title, font=dict(size=11, color=INK2)),
                      thickness=10, outlinewidth=0, tickfont=dict(size=10, color=MUTED)),
    ))
    height = 130 + 44 * len(df.index)
    fig = _layout(fig, height=max(220, height))
    fig.update_layout(plot_bgcolor=NEUTRAL_BG)   # el hueco NaN se lee como "no aplica"
    fig.update_yaxes(autorange="reversed")
    # todas las etiquetas de día visibles hasta ~5 semanas; después, una por semana
    fig.update_xaxes(tickangle=-55, dtick=1 if len(df.columns) <= 35 else 7,
                     tickfont=dict(size=10, color=INK2))
    return fig


def demand_heatmap(df: pd.DataFrame) -> go.Figure:
    """Demanda b_ts (magnitud -> rampa azul clara; gris = turno no ofrecido ese día)."""
    return _heatmap(df, colors=SEQ_BLUE[:4], hover_value="demanda", colorbar_title="b")


def deficit_heatmap(df: pd.DataFrame) -> go.Figure:
    """Déficit de cobertura (rampa roja clara; gris = celda sin demanda)."""
    return _heatmap(df, colors=SEQ_RED[:4], hover_value="déficit", colorbar_title="déficit")


def contributions_bar(terms: pd.DataFrame) -> go.Figure:
    """Contribución con signo de cada término a Z_modelo (barra horizontal, un solo tono;
    las preferencias satisfechas aparecen en negativo)."""
    df = terms.iloc[::-1]   # el primer término arriba
    fig = go.Figure(go.Bar(
        x=df["contribución"], y=df["término"], orientation="h",
        marker=dict(color=CAT[0]),
        text=[f"{v:.4f}" for v in df["contribución"]],
        textposition="outside", textfont=dict(color=INK2, size=11), cliponaxis=False,
        hovertemplate="%{y}<br>contribución: %{x:.4f}<extra></extra>",
        width=0.55,
    ))
    fig = _layout(fig, height=90 + 34 * len(df), x_title="contribución a Z")
    fig.update_xaxes(zeroline=True, zerolinecolor=AXIS, zerolinewidth=1)
    return fig


def hours_corridor_chart(df: pd.DataFrame) -> go.Figure:
    """Horas asignadas por médico frente al corredor [h_min, h_max] y a la carga media
    tau_i. Barras finas en azul; extremos del corredor como marcas de tinta; tau en aqua."""
    fig = go.Figure()
    fig.add_bar(
        x=df["horas"], y=df["médico"], orientation="h", name="horas asignadas",
        marker=dict(color=CAT[0]), width=0.45,
        hovertemplate="%{y}<br>horas: %{x:.1f}<extra></extra>",
    )
    corridor_x = list(df["h_min"]) + list(df["h_max"])
    corridor_y = list(df["médico"]) + list(df["médico"])
    fig.add_scatter(
        x=corridor_x, y=corridor_y, mode="markers", name="corredor [h^min, h^max]",
        marker=dict(symbol="line-ns-open", size=16, color=INK2, line=dict(width=2)),
        hovertemplate="%{y}<br>límite del corredor: %{x:.1f}<extra></extra>",
    )
    fig.add_scatter(
        x=df["tau"], y=df["médico"], mode="markers", name="τ_i (carga media)",
        marker=dict(symbol="diamond-open", size=9, color=CAT[1], line=dict(width=2)),
        hovertemplate="%{y}<br>τ_i: %{x:.1f}<extra></extra>",
    )
    fig = _layout(fig, height=110 + 30 * len(df), x_title="horas", legend=True)
    fig.update_yaxes(autorange="reversed")
    return fig


def guardias_chart(df: pd.DataFrame) -> go.Figure:
    """Guardias por médico frente a su cuota rho_i·tau_G y, si existe, el tope g^max."""
    fig = go.Figure()
    fig.add_bar(
        x=df["guardias"], y=df["médico"], orientation="h", name="guardias",
        marker=dict(color=CAT[0]), width=0.45,
        hovertemplate="%{y}<br>guardias: %{x}<extra></extra>",
    )
    fig.add_scatter(
        x=df["cuota_guardias"], y=df["médico"], mode="markers", name="cuota ρ_i·τ_G",
        marker=dict(symbol="diamond-open", size=9, color=CAT[1], line=dict(width=2)),
        hovertemplate="%{y}<br>cuota: %{x:.2f}<extra></extra>",
    )
    capped = df.dropna(subset=["g_max"])
    if len(capped):
        fig.add_scatter(
            x=capped["g_max"], y=capped["médico"], mode="markers", name="tope g^max",
            marker=dict(symbol="line-ns-open", size=16, color=INK2, line=dict(width=2)),
            hovertemplate="%{y}<br>g^max: %{x:.0f}<extra></extra>",
        )
    fig = _layout(fig, height=110 + 30 * len(df), x_title="guardias", legend=True)
    fig.update_yaxes(autorange="reversed")
    return fig


def compare_terms_chart(long_df: pd.DataFrame) -> go.Figure:
    """Contribuciones por término para varias soluciones (barras agrupadas; una serie por
    solución, colores categóricos en orden fijo). Espera columnas: solución, término,
    contribución."""
    fig = go.Figure()
    for k, name in enumerate(long_df["solución"].unique()):
        sub = long_df[long_df["solución"] == name]
        fig.add_bar(
            x=sub["término"], y=sub["contribución"], name=str(name),
            marker=dict(color=CAT[k % len(CAT)]),
            hovertemplate=str(name) + "<br>%{x}: %{y:.4f}<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.12)
    fig = _layout(fig, height=380, y_title="contribución a Z", legend=True)
    fig.update_yaxes(zeroline=True, zerolinecolor=AXIS, zerolinewidth=1)
    return fig


def box_by_group(groups: dict[str, list[float]], *, y_title: str) -> go.Figure:
    """Distribución de un valor por grupo (caja + puntos; un color por grupo, orden fijo).
    Sirve para réplicas del SA por semilla y para comparar dispersión de horas."""
    fig = go.Figure()
    for k, (label, values) in enumerate(groups.items()):
        fig.add_trace(go.Box(
            y=values, name=str(label), boxpoints="all", jitter=0.35, pointpos=0,
            marker=dict(color=CAT[k % len(CAT)], size=6),
            line=dict(color=CAT[k % len(CAT)], width=2),
            fillcolor="rgba(0,0,0,0)",
            hovertemplate="%{y:.4f}<extra>" + str(label) + "</extra>",
        ))
    fig = _layout(fig, height=360, y_title=y_title, legend=False)
    return fig


# ---------------------------------------------------------------------------
# Cuadrante (tabla estilizada, no figura)
# ---------------------------------------------------------------------------

def roster_styler(codes: pd.DataFrame, *, shift_names: list[str], guardia_names: set[str],
                  weekend_cols: list[str], holiday_cols: list[str],
                  marked: set[tuple[str, str]] | None = None):
    """Styler de pandas para el cuadrante médico x día: tinta por turno (la guardia manda
    en las celdas combinadas) y fondo neutro en fin de semana y festivo sin asignación.
    `marked` (pares etiqueta-médico, etiqueta-día) añade un anillo rojo interior a las
    celdas implicadas en alguna violación dura."""
    tint_of = {s: TINTS[k % len(TINTS)] for k, s in enumerate(shift_names)}
    off_cols = set(weekend_cols) | set(holiday_cols)
    marked = marked or set()

    def cell_css(value: str, row: str, col: str) -> str:
        base = "text-align:center; font-size:12px; white-space:nowrap;"
        if (row, col) in marked:
            base += "box-shadow: inset 0 0 0 2px #d03b3b;"
        if not value:
            bg = NEUTRAL_BG if col in off_cols else SURFACE
            return base + f"background-color:{bg};"
        parts = value.split("+")
        lead = next((s for s in parts if s in guardia_names), parts[0])
        weight = "font-weight:600;" if lead in guardia_names else ""
        return base + weight + f"background-color:{tint_of.get(lead, NEUTRAL_BG)}; color:{INK};"

    def apply_css(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [[cell_css(df.iloc[r, c], df.index[r], df.columns[c]) for c in range(df.shape[1])]
             for r in range(df.shape[0])],
            index=df.index, columns=df.columns,
        )

    return codes.style.apply(apply_css, axis=None)
