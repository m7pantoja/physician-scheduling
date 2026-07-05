"""Enrutador de la app: navegación por secciones, tema, logo y estilo global."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from core import ui

st.set_page_config(page_title="Physician scheduling", layout="wide")
ui.inject_style()

nav = st.navigation({
    "": [st.Page("views/inicio.py", title="Inicio", icon=":material/home:", default=True)],
    "Datos": [
        st.Page("views/generador.py", title="Generador", icon=":material/tune:"),
        st.Page("views/instancia.py", title="Instancia", icon=":material/edit_note:"),
    ],
    "Resolución": [
        st.Page("views/resolver.py", title="Resolver", icon=":material/play_circle:"),
        st.Page("views/experimento.py", title="Experimento", icon=":material/science:"),
    ],
    "Análisis": [
        st.Page("views/cuadrante.py", title="Cuadrante", icon=":material/calendar_month:"),
        st.Page("views/evaluacion.py", title="Evaluación", icon=":material/checklist:"),
        st.Page("views/comparador.py", title="Comparador", icon=":material/compare_arrows:"),
    ],
})
nav.run()
