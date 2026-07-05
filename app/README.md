# Interfaz sobre el paquete `rostering`

Aplicación Streamlit para trabajar de forma interactiva con el motor de *physician
scheduling* del TFG: generar instancias sintéticas, editarlas, resolverlas con los tres
métodos (greedy, *simulated annealing* y MILP exacto) y evaluar o comparar las soluciones
con la función objetivo del modelo (Caps. 5–7).

La app es una capa aparte: **no modifica el paquete `rostering`** (`src/rostering/`), que
sigue siendo la única fuente de la semántica del modelo (términos del objetivo,
restricciones duras, normalizadores).

## Lanzar

```bash
uv run --group ui streamlit run app/streamlit_app.py
```

Las dependencias de interfaz (`streamlit`, `plotly`, `pandas`) viven en el grupo `ui` de
`pyproject.toml`, separadas de las del motor.

## Vistas

| Sección | Vista | Qué hace |
|---|---|---|
| — | Inicio | Estado del workspace, flujo de trabajo, importar/exportar/borrar |
| Datos | Generador | Instancias sintéticas parametrizadas o construcción manual por recuentos de clase |
| Datos | Instancia | Inspección y edición validada: plantilla, demanda, reglas, preferencias |
| Resolución | Resolver | Greedy, SA (réplicas) y MILP (HiGHS/Gurobi); «los tres métodos» en un clic |
| Resolución | Experimento | Mini-rejilla n × días × ratio × semillas con greedy y SA (MILP opcional), CSV descargable |
| Análisis | Cuadrante | Cuadrante médico × día con violaciones duras señaladas celda a celda |
| Análisis | Evaluación | Las tres Z, desglose de términos, restricciones duras, re-puntuación |
| Análisis | Comparador | Varias soluciones con la misma política; Δ frente a la mejor |

## Datos

El espacio de trabajo vive en `data/workspace/` (ignorado por git):

- `instances/<nombre>.json` — instancia en el formato canónico de `rostering.storage`,
  legible directamente desde código; `<nombre>.meta.json` guarda procedencia y, si viene
  del generador, el blueprint `Settings` (reproducibilidad).
- `solutions/<nombre>.json` — roster disperso más la configuración empleada y las
  métricas medidas al resolver.

Los resultados de la página Experimento no entran en el workspace: se descargan como CSV.

## Estructura del código

```
app/
├── streamlit_app.py     # enrutador (st.navigation): tema y estilo global
├── views/               # una vista por página (inicio, generador, ..., experimento)
└── core/
    ├── store.py         # persistencia del workspace
    ├── services.py      # fachada sobre rostering (resúmenes, informes, edición, localización de violaciones duras)
    ├── charts.py        # figuras Plotly (paleta única, CVD-safe) y styler del cuadrante
    └── ui.py            # estilo global, cabeceras, badges y widgets compartidos
```
