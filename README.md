# Physician Scheduling

Código y datos de la parte experimental de mi Trabajo de Fin de Grado sobre *Physician
Scheduling* (Doble Grado en Matemáticas y Estadística, Universidad de Sevilla). El paquete
implementa el modelo de la memoria (asignación de médicos a turnos con cobertura blanda
dominante, corredor de jornada, descansos, tope de guardias y términos de equidad y
preferencias) y los tres métodos de resolución que compara el Cap. 7: heurística
constructiva, *simulated annealing* y MILP exacto.

## Estructura

```
src/rostering/       paquete Python
  domain.py          modelo de dominio (Pydantic): turnos, clases, médicos, calendario, instancia
  generator.py       generador de instancias sintéticas (caso base: hospital público andaluz)
  objective.py       evaluador: términos del objetivo, normalizadores, formulaciones de Z
  greedy.py          heurística constructiva (suelo de comparación y solución inicial del SA)
  sa.py, delta.py    simulated annealing con evaluación incremental de movimientos
  milp.py            formulación exacta con PuLP (resolución con Gurobi)
  experiment.py      harness: greedy + SA + exacto por instancia, volcado a CSV
  roster.py          representación de la solución
  storage.py         persistencia de instancias a JSON
experiments/         drivers de los experimentos y análisis
data/                CSV congelados: los resultados que usa la memoria
```

## Instalación

El entorno se gestiona con [uv](https://docs.astral.sh/uv/) (Python >= 3.12):

```
uv sync
```

El método exacto usa Gurobi (`gurobipy`; los experimentos se corrieron con licencia
académica). El greedy y el SA no dependen de ninguna licencia. `milp.solve_exact` admite
también HiGHS (open source) como solver, pero los tiempos y certificaciones de la memoria
corresponden a Gurobi.

## Datos

Dos rejillas con el mismo diseño factorial: 3 tamaños de plantilla x 3 horizontes
(14/21/28 días) x 3 holguras de cobertura (1.10/1.30/1.50) x 5 semillas de instancia
(0-4) = 135 celdas cada una. Presupuesto del solver exacto: 300 s por celda.

| Archivo | Contenido |
|---|---|
| `grid_base.csv` | rejilla base, n en {8, 10, 12}: greedy + SA + exacto por celda |
| `grid_extended.csv` | rejilla extendida, n en {20, 30, 40}: ídem |
| `sa_replicas_base.csv`, `sa_replicas_extended.csv` | 30 réplicas del SA por celda (semilla del SA 0-29, desacoplada de la de instancia) |
| `fairness_base.csv`, `fairness_extended.csv` | distribución por médico (horas, turnos, guardias, fines de semana, festivos) de cada réplica |

## Reproducción

```
uv run python experiments/run_grid.py base        # rejilla base (n 8-12), reanudable
uv run python experiments/run_grid.py ext         # rejilla extendida (n 20-40), reanudable
uv run python experiments/run_replicas.py base    # 30 réplicas SA por celda, paralelo (8 proc.)
uv run python experiments/run_replicas.py ext
uv run python experiments/run_replicas.py ext --dry   # control de fidelidad, no escribe
uv run python experiments/analyze.py              # todas las cifras del Cap. 7 + .dat de figuras (out/)
```

Las corridas escriben en `data/rerun/` para no pisar los CSV congelados; el análisis lee
siempre los congelados de `data/`.

## Reproducibilidad

- La generación de instancias, el greedy y el SA son deterministas por semilla: regenerar
  una celda con las mismas semillas reproduce exactamente los valores del CSV. El exacto
  reproduce los óptimos certificados; en las celdas donde el solver agota los 300 s sin
  certificar, el *incumbent* y la cota dual pueden variar con la versión de Gurobi y el
  hardware.
- La columna `z_sa` de `grid_base.csv` procede de una corrida preliminar, anterior a un
  ajuste de reproducibilidad del SA; el SA canónico de la memoria es el de
  `sa_replicas_*.csv`, corrido íntegro con la versión final del algoritmo. La rejilla
  extendida también se corrió tras el ajuste y sirve de control: la réplica cuya semilla
  del SA coincide con la de instancia reproduce el `z_sa` de `grid_extended.csv` en sus
  135 celdas. El análisis (`experiments/analyze.py`) lee los resultados del SA
  exclusivamente de las réplicas.

## Autor

Mario Pantoja Castro — Trabajo de Fin de Grado, Universidad de Sevilla.
