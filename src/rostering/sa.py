"""Simulated annealing (Sección 6.3, Alg. 6.2).

Minimiza el objetivo aumentado (cobertura blanda y hard constraints penalizadas con mu) sobre el
hipercubo {0,1}^n, con dos operadores de vecindario: FLIP enciende o apaga una asignación viva,
y SWAP reasigna un turno de un médico a otro en la misma celda, el ataque directo a la equidad
de carga sin cruzar la barrera de déficit. Acepta toda mejora y, con probabilidad de Metropolis
e^{-Delta/T}, también empeoramientos, con T en esquema geométrico descendente. Devuelve el mejor
roster visitado; el coste de cada movimiento lo da el `DeltaEvaluator` incremental."""

import math
import random

from pydantic import BaseModel, ConfigDict, Field

from rostering.delta import DeltaEvaluator
from rostering.domain import Instance
from rostering.objective import ObjectiveConfig
from rostering.roster import Roster


class SAConfig(BaseModel):
    """Parámetros del enfriamiento (Alg. 6.2). La calibración fina (T_0, alpha, L frente al tamaño n)
    es trabajo del Cap. 7; estos son defaults razonables para arrancar."""

    t_0: float = Field(default=100.0, gt=0)          # T_0: temperatura inicial
    alpha: float = Field(default=0.95, gt=0, lt=1)   # factor de enfriamiento geométrico (T <- alpha*T)
    chain_length: int = Field(default=200, ge=1)     # L: pasos por temperatura (equilibrio térmico)
    t_min: float = Field(default=1e-2, gt=0)         # parada: temperatura mínima
    max_iterations: int = Field(default=200_000, ge=1)    # parada: presupuesto de iteraciones
    stagnation_limit: int = Field(default=20_000, ge=1)   # parada: iteraciones sin mejorar x*
    p_swap: float = Field(default=0.5, ge=0, le=1)   # prob. de proponer SWAP (vs FLIP) por paso
    seed: int = 0


class SAResult(BaseModel):
    """Salida del SA: el mejor roster visitado y sus estadísticas de ejecución."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    roster: Roster
    score: float          # Z(x*): valor del objetivo aumentado en el mejor visitado
    iterations: int
    accepted: int         # movimientos aceptados (diagnóstico de la temperatura)
    stop_reason: str      # "t_min" | "max_iterations" | "stagnation"


def _live_structures(instance: Instance):
    """Precomputa, una sola vez, lo que el vecindario necesita por movimiento:
      - `live_triples`: ternas (i,t,s) que pasan gamma y eta, sin guardias de exentos -> candidatas de FLIP;
      - `live_phys[s]`: médicos que pueden cubrir s (gamma; un exento no entra en guardias) -> candidatos j de SWAP
        (la elegibilidad eta de la celda ya se cumple, porque el origen del swap es una asignación)."""
    days = instance.calendar.days
    is_guardia = {s.name: s.is_guardia for s in instance.shifts}
    class_name = {p.id: instance.class_of(p).name for p in instance.physicians}
    exempt = {p.id: p.is_exempt for p in instance.physicians}

    def is_live(i: int, t: int, s: str) -> bool:
        if not instance.eligibility.get(t, {}).get(s, False):                 # eta
            return False
        if not instance.qualification.get(class_name[i], {}).get(s, False):   # gamma
            return False
        if is_guardia[s] and exempt[i]:                                        # E
            return False
        return True

    live_triples = [(p.id, t, s.name)
                    for p in instance.physicians for t in range(days) for s in instance.shifts
                    if is_live(p.id, t, s.name)]
    live_phys = {
        s.name: [p.id for p in instance.physicians
                 if instance.qualification.get(class_name[p.id], {}).get(s.name, False)
                 and not (s.is_guardia and exempt[p.id])]
        for s in instance.shifts
    }
    return live_triples, live_phys


def schedule(instance: Instance, *, budget: int | None = None, chain_length: int = 200,
             t_0: float = 100.0, t_min: float = 1e-2, p_swap: float = 0.5, rounds: int = 400,
             stagnation_frac: float = 1.0, seed: int = 0) -> SAConfig:
    """Esquema de enfriamiento robusto para `instance`.

    `t_0`/`t_min` van FIJOS, situados en el HUECO entre dos escalas del objetivo aumentado: los
    términos soft (equidad, ergonomía) cambian en Delta~=O(1) ---a T=t_0 se aceptan y se RECUECEN---,
    mientras que violar una hard cuesta mu (10^3 o más) ---a T=t_0 se acepta con e^{-mu/t_0}~=0 y queda
    CONGELADA---. Es el método de penalización: la temperatura anela lo blando y respeta lo duro.
    (Calibrar t_0 por la media/mediana de los Delta falla: la cola mu la dispara fuera del hueco.)

    El único parámetro CALCULADO es alpha, fijado para que el descenso geométrico de t_0 a t_min
    GASTE el presupuesto de iteraciones (`alpha = (t_min/t_0)^{L/budget}`) en vez de apagarse antes de
    tiempo. El presupuesto, si no se da, escala con el espacio de búsqueda: ~=`rounds` pasadas
    sobre las ternas vivas (instancias mayores reciben más iteraciones)."""
    live_triples, _ = _live_structures(instance)
    if budget is None:
        budget = max(100_000, rounds * len(live_triples))
    levels = max(1, budget // chain_length)
    alpha = min(0.999999, max(1e-6, (t_min / t_0) ** (1.0 / levels)))
    return SAConfig(t_0=t_0, alpha=alpha, t_min=t_min, chain_length=chain_length,
                    max_iterations=budget, stagnation_limit=max(1, int(budget * stagnation_frac)),
                    p_swap=p_swap, seed=seed)


def simulated_annealing(instance: Instance, x0: Roster,
                        sa_config: SAConfig | None = None,
                        obj_config: ObjectiveConfig | None = None) -> SAResult:
    """Recoce desde el roster inicial `x0` (típicamente el del greedy). Minimiza Z con el
    esquema del Alg. 6.2 (vecindario flip+swap) y devuelve el mejor roster visitado."""
    sa_config = sa_config or SAConfig()
    obj_config = obj_config or ObjectiveConfig()
    rng = random.Random(sa_config.seed)
    live_triples, live_phys = _live_structures(instance)

    ev = DeltaEvaluator(instance, x0, obj_config)   # estado incremental de Z sobre el roster de trabajo
    fbest = ev.score
    best = set(ev.assigned)                          # mejor conjunto de asignaciones visitado (x*)
    iters = accepted = since_improve = 0
    T = sa_config.t_0

    def propose():
        """Propone y APLICA un movimiento; devuelve (Delta, undo) donde undo() lo deshace. Con prob.
        p_swap intenta un SWAP (si hay asignaciones y candidato j); en otro caso, un FLIP vivo."""
        if ev.assigned and rng.random() < sa_config.p_swap:
            # sorted, no list: el orden de iteración de un set de tuplas con nombre de turno
            # (str) depende de PYTHONHASHSEED; ordenarlo deja la trayectoria del SA fijada solo
            # por la semilla, reproducible entre procesos.
            i, t, s = rng.choice(sorted(ev.assigned))
            choices = [j for j in live_phys[s] if j != i and (j, t, s) not in ev.assigned]
            if choices:
                j = rng.choice(choices)
                delta = ev.apply(i, t, s) + ev.apply(j, t, s)     # apaga i, enciende j (izq->dcha)
                return delta, lambda: (ev.apply(j, t, s), ev.apply(i, t, s))   # inverso: apaga j, re-enciende i
        i, t, s = rng.choice(live_triples)
        return ev.apply(i, t, s), lambda: ev.apply(i, t, s)

    def finish(reason: str) -> SAResult:
        return SAResult(roster=Roster(assignments=set(best)), score=fbest,
                        iterations=iters, accepted=accepted, stop_reason=reason)

    while T > sa_config.t_min:
        for _ in range(sa_config.chain_length):
            delta, undo = propose()
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                accepted += 1                         # aceptar (el movimiento ya está aplicado)
                if ev.score < fbest:
                    fbest = ev.score
                    best = set(ev.assigned)
                    since_improve = 0
                else:
                    since_improve += 1
            else:
                undo()                                # rechazar: deshacer
                since_improve += 1
            iters += 1
            if iters >= sa_config.max_iterations:
                return finish("max_iterations")
            if since_improve >= sa_config.stagnation_limit:
                return finish("stagnation")
        T *= sa_config.alpha                          # enfriamiento geométrico

    return finish("t_min")
