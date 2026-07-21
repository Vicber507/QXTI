---
tags: [package, solvers]
updated: 2026-07-21
---

# 📦 qxti.solvers — integradores ODE

> Integradores adaptativos genéricos. **Hoja** (no importa nada de `qxti`).
> [[Home]] · [[Architecture Map]]

**Usado por:** [[qxti.core|simulation._build_solver]], [[qxti.response|CMD]] (lo guarda).

## Archivos

| Archivo | Rol | Símbolos |
| --- | --- | --- |
| [solver.py](../../qxti/solvers/solver.py) | ABC + RKF45 + Adams-Bashforth 2 | `Solver`, `RKF45Solver`, `AdamsBashforth2Solver` |
| time_domain_solver.py | ⚠️ **stub vacío** | — |
| frequency_domain_solver.py | ⚠️ **stub vacío** | — |
| perturbative_solver.py | ⚠️ **stub vacío** | — |

## Lo importante: quién integra de verdad

⚠️ **La recursión perturbativa NO usa estos solvers.** `CMD` guarda un `Solver` pero, en la
ruta perturbativa, integra la ODE lineal de cada orden con su **propio integrador exponencial
trapezoidal** (`cmd._solve_linear_order_band_on_grid`):
- malla uniforme + Nt>64 → **convolución por FFT** (O(Nt log Nt)),
- si no → iteración directa con propagadores cacheados.

`RKF45Solver`/`AdamsBashforth2Solver` quedan para una eventual ruta **no-perturbativa** (ODE
completa), hoy no cableada. Los `*_solver.py` vacíos son marcadores de esa intención.
Detalle en [[Concept - Perturbative Recursion]] y [[qxti.response]].

## RKF45 (cuando se use)

Runge-Kutta-Fehlberg 4(5) adaptativo. Params: `h_min/h_max`, `safety_factor`, `min/max_factor`,
`max_rejections`. Tolerancia **relativa, consciente de escala** (piso 1e-14). Opcional
`enforce_hermiticity` / `enforce_trace` (post-aceptación).
⛔ Si el error supera la tolerancia en `h_min`, **lanza** `RuntimeError` (no acepta pasos malos).

➕ **Extender:** nuevo integrador → subclasea `Solver` (implementa `solve(f,t0,tf,y0,h)`), y
regístralo en [[qxti.core|simulation._build_solver]].

---

Relacionado: [[Concept - Perturbative Recursion]] · [[qxti.response]]
