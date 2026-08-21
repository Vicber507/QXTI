# Integradores temporales del motor `tddm` (velocity gauge)

Cómo QXTI resuelve, en el tiempo, la ecuación no perturbativa del `tddm`; qué esquema usa
por defecto, por qué ese y no un Runge–Kutta genérico, cómo se valida con la literatura, y
cómo elegir el integrador desde el input. El código está en el módulo
[`qxti/analytics/propagators.py`](../qxti/analytics/propagators.py) y el motor en
[`qxti/analytics/tddm.py`](../qxti/analytics/tddm.py).

---

## 1. La ecuación diferencial que resolvemos

El motor `tddm` (no perturbativo, **velocity gauge**) propaga, **independientemente para cada
punto k**, la matriz densidad de un electrón bajo la sustitución de Peierls `k → k + A(t)`:

$$
\boxed{\;\frac{d\rho_{\mathbf k}}{dt} \;=\; -\,i\,\big[\,H(\mathbf k + \mathbf A(t)),\,\rho_{\mathbf k}\,\big]\;-\;\mathcal R[\rho_{\mathbf k};t]\;}
$$

y de ahí la corriente macroscópica

$$
\mathbf J(t) \;=\; -\sum_{\mathbf k} w_{\mathbf k}\,\mathrm{Tr}\!\big[\,\hat{\mathbf v}(\mathbf k+\mathbf A(t))\,\rho_{\mathbf k}(t)\,\big] \;-\; \mathbf J_{\rm DC},
\qquad \hat{\mathbf v} = \partial_{\mathbf k}H .
$$

Las **partes** de la ecuación, cada una con su tratamiento numérico:

| parte | expresión | carácter | cómo se trata |
|---|---|---|---|
| **coherente** (unitaria) | $-i[H(\mathbf k+\mathbf A(t)),\rho]$ | lineal, **rígida y oscilatoria** (frecuencia = gap de banda), preserva unitariedad | integrador exponencial (§3) |
| **relajación** RTA | $\mathcal R = \tfrac1{T_1}(\rho_{\rm diag}-f)\ +\ \tfrac1{T_2}\rho_{\rm off}$ | disipativa, hacia las ocupaciones instantáneas $f_n(\mathbf k+\mathbf A)$ | **Strang split** (medio paso a cada lado del paso coherente) |
| **campo** $\mathbf A(t)$ | $\mathbf A(t)=-\!\int^t \mathbf E\,dt'$ | no autónomo (depende de t) | **analítico** del láser (§2) |

La relajación por defecto está **apagada** ($T_1=T_2=\infty$): el uso principal (HHG) es
puramente coherente. Cuando está activa, el `_relax_half_` se aplica en la base propia
instantánea (Houston) del punto medio, con la misma estructura para todos los integradores.

---

## 2. El potencial vector `A(t)`: analítico, no integrado

El propagador solo "ve" `A(t)` (gauge de velocidad; el campo `E(t)` no entra en la dinámica,
solo se reporta y normaliza el espectro). Por eso **la precisión de `A(t)` limita el orden de
todo el esquema**.

Históricamente `A(t)` se construía por **trapecio** `A=−∫E dt` (`_vector_potential_from_field`),
que es **solo $O(\Delta t^2)$** — techo de 2º orden aunque el propagador fuese de orden mayor.
Ahora que el láser cumple $E=-dA/dt$ **exacto** (fix del signo de la derivada de envolvente,
ver `docs`/`qxti/physics/laser.py`), usamos la **A analítica** `laser_system.vector_potential(t)`,
que:

- es **consistente** con la misma `E(t)` de los motores perturbativos (sin discrepancia de gauge),
- **no tiene error de cuadratura** → no capa el orden del propagador,
- puede evaluarse en **cualquier instante** (puntos de malla, puntos medios, nodos de Gauss),
  que es justo lo que un integrador de orden alto necesita.

> La función trapezoidal `_vector_potential_from_field` queda **deprecada** (solo la usan
> algunas herramientas sueltas en `tools/`); el motor `tddm` ya no la usa.

---

## 3. Los integradores (elegibles desde el input)

Se elige con `[cmd] tddm_propagator = cfm2 | rkf45 | ab2`. **Por defecto `cfm2`.** Implementados
en [`qxti/analytics/propagators.py`](../qxti/analytics/propagators.py), vectorizados sobre bloques
de k.

### 3.1 `cfm2` — Magnus sin conmutadores de 2º orden (exponential-midpoint)  ← DEFAULT

Congela `H` en el **punto medio** del paso y aplica la exponencial exacta:

$$
\rho(t+\Delta t) = U\,\rho(t)\,U^\dagger,\qquad U = e^{-\,i\,H(t+\Delta t/2)\,\Delta t}
= V\,e^{-iE\Delta t}\,V^\dagger ,
$$

con $E,V$ la descomposición propia (Hermítica) de $H(\mathbf k+\mathbf A(t+\Delta t/2))$. Es el
**miembro de 2º orden de la expansión de Magnus** — lo que la literatura llama *"exponential
midpoint rule"* (Blanes, Casas, Oteo, Ros, *Phys. Rep.* **470** (2009) 151; Hochbruck &
Ostermann, *Acta Numer.* **19** (2010) 209; Gómez-Pueyo, Marques, Rubio, Castro, *JCTC* **14**
(2018) 3040, que lo nombra explícitamente *"the second-order Magnus expansion, also known as
the exponential midpoint rule"*).

**Propiedades** (verificadas en `tests/test_propagators.py`):
- **Unitario por construcción** ⇒ conserva $\mathrm{Tr}\,\rho$ y mantiene los autovalores de
  $\rho$ en $[0,1]$ (positividad) a precisión de máquina.
- Orden global **2** ($O(\Delta t^2)$), 1 exponencial/paso.
- Es **geométrico/estructura-preservante**: la razón por la que se prefiere en dinámica cuántica
  manejada por láser frente a un RK genérico (que a muchos ciclos deriva en fase/amplitud y
  rompe la norma).

### 3.2 `rkf45` — Runge–Kutta–Fehlberg 4(5)

Un paso Fehlberg (6 etapas, tabla clásica de 1969) por intervalo de malla, usando la solución de
**5º orden**; `H` se evalúa en los 6 nodos $t+c_j\Delta t$ (de ahí que necesitemos la `A`
analítica en esos sub-instantes). El par 4(5) da un estimador de error local embebido.
General-purpose y de orden alto, pero **no preserva unitariedad** (la norma de $\rho$ deriva
lentamente). Útil como referencia/validación cruzada.

### 3.3 `ab2` — Adams–Bashforth de 2 pasos

Multipaso lineal explícito, $\rho_{n+1}=\rho_n+\Delta t\,(\tfrac32 f_n-\tfrac12 f_{n-1})$ con
$f=-i[H(t),\rho]$, arrancado con un paso `cfm2`. Orden global **2**, barato (1 evaluación de RHS
por paso), pero tampoco unitario. Incluido por completitud.

**Comparación** (ver `tests/test_propagators.py`):

| propagador | orden | exp./RHS por paso | unitario | uso |
|---|---|---|---|---|
| **cfm2** (default) | 2 | 1 exp | **sí** | producción (HHG, respuesta no lineal) |
| rkf45 | ≥4 | 6 evals H | no | referencia/validación de orden alto |
| ab2 | 2 | 1 eval | no | referencia barata |

Solo `cfm2` es estructura-preservante. Para el problema (rígido, oscilatorio a muchos ciclos,
matrices pequeñas por k donde la exponencial es trivial) la literatura recomienda precisamente
integradores exponenciales/Magnus, no RK adaptativo — por eso `cfm2` es el default.

---

## 4. Relajación (T1/T2) con cualquier integrador

Para todos los propagadores la relajación se aplica por **Strang splitting** simétrico alrededor
del paso coherente (medio paso de $\mathcal R$ antes y después), en la base propia instantánea del
punto medio. Así el tratamiento de $T_1/T_2$ es idéntico y de 2º orden con los tres esquemas; el
único que cambia es el **avance coherente**.

---

## 5. Validación (tests)

`tests/test_propagators.py` (unidad, sobre un $H(t)$ 2×2 con componentes de frecuencias distintas
para que $[H(t),H(t')]\neq0$):
- **exactitud** en $H$ constante: `cfm2` reproduce $e^{-iHt}$ a máquina; `rkf45` a $<10^{-8}$.
- **orden de convergencia** empírico frente a una referencia fina: `cfm2`$\approx2$,
  `rkf45`$\ge4$, `ab2`$\approx2$.
- **preservación de estructura**: `cfm2` conserva $\mathrm{Tr}\,\rho$, hermiticidad y positividad
  ($\lambda(\rho)\in[0,1]$) a máquina; `rkf45`/`ab2` solo aproximadamente.
- **unitariedad** de $U=e^{-iH\Delta t}$: $\|U^\dagger U-\mathbb 1\|<10^{-12}$.

`tests/test_tddm.py` (motor completo):
- el **default es `cfm2`**;
- los tres son **seleccionables** desde el input y **coinciden** en la corriente sobre una malla
  resuelta (rkf45 y ab2 vs cfm2);
- un nombre de integrador inválido se **rechaza** al parsear el config.

---

## 6. Resumen

- Resolvemos $d\rho/dt=-i[H(\mathbf k+\mathbf A(t)),\rho]-\mathcal R$ por k, con `A(t)`
  **analítica** (sin trapecio) y relajación por Strang split.
- El integrador es elegible: **`[cmd] tddm_propagator = cfm2 | rkf45 | ab2`**, **default `cfm2`**
  (Magnus-2 / exponential-midpoint, unitario) — el recomendado por la literatura para esta clase
  de ecuación.
- `rkf45` y `ab2` quedan disponibles como alternativas/validación (no unitarias).
