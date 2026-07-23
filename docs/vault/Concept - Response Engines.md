---
tags: [concept, engine, validation]
updated: 2026-07-21
---

# 🧠 Concept — Los motores de respuesta y la cadena de validación

> El eje de motores tiene **3 nombres canónicos** (`pfddm`/`ptddm`/`tddm`) + la referencia per-k.
> Saber cuál es perturbativo, cuál es producción y cuál es referencia evita "arreglar" un motor
> comparándolo mal con otro. [[Home]] · [[qxti.analytics]] · [[qxti.response]]

## La taxonomía (nombres canónicos)

| Nombre | = viejo | Dónde | Dominio | Perturbativo | Rol / coste |
| --- | --- | --- | --- | --- | --- |
| **`pfddm`** | theory | [[qxti.analytics|mesh_response/theory_response]] | frecuencia (n·ω) | sí | **producción** cerrada, vectorizada; O(orden·Nk) |
| **`ptddm`** | simulation | [[qxti.response|cmd.py]] (CMD) | tiempo | sí | recursión ρ⁽ˢ⁾(k,t); referencia exacta; O(orden·Nk·Nt) |
| **`tddm`** | — (nuevo) | [[qxti.analytics|tddm.py]] | tiempo | **NO** | **full no-perturbativo** (gauge velocidad); el más caro; O(Nk·Nt·eigh) |
| `rho_analytic` | — | [[qxti.analytics|rho_analytic.py]] | frecuencia | sí | **referencia per-k** (la "verdad"); O(Nk·7^(s−1)) |

pfddm = *perturbative frequency-domain DM*; ptddm = *perturbative time-domain DM*; tddm =
*time-dependent DM* (completo). `both` = pfddm+ptddm; **`all`** = los 3 + comparación.
`time_domain_currents` (dentro de mesh_response) es la variante **pulsada/multi-láser** de pfddm.

## La cadena de validación (no la rompas)

```
rho_analytic (per-k, lento)  ══ precisión de máquina ══  mesh_response (producción)
      ║                                                        ║
      ╚═══ valida ═══▶  CMD (tiempo) ══ FFT en n·ω, uniforme ══╝
```

- `test_mesh_matches_perk_to_machine_precision` fija el acuerdo **mesh ↔ per-k** (con pasos de
  gradiente igualados).
- ⚠️ **Trampa de comparación:** a malla **gruesa**, mesh (gradiente Wilson con paso 2π/N) y per-k
  (`dk_grad=1e-3` local) usan **pasos distintos** → difieren en órdenes ≥2. Eso **no** es bug;
  convergen al refinar. Para comparar de verdad: iguala `dk_grad` al paso de malla o usa malla fina.

## Cuál usa cada flag

- `-cmd` `response_method`: `pfddm` | `ptddm` | `tddm` | `both` | `all` (alias: theory→pfddm,
  simulation→ptddm). Canonicaliza en [[qxti.core|config._canonical_method]].
- `-xtp` `susceptibility_method`: igual. tddm en `-xtp` extrae χ⁽ˢ⁾ por **escalado en amplitud**
  (corre M amplitudes E0, ajusta polinomio; el coef de E0^s = χ⁽ˢ⁾).
- Multi-láser ⇒ siempre rama tiempo (`time_domain_currents` para pfddm; tddm ya es tiempo).
- `all` guarda `engine_comparison.npz` (tabla de picos por armónico + error relativo + tiempos).

Detalle de despacho en [[qxti.core|simulation / susceptibility_scan]] y [[Data Flow]].

## El motor `tddm` (no perturbativo) — gauge de velocidad

`dρ_k/dt = −i[H(k+A(t)), ρ_k] − R`, independiente por-k (Peierls; sin gradiente covariante →
**streamable sin halo**, más ligero que el mesh). Integrador: exp-Euler en la base instantánea
(1 eigh/paso), Strang con la relajación T1/T2 en la **base Houston** (relaja hacia f_n(k+A(t)),
**no** hacia cero). Corriente `J=−Σ w Tr[v(k+A)ρ] − J_DC`.

⛔ **Correctitud (no omitir):**
- **A(t) = −∫E dt** del MISMO E (helper `_vector_potential_from_field`), no la A(t) del láser
  (difiere ~% por la envolvente y rompe la equivalencia de gauge con los perturbativos).
- **Restar J_DC** (corriente de equilibrio A=0): quita el pico espurio en ω=0.
- Relajación/split en la base **instantánea**, hacia las ocupaciones reales.

⚠️ **Comparación justa (leer):** tddm es **pulsado**; pfddm (CW-dressed) difiere por la envolvente.
Compara tddm contra el **perturbativo pulsado** (`time_domain_currents` o ptddm/CMD), mismo
E(t)/malla/dt/FFT, y con **ventana** (el H1 es ~3 órdenes mayor que H3; sin ventana su fuga domina el
bin de 3ω). El **H1 lineal coincide 0.1–5% y converge en k** en modelos gapped y gapless → el núcleo
(gauge, corriente, DC, propagación) es correcto.

**Estado de validación de tddm (leer con cuidado):**
- ✅ **H1 (respuesta lineal) coincide con el perturbativo a <0.1%** (Haldane 1.000; graphene converge
  0.87→1.05) **a todo campo** → el núcleo (gauge velocidad, corriente k+A, resta DC, A=−∫E,
  propagación exp-midpoint 2º orden) es **correcto**. Traza/hermiticidad conservadas; independiente
  de workers; H3 ∝ E0³ (orden-3 auto-consistente).
- 🔶 **H3: la referencia PERTURBATIVA no está convergida en malla (ésta es la historia real).** La
  convergencia en malla a campo débil (fig3) muestra que el **H3 perturbativo CRECE** con la malla-k
  (4.5e-7→5.7e-6, aún creciendo a 28²) mientras el **H3 de tddm es ESTABLE** (~1e-6). El ratio
  tddm/pert cae (3.1→0.09) **no porque tddm esté mal, sino porque el χ³ perturbativo es sensible a la
  malla** (se dispara cerca del gap pequeño; en gapless **diverge** en el Dirac: graphene 1.0→0.2→0.03).
  ⚠️ **Corrección:** mi interpretación previa de "no-invariancia de gauge" era **probablemente errónea**
  — en TB **finito** la base es completa y velocidad/longitud **deberían** coincidir; el problema es la
  **no-convergencia en malla** de la referencia. No se puede probar cuál χ³ es el verdadero sin una
  referencia independiente más fina, pero **tddm es el motor ESTABLE**. NO afirmar que tddm reproduce
  las amplitudes no-lineales perturbativas.
- ⚠️ **H2 (armónico par): artefacto ESPURIO pequeño en tddm.** En graphene y haldane_topological
  (ambos con inversión, M0=0 → pares prohibidos) tddm da un H2 no-nulo que **escala E0³** (potencia
  impar → numérico, no un orden-2 físico E0²) y vive en **J_x** (no en J_y, donde estaría un Hall
  físico). Causa: **gauge de velocidad en malla finita** — el desplazamiento k→k+A(t) rompe la
  cancelación discreta k→−k. Minúsculo a campo débil/malla fina (H2/H1~1e-6…1e-7), crece a campo
  fuerte/malla gruesa. Fix: simetrización k→−k de J(t) (OPCIONAL — mataría los pares Hall FÍSICOS en
  modelos con inversión ROTA/Chern) o malla densa.

⇒ **Para zanjar H3:** un full-solve en gauge de LONGITUD (reusa `cov_grad`) o una referencia
independiente muy fina. Figuras de validación en `outputs/tddm_validation/`. Ver `tools/compare_engines.py`.

## El ciclo de import (cuidado)

`theory_response` importa `simulation` en el **tope**; `simulation` importa `compute_hhg_spectrum`
de forma **lazy** (dentro de la función). ⛔ No lo muevas al tope → import circular. Es la razón de
que theory pueda reusar los *builders* de simulation sin duplicar código.

---

Relacionado: [[Concept - Perturbative Recursion]] · [[Concept - Inter-Intra Decomposition]] · [[Architecture Map]]
