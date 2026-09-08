---
tags: [concept, physics, recursion]
updated: 2026-07-21
---

# 🧠 Concept — Recursión perturbativa + gradiente covariante

> El corazón físico: cómo se genera ρ⁽ˢ⁾ orden a orden, y por qué el gradiente-k **debe** ser
> covariante "de un solo tiro". [[Home]] · [[qxti.response]] · [[qxti.analytics]]

## La recursión (Hipólito 2018, A1)

- **Orden 0:** ρ⁽⁰⁾ = `diag(f_n)` en base de banda (ocupación de equilibrio, ver
  [[Concept - Occupation Gauge Basis]]).
- **Orden s≥1:** ecuación lineal inhomogénea por elemento (n,m):
  ```
  dρ⁽ˢ⁾_nm/dt = −(i·ω_nm + γ_nm)·ρ⁽ˢ⁾_nm  +  E(t)·[D_k ρ⁽ˢ⁻¹⁾]_nm
  ```
  con ω_nm = E_m − E_n, γ_nm = γ_pop (diag) o γ_coh (off-diag). El `+E·D_kρ` es la carga del
  **electrón** (q = −1): H' = +E·r̂, [r̂,ρ] = i·D_kρ ⇒ −i[H',ρ] = +E·D_kρ. Coherente con `j = −v`.
- En **frecuencia** (mesh/per-k), convenio e^{−iωt}, la forma cerrada es
  ```
  ρ⁽ˢ⁾ = i · E·[D_k ρ⁽ˢ⁻¹⁾] / (s·ω + iγ − ω_nm)        (orden 1: ρ⁽¹⁾ = i·E·D_kρ⁽⁰⁾/(ω+iγ−ω_nm))
  ```
  ⛔ **La `i` es física** (sale de resolver la EDO). Hasta 2026-09 el mesh y `rho_order_s` no la
  llevaban y además usaban `A_mn` con el signo cambiado; el efecto neto era `ρ⁽ˢ⁾ = −(−i)^{s−1}·ρ_fís`:
  **orden 1 con el signo invertido** (`Re σ_xx < 0`, absorción negativa; `J(t)` de pfddm en
  contrafase con tddm/CMD) y fases entre armónicos rotadas. `time_domain_currents` lo parcheaba
  con `i^(s−1)` en la corriente. Todo eso está **corregido**; ya no hay parche.
- En **tiempo** (CMD) es el integrador exponencial trapezoidal (FFT o directo). CMD siempre estuvo
  en el convenio físico (la herramienta `compare_rho4_cmd_vs_analytic.py` medía `−i^(s−1)` de
  desfase con el analítico y lo llamaba "convención": era el bug).

## El gradiente covariante D_k (el *lazo* más delicado)

```
D_k ρ = ∂_k ρ − i[A, ρ]      (A = conexión de Berry, signo ESTÁNDAR: A_mn = i⟨m|∂_k n⟩ = i·v_mn/(E_n−E_m))
```
⛔ El signo de `A` importa: `operators.berry_connection`, `mesh_response.BandData.A` y
`rho_analytic._berry_offdiag` usan todos `i·v_mn/(E_n−E_m)`. Con el signo opuesto el orden 1
sale con la corriente invertida.

⛔ **Se calcula de UN SOLO TIRO** con **Wilson links** (transporte paralelo): se rota el ρ del
vecino a la base local antes de restar. Numéricamente:
```
D_k ρ ≈ (W₊ ρ(k+dk) W₊†  −  W₋ ρ(k−dk) W₋†) / (2 dk),   W± = U(k)† U(k±dk)
```
Esto **ya incluye** el `−i[A,ρ]`. **Nunca** sumes el conmutador de Berry por separado: en órdenes
≥2 con ρ fuera de la diagonal, hacerlo lo cuenta doble y rompe la simetría/gauge-invariancia.

Implementaciones (deben coincidir):
- CMD: `_covariant_gradient_for_k_index` / `_driving_components_for_k_index` ([[qxti.response|cmd.py]]).
- mesh: `cov_grad` (con `np.roll` de U) en [[qxti.analytics|mesh_response.py]].
- referencia: `_drho_dk_numerical` en [[qxti.analytics|rho_analytic.py]].

## Discretización (la sutileza)

- El gradiente Wilson usa el **paso de malla** (2π/N). La referencia per-k usa `dk_grad` local.
  Convergen al refinar, pero a malla gruesa **difieren** en órdenes altos → ver la trampa en
  [[Concept - Response Engines]].
- La malla-k necesita **≥2 puntos en cada eje activo** o CMD cae fuera de la ruta covariante.
- Residuos de orden alto suelen venir del **muestreo temporal**, no del paso-k.

## Corriente

`J⁽ˢ⁾ = Σ_k w_k Tr[v ρ⁽ˢ⁾]` (con el signo `j=−v` de [[qxti.physics|operators]]). El split
intra/inter sale de aquí → [[Concept - Inter-Intra Decomposition]].

---

Relacionado: [[Concept - Response Engines]] · [[Concept - Occupation Gauge Basis]] · [[MESH_RESPONSE]]
