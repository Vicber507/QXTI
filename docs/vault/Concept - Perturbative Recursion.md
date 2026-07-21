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
  con ω_nm = E_m − E_n, γ_nm = γ_pop (diag) o γ_coh (off-diag).
- En **frecuencia** (mesh/per-k) esto es forma cerrada en s·ω; en **tiempo** (CMD) es el
  integrador exponencial trapezoidal (FFT o directo).

## El gradiente covariante D_k (el *lazo* más delicado)

```
D_k ρ = ∂_k ρ − i[A, ρ]      (A = conexión de Berry)
```

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
