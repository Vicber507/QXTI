---
tags: [concept, physics, decomposition]
updated: 2026-07-21
---

# 🧠 Concept — Descomposición intra/inter, x/y/z, RCP/LCP

> Cómo se parte la corriente y qué es físico (aunque parezca raro). [[Home]] ·
> [[qxti.response]] · [[qxti.data]] · [[qxti.graphics]]

## Definición única (la misma en los 3 motores)

Con `J = Tr[v ρ] = Σ_mn v_mn ρ_nm`:
- **Intra (diagonal):** `J_intra = Σ_n v_nn ρ_nn` — velocidad de grupo × población (Bloch/Drude).
- **Inter (off-diagonal):** `J_inter = J_total − J_intra` — coherencias entre bandas.

Implementado idéntico en:
- [[qxti.data|StreamingCurrentAccumulator]]: `einsum("mn,tnm->t",v,ρ)` (total) y `einsum("n,tn->t",diag(v),diag(ρ))` (intra).
- [[qxti.response|XTP.current_decomposition]] (requiere `band_gauge_frame`).
- [[qxti.analytics|mesh_response]] (`return_intraband=True`, `_trace_J` vs `_trace_intra`).

⛔ **Invariante:** `intra + inter = total` **exacto** por construcción, con signos consistentes
(ambos con el `−` de la corriente). Si "arreglas" uno, verifica que la suma siga dando el total.

## RCP / LCP (base circular)

Sobre el espectro complejo a ω>0: `J_R = (J_x − i J_y)/√2`, `J_L = (J_x + i J_y)/√2`
(`plot_harmonics.py`). El tensor σ/χ se rota con `e_± = (x ± i y)/√2` (base de **helicidad**,
`plot_susceptibility_tensor.py`).

## ⚠️ El caso "el inter es más alto que el armónico" — ES FÍSICO

Observación típica (WSM, CMD): en armónicos **impares** H3/H5/H7, |inter| y |intra| son ambos
**grandes** y casi se cancelan → |total| pequeño. En **pares** H2/H4/H6, intra≈0 e inter=total.

Esto **no es bug**: es la **interferencia intra/interbanda** de HHG en sólidos (Vampa–Corkum):
- La velocidad de grupo `v_nn` es **impar en k** → el intra solo genera armónicos **impares**.
- En los impares, intra e inter (grandes, signo opuesto) se restan → el total es el **residuo**.
- Por eso el "pico" del interbanda queda por encima del armónico total pequeño.

Matiz honesto: el split diagonal/off-diagonal (base de banda) es **dependiente de gauge** y puede
mostrar cancelaciones grandes — se ve raro pero es correcto. Si se quisiera una versión que no
cancele, habría que implementar la descomposición **gauge-invariante** (corriente de polarización
`∂_t P`) como opción aparte. Ver [[Concept - Perturbative Recursion]] para el origen de v_nn.

## Claves del dataset

`current_*_intraband/interband` solo aparecen si `current_decomposition_available=True`
(streaming en `band` basis, o mesh con `return_intraband`). [[qxti.graphics]] las lee por nombre.

---

Relacionado: [[Concept - Occupation Gauge Basis]] · [[Concept - Response Engines]] · [[qxti.graphics]]
