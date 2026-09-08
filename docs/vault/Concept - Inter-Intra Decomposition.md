---
tags: [concept, physics, decomposition]
updated: 2026-09-07
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

Sobre el espectro de **numpy FFT en los bins +ω** (que guardan el **conjugado** de la amplitud
e^{−iωt}): `J_R = (J_1 + i J_2)/√2`, `J_L = (J_1 − i J_2)/√2`, donde `(1,2)` es el plano
**transverso a la propagación**, no el `(x,y)` de laboratorio. **R ≡ helicidad positiva (σ⁺)** =
el mismo sentido de giro que un drive con `ellip > 0` (`Laser.helicity > 0`); L ≡ σ⁻.
⛔ Sobre la amplitud e^{−iωt} los signos se intercambian — no mezclar. (Hasta 2026-09 la fórmula
`(J_1 − iJ_2)` se aplicaba al bin +ω ⇒ un drive `ellip=+1` salía etiquetado "L".) Helper único:
`helicity_components(spectrum, frame)` en `plot_harmonics.py` (devuelve R, L y la componente
**longitudinal**; cumple `|R|²+|L|²+|long|² = |J|²`). El tensor σ/χ usa `to_helicity_basis(T, dim,
frame)` con `M = frame @ conj(U)`, `U = [e_+, e_-, z]` (el `conj` porque el tensor se guarda en convenio FFT).

⛔ **`frame` = `Laser.rotation_matrix()`** (columnas `[xdir, ydir, zdir]`), resuelto por
`_laser_helicity_frame(config)` en `graphics.py`. La helicidad es el espín del fotón proyectado
sobre `zdir`: con el drive inclinado (`thetaz ≠ 0`, p.ej. una muestra (112)) el split en ejes de
laboratorio **mezcla la componente longitudinal en R/L** y da dicroísmo espurio (medido: una
corriente circular pura en el plano del drive sale como `−0.98` en vez de `∓1`).
El helper devuelve `None` cuando `zdir = +ẑ` → ruta legacy **bit-idéntica**; una rotación en el
plano (`phix`) solo cambia una **fase global** de R/L, nunca `|R|`,`|L|`.

⚠️ **Modelos 2D (`dimension < 3`) con el drive fuera del plano:** un tensor de 2 índices no puede
representar esa rotación — recortar el frame a 2×2 daría una transformación **no unitaria** (el mismo
recorte que Antelope deja comentado). `to_helicity_basis` lo detecta y **vuelve a los ejes del
modelo** avisando. Para las corrientes no aplica: `helicity_components` usa siempre el vector 3D
completo. Verificado unitario (norma de Frobenius, 1e-15) en 245 geometrías `thetaz × phiz × phix`,
y con `ellip=±1` la corriente que copia el campo sale 100% en R / 100% en L según `laser.helicity`.

⚠️ `tools/hhg_112_ellipticity_orientation.py` **no** está afectado: ya proyecta sobre `ê1,ê2` del
plano (112) por su cuenta (`plane_112()`).

### Convenio vs. Antelope (contrastar antes de comparar)

**El método es el mismo.** `AlsisPythonShort/AnalysisInterIntra.py:336-350` (y su gemelo
`AnalysisReader_PyDir.py:334-348`) proyectan sobre `trans_x0`,`trans_y0` de `laser.h` — el plano
transverso del drive — antes de formar R/L. Es exactamente `helicity_components(J, frame)`.
Verificado: con `Jz = 0` nuestro `R` == su `dJ_m/√2` y nuestro `L` == su `dJ_p/√2`, bit a bit.

Dos diferencias, ambas a nuestro favor:
- ⛔ En Antelope el término **`− sin(thetaz)·J[2]` está COMENTADO** (línea 341): su rotación no es
  ortonormal si `Jz ≠ 0`. Válido en sus modelos (`Dim = 2`), **mal en 3D con `thetaz ≠ 0`**
  (TaAs/WSM): discrepancia medida **32%** en un caso aleatorio. Descomentándolo coincide con nosotros
  exactamente. Nosotros además devolvemos la longitudinal (`|R|²+|L|²+|long|² = |J|²`); ellos la tiran.
- Ellos ignoran `phix`, nosotros lo incluimos → solo una **fase global**, `|R|`,`|L|` idénticos.

| | QXTI | Antelope |
|---|---|---|
| RCP | `(J_1 + i J_2)/√2` (bin +ω) | `J_1 + i J_2` en `AnalysisInterIntra.py:349` (bin +ω, etiquetado `J_rcp`) ⇒ **misma fórmula, misma etiqueta** |
| normalización | `1/√2` (unitaria) | ninguna |
| observable | `\|J(ω)\|` | `\|−iω·J(ω)\|²` (campo **radiado**, intensidad) → difiere en `ω²` |
| plano | frame del láser, 3D completo | frame del láser, truncado a 2D (ver arriba) |

⚠️ **Antelope no es autoconsistente en el signo**: la familia `PY_ANALYSIS/AnalysisReader_PyDir.py:419`
usa `dJ_p = x − i y` (= nuestro **L**) mientras `AnalysisInterIntra.py:349` usa `+i` (= nuestro R).
Al comparar, fíjate en el script concreto, no en la etiqueta "RCP".

El tensor σ/χ se guarda en el convenio FFT (`σ_fft = conj(σ_fís)`), así que `to_helicity_basis`
rota con `conj(U)` para que la etiqueta `+` sea la helicidad física (mismo criterio que `J_R`).

## ⚠️ El caso "el inter es más alto que el armónico" — ES FÍSICO

Observación típica (WSM, CMD): en armónicos **impares** H3/H5/H7, |inter| y |intra| son ambos
**grandes** y casi se cancelan → |total| pequeño. En **pares** H2/H4/H6, intra≈0 e inter=total.

Esto **no es bug**: es la **interferencia intra/interbanda** de HHG en sólidos (Vampa–Corkum):
- La velocidad de grupo `v_nn` es **impar en k** → el intra solo genera armónicos **impares**.
- En los impares, intra e inter (grandes, signo opuesto) se restan → el total es el **residuo**.
- Por eso el "pico" del interbanda queda por encima del armónico total pequeño.

### De qué gauge depende (matiz, importa al comparar motores)

El split **no** depende del gauge U(1) de banda: bajo `ρ → U†ρU` con `U = diag(e^{iθ_n(k)})`, tanto
`ρ_nn` como `v_nn = ∂_k E_n` son invariantes. Depende de tres cosas:

1. **Gauge electromagnético (longitud vs velocidad).** pfddm/ptddm son longitud (evalúan en `k+A`);
   `tddm` es velocidad (evalúa en `k` y necesita el término diamagnético). Antelope lo hace
   explícito en `SBEs.h:752 GenInterIntraCurrent`: mismo criterio `i==j → intra`, `i≠j → inter`,
   pero en velocity gauge **suma `Nval·A(t)` al intra**. ⇒ compara intra/inter **solo entre motores
   del mismo gauge**; entre gauges solo el **total** es comparable.
2. **Degeneraciones (no-abeliano).** En un nodo de Weyl o una degeneración de Kramers, "diagonal"
   dentro del subespacio degenerado depende de la base → relevante en `taas_spinful_4band` y
   `wsm_two_weyl`.
3. **Cancelaciones grandes** en armónicos impares (arriba) — correcto, pero poco informativo.

⛔ **Distinto y sí genuinamente gauge-dependiente:** las flags `include_intraband` /
`include_interband` ([[qxti.core|config.py]]) NO parten la corriente sino la **fuente**
`D_k ρ = ∂_k ρ − i[A,ρ]` en gradiente + conmutador de Berry. Cada término por separado **no** es
covariante (solo la suma lo es) — por eso `cmd.py` exige ambas activas para entrar en la ruta del
gradiente covariante. Apagar una da un resultado dependiente de la elección de fase. Ver
[[Concept - Perturbative Recursion]].

La versión **gauge-invariante** de verdad (corriente de polarización `∂_t P` vs Drude) sigue **sin
implementar** en los tres motores.

## Claves del dataset

`current_*_intraband/interband` solo aparecen si `current_decomposition_available=True`
(streaming en `band` basis, o mesh con `return_intraband`). [[qxti.graphics]] las lee por nombre.

---

Relacionado: [[Concept - Occupation Gauge Basis]] · [[Concept - Response Engines]] · [[qxti.graphics]]
