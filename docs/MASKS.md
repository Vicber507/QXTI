# Máscaras de zona de Brillouin en QXTI

Cómo se implementan las máscaras de la BZ, con las ecuaciones, dónde entran en las
integrales, su generalización a 3D, y por qué **solo** usamos la apodización radial del
borde (la de Γ está eliminada porque borra la señal Hall).

---

## 1. Dónde entra una máscara: la integral de BZ

Todo observable (corriente, susceptibilidad, AHC) es una integral sobre la zona de
Brillouin de dimensión `D`:

$$
O \;=\; \int_{\mathrm{BZ}} \frac{d^{D}k}{(2\pi)^{D}}\; o(\mathbf k)
\;\;\xrightarrow{\text{malla}}\;\;
O \;\approx\; \sum_{\mathbf k}\, \frac{(\Delta k)^{D}}{(2\pi)^{D}}\; o(\mathbf k)
\;=\; \sum_{\mathbf k} w^{\mathrm{BZ}}_{\mathbf k}\, o(\mathbf k),
\qquad
w^{\mathrm{BZ}}_{\mathbf k}=\frac{(\Delta k)^{D}}{(2\pi)^{D}} .
$$

Una **máscara** es un peso multiplicativo extra $W(\mathbf k)\in[0,1]$ sobre cada punto:

$$
\boxed{\,O_{\text{mask}} \;=\; \sum_{\mathbf k}\; W(\mathbf k)\; w^{\mathrm{BZ}}_{\mathbf k}\; o(\mathbf k)\, }
$$

En QXTI, `W(k)` multiplica los **pesos de integración** de la respuesta
(`build_k_integration_weights`), así que aparece dentro de **toda** suma sobre `k`:
corrientes $J_i(t)=-\sum_{\mathbf k} W(\mathbf k)\,w^{\mathrm{BZ}}_{\mathbf k}\,\mathrm{Tr}[\rho\,\hat v_i]$,
susceptibilidades $\sigma^{(n)}$, y la AHC del pump-probe.

Para el **pump-probe AHC** (probe DC $E_p$, diferencia central):

$$
\sigma_{xy}(t) \;=\; \frac{1}{2E_p}\sum_{\mathbf k} W(\mathbf k)\,\frac{(\Delta k)^{D}}{(2\pi)^{D}}\,
\Big(\, -\mathrm{Tr}\big[\big(\rho_{+E_p}-\rho_{-E_p}\big)\,\hat v_y(\mathbf k+\mathbf A(t))\big]\Big).
$$

---

## 2. Máscara A — apodización radial del borde de BZ  (**la que usamos**)

### Propósito
Suavizar la contribución de los `k` grandes (borde de zona), donde (i) el modelo de
2 bandas de superficie deja de valer y (ii) el espectro tiene ruido/leakage. **No toca
el centro Γ** — el peso en Γ es 1.

### Ecuación (forma correcta: super-gaussiana de orden 8, techo plano)

$$
\boxed{\,W_{\text{BZ}}(\mathbf k) \;=\; \exp\!\left[-\left(\frac{|\mathbf k|}{k_0}\right)^{8}\right]\,}
\qquad
|\mathbf k| = \sqrt{\textstyle\sum_{a=1}^{D} k_a^{2}} .
$$

El exponente 8 da un **techo casi plano** hasta $\sim k_0$ y una **caída de pared** (C¹,
sin corte duro → sin ringing en la FFT). El radio característico $k_0$ se fija en unidades
absolutas (ver §4): $k_0 = 0.2651$ a.u. para Bi₂Se₃ (= 0.444·kref del Fortran).

**Perfil** (a. u.):

| $|\mathbf k|$ | 0 | 0.10 | 0.20 | 0.2651 | 0.30 | 0.35 |
|---|---|---|---|---|---|---|
| $W_{\text{BZ}}$ | 1.000 | 0.9996 | 0.900 | 0.368 | 0.068 | 0.0001 |

### Implementación actual en QXTI (a mejorar)
`qxti/analytics/theory_response.py:94-112` (`_bz_radial_mask_weights`) y su duplicado
`qxti/response/xtp.py`:

```python
radial_distance = sqrt( sum_{a=0}^{dimension-1} k_a^2 )     # ya D-dimensional
weights = exp(-0.5 * (radial_distance / sigma)^2)          # <-- gaussiana ORDEN 2
weights = where(radial_distance <= radius, weights, 0.0)   # <-- corte DURO en 'radius'
```

Diferencias con la forma correcta (documentadas para arreglar):
- Es **orden 2** (gaussiana), no orden 8 → no tiene techo plano; empieza a decaer desde Γ.
- El `np.where(... , 0.0)` mete un **salto** ($\sim$0.61 → 0) en $|\mathbf k|=\text{radius}$
  → **ringing** en la FFT. La forma orden-8 no necesita corte (decae sola).
- Parámetros: `bz_mask_radius_percent`, `bz_mask_sigma` (`qxti/core/config.py`).

> **Objetivo:** reemplazar por `exp(-(|k|/k0)**8)` sin `np.where`, con `k0` absoluto.

### ✔ Ya está extendida a 3D
La línea `radial_distance = sqrt(sum over range(dimension) of coord^2)` recorre
`dimension` ejes, así que para `dimension=3` calcula
$|\mathbf k|=\sqrt{k_x^2+k_y^2+k_z^2}$ automáticamente; y el radio de referencia
(`_bz_mask_reference_radius`) toma el `min` sobre los `dimension` semianchos de la caja.
**No hace falta tocar nada para 3D — el proceso ya es D-dimensional.** (Para una
superficie 2D como Bi₂Se₃, `dimension=2` y se reduce a $\sqrt{k_x^2+k_y^2}$.)

---

## 3. Máscara B — hueco en Γ (`kmask2D`)  (**ELIMINADA — borraba la señal**)

### Qué era
Un hueco gaussiano **centrado en Γ**, aplicado como factor $(1-K)$ sobre la curvatura de
Berry $\Omega$ y los dipolos:

$$
K(\mathbf k) = f_0\,\exp\!\left(-\frac{|\mathbf k|^2}{w^2}\right),\qquad
\Omega(\mathbf k)\;\to\;\big(1-K(\mathbf k)\big)\,\Omega(\mathbf k),\qquad
w = \sigma_0\cdot\frac{4\pi}{3\,a_0}=0.00535\text{ a.u.}
$$

### Por qué la eliminamos (mata la AHC)
En Γ ($\mathbf k=0$): $K=f_0=1 \Rightarrow (1-K)=0 \Rightarrow \Omega(\Gamma)=0$.

| $|\mathbf k|$ | 0 (Γ) | 0.002 | 0.0054 | 0.02 |
|---|---|---|---|---|
| $(1-K)$ sobre $\Omega$ | **0.000** | 0.130 | 0.632 | 1.000 |

La curvatura de Berry del cono de Dirac de Bi₂Se₃ está **concentrada en Γ** (flujo
$-\pi$ casi singular). La velocidad anómala es $\mathbf v^{\text{an}}=-\mathbf E\times\Omega$;
si $\Omega(\Gamma)=0$, **no hay corriente Hall del cono** → el hueco borra exactamente la
señal que queremos medir.

### Por qué el Fortran la tenía (y por qué nosotros NO la necesitamos)
El Fortran calcula la corriente con **dipolos analíticos** $p_{cv}\propto 1/\mathrm{gap}$,
que **divergen** en el punto de Dirac (gap→0 en Γ). El hueco tapa esa divergencia
(necesario para el HHG). Pero QXTI calcula la corriente con el **operador velocidad**

$$
\hat v_a(\mathbf k) = \frac{\partial H(\mathbf k)}{\partial k_a}
\quad(\text{diferencias finitas}),
$$

que es **finito y suave en Γ** — no diverge. Por eso **no necesitamos regularizar Γ**:
sin hueco, la AHC de Γ se conserva.

| | dipolo analítico (Fortran) | $\hat v=\partial H/\partial k$ (QXTI) |
|---|---|---|
| ¿diverge en Γ? | sí ($1/\mathrm{gap}$) | no |
| ¿necesita hueco en Γ? | sí | **no** |
| efecto en la AHC | **la borra** | **la preserva** |

**Conclusión: la máscara B (hueco de Γ) queda eliminada del código.** No existe en el core
de QXTI (`qxti/`); solo estaba, apagada (`f0=0`), en la herramienta de comparación
`tools/bi2se3_pumpprobe_fortmask.py` — ya removida de ahí también.

---

## 4. Convención de unidades del radio (trampa al portar)

En el Fortran la super-gaussiana es `exp(-b·(a·kpar·kref)^8)`, con `a,b` números puros y
`kref` el tamaño de caja. Reordenando queda `exp(-(k/k0)^8)` con

$$
k_0 = a^{-1}\, b^{-1/8}\, k_{\text{ref}}^{-9/8}.
$$

$k_0$ escala como $k_{\text{ref}}^{-9/8}$ (¡inverso al tamaño de caja!). **Copiar `a,b`
directo a otra caja da una máscara inerte.** Hay que portar **$k_0$ en unidades absolutas**
(0.2651 a.u.), no `a,b`. Contra el `kref` de QXTI (0.4016) eso equivale a
`bz_mask_radius_percent ≈ 66`.

---

## 5. Resumen

- **Se usa solo la Máscara A** (apodización radial del borde): `exp(-(|k|/k0)^8)`, con Γ
  intacto (peso 1). Ya es **D-dimensional** (2D/3D automático vía `range(dimension)`).
- **La Máscara B (hueco de Γ) está eliminada**: borraba $\Omega$ en Γ, que es donde vive la
  AHC del cono. No hace falta porque usamos $\hat v=\partial H/\partial k$ (finito en Γ).
- Pendiente opcional: subir la Máscara A de orden 2 a **orden 8** y quitar el corte duro
  `np.where` en `_bz_radial_mask_weights` para evitar ringing.
