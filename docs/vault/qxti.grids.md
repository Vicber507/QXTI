---
tags: [package, grids]
updated: 2026-07-21
---

# 📦 qxti.grids — mallas k / tiempo / frecuencia

> Discretización pura. **Hojas** (no importan nada de `qxti`). [[Home]] · [[Architecture Map]]

**Usado por:** [[qxti.physics|band_gauge]], [[qxti.response]] (CMD, XTP), [[qxti.core]]

## Archivos

| Archivo | Rol | Símbolos |
| --- | --- | --- |
| [kgrid.py](../../qxti/grids/kgrid.py) | Malla en espacio recíproco | `KGrid`, `periodic_axis`, `shifted_axis` |
| [timegrid.py](../../qxti/grids/timegrid.py) | Malla temporal + FFT/ventanas | `TimeGrid`, `from_dt`, `frequency_axis` |
| [frequencygrid.py](../../qxti/grids/frequencygrid.py) | Eje de frecuencia/armónicos | `FrequencyGrid`, `harmonic_axis` |

## KGrid

`uniform()` construye la malla; `shape`, `points()`, `mesh()`. `shifted=True` usa offset
Monkhorst-Pack 0.5 (puntos medios, cuadratura de punto medio), que **nunca cae** en el borde
ni el centro de la BZ — clave para esquivar degeneraciones de Weyl/Dirac.

⛔ **Invariantes**
- `dimension ∈ {1,2,3}`; ejes inactivos deben tener 1 punto.
- La **simetría k→−k** de una malla `shifted` (offset exacto 0.5) es sagrada: la rompe un offset
  irracional (que sí esquiva todo punto racional, pero pierde la cancelación por simetría).
- Los **pesos de cuadratura** deben coincidir con los de [[qxti.response|XTP]] y [[qxti.analytics|dos]]
  (Simpson para N impar uniforme, trapezoidal, o punto-medio si shifted). Ver
  [[Concept - BZ Grid and Degeneracy Guard]].

## TimeGrid

`from_dt(t_min,t_max,dt)` fija `Nt`. `apply_window` (hann/hamming/blackman/none) antes de FFT;
`padded_signal` (zero-padding ×`padding_factor`); `frequency_axis` = `2π·fftfreq` (respeta padding).
⛔ El eje ω de todo el HHG sale de aquí — no recalcular a mano.

## FrequencyGrid

Mapea FFT→frecuencias físicas y arma `harmonic_axis` (n·ω₀). Usado por análisis espectral.

➕ **Extender:** nueva ventana FFT → `TimeGrid.apply_window`; nueva regla de malla k →
`KGrid.periodic_axis(offset=…)`. El **degeneracy guard** que elige N vive en
[[qxti.core|simulation.build_kgrid]], no aquí.

---

Relacionado: [[Concept - BZ Grid and Degeneracy Guard]] · [[qxti.response]]
