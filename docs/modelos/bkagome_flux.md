# BKagomeFlux

`models/bkagome_flux.py` y `models/bkagome_flux_2l.py` son adaptaciones directas
de `BKagomeFlux.h` y `BKagomeFlux2L.h` de Antelope. Exponen el contrato completo
de QXTI: `H`, `H_batch`, `dH_dk`, parámetros por defecto y metadatos de red/BZ.

## Convenciones

- Unidades atómicas: `a0` en Bohr, `k` en Bohr⁻¹ y `ta`, `tb` en Hartree.
- `phi_a`, `phi_b` y `rot` están en radianes.
- Monocapa: base `(A, B, C)` y matriz 3×3.
- Bicapa: base `(A1, B1, C1, A2, B2, C2)` y matriz 6×6.
- El `a0 = 7 Å` de los inputs históricos corresponde a
  `a0 = 13.22808287238039 Bohr` en QXTI.

Para un enlace con vector `d` y signo de flujo `s`, el elemento no diagonal es

```text
h_d(k) = ta exp(i s phi_a/3) exp(-i k·d)
       + tb exp(i s phi_b/3) exp(+i k·d).
```

Los tres enlaces `(AB, AC, BC)` usan `(a3, a2, a1)` y signos `(-,+,-)`, igual
que el código C++.

## Bicapa y `FB`

La bicapa conserva exactamente la implementación original: el segundo bloque
kagome se construye con los enlaces rotados por `rot` y los bloques fuera de la
diagonal son cero. Por tanto, no hay hopping interlayer.

`FB` no aparece en `H`. En Antelope solo seleccionaba el llenado:

| `FB` | Bandas ocupadas (`Nval`) |
| --- | ---: |
| `1` | 2 |
| `-1` | 4 |

El helper `occupied_bands_from_fb()` preserva esa traducción. En una ejecución
de QXTI, la ocupación se controla en `[cmd]` y `[xtp]`; el input de ejemplo para
`FB=1` usa Fermi-Dirac a temperatura cero y coloca el nivel de Fermi en el gap
que sigue al doblete inferior.

## Inputs

```bash
python main.py inputs/inputParams.bkagome_flux.cfg -cmd
python main.py inputs/inputParams.bkagome_flux_2l.cfg -ldos
python main.py inputs/inputParams.bkagome_flux_phi_0.25_pfddm.cfg -cmd
```

El tercer comando reproduce los parámetros físicos del job histórico
`LP_phi_LowF/phi_0.25` —material, pulso gaussiano de siete ciclos, malla
`200x200`, ventana temporal, `T1` y `T2`— usando la expansión perturbativa
`pfddm` hasta tercer orden en lugar de la propagación SBE completa.

La bicapa desacoplada tiene degeneraciones exactas entre capas cuando `rot=0`.
Por eso su input desactiva el guard automático de degeneración y recomienda la
base de trabajo/orbital para la evolución temporal.
