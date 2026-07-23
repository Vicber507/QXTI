---
tags: [models, inputs]
updated: 2026-07-21
---

# 🧬 Models and Inputs

> Los Hamiltonianos (`models/*.py`) y sus configs (`inputs/*.cfg`). Un modelo + un `.cfg`
> alimentan los 3 cálculos. [[Home]] · [[qxti.physics]] · [[Playbook - Add a Model]]

## Contrato de un modelo (`models/<x>.py`)

Un módulo cargado por [[qxti.physics|CustomHamiltonian]] debe exponer:
- `H(kx, ky, kz, params) -> (nb, nb) complex` — el Hamiltoniano (en eV/Å; el wrapper pasa a a.u.).
- Opcional pero recomendado: `MODEL_NAME`, `BASIS_SIZE`, `DIMENSION`, `BZaxis`/`default_lattice`,
  `default_params()`, y **`H_batch(kpts, params)`** vectorizado (los motores mesh lo usan; debe ser
  **bit-exacto** vs `H`), y `dH_dk(...)` analítico si lo tienes.

⛔ Unidades atómicas al final ([[Concept - Atomic Units]]). `H(k)` hermítico.

## Modelos existentes (`models/`)

| Modelo | Qué es | Notas |
| --- | --- | --- |
| `graphene`, `graphene_bilayer` | Grafeno mono/bicapa | casos de prueba 2D |
| `haldane` | Haldane 2D | topológico; único con LDOS `finite` |
| `wsm_two_weyl` | Semimetal de Weyl (2 nodos) | el WSM "de trabajo" del mapa de elipticidad |
| `wsm_orenstein` | WSM (Orenstein) | arcos de Fermi, susceptibilidad |
| `frank_zhang_8band` | TaAs Weyl 8 bandas (Frank-Zhang 2017) | `BZaxis=diag(4π,4π,8π)`; nodos en kz=±4π; converge lento (ver memoria) |
| `taas_tb` | TaAs tight-binding | χ⁽²⁾ zzz |
| `natphys_4band` | modelo 4 bandas | |
| `bi2se3_surface` | superficie Bi₂Se₃ | ver [[bi2se3_surface|nota en docs/modelos]] |
| `example_hamiltonian_template` | plantilla | punto de partida |

## Inputs (`inputs/inputParams.<x>.cfg`)

`frank8`, `frank_zhang_ldos`, `graphene`, `graphene_bilayer`, `haldane_topological`,
`haldane_trivial`, `taas_tb`, `tblg`, `wsm`, `wsm_orenstein`.

Secciones: `[hamiltonian]` (obligatoria), `[hamiltonian_plots]`, `[kgrid]`, `[timegrid]`,
`[laser]`, `[cmd]`, `[cmd_plots]`, `[xtp]`, `[ldos]`. El *flag* elige qué corre; un mismo `.cfg`
sirve para los 3. Detalle de campos en [[qxti.core|config.py]].

## Uso

```bash
python main.py inputs/inputParams.wsm.cfg -cmd      # HHG / armónicos
python main.py inputs/inputParams.wsm.cfg -xtp      # tensores σ/χ
python main.py inputs/inputParams.frank8.cfg -ldos  # DOS / arcos de Fermi
python qxti/graphics/graphics.py inputs/inputParams.wsm.cfg   # plotea lo que haya
```

**Motor de respuesta** (`[cmd] response_method` / `[xtp] susceptibility_method`):
`pfddm` (=theory) | `ptddm` (=simulation) | `tddm` (full no-perturbativo) | `both` | `all`.
Ver [[Concept - Response Engines]]. Para campo fuerte usa `tddm`; para comparar los 3, `all`.

## Convergencia / validez perturbativa

Cada modelo tiene su malla-k, dt, E0, ω máximos para seguir siendo perturbativo. `tools/
convergencia_perturbativa.py` ayuda. Los WSM gapless (frank8) necesitan mallas enormes cerca del
nodo — a veces la "no convergencia" es física, no un bug (ver memoria del proyecto).

➕ **Añadir un modelo:** [[Playbook - Add a Model]]. **Añadir un input:** copia un `.cfg` similar
y ajusta `[hamiltonian] source_file` + `[kgrid]`/`[laser]`.

---

Relacionado: [[qxti.physics]] · [[Concept - BZ Grid and Degeneracy Guard]] · [[Concept - Atomic Units]]
