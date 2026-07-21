---
tags: [playbook, howto]
updated: 2026-07-21
---

# 🛠️ Playbook — Añadir un modelo (Hamiltoniano)

> Meta: un `models/<x>.py` nuevo que funcione en `-cmd`/`-xtp`/`-ldos` sin tocar el núcleo.
> [[Home]] · [[qxti.physics]] · [[Models and Inputs]]

## Pasos

1. **Crea `models/<x>.py`.** Guíate de `models/example_hamiltonian_template.py` o de uno cercano
   (`wsm_two_weyl`, `frank_zhang_8band`). Debe exponer:
   - `H(kx, ky, kz, params) -> (nb, nb) complex` (en eV/Å; convierte a **a.u.** al final →
     [[Concept - Atomic Units]]).
   - `MODEL_NAME`, `BASIS_SIZE`, `DIMENSION`.
   - `default_params()` y `BZaxis`/`default_lattice()` (caja recíproca; para modelos periódicos
     la caja debe ser la **periodicidad exacta** de la matriz).
2. **(Muy recomendado) `H_batch(kpts, params) -> (nk, nb, nb)`** vectorizado. Los motores mesh lo
   usan para grids de millones de puntos. ⛔ Debe ser **bit-exacto** vs `H` escalar (compáralo con
   `np.allclose(..., atol=0)` en un puñado de k).
3. **(Opcional) `dH_dk(kx,ky,kz,dir,params)` analítico** — si no, se usa FD con `dk_derivative`.
4. **Crea `inputs/inputParams.<x>.cfg`.** Copia uno similar; ajusta `[hamiltonian] source_file`,
   `[kgrid]` (dimensión, puntos, `shifted`), `[laser]`, `[cmd]`/`[xtp]`/`[ldos]`.
5. **Prueba de humo:**
   ```bash
   python main.py inputs/inputParams.<x>.cfg -cmd
   python qxti/graphics/graphics.py inputs/inputParams.<x>.cfg
   ```

## Checklist de invariantes

- [ ] `H(k)` **hermítico** (`validate_hermiticity`).
- [ ] Unidades **a.u.** a la salida.
- [ ] `H_batch` bit-exacto vs `H`.
- [ ] La malla no cae sobre nodos (deja `auto_degeneracy_guard=true`) →
      [[Concept - BZ Grid and Degeneracy Guard]].
- [ ] `distribution` correcta (semimetal a neutralidad ⇒ `valence_occupation`) →
      [[Concept - Occupation Gauge Basis]].

## Dónde NO tocar

- No edites `qxti/physics/custom_hamiltonian.py` para un modelo concreto: todo va en `models/<x>.py`.
- No metas conversiones de unidad en el núcleo.

## Prueba formal

Añade un test tipo `tests/test_<x>_model.py` (guíate de `tests/test_graphene_bilayer_model.py`):
verifica hermiticidad, `BASIS_SIZE`, y `H_batch` vs `H`. Corre `pytest tests/ -k <x>`.

---

Relacionado: [[Playbook - Add a Config Option]] · [[Playbook - Invariants Not to Break]]
