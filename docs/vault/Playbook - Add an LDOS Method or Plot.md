---
tags: [playbook, howto]
updated: 2026-07-21
---

# 🛠️ Playbook — Añadir un método LDOS o un plot

> Dos extensiones frecuentes y de bajo riesgo. [[Home]] · [[qxti.analytics]] · [[qxti.graphics]]

## Añadir un método LDOS/DOS

1. **Cálculo** en [[qxti.analytics|dos.py]]: añade una función `compute_<algo>_spectrum(config)`
   que devuelva `{"dataset": {...}, "runtime_seconds", ...}`. Usa los **mismos pesos BZ** que XTP
   (regla de suma) → [[Concept - BZ Grid and Degeneracy Guard]].
2. **Despacho** en [[qxti.core|ldos_scan.py]]: añade la rama al `if ldos.method == ...`.
3. **Config** en `LDOSConfig` (`method` acepta el valor nuevo; campos extra si hacen falta) →
   [[Playbook - Add a Config Option]].
4. **Gráfica** en [[qxti.graphics|plot_dos.py]]: nueva `plot_<algo>(data, output_path)`.
5. Convierte a eV/Å **solo** en la gráfica.

Métodos actuales: `eigenvalues` (bulk), `surface` (López-Sancho, arcos de Fermi), `finite` (placa,
hoy solo Haldane). Prueba: `pytest tests/test_ldos.py`.

## Añadir un plot

1. Nueva función/método en el `plot_*.py` que corresponda (bandas → `plot_hamiltonian`, HHG →
   `plot_harmonics`, σ/χ → `plot_susceptibility_tensor`, DOS → `plot_dos`, poblaciones →
   `plot_response`).
2. Lee **solo** claves del dataset `.npz` (no calcules física → regla de oro 4).
3. Engánchalo en el despachador [[qxti.graphics|graphics.py]] (`plot_*_from_saved_data`) para que
   se genere al detectar datos de esa familia.
4. Estilo: reusa `apply_paper_style()` si aplica.

⛔ Si necesitas una cantidad que el dataset no tiene, créala **antes** en [[qxti.data]]/
[[qxti.response]], no en la gráfica.

---

Relacionado: [[qxti.graphics]] · [[qxti.analytics]] · [[Playbook - Invariants Not to Break]]
