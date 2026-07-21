---
tags: [package, data]
updated: 2026-07-21
---

# 📦 qxti.data — contenedores, streaming, I/O

> Ensamblado de observables y serialización `.npz`/`.npy`. **No** calcula física nueva.
> [[Home]] · [[Architecture Map]] · [[Data Flow]]

**Depende de:** [[qxti.physics]], [[qxti.response]], [[qxti.utils|io_utils]]
**Usado por:** [[qxti.core]] (runners), [[qxti.graphics]]

## Archivos

| Archivo | Rol | Símbolos |
| --- | --- | --- |
| [io.py](../../qxti/data/io.py) | Guardar/cargar datasets `.npz` (+ meta JSON) y ρ `.npy/.dat` | `save_dataset_npz`, `load_dataset_npz`, `load_rho_orders_*` |
| [streaming_current.py](../../qxti/data/streaming_current.py) | Acumula J(t)/P(t) intra/inter al vuelo | `StreamingCurrentAccumulator` |
| [harmonic_data.py](../../qxti/data/harmonic_data.py) | Dataset HHG (FFT, intra/inter) desde XTP | `HarmonicData` |
| [hamiltonian_data.py](../../qxti/data/hamiltonian_data.py) | Bandas/velocidades desde el Hamiltoniano | `HamiltonianData` |
| [response_data.py](../../qxti/data/response_data.py) | Poblaciones/coherencias desde CMD | `ResponseData` |
| [susceptibility_data.py](../../qxti/data/susceptibility_data.py) | Tensores σ/χ(ω) desde XTP | `SusceptibilityData` |
| exporters.py / loaders.py / pdg.py | ⚠️ **stubs vacíos** | — |

## StreamingCurrentAccumulator (clave para RAM)

Thread-safe. `make_callback(order)` se pasa a `CMD.solve_time_domain` para acumular, por k y sin
guardar ρ completo:
- **Total:** `J_i(t) = einsum("mn,tnm->t", v_i, ρ)`
- **Intra:** `J_intra_i(t) = einsum("n,tn->t", diag(v_i), diag(ρ))`
- **Inter:** `J − J_intra`

Es la **misma definición** que `XTP.current_decomposition` y `mesh_response` (traza diagonal =
intra). Ver [[Concept - Inter-Intra Decomposition]].

## io.py — esquema del dataset

`.npz` = arrays `.npy` + metadatos JSON en `__meta_json__` (`allow_pickle=False`). Claves típicas
de HHG: `omega_axis`, `current_spectrum`, `current_time`, y los splits
`current_*_intraband/interband` (solo si `current_decomposition_available=True`). ρ en
`(Nk, Nt, Nb, Nb)`; soporta layout comprimido `float16_complex` (ver [[qxti.utils|io_utils]]).

⛔ **Invariantes**
- No renombres claves del dataset sin actualizar [[qxti.graphics]] (que las lee por nombre).
- Los metadatos deben ser JSON-serializables (`_jsonify` cubre np.generic/Path/dict/list).

## ⚠️ PDG está vacío

El viejo [[ARCHITECTURE]] habla de `PDG` como organizador de datos, pero `pdg.py`/`exporters.py`/
`loaders.py` son **stubs**. Hoy la organización de salidas vive en
[[qxti.core|config.with_standard_output_dirs]] + los runners + [[qxti.graphics|graphics.py]].
Si algún día se centraliza, este es el lugar (sin romper: los runners ya escriben con `save_dataset_npz`).

➕ **Extender:** nuevo observable → método en `HarmonicData`/`ResponseData`; nuevo layout →
[[qxti.utils|io_utils]].

---

Relacionado: [[Concept - Inter-Intra Decomposition]] · [[qxti.graphics]] · [[Concept - Memory and Parallelism]]
