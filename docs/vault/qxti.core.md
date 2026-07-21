---
tags: [package, orchestration]
updated: 2026-07-21
---

# 📦 qxti.core — orquestación

> Config tipada + los tres *runners*. Coordina el flujo pero **no** contiene física.
> [[Home]] · [[Architecture Map]] · [[Data Flow]]

**Depende de:** [[qxti.physics]], [[qxti.grids]], [[qxti.solvers]], [[qxti.response]],
[[qxti.analytics]], [[qxti.data]], [[qxti.utils]]
**Usado por:** `main.py`, [[qxti.graphics]], [[qxti.analytics]] (theory_response/dos reusan sus *builders*)

## Archivos

| Archivo | Rol | Símbolos clave |
| --- | --- | --- |
| [config.py](../../qxti/core/config.py) | INI `.cfg` → dataclasses tipadas | `QXTIConfig`, `*Config`, `from_file`, `with_standard_output_dirs` |
| [simulation.py](../../qxti/core/simulation.py) | Runner `-cmd` (HHG/tiempo) | `QXTISimulation.run` |
| [susceptibility_scan.py](../../qxti/core/susceptibility_scan.py) | Runner `-xtp` (σ/χ) | `SusceptibilityScanRunner.run` |
| [ldos_scan.py](../../qxti/core/ldos_scan.py) | Runner `-ldos` (DOS) | `LDOSRunner.run` |
| [results.py](../../qxti/core/results.py) | ⚠️ **stub vacío** | — (los resultados son `dict[str,Path]`/`dict[str,ndarray]`) |

## config.py — el contrato de entrada

`QXTIConfig` agrupa: `hamiltonian`, `hamiltonian_plots`, `kgrid`, `timegrid`, `laser`,
`cmd`, `cmd_plots`, `xtp`, `ldos`, `susceptibility_solver`, `source_path`.

- `from_file()` usa `configparser` con `inline_comment_prefixes=("#",)`. Cada `[sección]`
  tiene su `_parse_<sección>_section()`. Claves con puntos (`param.a.b`) → dicts anidados.
- `with_standard_output_dirs()` reescribe **solo rutas por defecto** (`outputs/cmd` →
  `outputs/<run>/cmd`); rutas explícitas del usuario se respetan. Es **idempotente**.
- `[susceptibility_scan]` y `[susceptibility_solver]` se leen por compatibilidad y se
  remapean a `[xtp]` (los parámetros del solver del sweep viven **dentro de `[xtp]`**).

Los campos completos de cada dataclass están en el propio `config.py`. Los más "peligrosos":
`response_method`/`susceptibility_method`/`method` (case-insensitive, set fijo o `ValueError`),
`n_workers` (0=auto), `reserve_gb` (RAM guard). Ver [[Concept - Memory and Parallelism]].

## simulation.py — QXTISimulation (`-cmd`)

`.run()` ⇒ construye Hamiltoniano → datasets de bandas (si `hamiltonian_plots`) → `generate_cmd_outputs()`.

`generate_cmd_outputs()` despacha por `cmd.response_method`:
- `theory` → `_generate_hhg_theory()` → **lazy** `from qxti.analytics.theory_response import compute_hhg_spectrum`
- `simulation` → pipeline `CMD` en el tiempo (+ streaming) 
- `both` → ambos, reporta *speedup*

*Builders* reutilizables (también los usa `theory_response`): `build_hamiltonian`, `build_kgrid`
(con **degeneracy guard**, ver [[Concept - BZ Grid and Degeneracy Guard]]), `build_timegrid`,
`build_laser_system`, `build_cmd`, `build_xtp`, `_build_solver` (registra RKF45/AB2).

⛔ **Invariantes**
- El import de `compute_hhg_spectrum` **debe** quedar dentro de la función (lazy) — evita el
  ciclo `simulation ↔ theory_response`. Ver [[Concept - Response Engines]].
- Streaming requiere `cmd.basis == "band"` + `band_gauge_frame` no-None.
- El eje de frecuencia siempre sale de `cmd.timegrid.frequency_axis()` (dt, Nt, padding, ventana).

➕ **Extender:** nuevo tipo de plot de bandas → `generate_hamiltonian_datasets()`; nuevo solver →
`_build_solver()`; nueva rama de método → `generate_cmd_outputs()`.

## susceptibility_scan.py — SusceptibilityScanRunner (`-xtp`)

Despacha por `xtp.susceptibility_method` (theory/simulation/both). En `simulation`, paraleliza
**por frecuencia** con `ProcessPoolExecutor`; cada worker corre `CMD`+`XTP` en scratch temporal
(RAM segura). En `theory`, llama `compute_susceptibility_spectrum` de una sola vez sobre todo el eje ω.
⛔ Requiere input mono-pulso (`laser.pulses == []`) y `susceptibility_enabled=true`.
Ver [[Data Flow]].

## ldos_scan.py — LDOSRunner (`-ldos`)

Capa fina: elige `ldos.method` (`eigenvalues`/`surface`/`finite`) y delega en
[[qxti.analytics|dos.compute_dos_spectrum]]; guarda `ldos.npz` y reporta la regla de suma.

## Gotchas

- `results.py` está vacío: no hay contenedores fuertemente tipados; se usan dicts. Si quieres
  tipos fuertes, este es el lugar natural (sin romper nada, los runners devuelven `dict[str,Path]`).
- `run_name()` deriva de `inputParams.<x>.cfg` → `<x>`; controla el nombre de la carpeta de salida.

---

Relacionado: [[Playbook - Add a Config Option]] · [[Playbook - Add an Observable or Order]] · [[qxti.response]] · [[qxti.analytics]]
