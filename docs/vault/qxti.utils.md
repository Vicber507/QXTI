---
tags: [package, utils]
updated: 2026-07-21
---

# 📦 qxti.utils — RAM guard, I/O de arrays, progreso

> Utilidades transversales. **Hojas**. [[Home]] · [[Architecture Map]]

**Usado por:** casi todos (memory por los motores, io_utils por CMD/data, progress por los runners).

## Archivos

| Archivo | Rol | Símbolos |
| --- | --- | --- |
| [memory.py](../../qxti/utils/memory.py) | **RAM guard** multiplataforma | `available_bytes`, `memory_budget_bytes`, `pick_block_count`, `ensure_headroom`, `MemoryGuardError` |
| [io_utils.py](../../qxti/utils/io_utils.py) | memmaps `.npy`, layout `float16_complex` | `open_array_npy`, `save_array_npy`, `read_complex_slice`, `expand_rho_tensor_time_axis` |
| [progress.py](../../qxti/utils/progress.py) | Progreso + ETA | `ProgressTimer`, `DetailedProgressReporter`, `format_duration` |
| constants.py / math_utils.py / validators.py | ⚠️ **stubs vacíos** | — |

## memory.py — dejar ≥1 GB libre (mac/win/linux)

Cadena de detección: `psutil` → nativo (`/proc/meminfo` Linux, `vm_stat` macOS,
`GlobalMemoryStatusEx` Windows) → fallback. API que usan los motores:
- `memory_budget_bytes(reserve_gb=1.0, fraction=0.9)` → bytes que puedes pedir dejando `reserve_gb` libre.
- `pick_block_count(n_units, bytes_per_unit, reserve_gb, halo_units)` → `(units_per_block, n_blocks)`.
- `ensure_headroom(need_bytes, reserve_gb, label)` → `True`/`gc.collect()`+recheck/`MemoryGuardError`.

⛔ **Invariante:** `reserve_gb` nunca se viola a propósito. Si la RAM no es medible (`≤0`),
`ensure_headroom` **permite** (no bloquea). Ver [[Concept - Memory and Parallelism]].

## io_utils.py — almacenamiento compacto de ρ

`float16_complex` = layout `(*shape, 2)` float16 (4 bytes/elem vs 16 de complex128) para el ρ de
scratch. `expand_rho_tensor_time_axis` difunde `(Nk,1,Nb,Nb)` (equilibrio) a `(Nk,Nt,Nb,Nb)`.
⚠️ El ε de float16 (~1e-3) casa con la tolerancia del solver, pero es arriesgado para órdenes altos.

## progress.py

`DetailedProgressReporter` emite con *throttle* (`interval_seconds`, 15 s por defecto) y ETA — es
el "(floquet)"/timer que ves en corridas largas.

➕ **Extender:** nuevo backend de progreso → `emit` callable; nuevo layout de disco → aquí.
`constants.py` sigue vacío: las conversiones de unidad viven hoy en los modelos/gráficas
(ver [[Concept - Atomic Units]]); si centralizas constantes, este es el sitio.

---

Relacionado: [[Concept - Memory and Parallelism]] · [[Concept - Atomic Units]]
