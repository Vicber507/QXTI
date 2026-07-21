---
tags: [concept, performance, memory]
updated: 2026-07-21
---

# 🧠 Concept — RAM guard y paralelismo

> Cómo QXTI corre mallas grandes (p.ej. 120³ de 8 bandas) sin apagar la máquina, y cómo
> paraleliza. [[Home]] · [[qxti.utils]] · [[qxti.analytics]] · [[qxti.response]]

## RAM guard — dejar ≥1 GB libre (mac/win/linux)

[[qxti.utils|memory.py]] mide la RAM disponible de forma multiplataforma (psutil → `/proc/meminfo`
Linux, `vm_stat` macOS, `GlobalMemoryStatusEx` Windows). Los motores lo usan así:
- `memory_budget_bytes(reserve_gb)` — cuánto puedo pedir dejando `reserve_gb` libre.
- `pick_block_count(n_units, bytes_per_unit, reserve_gb, halo_units)` — cuántos planos-k por bloque.
- `ensure_headroom(need_bytes, reserve_gb)` — chequea; si falta, `gc.collect()` y recheck; si aún
  falta, `MemoryGuardError`.

⛔ **Invariante:** `reserve_gb` (config: `[cmd]/[ldos] reserve_gb`, default 1.0) **no se viola**.
Si la RAM no es medible, se **permite** (no bloquea). Es la solución permanente al "se apaga sola".

## Streaming por bloques de k (el truco de RAM)

[[qxti.analytics|mesh_response.harmonic_currents_meshed]] procesa la malla en **bloques de planos-k**
con `halo = max_order − 1` (los planos extra que necesita el gradiente). El **interior** de cada
bloque es **bit-exacto** vs la malla completa. Así 120³×8 bandas pasó de ~39 GB (thrashing) a un
"piso" de ~2 GB. `_plan_stream` decide el tamaño de bloque con el budget de RAM.

En [[qxti.response|CMD]], la ruta *streaming* acumula J(t) con
[[qxti.data|StreamingCurrentAccumulator]] y **no** escribe el ρ del último orden a disco; el scratch
de ρ puede ir en `float16_complex` (4 bytes/elem, ver [[qxti.utils|io_utils]]).

## Paralelismo (por dónde)

QXTI es **memoria compartida de un solo nodo** (multi-hilo), **no MPI** (a diferencia de
antelope, que es `mpirun` multi-nodo). Un proceso QXTI usa **todos los cores de su nodo**.

| Cálculo | Unidad paralela | Mecanismo |
| --- | --- | --- |
| mesh / theory (`-cmd`, `-xtp` o≥2) | **slabs** (bloques de planos-k) + halo, freq×dir | **ThreadPool** (NumPy suelta el GIL) |
| CMD tiempo | chunks de k | ThreadPool |
| `-xtp` simulation | **por frecuencia** | **ProcessPool** (cada worker: CMD+XTP en scratch, k-loop a 1 hilo) |

## Conteo de cores — fuente única: `qxti/utils/parallel.py`

`resolve_worker_count(requested, cap)` decide cuántos workers, con esta prioridad:
1. `n_workers` del `.cfg` si **>0** (config gana), 2. `QXTI_NUM_WORKERS`, 3. **SLURM**
(`SLURM_CPUS_PER_TASK` — completo, **nunca a la mitad**), 4. **todos los cores usables** (el
máximo local: `os.sched_getaffinity` en Linux —respeta cgroups/taskset—, `os.cpu_count()` en
mac/win). Es decir `n_workers=0` ⇒ **todos** los cores de la PC.
**Opt-in** (no default): `QXTI_MAC_PERF_CORES=1` limita a los performance cores en Apple Silicon
(a veces más rápido: e-cores + GIL).

Los 3 sitios (`cmd._default_worker_count`, `mesh_response.default_worker_count`,
`susceptibility._resolve_n_workers`) delegan aquí → mismo comportamiento en laptop, workstation y
cluster. `main.py` imprime el plan al arrancar (`[main] Parallelism: N workers (source: …)`).

⛔ **Invariantes**
- El resultado **no** depende de `n_workers` (paralelismo bit-exacto: los slabs llevan halo).
  Si cambia con los workers, hay bug de acumulación/halo.
- No sobre-suscribir: el worker de proceso (`-xtp`) fuerza `n_workers=1` internamente, y las
  librerías BLAS se **pinchan a 1 hilo** (`configure_thread_env`, `OMP/MKL/OPENBLAS_NUM_THREADS=1`)
  para que el pool de k sea el único que paraleliza. Override con `QXTI_BLAS_THREADS`.
- En cluster una asignación de N cores significa N (se usa entera). Ver [[Cluster and SLURM]].

## Cross-platform (mac/win/linux)

- RAM guard nativo por SO (`memory.py`). Cache de matplotlib vía `tempfile.gettempdir()` (respeta
  `$TMPDIR` de SLURM) — antes era `/private/tmp` (solo-macOS, rompía en Linux/Windows).
- Windows: `-xtp` usa ProcessPool con `spawn`; `main.py` tiene el guard `if __name__=="__main__"`.

---

Relacionado: [[qxti.utils]] · [[Concept - Perturbative Recursion]] · [[Playbook - Invariants Not to Break]]
