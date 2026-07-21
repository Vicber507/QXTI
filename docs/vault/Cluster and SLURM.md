---
tags: [playbook, cluster, slurm, performance]
updated: 2026-07-21
---

# 🛠️ Cluster and SLURM

> Cómo correr QXTI en un cluster (miles de núcleos) y cómo se elige el número de cores.
> [[Home]] · [[Concept - Memory and Parallelism]] · archivos en [cluster/](../../cluster/)

## Modelo de paralelización (importante)

QXTI = **memoria compartida, un solo nodo, multi-hilo** — **no MPI**. (antelope sí es MPI,
`mpirun -np 96` multi-nodo.) Un proceso QXTI vive en **un nodo** y usa **todos sus cores**.

Dentro del nodo: la malla-k se corta en **slabs** (bloques de planos-k contiguos) con un **halo**
de `max_order−1` planos a cada lado (la "región de solape"): es justo la vecindad que necesita el
gradiente covariante (Wilson), así el **interior de cada slab es bit-exacto** vs la malla completa.
Un `ThreadPoolExecutor` recorre los slabs (hasta `n_workers` a la vez; NumPy suelta el GIL). El
[[Concept - Memory and Parallelism|RAM guard]] dimensiona los slabs para dejar ≥1 GB libre.
→ **BZ → slabs con halo → pool de hilos → interiores exactos.** El solape es por corrección del
gradiente, no solo "para chequear".

## Cuántos cores usa (fuente única `qxti/utils/parallel.py`)

Prioridad: `n_workers` del `.cfg` (>0) → `QXTI_NUM_WORKERS` → **SLURM** (`SLURM_CPUS_PER_TASK`,
entero, nunca a la mitad) → **todos los cores usables** (el máximo). En local `n_workers=0` usa
**todos** los cores de la PC (o el del `.cfg`); en cluster usa la asignación SLURM completa.
Opt-in `QXTI_MAC_PERF_CORES=1` para limitar a perf-cores en Apple Silicon. BLAS pinchado a 1 hilo
para no sobre-suscribir. Ver [[Concept - Memory and Parallelism]].

## Uso rápido

```bash
# un run en un nodo con todos sus cores:
./cluster/submit.sh single inputs/inputParams.wsm.cfg -xtp

# MILES de núcleos = muchos runs a la vez (job array):
MAXCONC=200 CPUS=64 PART=normal ./cluster/submit.sh array cluster/joblist.txt
#   cores en vuelo = MAXCONC × CPUS  (200×64 = 12 800)

# uno tras otro (escalera de convergencia, "y así sucesivamente"):
./cluster/submit.sh chain cluster/joblist.txt
```

Deja `n_workers = 0` en el `.cfg` para usar la asignación completa. `PART`/`ACCOUNT`/`MEM`/`TIME`/
`CPUS`/`MAXCONC` se pasan por entorno a `submit.sh`.

## Archivos (`cluster/`)

| Archivo | Rol |
| --- | --- |
| `qxti_job.slurm` | 1 run, 1 nodo, todos los cores (`--cpus-per-task`). Opcional `PLOT=1`. |
| `qxti_array.slurm` | **job array**: cada tarea = 1 run de una línea del joblist. |
| `submit.sh` | wrapper `single` / `array` / `chain` (rellena cpus/time/partition). |
| `joblist.example.txt` | lista `<config> [mode]` por línea. |
| `README.md` | guía completa (incluye notas Windows/mac/Linux y el caveat MPI). |

## Escala y límites

- "Miles de núcleos" en QXTI = **muchos runs concurrentes** (arrays), ideal para barridos
  (frecuencia, orientación, parámetros). No es un run gigante multi-nodo.
- Para spread multi-nodo de **un** barrido de frecuencia: parte el rango en varios `.cfg` y lánzalos
  como array.
- **Futuro (opcional):** MPI real (`mpi4py`) para un run multi-nodo — los slabs ya mapean a bloques
  por-rank. Cambio mayor; hoy los arrays ya dan miles de cores para barridos.

⛔ No muevas el import lazy `simulation → theory_response` (ciclo). No rompas el bit-exacto por
workers. → [[Playbook - Invariants Not to Break]]

---

Relacionado: [[Concept - Memory and Parallelism]] · [[qxti.utils]] · [[Data Flow]]
