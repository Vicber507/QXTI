# Running QXTI on a SLURM cluster

This folder makes QXTI run on a computational cluster (mac/win/linux all behave
the same locally, and Linux/SLURM on the cluster). See also the vault note
[[Cluster and SLURM]] and [[Concept - Memory and Parallelism]] in `docs/vault/`.

---

## TL;DR

```bash
# one run on one node, using all its cores:
./cluster/submit.sh single inputs/inputParams.wsm.cfg -xtp

# many runs at once (this is how you use thousands of cores):
MAXCONC=50 CPUS=64 ./cluster/submit.sh array cluster/joblist.txt

# runs one-after-another (e.g. a convergence ladder):
./cluster/submit.sh chain cluster/joblist.txt
```

Set your cluster's partition/account once: `PART=normal ACCOUNT=abc ./cluster/submit.sh ...`.
Leave `n_workers = 0` in the `.cfg` so each run uses the **full** SLURM allocation.

---

## How QXTI parallelises (read this)

**QXTI is a shared-memory, single-node, multi-threaded code — not MPI.** This is
the key difference from Antelope (the C++ SBE reference), which is MPI
(`mpirun -np 96` across several nodes, distributed memory). One QXTI *process*
lives on **one node** and uses **all cores of that node**.

Within a node, the work is split like this (your "cubes with overlap" picture,
made precise):

- The Brillouin-zone k-mesh is cut into **slabs** — contiguous blocks of
  k-planes along one axis (not free-floating cubes; slabs tile the BZ).
- Each slab carries a **halo** of `max_order − 1` planes on **each** side. That
  halo is the "overlap region": it is exactly the neighbourhood the covariant
  k-gradient (Wilson links) needs, so the **interior of every slab is bit-exact**
  versus computing the whole mesh at once. Change the number of cores → identical
  numbers (only the tiling changes).
- A **thread pool** (`ThreadPoolExecutor`) chews through the slabs, up to
  `n_workers` at a time. NumPy releases the GIL inside BLAS, so the threads run
  truly in parallel. Workers don't each own a fixed cube; they pull the next slab
  when free (load-balanced).
- The **RAM guard** sizes the slabs so that
  `n_workers × (slab + 2·halo) × bytes_per_plane ≤ (free RAM − reserve)`, keeping
  ≥ `reserve_gb` free at all times (default 1 GB) on mac/win/linux.

So: **BZ → slabs with halo overlap → a pool of worker threads → bit-exact
interiors.** The overlap exists for gradient correctness, not just "to check
things line up."

Two engines, two parallel units:

| Calculation | Parallel unit | Mechanism |
| --- | --- | --- |
| `-cmd`, `-xtp`/`-cmd` theory (mesh) | slabs of k-planes | ThreadPool (1 node) |
| `-cmd` simulation (CMD time-domain) | chunks of k | ThreadPool (1 node) |
| `-xtp` simulation | **frequencies** | ProcessPool (1 node) |

**BLAS threads are pinned to 1** (the SLURM scripts export `OMP_NUM_THREADS=1`
etc.) so the k-loop pool owns the parallelism — otherwise `cores × BLAS_threads`
would oversubscribe the node. Override with `QXTI_BLAS_THREADS` if you ever run a
few very large dense diagonalisations.

---

## How many cores does a run use?

Resolved by `qxti/utils/parallel.py` in this priority (the fix that made cluster
runs use the whole node instead of a local fraction):

1. `n_workers` in the `.cfg` (`[cmd]`/`[xtp]`/`[ldos]`) if **> 0** — explicit wins.
2. `QXTI_NUM_WORKERS` environment variable.
3. **SLURM allocation** — `SLURM_CPUS_PER_TASK` (used in *full*, never halved).
4. **All usable cores** — the maximum (`os.sched_getaffinity` on Linux; respects
   cgroups/taskset; `os.cpu_count()` on mac/win). So `n_workers = 0` locally = **all** cores.

Opt-in: `QXTI_MAC_PERF_CORES=1` limits the default to performance cores on Apple
Silicon (sometimes faster: efficiency cores + the GIL). Not the default.

The SLURM scripts export `QXTI_NUM_WORKERS=$SLURM_CPUS_PER_TASK`, so
`--cpus-per-task=64` ⇒ QXTI uses 64. `main.py` prints the plan at startup:
`[main] Parallelism: 64 workers (source: SLURM allocation; usable CPUs=64)`.

Locally you get all your PC's cores automatically (or the value you set in the
`.cfg`). Nothing to configure.

---

## Using thousands of cores

QXTI does not spread **one** run across many nodes, so "thousands of cores" means
**many runs at once** — a natural fit for QXTI's work (frequency sweeps,
ellipticity/orientation sweeps, model/parameter scans, per-config HHG):

```
cores in flight = (concurrent array tasks) × (--cpus-per-task)
```

e.g. `MAXCONC=200 CPUS=64` ⇒ up to **12,800 cores** working at once. Put one run
per line in a job list and submit an array:

```bash
# cluster/joblist.txt  (see joblist.example.txt):
#   inputs/inputParams.wsm.cfg     -xtp
#   inputs/inputParams.frank8.cfg  -cmd
MAXCONC=200 CPUS=64 TIME=12:00:00 PART=normal ./cluster/submit.sh array cluster/joblist.txt
```

For a big **frequency sweep** in a single `-xtp` run, the sweep already
parallelises over frequencies on the node (ProcessPool) — give it a fat node
(`CPUS=128`). To spread the *same* sweep across many nodes, split the frequency
range into several `.cfg` files (each a sub-range) and submit them as an array.

---

## Chaining ("un job y luego otro y así sucesivamente")

`./cluster/submit.sh chain cluster/joblist.txt` submits each line as its own job
with `--dependency=afterok:<prev>`, so they run **strictly in sequence** (each
starts only if the previous finished OK). Useful for convergence ladders (grid
28³ → 32³ → 40³ …, cf. `run_overnight.sh`) or compute-then-postprocess pipelines.
Add figures to any single run with `PLOT=1`.

---

## Cluster-specific setup (edit once)

In `qxti_job.slurm` and `qxti_array.slurm` uncomment/adjust:

- the environment block: `module load python/3.11` and/or
  `source /path/to/venv/bin/activate` (or `conda activate qxti`);
- `--partition`, `--account`, `--mem` (or pass `PART=`/`ACCOUNT=`/`MEM=` to
  `submit.sh`). `--mem=0` (whole node) is recommended so the RAM guard has the
  full node budget.

Install QXTI on the cluster once: `pip install -e .` (only needs NumPy;
`matplotlib` for `PLOT=1`).

---

## Windows / macOS / Linux notes

- **Cross-platform RAM guard** (`qxti/utils/memory.py`): `/proc/meminfo` (Linux),
  `vm_stat` (macOS), `GlobalMemoryStatusEx` (Windows), psutil if present.
- **matplotlib cache** now uses `tempfile.gettempdir()` (honours `$TMPDIR`, which
  SLURM sets per job) instead of the old hard-coded macOS `/private/tmp`.
- **Windows**: the `-xtp` ProcessPool uses `spawn`; `main.py` is guarded by
  `if __name__ == "__main__"`, and the worker is a module-level function, so it
  works. Run via `python main.py ...`.

---

## Future: true multi-node MPI (optional)

To make a **single** run span many nodes (distributed memory, like Antelope),
QXTI would need an MPI layer (`mpi4py`) that partitions the k-mesh across ranks
and reduces the BZ sum with `Allreduce` — the streaming slab logic already maps
cleanly onto rank-local k-blocks. That is a larger change; the job-array approach
above already delivers thousands of cores for sweep-type workloads today.
