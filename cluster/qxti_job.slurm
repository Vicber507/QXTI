#!/bin/bash
# =============================================================================
# QXTI — single job on ONE node, using ALL cores allocated to the task.
#
# QXTI is a shared-memory (multi-threaded) code: one process uses every core on
# its node via a ThreadPool over the k-grid.  So one SLURM task = one node, and
# --cpus-per-task = how many cores QXTI parallelises over.  (To use thousands of
# cores, launch MANY of these as a job array — see qxti_array.slurm / README.)
#
# Submit with the helper (fills in partition/account/time/cpus for you):
#   ./cluster/submit.sh single inputs/inputParams.wsm.cfg -xtp
# or directly:
#   CONFIG=inputs/inputParams.wsm.cfg MODE=-xtp sbatch --cpus-per-task=64 cluster/qxti_job.slurm
# =============================================================================
#SBATCH --job-name=qxti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64          # cores for the shared-memory k-loop (edit/override)
#SBATCH --time=24:00:00
#SBATCH --output=logs/qxti-%j.out
#SBATCH --error=logs/qxti-%j.err
## --- set these for YOUR cluster (or pass via submit.sh) ---
## #SBATCH --partition=normal
## #SBATCH --account=your_account
## #SBATCH --mem=0                   # request ALL node memory (best for the RAM guard)

set -euo pipefail

# ---- environment (edit for your cluster) -----------------------------------
# module load python/3.11            # or: module load anaconda
# source /path/to/venv/bin/activate  # or: conda activate qxti
# If QXTI is installed editable (pip install -e .) `python -m`/main.py just work.

# ---- inputs ----------------------------------------------------------------
CONFIG="${CONFIG:?Set CONFIG=inputs/inputParams.<model>.cfg}"
MODE="${MODE:--cmd}"                 # -cmd | -xtp | -ldos
PLOT="${PLOT:-0}"                    # 1 -> also render figures after computing

# ---- parallelism / threads -------------------------------------------------
# Pin BLAS to 1 thread: the ThreadPool over k owns the parallelism, so we must
# NOT also let each NumPy call spawn threads (that would oversubscribe cores).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# QXTI auto-detects SLURM_CPUS_PER_TASK; export it explicitly so it is obvious
# and overridable.  (A positive n_workers in the .cfg still takes precedence.)
export QXTI_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-${QXTI_NUM_WORKERS:-1}}"

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
echo "[qxti] host=$(hostname) cpus_per_task=${SLURM_CPUS_PER_TASK:-?} \
mem=${SLURM_MEM_PER_NODE:-?}MB config=${CONFIG} mode=${MODE}"
date

srun --cpu-bind=cores python3 main.py "${CONFIG}" "${MODE}"

if [ "${PLOT}" = "1" ]; then
    echo "[qxti] rendering figures ..."
    python3 qxti/graphics/graphics.py "${CONFIG}" || echo "[qxti] plotting skipped/failed"
fi
date
echo "[qxti] done."
