#!/bin/bash
# =============================================================================
# QXTI — JOB ARRAY: many independent single-node runs at once.
#
# This is how QXTI scales to THOUSANDS of cores on a cluster: each array task is
# one QXTI run on one node using all its cores.  Total cores in flight =
#   (number of array tasks running concurrently) x (--cpus-per-task).
# e.g. 100 tasks x 64 cores = 6400 cores working in parallel.
#
# Each array task runs one line of a job list file:  "<config> [mode]"
#   inputs/inputParams.wsm.cfg      -xtp
#   inputs/inputParams.frank8.cfg   -ldos
#
# Submit with the helper (counts the lines and sets --array for you):
#   ./cluster/submit.sh array cluster/joblist.txt
# or directly (must set --array to match the number of lines):
#   JOBLIST=cluster/joblist.txt sbatch --array=1-100%50 --cpus-per-task=64 cluster/qxti_array.slurm
# =============================================================================
#SBATCH --job-name=qxti-arr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64          # cores per task (edit/override)
#SBATCH --time=12:00:00
#SBATCH --array=1-1%16             # OVERRIDDEN by submit.sh; %N caps concurrent tasks
#SBATCH --output=logs/qxti-%A_%a.out
#SBATCH --error=logs/qxti-%A_%a.err
## #SBATCH --partition=normal
## #SBATCH --account=your_account
## #SBATCH --mem=0

set -euo pipefail

# module load python/3.11 ; source /path/to/venv/bin/activate   # edit for your cluster

JOBLIST="${JOBLIST:-cluster/joblist.txt}"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# Pick this task's line (skip blank lines and #comments).
LINE=$(grep -vE '^\s*(#|$)' "${JOBLIST}" | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "${LINE}" ]; then
    echo "[qxti array ${SLURM_ARRAY_TASK_ID}] no job on that line of ${JOBLIST}; nothing to do."
    exit 0
fi
read -r CONFIG MODE _ <<< "${LINE}"
MODE="${MODE:--cmd}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export QXTI_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-1}"

echo "[qxti array ${SLURM_ARRAY_TASK_ID}] host=$(hostname) \
cpus=${SLURM_CPUS_PER_TASK:-?} config=${CONFIG} mode=${MODE}"
date
srun --cpu-bind=cores python3 main.py "${CONFIG}" "${MODE}"
date
echo "[qxti array ${SLURM_ARRAY_TASK_ID}] done: ${CONFIG} ${MODE}"
