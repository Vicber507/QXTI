#!/usr/bin/env bash
# =============================================================================
# QXTI SLURM submission helper.  Fills in cpus/time/partition/account and picks
# the right batch script.  Run from the repo root.
#
#   ./cluster/submit.sh single <config> [mode]        one run on one node
#   ./cluster/submit.sh array  <joblist>              many runs at once (array)
#   ./cluster/submit.sh chain  <joblist>              runs one-after-another
#
# Examples:
#   ./cluster/submit.sh single inputs/inputParams.wsm.cfg -xtp
#   CPUS=128 TIME=48:00:00 PART=bigmem ./cluster/submit.sh single inputs/inputParams.frank8.cfg -cmd
#   MAXCONC=50 ./cluster/submit.sh array cluster/joblist.txt
#   ./cluster/submit.sh chain cluster/joblist.txt      # e.g. a convergence ladder
#
# Tunables via environment:
#   CPUS=64            cores per task (--cpus-per-task); QXTI uses all of them
#   TIME=24:00:00      walltime
#   PART=...           partition   (omitted if unset)
#   ACCOUNT=...        account     (omitted if unset)
#   MEM=0              memory (0 = whole node, best for the RAM guard; omitted if unset)
#   MAXCONC=16         max concurrent array tasks (the %N in --array)
#   PLOT=1             also render figures after each run
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
cd "${ROOT}"

CPUS="${CPUS:-64}"
TIME="${TIME:-24:00:00}"
MAXCONC="${MAXCONC:-16}"
PLOT="${PLOT:-0}"

# Assemble the common sbatch options.
SB_OPTS=(--cpus-per-task="${CPUS}" --time="${TIME}")
[ -n "${PART:-}" ]    && SB_OPTS+=(--partition="${PART}")
[ -n "${ACCOUNT:-}" ] && SB_OPTS+=(--account="${ACCOUNT}")
[ -n "${MEM:-}" ]     && SB_OPTS+=(--mem="${MEM}")

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found (are you on the cluster?)"; exit 1; }
mkdir -p logs

sub="${1:-}"; shift || true
case "${sub}" in
  single)
    CONFIG="${1:?usage: submit.sh single <config> [mode]}"
    MODE="${2:--cmd}"
    echo "[submit] single: ${CONFIG} ${MODE}  (cpus=${CPUS}, time=${TIME})"
    sbatch "${SB_OPTS[@]}" \
      --export=ALL,CONFIG="${CONFIG}",MODE="${MODE}",PLOT="${PLOT}" \
      cluster/qxti_job.slurm
    ;;

  array)
    JOBLIST="${1:?usage: submit.sh array <joblist>}"
    [ -f "${JOBLIST}" ] || { echo "ERROR: no such joblist: ${JOBLIST}"; exit 1; }
    N=$(grep -cvE '^\s*(#|$)' "${JOBLIST}")
    [ "${N}" -ge 1 ] || { echo "ERROR: joblist has no runnable lines"; exit 1; }
    echo "[submit] array: ${N} tasks from ${JOBLIST} (cpus=${CPUS} each, up to ${MAXCONC} at once)"
    echo "         => up to ${MAXCONC} x ${CPUS} = $(( MAXCONC * CPUS )) cores in flight."
    sbatch "${SB_OPTS[@]}" \
      --array="1-${N}%${MAXCONC}" \
      --export=ALL,JOBLIST="${JOBLIST}" \
      cluster/qxti_array.slurm
    ;;

  chain)
    # Submit each line as its own job, each starting only after the previous one
    # finishes OK (afterok).  Good for sequential pipelines / convergence ladders
    # ("un job y luego este mismo se corra y así sucesivamente").
    JOBLIST="${1:?usage: submit.sh chain <joblist>}"
    [ -f "${JOBLIST}" ] || { echo "ERROR: no such joblist: ${JOBLIST}"; exit 1; }
    prev=""
    i=0
    while IFS= read -r line; do
      case "${line}" in ''|\#*) continue;; esac
      read -r CONFIG MODE _ <<< "${line}"
      MODE="${MODE:--cmd}"
      i=$((i+1))
      dep=()
      [ -n "${prev}" ] && dep=(--dependency="afterok:${prev}")
      jid=$(sbatch --parsable "${SB_OPTS[@]}" "${dep[@]}" \
              --export=ALL,CONFIG="${CONFIG}",MODE="${MODE}",PLOT="${PLOT}" \
              cluster/qxti_job.slurm)
      echo "[submit] chain step ${i}: job ${jid}  ${CONFIG} ${MODE}${prev:+  (after ${prev})}"
      prev="${jid}"
    done < "${JOBLIST}"
    [ "${i}" -ge 1 ] || { echo "ERROR: joblist has no runnable lines"; exit 1; }
    echo "[submit] chained ${i} jobs; last id ${prev}."
    ;;

  *)
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
