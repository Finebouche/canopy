#!/bin/bash
# Canary songs on PlaFRIM (canopy.next_phrase_prediction: boundaries and next phrase; canopy.syllables_classification), from a
# checkout of canopy:
#   - HAG_CANARY_MODE=hpo (default): hyperparameter optimization python -m canopy.next_phrase_prediction.hpo <model> <size>
#     (python -m canopy.syllables_classification.hpo with HAG_CANARY_TASK=syllables), HAG_CANARY_WORKERS processes sharing
#     the cores; in a job array, "<model> <size>" is the line SLURM_ARRAY_TASK_ID (from 0) of the file HAG_CANARY_TASKS;
#   - HAG_CANARY_MODE=evaluate: final evaluation python -m canopy.next_phrase_prediction.evaluate [sizes] [nb_seeds];
#   - HAG_CANARY_MODE=syllables: frame-by-frame syllable classification python -m
#     canopy.syllables_classification.classification [sizes] [nb_seeds] (reservoirs of the syllable studies);
#   - HAG_CANARY_MODE=next_phrase: prediction of the next phrase python -m canopy.next_phrase_prediction.prediction [sizes]
#     [nb_seeds] (reservoirs of the next-phrase studies, HAG_CANARY_TASK=next_phrase);
#   - HAG_CANARY_MODE=readout_rules: learning rules of the readout of the next phrase python -m
#     canopy.next_phrase_prediction.readout_rules (slurm/submit_readout_rules.sh);
#   - HAG_CANARY_MODE=certainty: certainty along the phrases python -m canopy.syllables_classification.certainty [size]
#     [conditions] [nb_seeds] (canopy.next_phrase_prediction.certainty with HAG_CANARY_TASK=next_phrase).
# The Optuna databases are in /dev/shm during the job (SQLite does not support the network file systems well, and the
# suroit nodes have no local disk), copied from and back to outputs/canary/studies (every 10 minutes and at the
# end, without replacing more recent files). Submit everything with slurm/submit_canary.sh.
#SBATCH --job-name=canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
set -euo pipefail
if [ -n "${HAG_CANARY_TASKS:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    read -r MODEL SIZE EXTRA < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$HAG_CANARY_TASKS")
    if [[ "$EXTRA" == *=* ]]; then
        # (optional fields KEY=VALUE: variables of the study, e.g. slurm/submit_hpo_rules.sh)
        for assignment in $EXTRA; do export "$assignment"; done
    elif [ -n "$EXTRA" ]; then
        # (optional third and fourth fields: the readout of the study and the leak, slurm/submit_hpo_readouts.sh)
        read -r READOUT LEAK <<< "$EXTRA"
        export HAG_CANARY_HPO_READOUT=$READOUT HAG_CANARY_LEAK=$LEAK
        # (training rows "all": all the states of the phrases, ~1 million rows, ~9 GB at 1000 neurons, more memory per
        # worker)
        case "$READOUT" in all:*) export HAG_CANARY_WORKER_MEM=${HAG_CANARY_WORKER_MEM_ALL:-28} ;; esac
    fi
    set -- "$MODEL" "$SIZE" "$@"
fi
set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
cd "$SLURM_SUBMIT_DIR"
# this checkout and hag (HAG_DIR, default ~/hag_core, branch main), not the editable install of ~/hag (branch rl)
export PYTHONPATH=$PWD:${HAG_DIR:-$HOME/hag_core}

# threads of each worker (NumPy and JAX), on the cores of the job (all those of the node with sbatch --exclusive);
# HAG_CANARY_WORKERS=auto: as many workers as the cores (2 threads each) and the memory of the job
# (HAG_CANARY_WORKER_MEM GB each, default 11: about 9.5 GB at 1000 neurons) allow
CPUS=${SLURM_CPUS_ON_NODE:-$SLURM_CPUS_PER_TASK}
WORKERS=${HAG_CANARY_WORKERS:-4}
if [ "$WORKERS" = auto ]; then
    WORKERS=$((CPUS / 2))
    # memory of the job in MB (that of the node with sbatch --mem=0)
    MEMORY=${SLURM_MEM_PER_NODE:-0}
    if [ "$MEMORY" -eq 0 ]; then MEMORY=$(($(awk '/^MemTotal/ {print $2}' /proc/meminfo) / 1024)); fi
    BY_MEMORY=$((MEMORY / 1024 / ${HAG_CANARY_WORKER_MEM:-11}))
    if [ "$BY_MEMORY" -lt "$WORKERS" ]; then WORKERS=$((BY_MEMORY > 0 ? BY_MEMORY : 1)); fi
fi
THREADS=$((CPUS / WORKERS))
export HAG_CANARY_WORKERS=$WORKERS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$THREADS"
# numba cache of librosa per job: the jobs writing it at the same time in the shared environment fail (stale file handle)
export NUMBA_CACHE_DIR=/dev/shm/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}

STUDIES=$PWD/outputs/canary/studies
export HAG_CANARY_STUDIES=/dev/shm/hag_canary_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$STUDIES" "$HAG_CANARY_STUDIES" slurm/logs
rsync -a "$STUDIES/" "$HAG_CANARY_STUDIES/"
sync_back() { rsync -a --update "$HAG_CANARY_STUDIES/" "$STUDIES/"; }
trap 'sync_back; rm -rf "$HAG_CANARY_STUDIES" "$NUMBA_CACHE_DIR"' EXIT
trap 'exit 143' TERM
(while true; do sleep 600; sync_back; done) &

echo "[slurm] job ${SLURM_JOB_ID} ${SLURM_ARRAY_TASK_ID:-}: ${HAG_CANARY_MODE:-hpo} $*, on $(hostname)," \
    "$CPUS cores, $WORKERS workers"
if [ "${HAG_CANARY_MODE:-hpo}" = evaluate ]; then
    python -m canopy.next_phrase_prediction.evaluate "$@"
elif [ "${HAG_CANARY_MODE:-hpo}" = certainty ]; then
    if [ "${HAG_CANARY_TASK:-}" = next_phrase ]; then
        python -m canopy.next_phrase_prediction.certainty "$@"
    else
        python -m canopy.syllables_classification.certainty "$@"
    fi
elif [ "${HAG_CANARY_MODE:-hpo}" = syllables ]; then
    # in a job array: one condition ("<condition> <size>" of HAG_CANARY_TASKS)
    if [ -n "${HAG_CANARY_TASKS:-}" ]; then set -- run "$@"; fi
    python -m canopy.syllables_classification.classification "$@"
elif [ "${HAG_CANARY_MODE:-hpo}" = next_phrase ]; then
    # in a job array: one condition ("<condition> <size>" of HAG_CANARY_TASKS)
    if [ -n "${HAG_CANARY_TASKS:-}" ]; then set -- run "$@"; fi
    python -m canopy.next_phrase_prediction.prediction "$@"
elif [ "${HAG_CANARY_MODE:-hpo}" = readout_rules ]; then
    if [ -n "${HAG_CANARY_TASKS:-}" ]; then set -- run "$@"; fi
    python -m canopy.next_phrase_prediction.readout_rules "$@"
elif [ "${HAG_CANARY_TASK:-}" = syllables ]; then
    python -m canopy.syllables_classification.hpo "$@"
else
    python -m canopy.next_phrase_prediction.hpo "$@"
fi
