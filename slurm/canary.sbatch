#!/bin/bash
# Canary songs on PlaFRIM (the studies of canopy.studies), from a checkout of canopy:
#   - HAG_CANARY_MODE=hpo (default): study of the reservoirs python -m canopy.studies.<study>.study <model> <size>, the
#     study given by HAG_CANARY_TASK (boundaries, default: phrase_boundaries; next_phrase; syllables:
#     syllables_classification),
#     HAG_CANARY_WORKERS processes sharing the cores; in a job array, "<model> <size>" is the line SLURM_ARRAY_TASK_ID
#     (from 0) of the file HAG_CANARY_TASKS, optionally followed by variables KEY=VALUE of the study;
#   - HAG_CANARY_MODE=evaluate: final evaluation of the boundaries python -m
#     canopy.studies.phrase_boundaries.evaluation [sizes] [nb_seeds];
#   - HAG_CANARY_MODE=next_phrase: python -m canopy.studies.next_phrase.evaluation [sizes] [nb_seeds] (in a job
#     array: one condition, "<condition> <size>" of HAG_CANARY_TASKS; slurm/submit_next_phrase.sh);
#   - HAG_CANARY_MODE=syllables: python -m canopy.studies.syllables_classification.evaluation (the same;
#     slurm/submit_syllables.sh);
#   - HAG_CANARY_MODE=certainty: certainty along the phrases python -m
#     canopy.studies.syllables_classification.certainty [size] [conditions] [nb_seeds]
#     (canopy.studies.next_phrase.certainty with HAG_CANARY_TASK=next_phrase).
# The Optuna databases are in /dev/shm during the job (SQLite does not support the network file systems well, and the
# suroit nodes have no local disk), copied from and back to outputs/<study>/hpo, the study of HAG_CANARY_TASK (every
# 10 minutes and at the end, without replacing more recent files). Submit everything with slurm/submit_canary.sh.
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
    # (optional fields KEY=VALUE: variables of the study, e.g. HAG_CANARY_LEAK=1)
    for assignment in $EXTRA; do export "$assignment"; done
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

case "${HAG_CANARY_TASK:-boundaries}" in  # (the folder of the study, canopy.settings.outputs_dir)
    boundaries) STUDY=phrase_boundaries ;;
    syllables) STUDY=syllables_classification ;;
    *) STUDY=$HAG_CANARY_TASK ;;
esac
case "${HAG_CANARY_MODE:-hpo}" in evaluate) STUDY=phrase_boundaries ;; next_phrase) STUDY=next_phrase ;;
    syllables) STUDY=syllables_classification ;; esac
STUDIES=$PWD/outputs/$STUDY/hpo
export HAG_CANARY_STUDIES=/dev/shm/hag_canary_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$STUDIES" "$HAG_CANARY_STUDIES" slurm/logs
rsync -a "$STUDIES/" "$HAG_CANARY_STUDIES/"
sync_back() { rsync -a --update "$HAG_CANARY_STUDIES/" "$STUDIES/"; }
trap 'sync_back; rm -rf "$HAG_CANARY_STUDIES" "$NUMBA_CACHE_DIR"' EXIT
trap 'exit 143' TERM
(while true; do sleep 600; sync_back; done) &

echo "[slurm] job ${SLURM_JOB_ID} ${SLURM_ARRAY_TASK_ID:-}: ${HAG_CANARY_MODE:-hpo} $*, on $(hostname)," \
    "$CPUS cores, $WORKERS workers"
case "${HAG_CANARY_MODE:-hpo}" in
    evaluate) python -m canopy.studies.phrase_boundaries.evaluation "$@" ;;
    next_phrase|syllables)
        # in a job array: one condition ("<condition> <size>" of HAG_CANARY_TASKS)
        if [ -n "${HAG_CANARY_TASKS:-}" ]; then set -- run "$@"; fi
        case "$HAG_CANARY_MODE" in
            next_phrase) python -m canopy.studies.next_phrase.evaluation "$@" ;;
            syllables) python -m canopy.studies.syllables_classification.evaluation "$@" ;;
        esac ;;
    certainty)
        if [ "${HAG_CANARY_TASK:-}" = next_phrase ]; then python -m canopy.studies.next_phrase.certainty "$@"
        else python -m canopy.studies.syllables_classification.certainty "$@"; fi ;;
    hpo)
        case "${HAG_CANARY_TASK:-boundaries}" in
            boundaries|next_phrase|syllables) python -m "canopy.studies.$STUDY.study" "$@" ;;
            *) echo "unknown HAG_CANARY_TASK=$HAG_CANARY_TASK" >&2; exit 1 ;;
        esac ;;
    *) echo "unknown HAG_CANARY_MODE=$HAG_CANARY_MODE" >&2; exit 1 ;;
esac
