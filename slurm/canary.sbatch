#!/bin/bash
# Canary songs on PlaFRIM (the studies of canopy.studies), from a checkout of canopy:
#   - CANOPY_JOB_MODE=hpo (default): study of the reservoirs python -m canopy.studies.<study>.study <model> <size>, the
#     study given by CANOPY_TASK (boundaries, default: phrase_boundaries; next_phrase; next_phrase_labels; syllables:
#     syllables_classification),
#     CANOPY_WORKERS processes sharing the cores; in a job array, "<model> <size>" is the line SLURM_ARRAY_TASK_ID
#     (from 0) of the file CANOPY_TASK_FILE, optionally followed by variables KEY=VALUE of the study;
#   - CANOPY_JOB_MODE=evaluate: final evaluation of the boundaries python -m
#     canopy.studies.phrase_boundaries.evaluation [sizes] [nb_seeds];
#   - CANOPY_JOB_MODE=next_phrase: python -m canopy.studies.next_phrase.evaluation [sizes] [nb_seeds] (in a job
#     array: one condition, "<condition> <size>" of CANOPY_TASK_FILE; slurm/submit_next_phrase.sh);
#   - CANOPY_JOB_MODE=syllables: python -m canopy.studies.syllables_classification.evaluation (the same;
#     slurm/submit_syllables.sh);
#   - CANOPY_JOB_MODE=next_phrase_labels: python -m canopy.studies.next_phrase_labels.evaluation [sizes] [nb_seeds]
#     (slurm/submit_next_phrase_labels.sh);
#   - CANOPY_JOB_MODE=certainty: certainty along the phrases python -m
#     canopy.studies.syllables_classification.certainty [size] [conditions] [nb_seeds]
#     (canopy.studies.next_phrase.certainty with CANOPY_TASK=next_phrase).
# The Optuna database of a study is in /dev/shm during its job (SQLite does not support the network file systems well,
# and the suroit nodes have no local disk), copied from and back to its part, outputs/<study>/hpo/parts/<dataset>/
# <Optuna study>/ (every 10 minutes and at the end); the parts are merged into the databases of the datasets,
# outputs/<study>/hpo/marron1_db.sqlite3, by slurm/merge_studies.sh (python -m canopy.studies.common merge), and the
# evaluations read these databases. Submit everything with slurm/submit_canary.sh.
#SBATCH --job-name=canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
set -euo pipefail
if [ -n "${CANOPY_TASK_FILE:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    read -r MODEL SIZE EXTRA < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CANOPY_TASK_FILE")
    # (optional fields KEY=VALUE: variables of the study, e.g. CANOPY_LEAKY_RESERVOIRS=1)
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
# CANOPY_WORKERS=auto: as many workers as the cores (2 threads each) and the memory of the job
# (CANOPY_WORKER_MEM_GB GB each, default 11: about 9.5 GB at 1000 neurons) allow
CPUS=${SLURM_CPUS_ON_NODE:-$SLURM_CPUS_PER_TASK}
WORKERS=${CANOPY_WORKERS:-4}
if [ "$WORKERS" = auto ]; then
    WORKERS=$((CPUS / 2))
    # memory of the job in MB (that of the node with sbatch --mem=0)
    MEMORY=${SLURM_MEM_PER_NODE:-0}
    if [ "$MEMORY" -eq 0 ]; then MEMORY=$(($(awk '/^MemTotal/ {print $2}' /proc/meminfo) / 1024)); fi
    BY_MEMORY=$((MEMORY / 1024 / ${CANOPY_WORKER_MEM_GB:-11}))
    if [ "$BY_MEMORY" -lt "$WORKERS" ]; then WORKERS=$((BY_MEMORY > 0 ? BY_MEMORY : 1)); fi
fi
THREADS=$((CPUS / WORKERS))
export CANOPY_WORKERS=$WORKERS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$THREADS"
# numba cache of librosa per job: the jobs writing it at the same time in the shared environment fail (stale file handle)
export NUMBA_CACHE_DIR=/dev/shm/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}

case "${CANOPY_TASK:-boundaries}" in  # (the folder of the study, canopy.settings.outputs_dir)
    boundaries) STUDY=phrase_boundaries ;;
    syllables) STUDY=syllables_classification ;;
    *) STUDY=$CANOPY_TASK ;;
esac
case "${CANOPY_JOB_MODE:-hpo}" in evaluate) STUDY=phrase_boundaries ;; next_phrase) STUDY=next_phrase ;;
    next_phrase_labels) STUDY=next_phrase_labels ;; syllables) STUDY=syllables_classification ;; esac
mkdir -p slurm/logs
if [ "${CANOPY_JOB_MODE:-hpo}" = hpo ]; then
    # the study in a database of its own (several jobs writing the same SQLite file would corrupt it), its part, merged
    # into the database of the dataset afterwards; the study starts from that of the database of the dataset if it is
    # there (canopy.studies.common.run_study)
    PART=$(python -m canopy.studies.common part "$STUDY" "$1" "$2" 2>/dev/null | tail -n 1)
    export CANOPY_JOB_STUDIES_DIR=/dev/shm/canopy_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
    mkdir -p "$PART" "$CANOPY_JOB_STUDIES_DIR"
    rsync -a "$PART/" "$CANOPY_JOB_STUDIES_DIR/"
    sync_back() { rsync -a "$CANOPY_JOB_STUDIES_DIR/" "$PART/"; }
    trap 'sync_back; rm -rf "$CANOPY_JOB_STUDIES_DIR" "$NUMBA_CACHE_DIR"' EXIT
    trap 'exit 143' TERM
    (while true; do sleep 600; sync_back; done) &
else
    trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT
fi

echo "[slurm] job ${SLURM_JOB_ID} ${SLURM_ARRAY_TASK_ID:-}: ${CANOPY_JOB_MODE:-hpo} $*, on $(hostname)," \
    "$CPUS cores, $WORKERS workers"
case "${CANOPY_JOB_MODE:-hpo}" in
    evaluate) python -m canopy.studies.phrase_boundaries.evaluation "$@" ;;
    next_phrase|syllables)
        # in a job array: one condition ("<condition> <size>" of CANOPY_TASK_FILE)
        if [ -n "${CANOPY_TASK_FILE:-}" ]; then set -- run "$@"; fi
        case "$CANOPY_JOB_MODE" in
            next_phrase) python -m canopy.studies.next_phrase.evaluation "$@" ;;
            syllables) python -m canopy.studies.syllables_classification.evaluation "$@" ;;
        esac ;;
    next_phrase_labels) python -m canopy.studies.next_phrase_labels.evaluation "$@" ;;
    certainty)
        if [ "${CANOPY_TASK:-}" = next_phrase ]; then python -m canopy.studies.next_phrase.certainty "$@"
        else python -m canopy.studies.syllables_classification.certainty "$@"; fi ;;
    hpo)
        case "${CANOPY_TASK:-boundaries}" in
            boundaries|next_phrase|next_phrase_labels|syllables) python -m "canopy.studies.$STUDY.study" "$@" ;;
            *) echo "unknown CANOPY_TASK=$CANOPY_TASK" >&2; exit 1 ;;
        esac ;;
    *) echo "unknown CANOPY_JOB_MODE=$CANOPY_JOB_MODE" >&2; exit 1 ;;
esac
