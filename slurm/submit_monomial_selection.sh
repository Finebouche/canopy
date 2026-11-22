#!/bin/bash
# Selection of the monomials of degree 2 of the readout of the next phrase
# (canopy.next_phrase_prediction.monomial_selection) on the reservoirs of the studies of a rule (default
# HAG_CANARY_HPO_RULE=ridge:2, the reservoirs optimized for the full readout of order 2), the pure models, with and
# without leak: one job per condition and leak, then the merge of their results (one job per leak). Run from the root of a checkout of
# canopy with the audio, the MFCC caches and the studies:
#   [MODULE=monomial_selection] [LEAKS="0 1"] [BEGIN=now] [MAX_RUNNING=8] [MAX_RUNNING_NIGHT=16] \
#   bash slurm/submit_monomial_selection.sh [sizes] [nb_seeds]
# Jobs running at once: MAX_RUNNING on the working days from 9:00 to 19:00 (the hours of the charter), MAX_RUNNING_NIGHT
# at night and on the weekends. The jobs start with the limit of their hour of start (BEGIN); small jobs then change
# it (scontrol update ArrayTaskThrottle): raised to MAX_RUNNING_NIGHT at NIGHT_START (default 19:00) if they start in
# the day, lowered to MAX_RUNNING at the next working day at NIGHT_END (default 09:00; the jobs already running go on,
# only the pending ones wait). After that morning the limit stays MAX_RUNNING. DRY_RUN=1: the limits, nothing
# submitted.
#   (sizes: comma-separated, default 500: the tests of the partial readouts of order 2 are made on the reservoirs of
#   500 units only; nb_seeds: default 10)
# MODULE=monomial_selection_rules: the online rules on the selected monomials
# (canopy.next_phrase_prediction.monomial_selection_rules; e.g. WORKERS=8 MEM=96G TIME=1-00:00:00, 5 seeds), with the
# variables of the module exported (HAG_CANARY_SELECTION_RULES, HAG_CANARY_SELECTION_METHODS,
# HAG_CANARY_RULES_MAX_EPOCHS).
# Each job: CPUS cores (default 16), MEM memory (default 64G), WORKERS processes (default 4: the folds, then the
# seeds, the threads of the linear algebra shared between them), time limit TIME (default 12 h), on zonda.
set -euo pipefail
SIZES=${1:-500}
NB_SEEDS=${2:-10}
read -r -a LEAKS <<< "${LEAKS:-0 1}"
MAX_RUNNING=${MAX_RUNNING:-8}
MAX_RUNNING_NIGHT=${MAX_RUNNING_NIGHT:-16}
NIGHT_START=${NIGHT_START:-19:00}
NIGHT_END=${NIGHT_END:-09:00}
CPUS=${CPUS:-16}
MEM=${MEM:-64G}
WORKERS=${WORKERS:-4}
TIME=${TIME:-12:00:00}
BEGIN=${BEGIN:-now}
MODULE=${MODULE:-monomial_selection}
NODES=(--constraint=zonda)
export HAG_CANARY_TASK=next_phrase HAG_CANARY_CALLS=1 HAG_CANARY_TAIL_HPO=0
export HAG_CANARY_HPO_RULE=${HAG_CANARY_HPO_RULE:-ridge:2}
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks

set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
# one line per condition and leak: "<condition> <size> HAG_CANARY_LEAK=<leak>" (slurm/canary.sbatch)
TASKS=slurm/tasks/${MODULE}_$(date +%Y%m%d_%H%M%S).txt
for leak in "${LEAKS[@]}"; do
    HAG_CANARY_LEAK=$leak HAG_CANARY_STUDIES=$PWD/outputs/canary/studies PYTHONPATH=$PWD:${HAG_DIR:-$HOME/hag_core} \
        python -m canopy.next_phrase_prediction.monomial_selection list "$SIZES" 2>/dev/null \
        | grep -E "^[^ ]+ [0-9]+$" | sed "s/$/ HAG_CANARY_LEAK=$leak/"
done > "$TASKS"
n=$(wc -l < "$TASKS")
[ "$n" -gt 0 ] || { echo "no condition (studies of $HAG_CANARY_HPO_RULE missing?)" >&2; exit 1; }

# start of the jobs (BEGIN: now, or an hour, tomorrow if it is past), and the end of its night: the next working day
# at NIGHT_END
now=$(date +%s)
start=$now
if [ "$BEGIN" != now ]; then
    start=$(date -d "$BEGIN" +%s)
    if [ "$start" -lt "$now" ]; then start=$((start + 86400)); fi
fi
night_end=$(date -d "$(date -d "@$start" +%F) $NIGHT_END" +%s)
if [ "$night_end" -le "$start" ]; then night_end=$((night_end + 86400)); fi
while [ "$(date -d "@$night_end" +%u)" -ge 6 ]; do night_end=$((night_end + 86400)); done
hour=$(date -d "@$start" +%-H)
if [ "$(date -d "@$start" +%u)" -ge 6 ] || [ "$hour" -ge 19 ] || [ "$hour" -lt 9 ]; then
    LIMIT=$MAX_RUNNING_NIGHT
else
    LIMIT=$MAX_RUNNING
fi
if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "$MODULE: $n conditions, $LIMIT at once from $(date -d "@$start" "+%a %F %H:%M")," \
        "$MAX_RUNNING from $(date -d "@$night_end" "+%a %F %H:%M") (nothing submitted)"
    exit 0
fi

array=$(sbatch --parsable "${NODES[@]}" --begin="$BEGIN" --array=0-$((n - 1))%$LIMIT --cpus-per-task=$CPUS \
    --mem=$MEM --time=$TIME --job-name=${MODULE//_/-} ${DEPENDENCY:+--dependency=$DEPENDENCY --kill-on-invalid-dep=yes} \
    --export=ALL,HAG_CANARY_MODE=$MODULE,HAG_CANARY_WORKERS=$WORKERS,HAG_CANARY_RULES_JOBS=$WORKERS,HAG_CANARY_TASKS="$TASKS" \
    slurm/canary.sbatch "$NB_SEEDS")
echo "$MODULE: job array $array ($n conditions, $LIMIT at once, $TASKS)"
limit_job() {  # <limit> <time (epoch)>: a small job setting the number of jobs of the array running at once
    sbatch --parsable "${NODES[@]}" --begin="$(date -d "@$2" +%FT%T)" --cpus-per-task=1 --mem=1G --time=00:05:00 \
        --job-name=${MODULE//_/-}-limit --output=slurm/logs/%x-%j.out \
        --wrap="scontrol update JobId=$array ArrayTaskThrottle=$1 || true"
}
if [ "$MAX_RUNNING_NIGHT" -gt "$MAX_RUNNING" ]; then
    if [ "$LIMIT" = "$MAX_RUNNING" ]; then  # (started in the day)
        night_start=$(date -d "$(date -d "@$start" +%F) $NIGHT_START" +%s)
        echo "limit of the night ($MAX_RUNNING_NIGHT at once): job $(limit_job "$MAX_RUNNING_NIGHT" "$night_start")," \
            "at $(date -d "@$night_start" "+%a %F %H:%M")"
    fi
    echo "limit of the day ($MAX_RUNNING at once): job $(limit_job "$MAX_RUNNING" "$night_end")," \
        "at $(date -d "@$night_end" "+%a %F %H:%M")"
fi
for leak in "${LEAKS[@]}"; do
    merge=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=1 --mem=8G --time=01:00:00 --dependency=afterany:$array \
        --job-name=${MODULE//_/-}-merge --output=slurm/logs/%x-%j.out \
        --export=ALL,HAG_CANARY_MODE=$MODULE,HAG_CANARY_WORKERS=1,HAG_CANARY_LEAK=$leak \
        slurm/canary.sbatch merge)
    echo "merge (leak $leak): job $merge (after $array)"
done
