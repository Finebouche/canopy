#!/bin/bash
# Optimization of the reservoirs of the next-phrase prediction for each readout (canopy.next_phrase_prediction.hpo with
# HAG_CANARY_HPO_READOUT: the readout fixed, studies next_phrase_calls_[leak_]readout-<points>-<decision>[-<tail>]_
# <model>_<size>), so that the readouts are compared on reservoirs optimized for each of them, then their final
# evaluation (slurm/submit_next_phrase.sh for each readout and leak, 10 seeds, the hybrids from the parts, one job at
# a time for each, from the next BEGIN) once all the studies are done. A single job array of all the studies (one per job, MAX_RUNNING at once). Run from the root of a
# checkout of canopy with the audio and the MFCC caches (the compute nodes have no internet):
#   [READOUTS="mean:mean tail_mean:tail_mean:10 ..."] [PURE="mean_hag var_hag random_ee random_ei"] [SIZES="500 1000"] \
#   [LEAKS="0 1"] [N_TRIALS=400] [BEGIN=19:00] [MAX_RUNNING=8] bash slurm/submit_hpo_readouts.sh
# READOUTS: "<points>:<decision>[:<tail>]" (canopy.next_phrase_prediction.prediction.readout_of), default all the
# combinations of the training rows last, mean, all, tail 5 and 10, tail_mean 5 and 10 and of the decisions last, mean,
# tail_mean (the tail of the training rows, or 5 and 10), but "last" -> "last" (the studies without readout). Each job:
# a whole node (CPUS=all, sbatch --exclusive) of the CPU nodes CONSTRAINT (default zonda, suroit and bora), WORKERS
# processes optimizing the study (default auto, see slurm/canary.sbatch), or CPUS cores and MEM memory; time limit TIME
# (default 3 days). The studies resume where they stopped if resubmitted. BEGIN: start time of the jobs (sbatch
# --begin, the big campaigns at night).
set -euo pipefail
DEFAULT_READOUTS="last:mean last:tail_mean:5 last:tail_mean:10"
for points in mean all; do DEFAULT_READOUTS+=" $points:last $points:mean $points:tail_mean:5 $points:tail_mean:10"; done
for points in tail tail_mean; do for tail in 5 10; do
    DEFAULT_READOUTS+=" $points:last:$tail $points:mean:$tail $points:tail_mean:$tail"
done; done
read -r -a READOUTS <<< "${READOUTS:-$DEFAULT_READOUTS}"
read -r -a PURE <<< "${PURE:-mean_hag var_hag random_ee random_ei}"
read -r -a SIZES <<< "${SIZES:-500 1000}"
read -r -a LEAKS <<< "${LEAKS:-0 1}"
N_TRIALS=${N_TRIALS:-400}
WORKERS=${WORKERS:-auto}
CPUS=${CPUS:-all}
MEM=${MEM:-90G}
if [ "$CPUS" = all ]; then RESOURCES=(--exclusive --mem=0); else RESOURCES=(--cpus-per-task="$CPUS" --mem="$MEM"); fi
CONSTRAINT=${CONSTRAINT:-zonda|suroit|bora}
TIME=${TIME:-3-00:00:00}
BEGIN=${BEGIN:-19:00}
MAX_RUNNING=${MAX_RUNNING:-8}
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks outputs/canary/studies
if [ -z "$(find canopy/datasets/marron1/audio -name "*.wav" -print -quit)" ] || [ ! -f canopy/datasets/marron1/boundary_features_44100_no_floor.pkl ]; then
    echo "canopy/datasets/marron1: audio or MFCC cache missing (the compute nodes have no internet)" >&2
    exit 1
fi

# one line per study: "<model> <size> <readout> <leak>" (slurm/canary.sbatch), the largest sizes first
TASKS=slurm/tasks/hpo_readouts_$(date +%Y%m%d_%H%M%S).txt
for size in $(printf '%s\n' "${SIZES[@]}" | sort -rn); do for leak in "${LEAKS[@]}"; do
    for readout in "${READOUTS[@]}"; do for model in "${PURE[@]}"; do echo "$model $size $readout $leak"; done; done
done; done > "$TASKS"
n=$(wc -l < "$TASKS")
array=$(HAG_CANARY_TASK=next_phrase HAG_CANARY_CALLS=1 HAG_CANARY_N_TRIALS=$N_TRIALS \
    sbatch --parsable --constraint="$CONSTRAINT" --begin="$BEGIN" --array=0-$((n - 1))%$MAX_RUNNING "${RESOURCES[@]}" \
    --time=$TIME --job-name=hpo-readouts --export=ALL,HAG_CANARY_TASKS="$TASKS",HAG_CANARY_WORKERS=$WORKERS \
    slurm/canary.sbatch)
echo "studies: job array $array ($n studies, $MAX_RUNNING at once, from $BEGIN; $TASKS)"

SIZE_LIST=$(IFS=,; echo "${SIZES[*]}")
EVALUATIONS=""
for leak in "${LEAKS[@]}"; do for readout in "${READOUTS[@]}"; do
    EVALUATIONS+="HAG_CANARY_TASK=next_phrase HAG_CANARY_CALLS=1 HAG_CANARY_LEAK=$leak HAG_CANARY_HPO_READOUT=$readout"
    EVALUATIONS+=" MAX_RUNNING=1 SBATCH_BEGIN=$BEGIN bash slurm/submit_next_phrase.sh $SIZE_LIST 10; "
done; done
evaluation=$(sbatch --parsable --constraint="$CONSTRAINT" --cpus-per-task=1 --mem=4G --time=02:00:00 \
    --dependency=afterany:$array --job-name=submit-readout-evaluations --output=slurm/logs/%x-%j.out \
    --wrap="$EVALUATIONS")
echo "final evaluations (one submit_next_phrase.sh per readout and leak): submitted by job $evaluation, after $array"
