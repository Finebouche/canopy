#!/bin/bash
# Optimization of the reservoirs of the next-phrase prediction for each learning rule of the readout "last" -> "last"
# (canopy.next_phrase_prediction.hpo with HAG_CANARY_HPO_RULE, the hyperparameters of the rule in the search: rate of
# LMS, NLMS and EH, noise of EH, alpha of RLS; studies next_phrase_calls_[leak_]<rule>_<model>_<size>, "<rule>-order<n>"
# for the rules on the monomials "<rule>:<n>"), so that the rules are compared on reservoirs optimized for each of them
# (the rule "ridge": the studies of the linear ridge readout, without rule, done again here with the same annotations),
# then their final evaluation on the reservoirs of their studies (slurm/submit_readout_rules.sh for each rule and leak,
# slurm/submit_next_phrase.sh for ridge, 10 seeds, two conditions at a time each, from the next BEGIN)
# once all the studies are done. A single job array of all the studies (one per job, MAX_RUNNING at once), the rules on
# the monomials and the largest reservoirs first. Run from the root of a checkout of canopy with the audio and the MFCC
# caches (the compute nodes have no internet):
#   [RULES="ridge lms eh_np_local rls nlms neh ridge:2 nlms:2 neh:2"] [PURE="mean_hag var_hag random_ee random_ei"] \
#   [SIZES="500 1000"] [LEAKS="0 1"] [N_TRIALS=400] [MAX_EPOCHS=3000] [BEGIN=20:00] [MAX_RUNNING=30] \
#   bash slurm/submit_hpo_rules.sh
# Each job: a whole node (CPUS=all, sbatch --exclusive) of the CPU nodes CONSTRAINT (default zonda, suroit and bora, no
# GPU nor big memory), WORKERS processes optimizing the study (default auto: as many as the cores, 2 threads each, and
# the memory, 11 GB each, 14 for the rules on the monomials, allow), or CPUS cores and MEM memory; time limit TIME
# (default 3 days, the limit outside the weekends). The studies resume where they stopped if resubmitted. MAX_EPOCHS:
# the number of epochs of the online rules chosen up to this one (HAG_CANARY_HPO_RULE_MAX_EPOCHS).
set -euo pipefail
read -r -a RULES <<< "${RULES:-ridge lms eh_np_local rls nlms neh ridge:2 nlms:2 neh:2}"
read -r -a PURE <<< "${PURE:-mean_hag var_hag random_ee random_ei}"
read -r -a SIZES <<< "${SIZES:-500 1000}"
read -r -a LEAKS <<< "${LEAKS:-0 1}"
N_TRIALS=${N_TRIALS:-400}
MAX_EPOCHS=${MAX_EPOCHS:-3000}
WORKERS=${WORKERS:-auto}
CPUS=${CPUS:-all}
MEM=${MEM:-90G}
if [ "$CPUS" = all ]; then RESOURCES=(--exclusive --mem=0); else RESOURCES=(--cpus-per-task="$CPUS" --mem="$MEM"); fi
CONSTRAINT=${CONSTRAINT:-zonda|suroit|bora}
TIME=${TIME:-3-00:00:00}
BEGIN=${BEGIN:-20:00}
MAX_RUNNING=${MAX_RUNNING:-30}
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks outputs/canary/studies
if [ -z "$(find canopy/datasets/marron1/audio -name "*.wav" -print -quit)" ] || [ ! -f canopy/datasets/marron1/boundary_features_44100_no_floor.pkl ]; then
    echo "canopy/datasets/marron1: audio or MFCC cache missing (the compute nodes have no internet)" >&2
    exit 1
fi

# one line per study: "<model> <size> HAG_CANARY_HPO_RULE=<rule> HAG_CANARY_LEAK=<leak> [HAG_CANARY_WORKER_MEM=14]"
# (slurm/canary.sbatch), the rules on the monomials first, then the largest sizes
TASKS=slurm/tasks/hpo_rules_$(date +%Y%m%d_%H%M%S).txt
for order in 2 1; do for size in $(printf '%s\n' "${SIZES[@]}" | sort -rn); do for leak in "${LEAKS[@]}"; do
    for rule in "${RULES[@]}"; do
        case "$rule" in *:*) [ "$order" = 2 ] || continue; memory=" HAG_CANARY_WORKER_MEM=14" ;;
                        *) [ "$order" = 1 ] || continue; memory="" ;; esac
        assignment="HAG_CANARY_HPO_RULE=$rule "
        [ "$rule" = ridge ] && assignment=""  # (the studies of the linear ridge readout)
        for model in "${PURE[@]}"; do echo "$model $size ${assignment}HAG_CANARY_LEAK=$leak$memory"; done
    done
done; done; done > "$TASKS"
n=$(wc -l < "$TASKS")
array=$(HAG_CANARY_TASK=next_phrase HAG_CANARY_CALLS=1 HAG_CANARY_TAIL_HPO=0 HAG_CANARY_N_TRIALS=$N_TRIALS \
    HAG_CANARY_HPO_RULE_MAX_EPOCHS=$MAX_EPOCHS \
    sbatch --parsable --constraint="$CONSTRAINT" --begin="$BEGIN" --array=0-$((n - 1))%$MAX_RUNNING "${RESOURCES[@]}" \
    --time=$TIME --job-name=hpo-rules --export=ALL,HAG_CANARY_TASKS="$TASKS",HAG_CANARY_WORKERS=$WORKERS \
    slurm/canary.sbatch)
echo "studies: job array $array ($n studies, $MAX_RUNNING at once, from $BEGIN; $TASKS)"

SIZE_LIST=$(IFS=,; echo "${SIZES[*]}")
EVALUATIONS=""
for leak in "${LEAKS[@]}"; do for rule in "${RULES[@]}"; do
    if [ "$rule" = ridge ]; then
        EVALUATIONS+="HAG_CANARY_TASK=next_phrase HAG_CANARY_CALLS=1 HAG_CANARY_LEAK=$leak MAX_RUNNING=2"
        EVALUATIONS+=" SBATCH_BEGIN=$BEGIN bash slurm/submit_next_phrase.sh $SIZE_LIST 10; "
        continue
    fi
    EVALUATIONS+="HAG_CANARY_TASK=next_phrase HAG_CANARY_CALLS=1 HAG_CANARY_TAIL_HPO=0 HAG_CANARY_LEAK=$leak"
    EVALUATIONS+=" HAG_CANARY_HPO_RULE=$rule HAG_CANARY_RULES_MAX_EPOCHS=$MAX_EPOCHS CPUS=16 MEM=96G MAX_RUNNING=2"
    EVALUATIONS+=" SBATCH_BEGIN=$BEGIN bash slurm/submit_readout_rules.sh $SIZE_LIST 10; "
done; done
evaluation=$(sbatch --parsable --constraint="$CONSTRAINT" --cpus-per-task=1 --mem=4G --time=02:00:00 \
    --dependency=afterany:$array --job-name=submit-rule-evaluations --output=slurm/logs/%x-%j.out \
    --wrap="$EVALUATIONS")
echo "final evaluations (one submit_readout_rules.sh or submit_next_phrase.sh per rule and leak): submitted by job $evaluation, after $array"
