#!/bin/bash
# Optimization of the reservoirs and of the readout of the next phrase with the readout trained by an online rule
# (canary.next_phrase_prediction.hpo with HAG_CANARY_HPO_RULE: studies next_phrase_calls_[leak_]tail_<rule>_<model>_<size>)
# on PlaFRIM, on the zonda nodes (no rare resource): a job array for each rule and leak, one study per job. Run from
# the root of a checkout of canopy with the audio and the MFCC caches (the compute nodes have no internet):
#   [RULES="lms eh_np_local"] [PURE="var_hag random_ei"] [SIZES="500 1000"] [LEAKS="0 1"] [N_TRIALS=400] \
#   [BEGIN=20:00] bash slurm/submit_hpo_rules.sh
# Each job: WORKERS processes optimizing the study (default 8, 2 threads each on CPUS=16 cores: a worker uses about 11 GB
# at 1000 neurons, MEM=128G), time limit TIME (default 3 days, the limit outside the weekends). The studies resume
# where they stopped if resubmitted. BEGIN: start time of the jobs (sbatch --begin, the big campaigns at night).
set -euo pipefail
read -r -a RULES <<< "${RULES:-lms eh_np_local}"
read -r -a PURE <<< "${PURE:-var_hag random_ei}"
read -r -a SIZES <<< "${SIZES:-500 1000}"
read -r -a LEAKS <<< "${LEAKS:-0 1}"
N_TRIALS=${N_TRIALS:-400}
WORKERS=${WORKERS:-8}
CPUS=${CPUS:-16}
MEM=${MEM:-128G}
TIME=${TIME:-3-00:00:00}
BEGIN=${BEGIN:-20:00}
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks outputs/canary/studies
if ! find canary/datasets/Canary/audio -name "*.wav" | grep -q . || [ ! -f canary/datasets/Canary/boundary_features_44100_no_floor.pkl ]; then
    echo "canary/datasets/Canary: audio or MFCC cache missing (the compute nodes have no internet)" >&2
    exit 1
fi

TASKS=slurm/tasks/hpo_rules_$(date +%Y%m%d_%H%M%S).txt
for size in "${SIZES[@]}"; do for model in "${PURE[@]}"; do echo "$model $size"; done; done > "$TASKS"
n=$(wc -l < "$TASKS")
for leak in "${LEAKS[@]}"; do
    for rule in "${RULES[@]}"; do
        array=$(HAG_CANARY_TASK=next_phrase HAG_CANARY_CALLS=1 HAG_CANARY_TAIL_HPO=1 HAG_CANARY_LEAK=$leak \
            HAG_CANARY_HPO_RULE=$rule HAG_CANARY_N_TRIALS=$N_TRIALS \
            sbatch --parsable --constraint=zonda --begin="$BEGIN" --array=0-$((n - 1)) --cpus-per-task=$CPUS \
            --mem=$MEM --time=$TIME --job-name=hpo-$rule-leak$leak \
            --export=ALL,HAG_CANARY_TASKS="$TASKS",HAG_CANARY_WORKERS=$WORKERS slurm/canary.sbatch)
        echo "$rule, leak $leak: job array $array ($n studies, $TASKS)"
    done
done
