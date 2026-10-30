#!/bin/bash
# Learning rules of the readout of the next phrase (canary.next_phrase_prediction.readout_rules) on PlaFRIM, one job per
# condition (reservoirs and readouts of the studies of the readout, HAG_CANARY_TAIL_HPO=1), on the zonda nodes (no rare
# resource), then the merge of their results and the paired comparisons with ridge. Run from the root of a checkout of
# canopy:
#   HAG_CANARY_TASK=next_phrase HAG_CANARY_CALLS=1 HAG_CANARY_TAIL_HPO=1 [HAG_CANARY_LEAK=1] bash slurm/submit_readout_rules.sh [sizes] [nb_seeds]
#   (default 500,1000 and 10 seeds; HAG_CANARY_READOUT_RULES and HAG_CANARY_RULES_SUFFIX passed to the jobs, TIME the
#   time limit of each job, default 1 day)
set -euo pipefail
SIZES=${1:-500,1000}
NB_SEEDS=${2:-10}
MAX_RUNNING=${MAX_RUNNING:-40}
CPUS=${CPUS:-8}
TIME=${TIME:-1-00:00:00}
NODES=(--constraint=zonda)
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks

set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
TASKS=slurm/tasks/readout_rules_$(date +%Y%m%d_%H%M%S)_${HAG_CANARY_LEAK:-0}.txt
HAG_CANARY_STUDIES=$PWD/outputs/canary/studies PYTHONPATH=$PWD:${HAG_DIR:-$HOME/hag_core} \
    python -m canary.next_phrase_prediction.readout_rules list "$SIZES" 2>/dev/null | grep -E "^[^ ]+ [0-9]+$" > "$TASKS"
n=$(wc -l < "$TASKS")

array=$(sbatch --parsable "${NODES[@]}" --array=0-$((n - 1))%$MAX_RUNNING --cpus-per-task=$CPUS --mem=48G \
    --time=$TIME --job-name=canary-readout-rules \
    --export=ALL,HAG_CANARY_MODE=readout_rules,HAG_CANARY_WORKERS=1,HAG_CANARY_TASKS="$TASKS" \
    slurm/canary.sbatch "$NB_SEEDS")
echo "readout rules: job array $array ($n conditions, $TASKS)"
merge=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=1 --mem=8G --time=01:00:00 --dependency=afterany:$array \
    --job-name=readout-rules-merge --export=ALL,HAG_CANARY_MODE=readout_rules,HAG_CANARY_WORKERS=1 \
    --output=slurm/logs/%x-%j.out slurm/canary.sbatch merge)
echo "merge and comparisons: job $merge (after $array)"
