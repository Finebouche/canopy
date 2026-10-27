#!/bin/bash
# Full canary boundary detection experiment (canary.next_phrase_prediction), run from the root of a checkout of canopy:
#   1. optimization of the pure models (canary.models.PURE_MODELS) at each size of SIZES, job array;
#   2. final evaluation with NB_SEEDS seeds and the paired comparisons (after 1), the hybrids <hag>+<esn> of each size
#      combining the parts optimized alone at half the size (they are not optimized).
# canary.next_phrase_prediction.hpo.N_TRIALS trials per study (Optuna, TPE). With HAG_CANARY_TASK=syllables, the studies optimize
# the frame accuracy of the syllable classification ("syllables_<model>_<size>") and the final evaluation is
# slurm/submit_syllables.sh.
# With HAG_CANARY_TASK=next_phrase, they optimize the accuracy of the prediction of the next phrase
# ("next_phrase_<model>_<size>") and the final evaluation is slurm/submit_next_phrase.sh.
# With HAG_CANARY_INPUT_GAINS=1 (syllables), the gain of each group of input features is searched too, and with
# HAG_CANARY_LEAK=1 (syllables, next_phrase), the leak rate of each reservoir.
# Usage: [HAG_CANARY_TASK=syllables|next_phrase] [HAG_CANARY_INPUT_GAINS=1] [HAG_CANARY_LEAK=1] [PURE="..."]
#        [SIZES="..."] bash slurm/submit_canary.sh
set -euo pipefail
NB_SEEDS=${NB_SEEDS:-10}
MAX_RUNNING=${MAX_RUNNING:-12}
read -r -a SIZES <<< "${SIZES:-250 500 1000}"
# models (space-separated list, PURE environment variable)
read -r -a PURE <<< "${PURE:-mean_hag var_hag random_ee random_ee_dense}"
NODES=(--constraint="zen4|zen3|zonda" --exclude="sirocco[21-25],enbata[01-02]")
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks outputs/canary/studies

# the MFCC caches, once, before the jobs (canary/datasets/Canary/boundary_features_44100.pkl, and ..._no_floor.pkl of the
# next-phrase prediction)
set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
PYTHONPATH=$PWD:${HAG_DIR:-$HOME/hag_core} python -c "from canary.common import load_dataset; load_dataset(); load_dataset(top_db=None)" > /dev/null

stamp=$(date +%Y%m%d_%H%M%S)
PURE_TASKS=slurm/tasks/canary_pure_$stamp.txt
for size in "${SIZES[@]}"; do for model in "${PURE[@]}"; do echo "$model $size"; done; done > "$PURE_TASKS"

n=$(wc -l < "$PURE_TASKS")
pure=$(sbatch --parsable "${NODES[@]}" --array=0-$((n - 1))%$MAX_RUNNING --job-name=canary-pure \
    --export=ALL,HAG_CANARY_TASKS="$PURE_TASKS" slurm/canary.sbatch)
echo "pure models: job array $pure ($n studies, $PURE_TASKS)"

if [ "${HAG_CANARY_TASK:-boundaries}" = syllables ]; then
    # syllable classification: one job per condition, submitted once the studies exist (slurm/submit_syllables.sh)
    evaluation=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=1 --mem=4G --time=00:30:00 --dependency=afterok:$pure \
        --job-name=submit-syllables --output=slurm/logs/%x-%j.out --wrap="bash slurm/submit_syllables.sh $(IFS=,; echo "${SIZES[*]}") $NB_SEEDS")
    echo "final evaluation (syllable classification, one job per condition): submitted by job $evaluation, after $pure"
elif [ "${HAG_CANARY_TASK:-boundaries}" = next_phrase ]; then
    evaluation=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=1 --mem=4G --time=00:30:00 --dependency=afterok:$pure \
        --job-name=submit-next-phrase --output=slurm/logs/%x-%j.out --wrap="bash slurm/submit_next_phrase.sh $(IFS=,; echo "${SIZES[*]}") $NB_SEEDS")
    echo "final evaluation (next phrase, one job per condition): submitted by job $evaluation, after $pure"
else
    evaluation=$(sbatch --parsable "${NODES[@]}" --job-name=canary-evaluate --dependency=afterok:$pure \
        --export=ALL,HAG_CANARY_MODE=evaluate,HAG_CANARY_WORKERS=1 --output=slurm/logs/%x-%j.out \
        slurm/canary.sbatch "$(IFS=,; echo "${SIZES[*]}")" "$NB_SEEDS")
    echo "final evaluation: job $evaluation, after $pure"
fi
