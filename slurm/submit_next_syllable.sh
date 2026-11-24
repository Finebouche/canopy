#!/bin/bash
# Control of the next phrase on PlaFRIM: the reservoirs that read one syllable per step
# (canopy.studies.next_syllable), its three variants (VARIANTS: the readout of the syllables scored raw, "raw", or
# knowing that the phrase changes, "change_known", and the readout of the phrases, "phrases"), with and without leak
# (LEAKS): one job per study of the pure models at each size (one job array, MAX_RUNNING at once), then the merge of their
# parts into outputs/next_syllable/hpo/marron1_db.sqlite3 and the final evaluation of each variant and leak on the test
# songs (NB_SEEDS seeds). Run from the root of a checkout of canopy with the MFCC cache, the annotations and
# outputs/next_syllable/analysis/syllable_counts.csv (canopy.studies.next_syllable.segmentation):
#   [VARIANTS="raw change_known phrases"] [LEAKS="0 1"] [PURE="..."] [SIZES="1000 500"] [NB_SEEDS=10]
#       [MAX_RUNNING=12] [CONSTRAINT="zonda|suroit"] bash slurm/submit_next_syllable.sh
# A study (about 250 steps per song) takes 1 to 2 h on 16 cores with 4 workers.
set -euo pipefail
NB_SEEDS=${NB_SEEDS:-10}
read -r -a VARIANTS <<< "${VARIANTS:-raw change_known phrases}"
read -r -a LEAKS <<< "${LEAKS:-0 1}"
read -r -a SIZES <<< "${SIZES:-1000 500}"
read -r -a PURE <<< "${PURE:-mean_hag var_hag random_ee random_ei}"
MAX_RUNNING=${MAX_RUNNING:-12}
NODES=(--constraint="${CONSTRAINT:-zonda|suroit}")  # CPU nodes, no GPU nor big memory
RESOURCES=(--cpus-per-task=16 --mem=48G --time=1-00:00:00)
cd "$(dirname "$0")/.."
[ -f outputs/next_syllable/analysis/syllable_counts.csv ] || { echo "syllable_counts.csv missing" >&2; exit 1; }
mkdir -p slurm/logs slurm/tasks

variables() {  # the variables of a variant and a leak
    case "$1" in
        raw) echo "CANOPY_SYLLABLE_READOUT=syllables CANOPY_SYLLABLE_SCORE=raw CANOPY_LEAKY_RESERVOIRS=$2" ;;
        change_known) echo "CANOPY_SYLLABLE_READOUT=syllables CANOPY_SYLLABLE_SCORE=change_known CANOPY_LEAKY_RESERVOIRS=$2" ;;
        phrases) echo "CANOPY_SYLLABLE_READOUT=phrases CANOPY_SYLLABLE_SCORE=raw CANOPY_LEAKY_RESERVOIRS=$2" ;;
        *) echo "unknown variant $1" >&2; exit 1 ;;
    esac
}

TASKS=slurm/tasks/next_syllable_$(date +%Y%m%d_%H%M%S).txt
for size in "${SIZES[@]}"; do for model in "${PURE[@]}"; do for variant in "${VARIANTS[@]}"; do for leak in "${LEAKS[@]}"; do
    echo "$model $size CANOPY_WITH_CALLS=1 $(variables "$variant" "$leak")"
done; done; done; done > "$TASKS"
n=$(wc -l < "$TASKS")
studies=$(sbatch --parsable "${NODES[@]}" "${RESOURCES[@]}" --array=0-$((n - 1))%$MAX_RUNNING --job-name=syllable-studies \
    --export=ALL,CANOPY_TASK=next_syllable,CANOPY_TASK_FILE="$TASKS",CANOPY_WORKERS=4 slurm/canary.sbatch)
echo "studies: job array $studies ($n studies, $MAX_RUNNING at once, $TASKS)"
merge=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=1 --mem=4G --time=00:30:00 --dependency=afterany:$studies \
    --job-name=syllable-merge --output=slurm/logs/%x-%j.out --wrap="bash slurm/merge_studies.sh next_syllable")
echo "merge of the studies: job $merge (after $studies)"
for variant in "${VARIANTS[@]}"; do for leak in "${LEAKS[@]}"; do
    exports="ALL,CANOPY_JOB_MODE=next_syllable,CANOPY_WORKERS=1,CANOPY_WITH_CALLS=1,$(variables "$variant" "$leak" | tr ' ' ',')"
    evaluation=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=16 --mem=64G --time=1-00:00:00 \
        --dependency=afterok:$merge --job-name=syllable-evaluation --export="$exports" --output=slurm/logs/%x-%j.out \
        slurm/canary.sbatch "$(IFS=,; echo "${SIZES[*]}")" "$NB_SEEDS")
    echo "final evaluation $variant, leak $leak: job $evaluation (after $merge)"
done; done
